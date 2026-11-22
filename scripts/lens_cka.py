"""Layer-by-layer CKA (linear or Gaussian/RBF kernel) of workspace-lens vectors (cf. Fig. 27 of Gurnee et al.,
"Verbalizable representations form a global workspace").

For each lens and source layer l, the lens vectors are the rows of X_l = W_U @ J_l
(one d_model vector per vocabulary token; the repo's ``jlens_vectors`` convention,
pre-final-norm). For every pair of layers we compute linear CKA with tokens as
samples over the full vocabulary:

    CKA(l, m) = ||X~_l^T X~_m||_F^2 / (||X~_l^T X~_l||_F ||X~_m^T X~_m||_F)

with X~ centred over tokens. This is computed exactly without materialising X_l:
X~_l^T X~_m = J_l^T C J_m with C = (W_U - mu)^T (W_U - mu) = L L^T, so
||J_l^T C J_m||_F^2 = <P_l, P_m> with P_l = L^T J_l J_l^T L, and CKA is the cosine
between the flattened P_l.

With ``--kernel rbf`` the similarity between lens vectors is a Gaussian kernel
K_ij = exp(-||x_i - x_j||^2 / (2 sigma^2)), with sigma^2 = s^2 * median_{i!=j} ||x_i - x_j||^2
(s = ``--sigma``, chosen per layer; Kornblith et al. 2019). This needs explicit
token x token Gram matrices, so it runs on a fixed random subsample of ``--n-tokens``
vocabulary tokens (the same tokens for every layer and lens). CKA is then the cosine
between the centred Gram matrices HKH. Outputs are prefixed ``cka_rbf`` instead of ``cka``.

Usage:
    python scripts/lens_cka.py                         # linear kernel, full vocab
    python scripts/lens_cka.py --kernel rbf            # Gaussian kernel, 8192-token subsample
    python scripts/lens_cka.py --kernel rbf --n-tokens 16384 --sigma 0.5
    python scripts/lens_cka.py --self-test             # check both kernels against reference code
    python scripts/lens_cka.py --plot-only [--kernel rbf]   # redraw from saved npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch as t

# Allow running without the package installed (e.g. from a different venv).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

LENS_LABELS = {"j_lens": "J-Lens", "r_lens": "R-Lens", "jpp_lens": "J++ Lens"}


### CKA CORE


def centred_unembed_covariance(W_VN: t.Tensor, device: str, chunk: int = 16384) -> t.Tensor:
    """C = (W - mu)^T (W - mu), [N, N] fp64, accumulated over token chunks."""
    V, N = W_VN.shape
    S = t.zeros(N, N, dtype=t.float64, device=device)
    s = t.zeros(N, dtype=t.float64, device=device)
    for i in range(0, V, chunk):
        w = W_VN[i : i + chunk].to(device=device, dtype=t.float32)
        S += (w.T @ w).double()
        s += w.sum(0).double()
    mu = s / V
    return S - V * t.outer(mu, mu)


def psd_sqrt_factor(C: t.Tensor) -> t.Tensor:
    """L with C = L L^T (eigendecomposition, negative eigenvalues clamped to 0)."""
    evals, evecs = t.linalg.eigh(C.double().cpu())
    return (evecs * evals.clamp_min(0).sqrt()).to(t.float32)


def linear_cka_matrix(
    jacobians: dict[int, t.Tensor], layers: list[int], L_NN: t.Tensor, device: str
) -> np.ndarray:
    """[len(layers), len(layers)] linear CKA between X_l = W_U @ J_l over all tokens."""
    L = L_NN.to(device)
    P_flat = []
    for layer in layers:
        J = jacobians[layer].to(device=device, dtype=t.float32)  # [F, N]; X = W_U @ J
        # X~^T X~ uses J^T C J (J is [F,N], X = W J -> X^T X = J^T W^T W J).
        A = L.T @ J  # [N, N]
        P = A @ A.T  # L^T J J^T L, symmetric
        P_flat.append(P.reshape(-1))
        del J, A
    return cosine_gram(P_flat)


def cosine_gram(flat: list[t.Tensor], step: int = 1 << 20) -> np.ndarray:
    """Pairwise cosine similarity between equal-length flat vectors, accumulated in fp64
    over column chunks (so the vectors are never stacked into one big tensor)."""
    n = len(flat)
    G = t.zeros(n, n, dtype=t.float64, device=flat[0].device)
    for i in range(0, flat[0].numel(), step):
        f = t.stack([v[i : i + step] for v in flat]).double()
        G += f @ f.T
    d = G.diag().sqrt()
    return (G / t.outer(d, d)).cpu().numpy()


def centred_rbf_gram(X: t.Tensor, sigma: float) -> tuple[t.Tensor, float]:
    """H K H for the Gaussian kernel on rows of X [n, d]. Bandwidth:
    2 sigma_eff^2 = 2 * sigma^2 * median off-diagonal squared distance.
    Returns (centred Gram [n, n] fp32, median squared distance)."""
    X = X.float()
    sq = X.pow(2).sum(1)
    D = (sq[:, None] + sq[None, :] - 2 * (X @ X.T)).clamp_min_(0)
    D.fill_diagonal_(0)
    n = D.shape[0]
    med = float(D[~t.eye(n, dtype=t.bool, device=D.device)].median())
    K = t.exp_(D.mul_(-1.0 / (2 * sigma**2 * med)))  # in place: D becomes K
    K -= K.mean(0, keepdim=True)
    K -= K.mean(1, keepdim=True)
    return K, med


def rbf_cka_matrix(
    jacobians: dict[int, t.Tensor], layers: list[int], W_sub: t.Tensor, sigma: float, device: str
) -> tuple[np.ndarray, list[float]]:
    """[len(layers), len(layers)] RBF-kernel CKA between X_l = W_sub @ J_l
    (W_sub: a token subsample of W_U). Also returns each layer's median squared distance."""
    Ws = W_sub.to(device=device, dtype=t.float32)
    grams, meds = [], []
    for layer in layers:
        X = Ws @ jacobians[layer].to(device=device, dtype=t.float32)  # [n, N]
        Kc, med = centred_rbf_gram(X, sigma)
        grams.append(Kc.reshape(-1))
        meds.append(med)
        del X
    M = cosine_gram(grams)
    del grams
    return M, meds


def explicit_linear_cka(X: t.Tensor, Y: t.Tensor) -> float:
    """Reference linear CKA on explicit [n, d] representations (feature-space form)."""
    X = (X - X.mean(0)).double()
    Y = (Y - Y.mean(0)).double()
    hsic = (X.T @ Y).pow(2).sum()
    return float(hsic / ((X.T @ X).norm() * (Y.T @ Y).norm()))


def explicit_rbf_cka(X: np.ndarray, Y: np.ndarray, sigma: float) -> float:
    """Reference RBF CKA (biased HSIC, trace form) in numpy fp64."""

    def gram(Z):
        D = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
        med = np.median(D[~np.eye(len(Z), dtype=bool)])
        return np.exp(-D / (2 * sigma**2 * med))

    n = len(X)
    H = np.eye(n) - 1.0 / n
    K, L = gram(X), gram(Y)
    hsic = lambda A, B: np.trace(A @ H @ B @ H)
    return float(hsic(K, L) / np.sqrt(hsic(K, K) * hsic(L, L)))


### IO


def load_unembedding(model_id: str) -> t.Tensor:
    """Read only ``lm_head.weight`` from the cached safetensors shards."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    index_path = hf_hub_download(model_id, "model.safetensors.index.json")
    weight_map = json.loads(Path(index_path).read_text())["weight_map"]
    shard = hf_hub_download(model_id, weight_map["lm_head.weight"])
    with safe_open(shard, framework="pt") as f:
        return f.get_tensor("lm_head.weight")


def load_jacobians(path: Path, model_id: str) -> tuple[dict[int, t.Tensor], list[int]]:
    from workspace_lens.utils import load_lens_file

    lens = load_lens_file(str(path), hf_model_name=model_id)
    assert lens.jacobians_L_dict_FN is not None
    return lens.jacobians_L_dict_FN, list(lens.source_layers)


### PLOTTING


def plot_heatmaps(
    results: dict[str, np.ndarray],
    layers: list[int],
    model_id: str,
    out: Path,
    prefix: str = "cka",
    cbar_label: str = "Linear CKA",
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = matplotlib.colormaps["viridis"]
    tick_idx = [i for i, l in enumerate(layers) if l % 5 == 0 or i == len(layers) - 1]
    tick_lab = [str(layers[i]) for i in tick_idx]

    def style(ax, title):
        ax.set_title(title, fontsize=12, color="#222")
        ax.set_xticks(tick_idx, tick_lab, fontsize=8, color="#555")
        ax.set_yticks(tick_idx, tick_lab, fontsize=8, color="#555")
        ax.set_xlabel("Layer", fontsize=9, color="#555")
        ax.set_ylabel("Layer", fontsize=9, color="#555")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n + 0.8, 4.8), constrained_layout=True)
    axes = np.atleast_1d(axes)
    im = None
    for ax, (name, M) in zip(axes, results.items()):
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, origin="lower", interpolation="nearest")
        style(ax, LENS_LABELS.get(name, name))
    cb = fig.colorbar(im, ax=list(axes), shrink=0.85, pad=0.02)
    cb.set_label(cbar_label, fontsize=9, color="#555")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors="#555", length=0)
    fig.suptitle(f"Layer-by-layer {cbar_label} of lens vectors (rows of W_U·J_l), {model_id}", fontsize=13)
    fig.savefig(out / f"{prefix}_heatmaps.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(out / f"{prefix}_heatmaps.pdf", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    for name, M in results.items():
        fig, ax = plt.subplots(figsize=(5.6, 4.8), constrained_layout=True)
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, origin="lower", interpolation="nearest")
        style(ax, f"{LENS_LABELS.get(name, name)}: layer-by-layer {cbar_label}")
        cb = fig.colorbar(im, ax=ax, shrink=0.85)
        cb.set_label(cbar_label, fontsize=9, color="#555")
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=8, colors="#555", length=0)
        fig.savefig(out / f"{prefix}_{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)


### CHECKS


def self_test(device: str) -> None:
    g = t.Generator().manual_seed(0)
    V, N, nl = 500, 32, 4
    W = t.randn(V, N, generator=g) + 0.5  # non-zero mean so centring matters
    Js = {i: t.randn(N, N, generator=g) * (1 + i) + t.eye(N) * i for i in range(nl)}
    L = psd_sqrt_factor(centred_unembed_covariance(W, device))
    fast = linear_cka_matrix(Js, list(range(nl)), L, device)
    ref = np.array([[explicit_linear_cka(W @ Js[a], W @ Js[b]) for b in range(nl)] for a in range(nl)])
    err = np.abs(fast - ref).max()
    print(f"self-test linear: max |fast - explicit| = {err:.2e}")
    assert err < 1e-5, (fast, ref)

    # RBF: fast GPU path vs numpy trace-form reference, on a token subsample.
    Wn = W[:200]
    for sigma in (0.5, 1.0):
        fast_r, _ = rbf_cka_matrix(Js, list(range(nl)), Wn, sigma, device)
        Xs = {i: (Wn @ Js[i]).double().numpy() for i in range(nl)}
        ref_r = np.array([[explicit_rbf_cka(Xs[a], Xs[b], sigma) for b in range(nl)] for a in range(nl)])
        err_r = np.abs(fast_r - ref_r).max()
        print(f"self-test rbf (sigma={sigma}): max |fast - explicit| = {err_r:.2e}")
        assert err_r < 1e-4, (fast_r, ref_r)
    # Very wide bandwidth: RBF CKA -> linear CKA on the same tokens.
    wide, _ = rbf_cka_matrix(Js, list(range(nl)), Wn, 300.0, device)
    lin = np.array([[explicit_linear_cka(Wn @ Js[a], Wn @ Js[b]) for b in range(nl)] for a in range(nl)])
    err_w = np.abs(wide - lin).max()
    print(f"self-test rbf (sigma=300) vs linear on same tokens: max diff = {err_w:.2e}")
    assert err_w < 1e-3, (wide, lin)
    print("self-test passed")


def spot_check(W, jacobians, layers, M, device, n_tokens=20000, check_layers=(0, 15, 30)):
    g = t.Generator().manual_seed(0)
    idx = t.randperm(W.shape[0], generator=g)[:n_tokens]
    Ws = W[idx].to(device=device, dtype=t.float32)
    ls = [l for l in check_layers if l in layers]
    X = {l: Ws @ jacobians[l].to(device=device, dtype=t.float32) for l in ls}
    out = {}
    for a in ls:
        for b in ls:
            if a < b:
                sub = explicit_linear_cka(X[a], X[b])
                full = float(M[layers.index(a), layers.index(b)])
                out[f"{a}-{b}"] = {"subsample_20k": sub, "full_vocab": full}
                print(f"  spot-check layers {a:>2}-{b:<2}: full={full:.4f}  20k-subsample={sub:.4f}")
    return out


### MAIN


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lens-dir", default="lens-params/qwen3.5-9b")
    p.add_argument("--lenses", nargs="+", default=["j_lens", "r_lens", "jpp_lens"])
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--out", default="results/cka/qwen3.5-9b")
    p.add_argument("--device", default="cuda" if t.cuda.is_available() else "cpu")
    p.add_argument("--kernel", choices=["linear", "rbf"], default="linear")
    p.add_argument("--n-tokens", type=int, default=8192, help="rbf only: vocab tokens to subsample")
    p.add_argument("--sigma", type=float, default=1.0, help="rbf only: bandwidth multiplier on median distance")
    p.add_argument("--seed", type=int, default=0, help="rbf only: token-subsample seed")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--no-spot-check", action="store_true")
    p.add_argument("--plot-only", action="store_true", help="re-plot from saved <prefix>_all.npz")
    args = p.parse_args()

    t.backends.cuda.matmul.allow_tf32 = False
    t.backends.cudnn.allow_tf32 = False

    if args.self_test:
        self_test(args.device)
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rbf = args.kernel == "rbf"
    prefix = "cka_rbf" if rbf else "cka"
    cbar_label = f"RBF CKA (σ = {args.sigma:g}×median dist)" if rbf else "Linear CKA"

    if args.plot_only:
        data = np.load(out / f"{prefix}_all.npz")
        results = {n: data[n] for n in args.lenses if n in data}
        if rbf and "sigma" in data:
            cbar_label = f"RBF CKA (σ = {float(data['sigma']):g}×median dist)"
        plot_heatmaps(results, data["layers"].tolist(), args.model, out, prefix, cbar_label)
        print(f"Re-plotted to {out}/")
        return

    t0 = time.time()
    W = load_unembedding(args.model)
    print(f"W_U {tuple(W.shape)} {W.dtype}  ({time.time() - t0:.1f}s)")
    summary: dict = {
        "model": args.model,
        "kernel": args.kernel,
        "d_model": int(W.shape[1]),
        "lens_vectors": "W_U @ J_l (pre-final-norm; repo jlens_vectors convention)",
        "lenses": {},
    }
    if rbf:
        g = t.Generator().manual_seed(args.seed)
        token_idx = t.randperm(W.shape[0], generator=g)[: args.n_tokens].sort().values
        W_sub = W[token_idx]
        summary.update(
            samples=f"random subsample of {len(token_idx)} vocabulary tokens (same for all layers/lenses)",
            n_tokens=len(token_idx),
            vocab_size=int(W.shape[0]),
            seed=args.seed,
            sigma=args.sigma,
            bandwidth="2*sigma_eff^2 = 2*sigma^2*median_{i!=j}||x_i-x_j||^2, per layer",
        )
    else:
        C = centred_unembed_covariance(W, args.device)
        L = psd_sqrt_factor(C)
        del C
        print(f"centred covariance + factor done ({time.time() - t0:.1f}s)")
        summary.update(
            samples="all vocabulary tokens (rows of W_U @ J_l), centred over tokens",
            n_tokens=int(W.shape[0]),
        )

    results: dict[str, np.ndarray] = {}
    layers_ref = None
    for name in args.lenses:
        path = Path(args.lens_dir) / f"{name}.pt"
        jac, layers = load_jacobians(path, args.model)
        layers_ref = layers_ref or layers
        assert layers == layers_ref, f"{name}: layers differ from first lens"
        if rbf:
            M, meds = rbf_cka_matrix(jac, layers, W_sub, args.sigma, args.device)
        else:
            M = linear_cka_matrix(jac, layers, L, args.device)
        M = (M + M.T) / 2
        results[name] = M
        print(f"\n== {LENS_LABELS.get(name, name)} ({path})  ({time.time() - t0:.1f}s)")
        with np.printoptions(precision=2, suppress=True, linewidth=250):
            print(M)
        off = M[~np.eye(len(layers), dtype=bool)]
        info = {
            "file": str(path),
            "mean_offdiag_cka": float(off.mean()),
            "min_cka": float(M.min()),
            "adjacent_layer_cka": [float(M[i, i + 1]) for i in range(len(layers) - 1)],
            "max_abs_diag_minus_1": float(np.abs(np.diag(M) - 1).max()),
        }
        if rbf:
            info["median_sq_dist_per_layer"] = meds
        if name == "j_lens" and not rbf and not args.no_spot_check:
            info["spot_check"] = spot_check(W, jac, layers, M, args.device)
        summary["lenses"][name] = info
        print(f"  mean off-diagonal CKA = {info['mean_offdiag_cka']:.3f}, min = {info['min_cka']:.3f}")

        header = "layer," + ",".join(str(l) for l in layers)
        rows = [f"{l}," + ",".join(f"{v:.6f}" for v in M[i]) for i, l in enumerate(layers)]
        (out / f"{prefix}_{name}.csv").write_text(header + "\n" + "\n".join(rows) + "\n")
        del jac
        if args.device == "cuda":
            t.cuda.empty_cache()

    summary["layers"] = layers_ref
    extra = {"token_idx": token_idx.numpy(), "sigma": np.array(args.sigma)} if rbf else {}
    np.savez(out / f"{prefix}_all.npz", layers=np.array(layers_ref), **extra, **results)
    (out / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2))
    plot_heatmaps(results, layers_ref, args.model, out, prefix, cbar_label)
    print(f"\nSaved to {out}/  ({time.time() - t0:.1f}s total)")


if __name__ == "__main__":
    main()
