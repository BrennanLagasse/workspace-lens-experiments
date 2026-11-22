"""Optimal contiguous segmentation of layer-by-layer CKA matrices (Approach 1).

Tests the sensory / workspace / motor phase claim of Gurnee et al. ("Verbalizable
representations form a global workspace") on the lens-vector CKA matrices written by
``scripts/lens_cka.py``. Layers are ordered, so instead of generic clustering we split
layers 0..L-1 into K contiguous blocks and find the globally optimal split by dynamic
programming.

Block cost (``--cost``):
  sse     sum over pairs i<j inside the block of (M_ij - block mean)^2  (default;
          a piecewise-constant model of the block's diagonal square)
  dissim  sum over pairs i<j inside the block of (1 - M_ij)
The diagonal (CKA = 1 by construction) is excluded, so a single-layer block costs 0.
Caveat: both costs sum over ~n^2 within-block pairs, which biases the split towards
equal-sized blocks. For K <= 4 we therefore also brute-force the boundaries that
minimise the full block-model RSS (below) and report/overlay them.

For every K = 1..K_max the DP gives the exact optimum in O(K L^2). Choosing K:
  * elbow: the optimal DP cost against K;
  * BIC: the DP boundaries define a full block model of M (one mean per pair of
    blocks, including the off-diagonal blocks), fitted to all L(L-1)/2 upper-triangle
    entries.  BIC = n log(RSS/n) + p log n, p = K(K+1)/2 means + (K-1) boundaries.
    Entries of M are not independent, so treat BIC as a relative ranking, not a test.
  * reference: a phase-free "distance decay" model (one mean per lag |i-j|, L-1
    parameters). A phase model that cannot beat it on RSS is weak evidence for phases.

Outputs go to ``<out>/segmentation/`` with prefix ``seg_<kernel>``; the CKA heatmaps
and CSVs written by lens_cka.py are left untouched.

Usage:
    python scripts/cka_segmentation.py                    # linear CKA
    python scripts/cka_segmentation.py --kernel rbf
    python scripts/cka_segmentation.py --k-max 8 --cost dissim
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LENS_LABELS = {"j_lens": "J-Lens", "r_lens": "R-Lens", "jpp_lens": "J++ Lens"}
LENS_COLORS = {"j_lens": "#2a78d6", "r_lens": "#eb6834", "jpp_lens": "#1baf7a"}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"


### SEGMENTATION


def block_costs(M: np.ndarray, cost: str) -> np.ndarray:
    """C[a, b] = cost of the block of layers a..b inclusive (inf for a > b)."""
    L = len(M)
    C = np.full((L, L), np.inf)
    iu = np.triu_indices(L, 1)
    for a in range(L):
        for b in range(a, L):
            if a == b:
                C[a, b] = 0.0
                continue
            sub = M[a : b + 1, a : b + 1]
            v = sub[np.triu_indices(b - a + 1, 1)]
            C[a, b] = ((v - v.mean()) ** 2).sum() if cost == "sse" else (1 - v).sum()
    del iu
    return C


def optimal_segmentation(C: np.ndarray, k_max: int, min_size: int = 1):
    """Exact DP. Returns {K: (cost, [start index of each block])}."""
    L = len(C)
    INF = np.inf
    # D[k, e] = best cost of splitting layers 0..e-1 into k blocks; back[k, e] = start of last block
    D = np.full((k_max + 1, L + 1), INF)
    back = np.zeros((k_max + 1, L + 1), dtype=int)
    D[0, 0] = 0.0
    for k in range(1, k_max + 1):
        for e in range(k * min_size, L + 1):
            best, arg = INF, -1
            for s in range((k - 1) * min_size, e - min_size + 1):
                c = D[k - 1, s] + C[s, e - 1]
                if c < best:
                    best, arg = c, s
            D[k, e], back[k, e] = best, arg
    out = {}
    for k in range(1, k_max + 1):
        if not np.isfinite(D[k, L]):
            continue
        starts, e = [], L
        for kk in range(k, 0, -1):
            s = back[kk, e]
            starts.append(int(s))
            e = s
        out[k] = (float(D[k, L]), starts[::-1])
    return out


def labels_from_starts(starts: list[int], L: int) -> np.ndarray:
    lab = np.zeros(L, dtype=int)
    for i, s in enumerate(starts):
        lab[s:] = i
    return lab


def full_block_rss(M: np.ndarray, starts: list[int]) -> tuple[float, int]:
    """RSS of the piecewise-constant block model over all upper-triangle entries,
    and its number of free means (block pairs that contain at least one entry)."""
    L = len(M)
    lab = labels_from_starts(starts, L)
    iu = np.triu_indices(L, 1)
    v, gi, gj = M[iu], lab[iu[0]], lab[iu[1]]
    rss, n_means = 0.0, 0
    for p in range(len(starts)):
        for q in range(p, len(starts)):
            m = (gi == p) & (gj == q)
            if m.any():
                rss += ((v[m] - v[m].mean()) ** 2).sum()
                n_means += 1
    return float(rss), n_means


def exhaustive_full_block(M: np.ndarray, k: int, min_size: int = 1) -> tuple[float, list[int]]:
    """Brute-force the boundaries that minimise the *full* block-model RSS (all block
    pairs, not just the diagonal squares). Feasible for small K (C(L-1, K-1) splits)."""
    from itertools import combinations

    L = len(M)
    best = (np.inf, None)
    for cuts in combinations(range(1, L), k - 1):
        starts = [0, *cuts]
        if min(np.diff(starts + [L])) < min_size:
            continue
        rss, _ = full_block_rss(M, starts)
        if rss < best[0]:
            best = (rss, starts)
    return best


def lag_model_rss(M: np.ndarray) -> tuple[float, int]:
    """Phase-free reference: M_ij ~ f(|i - j|), one mean per lag."""
    L = len(M)
    rss = 0.0
    for d in range(1, L):
        v = np.diagonal(M, d)
        rss += ((v - v.mean()) ** 2).sum()
    return float(rss), L - 1


def bic(rss: float, n: int, p: int) -> float:
    return n * np.log(max(rss, 1e-300) / n) + p * np.log(n)


def analyse(M: np.ndarray, k_max: int, cost: str, min_size: int, exhaustive_k_max: int = 4) -> dict:
    L = len(M)
    n = L * (L - 1) // 2
    segs = optimal_segmentation(block_costs(M, cost), k_max, min_size)
    rows = []
    for k, (c, starts) in segs.items():
        rss, n_means = full_block_rss(M, starts)
        p = n_means + (k - 1)
        rows.append(
            dict(K=k, dp_cost=c, starts=starts, sizes=np.diff(starts + [L]).tolist(),
                 full_rss=rss, n_params=p, bic=bic(rss, n, p))
        )
    for r in rows:
        if r["K"] <= exhaustive_k_max and r["K"] > 1:
            rss, starts = exhaustive_full_block(M, r["K"], min_size)
            r["exhaustive_full"] = dict(starts=starts, sizes=np.diff(starts + [L]).tolist(), full_rss=rss,
                                        bic=bic(rss, n, r["n_params"]))
    lag_rss, lag_p = lag_model_rss(M)
    best = min(rows, key=lambda r: r["bic"])
    return dict(
        segmentations=rows,
        best_bic_K=best["K"],
        lag_model=dict(rss=lag_rss, n_params=lag_p, bic=bic(lag_rss, n, lag_p)),
        n_entries=n,
    )


### PLOTTING


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c3")
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(axis="y", color="#ebeae6", lw=0.8)
    ax.set_axisbelow(True)


def _draw_blocks(ax, starts, L, color="white", lw=1.6, ls="-"):
    """Outline the diagonal blocks on an imshow (origin='lower', cell centres at ints)."""
    from matplotlib.patches import Rectangle

    ends = starts[1:] + [L]
    for s, e in zip(starts, ends):
        ax.add_patch(Rectangle((s - 0.5, s - 0.5), e - s, e - s, fill=False, ec=color, lw=lw, ls=ls))


def _block_label(starts, layers):
    L = len(layers)
    ends = starts[1:] + [L]
    return " | ".join(f"{layers[s]}–{layers[e - 1]}" for s, e in zip(starts, ends))


def plot_all(results: dict, mats: dict, layers: list[int], model: str, kernel_label: str,
             out: Path, prefix: str, k_show: list[int]):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = len(layers)
    names = list(results)
    tick_idx = [i for i, l in enumerate(layers) if l % 5 == 0 or i == L - 1]
    tick_lab = [str(layers[i]) for i in tick_idx]

    # 1) heatmaps with optimal boundaries: rows = K values, cols = lenses
    rows_k = k_show
    fig, axes = plt.subplots(len(rows_k), len(names), figsize=(4.2 * len(names) + 0.8, 4.1 * len(rows_k)),
                             constrained_layout=True, squeeze=False)
    im = None
    for r, k in enumerate(rows_k):
        for c, name in enumerate(names):
            ax = axes[r, c]
            im = ax.imshow(mats[name], cmap="viridis", vmin=0, vmax=1, origin="lower", interpolation="nearest")
            seg = next(s for s in results[name]["segmentations"] if s["K"] == k)
            _draw_blocks(ax, seg["starts"], L)
            alt = ""
            if "exhaustive_full" in seg and seg["exhaustive_full"]["starts"] != seg["starts"]:
                _draw_blocks(ax, seg["exhaustive_full"]["starts"], L, color="#eb6834", lw=1.4, ls="--")
                alt = f"\nfull-model optimum (orange): {_block_label(seg['exhaustive_full']['starts'], layers)}"
            star = " ★ best BIC" if k == results[name]["best_bic_K"] else ""
            ax.set_title(f"{LENS_LABELS.get(name, name)}, K = {k}{star}\n{_block_label(seg['starts'], layers)}{alt}",
                         fontsize=9.5, color=INK)
            ax.set_xticks(tick_idx, tick_lab, fontsize=8, color=INK2)
            ax.set_yticks(tick_idx, tick_lab, fontsize=8, color=INK2)
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            if r == len(rows_k) - 1:
                ax.set_xlabel("Layer", fontsize=9, color=INK2)
            if c == 0:
                ax.set_ylabel("Layer", fontsize=9, color=INK2)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
    cb.set_label(kernel_label, fontsize=9, color=INK2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=INK2, length=0)
    fig.suptitle(f"Optimal contiguous segmentation of {kernel_label} (white boxes), {model}", fontsize=13)
    fig.savefig(out / f"{prefix}_heatmaps.png", dpi=180, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(out / f"{prefix}_heatmaps.pdf", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    # 2) model selection: DP cost (elbow) and BIC vs K, one line per lens
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 4.0), constrained_layout=True)
    for name in names:
        segs = results[name]["segmentations"]
        ks = [s["K"] for s in segs]
        col = LENS_COLORS.get(name)
        lab = LENS_LABELS.get(name, name)
        dp = np.array([s["dp_cost"] for s in segs])
        a1.plot(ks, dp / dp[0], "-o", color=col, lw=2, ms=5, label=lab)
        rss = np.array([s["full_rss"] for s in segs])
        a2.plot(ks, rss / rss[0], "-o", color=col, lw=2, ms=5, label=lab)
        a2.axhline(results[name]["lag_model"]["rss"] / rss[0], color=col, lw=1.2, ls="--")
        b = np.array([s["bic"] for s in segs])
        a3.plot(ks, b - b.min(), "-o", color=col, lw=2, ms=5, label=lab)
        kb = results[name]["best_bic_K"]
        a3.plot([kb], [0], "o", ms=11, mfc="none", mec=col, mew=2)
    for ax, t, yl in (
        (a1, "Within-block DP cost (elbow)", "cost / cost(K=1)"),
        (a2, "Full block-model RSS\n(dashed: phase-free lag model, L−1 params)", "RSS / RSS(K=1)"),
        (a3, "BIC of full block model (circle = minimum)", "BIC − min BIC"),
    ):
        _style(ax)
        ax.set_title(t, fontsize=10, color=INK)
        ax.set_xlabel("Number of blocks K", fontsize=9, color=INK2)
        ax.set_ylabel(yl, fontsize=9, color=INK2)
        ax.set_xticks(range(1, max(len(results[n]["segmentations"]) for n in names) + 1))
    a1.legend(frameon=False, fontsize=9)
    fig.suptitle(f"Choosing K — {kernel_label}, {model}", fontsize=12)
    fig.savefig(out / f"{prefix}_model_selection.png", dpi=180, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    # 3) boundary stability: for each K, the optimal blocks drawn as a row of segments
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 3.8), constrained_layout=True, squeeze=False)
    for ax, name in zip(axes[0], names):
        segs = results[name]["segmentations"]
        for s in segs:
            k, starts = s["K"], s["starts"]
            ends = starts[1:] + [L]
            for i, (a, e) in enumerate(zip(starts, ends)):
                shade = "#cde2fb" if i % 2 == 0 else "#86b6ef"
                ax.barh(k, e - a - 0.12, left=a - 0.5 + 0.06, height=0.72, color=shade, lw=0)
            for a in starts[1:]:
                ax.plot([a - 0.5, a - 0.5], [k - 0.42, k + 0.42], color=INK, lw=1.6)
        kb = results[name]["best_bic_K"]
        ax.text(L - 0.3, kb, " ◀ BIC", va="center", fontsize=8, color=INK2)
        ax.set_xlim(-0.6, L + 2.2)
        ax.set_ylim(0.4, len(segs) + 0.6)
        ax.invert_yaxis()
        ax.set_yticks([s["K"] for s in segs])
        ax.set_xticks(tick_idx, tick_lab)
        _style(ax)
        ax.grid(False)
        ax.set_title(LENS_LABELS.get(name, name), fontsize=11, color=INK)
        ax.set_xlabel("Layer (black ticks = block boundaries)", fontsize=9, color=INK2)
        ax.set_ylabel("K", fontsize=9, color=INK2)
    fig.suptitle(f"Stability of optimal boundaries across K — {kernel_label}, {model}", fontsize=12)
    fig.savefig(out / f"{prefix}_boundaries.png", dpi=180, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


### MAIN


def self_test():
    rng = np.random.default_rng(0)
    L, true = 30, [0, 4, 22]
    lab = labels_from_starts(true, L)
    B = np.array([[0.95, 0.5, 0.3], [0.5, 0.9, 0.55], [0.3, 0.55, 0.92]])
    M = B[lab][:, lab] + rng.normal(0, 0.02, (L, L))
    M = (M + M.T) / 2
    np.fill_diagonal(M, 1)
    res = analyse(M, 6, "sse", 1)
    s3 = next(s for s in res["segmentations"] if s["K"] == 3)
    assert s3["starts"] == true, s3
    assert res["best_bic_K"] == 3, res["best_bic_K"]
    # brute-force check of the DP for K=3
    C = block_costs(M, "sse")
    bf = min(C[0, a - 1] + C[a, b - 1] + C[b, L - 1] for a in range(1, L) for b in range(a + 1, L))
    assert abs(bf - s3["dp_cost"]) < 1e-9
    print("self-test passed: recovered", true, "with best-BIC K =", res["best_bic_K"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cka-dir", default="results/cka/qwen3.5-9b")
    p.add_argument("--kernel", choices=["linear", "rbf"], default="linear")
    p.add_argument("--lenses", nargs="+", default=["j_lens", "r_lens", "jpp_lens"])
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--k-max", type=int, default=8)
    p.add_argument("--k-show", type=int, nargs="+", default=[3], help="K values drawn on the heatmaps "
                   "(the best-BIC K per lens is added automatically)")
    p.add_argument("--cost", choices=["sse", "dissim"], default="sse")
    p.add_argument("--min-size", type=int, default=1, help="minimum layers per block")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        self_test()
        return

    src = Path(args.cka_dir)
    tag = "cka_rbf" if args.kernel == "rbf" else "cka"
    data = np.load(src / f"{tag}_all.npz")
    layers = data["layers"].tolist()
    mats = {n: np.asarray(data[n], dtype=np.float64) for n in args.lenses if n in data}
    kernel_label = "RBF CKA" if args.kernel == "rbf" else "Linear CKA"

    out = src / "segmentation"
    out.mkdir(exist_ok=True)
    prefix = f"seg_{args.kernel}" + ("" if args.cost == "sse" else f"_{args.cost}")

    results = {n: analyse(M, args.k_max, args.cost, args.min_size) for n, M in mats.items()}

    for n, r in results.items():
        print(f"\n== {LENS_LABELS.get(n, n)} ({kernel_label}, cost={args.cost})")
        print(f"   {'K':>2} {'dp_cost':>9} {'full_RSS':>9} {'BIC':>9}  blocks (layers)")
        for s in r["segmentations"]:
            mark = " *" if s["K"] == r["best_bic_K"] else ""
            print(f"   {s['K']:>2} {s['dp_cost']:9.4f} {s['full_rss']:9.4f} {s['bic']:9.1f}  "
                  f"{_block_label(s['starts'], layers)}{mark}")
            if "exhaustive_full" in s:
                x = s["exhaustive_full"]
                print(f"   {'':>2} {'':>9} {x['full_rss']:9.4f} {x['bic']:9.1f}  "
                      f"{_block_label(x['starts'], layers)}  (exhaustive, full-model objective)")
        lm = r["lag_model"]
        print(f"   lag model (phase-free, {lm['n_params']} params): RSS={lm['rss']:.4f}  BIC={lm['bic']:.1f}")

    k_show = sorted(set(args.k_show) | {r["best_bic_K"] for r in results.values()})
    plot_all(results, mats, layers, args.model, kernel_label, out, prefix, k_show)

    for r in results.values():
        for s in r["segmentations"]:
            s["blocks_layers"] = [[layers[a], layers[e - 1]] for a, e in zip(s["starts"], s["starts"][1:] + [len(layers)])]
    meta = dict(model=args.model, kernel=args.kernel, cost=args.cost, min_size=args.min_size,
                layers=layers, source=str(src / f"{tag}_all.npz"), lenses=results)
    (out / f"{prefix}_summary.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved to {out}/ ({prefix}_*)")


if __name__ == "__main__":
    main()
