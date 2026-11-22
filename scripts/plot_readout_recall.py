"""Plot readout recall@k for the J-, R- and J++ Lenses, in the style of Figure 2 (a, b)
of the J++ Lens paper.

Reads the rank table written by ``jpp_cli.py evaluate`` (``readout_ranks.csv``) and writes:

* ``recall_at_{k}_by_task.{png,pdf}``: grouped bars, one group per task, of recall@k with
  the rank taken as the minimum over layers (the same item-weighted number as the
  ``readout_ranks_pass_at_k.csv`` table, which the script cross-checks).
* ``recall_at_{k}_by_layer.{png,pdf}``: a grid of line charts, one per task plus a macro
  panel (unweighted mean over tasks), of recall@k at each layer, with shaded 95%
  bootstrap confidence bands over items.

Usage:
    python scripts/plot_readout_recall.py results/qwen3.5-9b
    python scripts/plot_readout_recall.py results/qwen3.5-9b --k 1 --out-dir some/dir
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Lens key in the CSV -> (label, colour, marker), matching Figure 2 of the paper.
LENS_STYLE = {
    "j_lens": ("J-Lens", "#4d4d4d", "s"),
    "r_lens": ("R-Lens", "#d97a00", "^"),
    "jpp_lens": ("J++ Lens", "#1f77b4", "o"),
}
TASK_ORDER = ["multihop", "multilingual", "typo", "association", "poetry"]
N_BOOT = 2000


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Palatino", "TeX Gyre Pagella", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": "#e0e0e0",
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def _item_hits(ranks_df: pd.DataFrame, k: int, by_layer: bool) -> pd.DataFrame:
    """Per-item recall: fraction of the item's intermediates with rank <= k, using the
    min rank over layers (``by_layer=False``) or the rank at each layer."""
    keys = ["eval", "lens", "item", "intermediate"] + (["layer"] if by_layer else [])
    ranks = ranks_df.groupby(keys)["rank"].min().reset_index()
    ranks["hit"] = ranks["rank"] <= k
    item_keys = ["eval", "lens", "item"] + (["layer"] if by_layer else [])
    return ranks.groupby(item_keys)["hit"].mean().reset_index()


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _layer_curves(item_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Mean and 95% bootstrap CI (over items) per (eval, lens, layer), plus a macro
    row per (lens, layer): the unweighted mean over tasks, with the CI from
    resampling items within each task independently."""
    rows = []
    for (ev, lens, layer), grp in item_df.groupby(["eval", "lens", "layer"]):
        vals = grp["hit"].to_numpy(dtype=float)
        lo, hi = _bootstrap_ci(vals, rng)
        rows.append(dict(eval=ev, lens=lens, layer=layer, mean=vals.mean(), lo=lo, hi=hi))
    for (lens, layer), grp in item_df.groupby(["lens", "layer"]):
        per_task = [grp.loc[grp["eval"] == t, "hit"].to_numpy(float) for t in TASK_ORDER]
        if any(len(v) == 0 for v in per_task):
            continue
        boot = np.mean(
            [v[rng.integers(0, len(v), size=(N_BOOT, len(v)))].mean(axis=1) for v in per_task],
            axis=0,
        )
        rows.append(
            dict(
                eval="macro",
                lens=lens,
                layer=layer,
                mean=float(np.mean([v.mean() for v in per_task])),
                lo=float(np.percentile(boot, 2.5)),
                hi=float(np.percentile(boot, 97.5)),
            )
        )
    return pd.DataFrame(rows)


def _legend(fig: plt.Figure, lenses: list[str]) -> None:
    handles = []
    for lens in lenses:
        label, color, marker = LENS_STYLE[lens]
        handles.append(
            (Patch(color=color), Line2D([], [], color=color, marker=marker, lw=2.2, ms=7))
        )
    from matplotlib.legend_handler import HandlerTuple

    fig.legend(
        handles,
        [LENS_STYLE[lens][0] for lens in lenses],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)},
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(lenses),
        frameon=False,
        handlelength=4,
        columnspacing=2.0,
    )


def plot_by_task(summary: pd.DataFrame, k: int, lenses: list[str], tasks: list[str], out: Path):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    width = 0.8 / len(lenses)
    x = np.arange(len(tasks))
    for i, lens in enumerate(lenses):
        label, color, _ = LENS_STYLE[lens]
        vals = [100 * summary.loc[(t, lens)] for t in tasks]
        ax.bar(x + (i - (len(lenses) - 1) / 2) * width, vals, width * 0.92, color=color, label=label)
    ax.set_xticks(x)
    # Staggered tick labels, as in Figure 2a.
    ax.set_xticklabels(["\n" + t.capitalize() if j % 2 == 0 else t.capitalize() for j, t in enumerate(tasks)])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel(f"Recall@{k} (%)")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    _legend(fig, lenses)
    _save(fig, out)


def plot_by_layer(curves: pd.DataFrame, k: int, lenses: list[str], panels: list[str], out: Path):
    ncols = 3
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.2 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    layers = sorted(curves["layer"].unique())
    for ax, panel in zip(axes, panels):
        sub = curves[curves["eval"] == panel]
        for lens in lenses:
            label, color, marker = LENS_STYLE[lens]
            c = sub[sub["lens"] == lens].sort_values("layer")
            ax.fill_between(c["layer"], 100 * c["lo"], 100 * c["hi"], color=color, alpha=0.2, lw=0)
            ax.plot(c["layer"], 100 * c["mean"], color=color, marker=marker, lw=2.2, ms=6, label=label)
        ax.set_title("Macro (mean over tasks)" if panel == "macro" else panel.capitalize())
        ax.set_xticks(layers)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="x", labelbottom=True)
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    for r in range(nrows):
        axes[r * ncols].set_ylabel(f"Recall@{k} at layer (%)")
    for ax in axes[(nrows - 1) * ncols :]:
        ax.set_xlabel("Layer")
    fig.tight_layout()
    _legend(fig, lenses)
    _save(fig, out)


def _save(fig: plt.Figure, out: Path) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"))
    plt.close(fig)
    print(f"wrote {out.with_suffix('.png')} (+ .pdf)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir", type=Path, help="dir holding readout_ranks.csv")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out-dir", type=Path, default=None, help="default: <results_dir>/figures")
    p.add_argument("--no-macro-panel", action="store_true", help="omit the macro panel from the layer grid")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ranks_df = pd.read_csv(args.results_dir / "readout_ranks.csv")
    lenses = [lens for lens in LENS_STYLE if lens in set(ranks_df["lens"])]
    present = set(ranks_df["eval"])
    tasks = [t for t in TASK_ORDER if t in present] + sorted(present - set(TASK_ORDER))
    out_dir = args.out_dir or args.results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    _style()

    # (a) recall@k over layers (min rank across layers), per task.
    task_items = _item_hits(ranks_df, args.k, by_layer=False)
    summary = task_items.groupby(["eval", "lens"])["hit"].mean()
    pass_path = args.results_dir / "readout_ranks_pass_at_k.csv"
    if pass_path.exists():  # sanity check against the CLI's own table
        ref = pd.read_csv(pass_path)
        ref = ref[(ref["k"] == args.k) & (ref["weighting"] == "item") & (ref["eval"] != "macro")]
        ref = ref.set_index(["eval", "lens"])["pass_at_k"]
        diff = (summary.reindex(ref.index) - ref).abs().max()
        assert diff < 1e-9, f"recall disagrees with {pass_path} (max diff {diff})"
    plot_by_task(summary, args.k, lenses, tasks, out_dir / f"recall_at_{args.k}_by_task")

    # (b) recall@k at each layer, one panel per task (+ macro).
    layer_items = _item_hits(ranks_df, args.k, by_layer=True)
    curves = _layer_curves(layer_items, np.random.default_rng(args.seed))
    panels = tasks + ([] if args.no_macro_panel else ["macro"])
    plot_by_layer(curves, args.k, lenses, panels, out_dir / f"recall_at_{args.k}_by_layer")

    curves.to_csv(out_dir / f"recall_at_{args.k}_by_layer.csv", index=False)
    print(json.dumps({t: {LENS_STYLE[l][0]: round(100 * summary[(t, l)], 1) for l in lenses} for t in tasks}, indent=1))


if __name__ == "__main__":
    main()
