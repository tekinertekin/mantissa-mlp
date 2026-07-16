"""Benchmark plots. Consumes bench/results/speed.json (produced by
bench.speed) and writes PNGs to assets/. Never invents data: exits with a
message if the JSON is missing.

Figures (matplotlib, no seaborn, one chart per file):
  assets/fit_time.png   median fit seconds per dataset, grouped bars per
                        contender, log scale
  assets/accuracy.png   test accuracy per dataset per contender (top panel)
                        + higgsml AMS at the top-15% operating point with the
                        select-everything baseline drawn in (bottom panel) —
                        accuracy and AMS deliberately disagree, so they get
                        separate axes
  assets/peak_rss.png   peak RSS per (dataset, contender), fresh-process,
                        import + one fit

Run from the repo root:  python -m bench.plots
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "bench" / "results"
ASSETS = REPO_ROOT / "assets"

# One stable color per contender across every plot — the family palette (the
# categorical slots were validated in the cnn repo with the dataviz six-checks
# script against this light surface; the lower-contrast hues are relieved by
# the direct value label every bar carries). sklearn joins as a full contender
# here, so it takes the family's fifth slot (amber).
COLORS = {
    "ours": "#2a78d6",           # blue
    "vanilla_numpy": "#1baf7a",  # aqua
    "torch": "#e34948",          # red
    "tensorflow": "#d96b2f",     # orange
    "sklearn": "#eda100",        # amber
}
LABELS = {
    "ours": "ours (mantissa)",
    "vanilla_numpy": "vanilla numpy",
    "torch": "torch",
    "tensorflow": "tensorflow",
    "sklearn": "scikit-learn",
}
ORDER = ["ours", "vanilla_numpy", "torch", "tensorflow", "sklearn"]

# Opaque light surface so the PNG reads on GitHub light AND dark themes.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": AXIS,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
    })


def _load():
    path = RESULTS / "speed.json"
    if not path.is_file():
        raise SystemExit(f"{path.relative_to(REPO_ROOT)} missing — run "
                         f"python -m bench.speed first")
    return json.loads(path.read_text())


def _short_env(env) -> str:
    bits = [env.get("cpu", "?"), f"Python {env.get('python', '?')}"]
    if env.get("date"):
        bits.append(env["date"])
    return "  ·  ".join(bits)


def _contender_order(per_ds):
    seen = {c for row in per_ds.values() for c in row}
    return [c for c in ORDER if c in seen] + sorted(seen - set(ORDER))


def _grouped_bars(ax, contenders, datasets, values, log=False, fmt="{:.2f}"):
    """values[contender][dataset] -> float. One group per dataset."""
    n_series = len(contenders)
    x = np.arange(len(datasets))
    width = 0.8 / n_series
    all_h = []
    for si, c in enumerate(contenders):
        heights = [values[c].get(d, 0.0) for d in datasets]
        offset = (si - (n_series - 1) / 2) * width
        bars = ax.bar(x + offset, heights, width, label=LABELS[c],
                      color=COLORS[c], edgecolor=SURFACE, linewidth=0.6,
                      zorder=3)
        for rect, h in zip(bars, heights):
            if h <= 0:
                continue
            all_h.append(h)
            ax.annotate(fmt.format(h), (rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=5.6, rotation=90,
                        color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    if log:
        ax.set_yscale("log")
        ax.set_ylim(min(all_h) / 3.0, max(all_h) * 16.0)
    else:
        ax.margins(y=0.18)


def plot_fit_time(speed):
    datasets = speed["protocol"]["datasets"]
    per_ds = {d: {c: v["median"] for c, v in speed["fit_s"][d].items()}
              for d in datasets}
    contenders = _contender_order(per_ds)
    values = {c: {d: per_ds[d][c] for d in datasets} for c in contenders}
    r = speed["protocol"]["repeats"]
    e = speed["protocol"]["epochs"]

    fig, ax = plt.subplots(figsize=(11.0, 4.6), dpi=150)
    _grouped_bars(ax, contenders, datasets, values, log=True, fmt="{:.3f}")
    ax.set_ylabel("median fit time — s (log scale)")
    ax.set_title(f"Training time per dataset — median of {r} interleaved "
                 f"fits, {e} epochs", color=INK, fontsize=13,
                 fontweight="bold", pad=32, loc="left")
    ax.text(0, 1.05, _short_env(speed["env"]), transform=ax.transAxes,
            fontsize=8, color=INK2, va="bottom")
    ax.legend(loc="upper right", framealpha=0.9, facecolor=SURFACE,
              edgecolor=GRID, fontsize=7.5, ncol=len(contenders))
    _save(fig, "fit_time.png")


def plot_accuracy(speed):
    """Test accuracy for every dataset (top) and higgsml AMS (bottom). AMS is
    a weighted physics significance, not an accuracy, so it gets its own axis
    and its own baseline (select everything) — the two metrics disagree on
    purpose, which is the whole point of the higgsml row."""
    datasets = speed["protocol"]["datasets"]
    per_ds = {d: {c: v for c, v in speed["test_acc"][d].items()}
              for d in datasets}
    contenders = _contender_order(per_ds)
    values = {c: {d: per_ds[d][c] for d in datasets} for c in contenders}
    e = speed["protocol"]["epochs"]
    ams = speed.get("ams", {}).get("higgsml", {})

    fig, (ax, axa) = plt.subplots(
        2, 1, figsize=(11.0, 6.6), dpi=150,
        gridspec_kw={"height_ratios": [2.4, 1.0]})

    _grouped_bars(ax, contenders, datasets, values, log=False, fmt="{:.3f}")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1.30)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_title(f"Test accuracy — {e} epochs, identical architecture per "
                 f"dataset", color=INK, fontsize=13, fontweight="bold",
                 pad=32, loc="left")
    ax.text(0, 1.03, _short_env(speed["env"]), transform=ax.transAxes,
            fontsize=8, color=INK2, va="bottom")
    ax.legend(loc="upper center", framealpha=0.9, facecolor=SURFACE,
              edgecolor=GRID, fontsize=7.5, ncol=len(contenders))

    # AMS panel — higgsml only.
    if ams:
        cs = [c for c in ORDER if c in ams]
        x = np.arange(len(cs))
        heights = [ams[c]["value"] for c in cs]
        bars = axa.bar(x, heights, 0.62, color=[COLORS[c] for c in cs],
                       edgecolor=SURFACE, linewidth=0.6, zorder=3)
        for rect, h in zip(bars, heights):
            axa.annotate(f"{h:.2f}",
                         (rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 2), textcoords="offset points",
                         ha="center", va="bottom", fontsize=8, color=INK2)
        base = ams[cs[0]]["select_all_ams"]
        frac = ams[cs[0]]["select_frac"]
        axa.axhline(base, color=INK2, linewidth=1.0, linestyle="--", zorder=2)
        axa.annotate(f"select everything ({base:.2f})", (0.015, base),
                     xycoords=("axes fraction", "data"), xytext=(0, 2),
                     textcoords="offset points", fontsize=7, color=INK2,
                     va="bottom")
        axa.set_xticks(x)
        axa.set_xticklabels([LABELS[c] for c in cs], fontsize=8)
        axa.set_ylabel("AMS")
        axa.set_axisbelow(True)
        axa.grid(axis="x", visible=False)
        axa.margins(y=0.28)
        axa.set_title(f"higgsml AMS — top-{frac:.0%} selection by P(signal), "
                      f"CERN challenge metric (higher is better)", color=INK,
                      fontsize=11, fontweight="bold", loc="left")
    _save(fig, "accuracy.png")


def plot_peak_rss(speed):
    datasets = speed["protocol"]["datasets"]
    per_ds = speed["peak_rss_mb"]
    contenders = _contender_order(per_ds)
    values = {c: {d: per_ds[d].get(c, 0.0) for d in datasets}
              for c in contenders}

    fig, ax = plt.subplots(figsize=(11.0, 4.6), dpi=150)
    _grouped_bars(ax, contenders, datasets, values, log=False, fmt="{:.0f}")
    ax.set_ylabel("peak RSS — MB")
    ax.set_title("Peak memory — import + one fit, fresh process each",
                 color=INK, fontsize=13, fontweight="bold", pad=32, loc="left")
    ax.text(0, 1.05, _short_env(speed["env"]), transform=ax.transAxes,
            fontsize=8, color=INK2, va="bottom")
    ax.legend(loc="upper right", framealpha=0.9, facecolor=SURFACE,
              edgecolor=GRID, fontsize=7.5, ncol=len(contenders))
    _save(fig, "peak_rss.png")


def _save(fig, name):
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    # metadata=Software:None -> byte-stable across runs (no version/timestamp).
    fig.savefig(ASSETS / name, dpi=150, metadata={"Software": None})
    plt.close(fig)
    print(f"wrote assets/{name}")


def main() -> int:
    _style()
    speed = _load()
    plot_fit_time(speed)
    plot_accuracy(speed)
    plot_peak_rss(speed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
