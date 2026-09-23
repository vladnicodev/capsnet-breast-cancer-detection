"""Aggregate all runs into tables and figures under ``results/``.

    python -m capsnet_pcam.report

Writes ``results/runs.csv`` (one row per run), ``results/summary.csv`` (mean/std per model and
training-set size), ``results/summary.md`` (the tables used in the README) and every figure in
``results/figures/`` in a light and a dark variant.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from .models import DISPLAY_NAMES  # noqa: E402
from .perturbations import PERTURBATIONS  # noqa: E402

MAIN_MODELS = ("capsnet", "matched_cnn", "simple_cnn")
MARKERS = {"capsnet": "o", "matched_cnn": "s", "simple_cnn": "^"}
ABLATIONS = {
    "capsnet": "CapsNet (3 routing iterations + decoder)",
    "capsnet-no-routing": "CapsNet without routing (1 iteration)",
    "capsnet-no-recon": "CapsNet without reconstruction decoder",
    "matched_cnn": "Matched CNN",
}

# Validated categorical slots 1-3 (light / dark steps) and chart chrome for both themes.
THEMES = {
    "light": {
        "series": {"capsnet": "#2a78d6", "matched_cnn": "#eb6834", "simple_cnn": "#1baf7a"},
        "surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7",
    },
    "dark": {
        "series": {"capsnet": "#3987e5", "matched_cnn": "#d95926", "simple_cnn": "#199e70"},
        "surface": "#1a1a19", "primary": "#ffffff", "secondary": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835",
    },
}  # fmt: skip


# --------------------------------------------------------------------------- loading


def load_runs(runs_dir: Path) -> pd.DataFrame:
    rows = [{"variant": "", **json.loads(p.read_text())} for p in sorted(runs_dir.glob("*/*/*/metrics.json"))]
    if not rows:
        raise FileNotFoundError(f"no finished runs under {runs_dir}")
    df = pd.DataFrame(rows)
    df["name"] = np.where(df["variant"] == "", df["model"], df["model"] + "-" + df["variant"])
    df["run_dir"] = [
        str(runs_dir / n / f"frac{f:g}" / f"seed{s}")
        for n, f, s in zip(df["name"], df["fraction"], df["seed"], strict=True)
    ]
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in df.columns if c.startswith("test_")] + [
        "val_auc", "train_time_s", "train_steps_per_s", "inference_img_per_s", "peak_gpu_mem_mb",
    ]  # fmt: skip
    g = df.groupby(["name", "fraction"])
    out = g[metrics].agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out.insert(0, "n_train", g["n_train"].first())
    out.insert(1, "params", g["params"].first())
    out.insert(2, "params_with_decoder", g["params_with_decoder"].first())
    out.insert(3, "seeds", g.size())
    return out.reset_index()


# --------------------------------------------------------------------------- tables


def _pm(mean: float, std: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}" if np.isfinite(std) else f"{mean:.{digits}f}"


def markdown_tables(summary: pd.DataFrame, robustness: pd.DataFrame | None) -> str:
    s = summary.set_index(["name", "fraction"])
    fractions = sorted(summary["fraction"].unique())
    models = [m for m in MAIN_MODELS if m in summary["name"].values]
    n_train = summary.groupby("fraction")["n_train"].first()
    out = []

    out.append("### Test ROC-AUC vs. training-set size\n")
    header = "| Model | " + " | ".join(f"{f:.0%} ({n_train[f]:,})" for f in fractions) + " |"
    out += [header, "|---|" + "---:|" * len(fractions)]
    main = summary[summary["name"].isin(models)]
    best = main.loc[main.groupby("fraction")["test_auc_mean"].idxmax(), ["name", "fraction"]]
    best = set(zip(best["name"], best["fraction"], strict=True))
    for m in models:
        cells = []
        for f in fractions:
            cell = _pm(s.loc[(m, f), "test_auc_mean"], s.loc[(m, f), "test_auc_std"]) if (m, f) in s.index else "–"
            cells.append(f"**{cell}**" if (m, f) in best else cell)
        out.append(f"| {DISPLAY_NAMES[m]} | " + " | ".join(cells) + " |")

    full = max(fractions)
    out.append(f"\n### All test metrics at {full:.0%} of the training data\n")
    out += ["| Model | ROC-AUC | Accuracy | Sensitivity | Specificity | F1 |", "|---|---:|---:|---:|---:|---:|"]
    for m in models:
        r = s.loc[(m, full)]
        cells = [
            _pm(r[f"test_{k}_mean"], r[f"test_{k}_std"])
            for k in ("auc", "accuracy", "sensitivity", "specificity", "f1")
        ]
        out.append(f"| {DISPLAY_NAMES[m]} | " + " | ".join(cells) + " |")

    out.append("\n### Model size and compute (RTX 5080, bf16 autocast, batch 128)\n")
    out += [
        "| Model | Parameters (inference) | + decoder (training only) | Training steps/s | Inference images/s |",
        "|---|---:|---:|---:|---:|",
    ]
    for m in models:
        r = s.loc[(m, full)]
        params, decoder = int(r["params"]), int(r["params_with_decoder"] - r["params"])
        out.append(
            f"| {DISPLAY_NAMES[m]} | {params:,} | {f'{decoder:,}' if decoder else '–'} "
            f"| {r['train_steps_per_s_mean']:.0f} | {r['inference_img_per_s_mean']:,.0f} |"
        )

    ablations = [a for a in ABLATIONS if a in summary["name"].values]
    if any(a.startswith("capsnet-") for a in ablations):
        abl_fracs = sorted(summary[summary["name"].str.startswith("capsnet-")]["fraction"].unique())
        out.append("\n### CapsNet ablations (test ROC-AUC)\n")
        out += ["| Variant | " + " | ".join(f"{f:.0%}" for f in abl_fracs) + " |", "|---|" + "---:|" * len(abl_fracs)]
        for a in ablations:
            cells = [
                _pm(s.loc[(a, f), "test_auc_mean"], s.loc[(a, f), "test_auc_std"]) if (a, f) in s.index else "–"
                for f in abl_fracs
            ]
            out.append(f"| {ABLATIONS[a]} | " + " | ".join(cells) + " |")

    if robustness is not None:
        r = robustness[robustness["variant"].fillna("") == ""]
        out.append(f"\n### Robustness: test ROC-AUC under the strongest shift (models trained on {full:.0%})\n")
        cols = [(name, sev[-1]) for name, (_, sev, _) in PERTURBATIONS.items()]
        out += [
            "| Model | Clean | " + " | ".join(f"{n} ({v:g})" for n, v in cols) + " |",
            "|---|---:|" + "---:|" * len(cols),
        ]
        for m in models:
            rm = r[(r["model"] == m) & (r["fraction"] == full)]
            if rm.empty:
                continue
            first, (_, severities, _) = next(iter(PERTURBATIONS.items()))
            clean = rm[(rm["perturbation"] == first) & (rm["severity"] == severities[0])]["auc"]
            cells = [_pm(clean.mean(), clean.std())]
            for n, v in cols:
                x = rm[(rm["perturbation"] == n) & (rm["severity"] == v)]["auc"]
                cells.append(_pm(x.mean(), x.std()))
            out.append(f"| {DISPLAY_NAMES[m]} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- figure helpers


def _style(theme: dict) -> dict:
    return {
        "figure.facecolor": theme["surface"], "axes.facecolor": theme["surface"], "savefig.facecolor": theme["surface"],
        "font.family": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 10,
        "text.color": theme["primary"], "axes.labelcolor": theme["secondary"], "axes.titlecolor": theme["primary"],
        "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.edgecolor": theme["axis"], "axes.linewidth": 1, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": theme["grid"], "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.axisbelow": True, "xtick.color": theme["muted"], "ytick.color": theme["muted"],
        "xtick.labelcolor": theme["muted"], "ytick.labelcolor": theme["muted"],
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.minor.size": 0,
        "legend.frameon": False, "legend.labelcolor": theme["primary"], "legend.fontsize": 9.5,
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    }  # fmt: skip


def _line(ax, x, mean, std, model: str, theme: dict, label: str | None = None) -> None:
    color = theme["series"][model]
    if std is not None:
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
    ax.plot(
        x, mean, color=color, label=label or DISPLAY_NAMES[model], marker=MARKERS[model], markersize=7,
        markeredgecolor=theme["surface"], markeredgewidth=1.5,
    )  # fmt: skip


def _save(fig, out_dir: Path, name: str, theme_name: str) -> None:
    fig.savefig(out_dir / f"{name}{'' if theme_name == 'light' else '_dark'}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- figures


def fig_data_efficiency(summary: pd.DataFrame, theme: dict):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, title in zip(axes, ("test_auc", "test_accuracy"), ("Test ROC-AUC", "Test accuracy"), strict=True):
        for m in MAIN_MODELS:
            d = summary[summary["name"] == m].sort_values("n_train")
            if not d.empty:
                _line(ax, d["n_train"], d[f"{metric}_mean"], d[f"{metric}_std"].fillna(0), m, theme)
        ax.set_xscale("log")
        ticks = sorted(summary["n_train"].unique())
        fracs = summary.groupby("n_train")["fraction"].first()
        ax.set_xticks(ticks, [f"{fracs[t]:.0%}\n{t / 1000:g}k" for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("Training patches (share of PCam train set)")
        ax.set_title(title)
    axes[0].legend(loc="lower right")
    fig.suptitle(
        "Data efficiency: mean ± 1 s.d. over 3 seeds", x=0.01, ha="left", fontsize=10, color=theme["secondary"], y=1.0
    )
    fig.tight_layout()
    return fig


def fig_roc(df: pd.DataFrame, theme: dict, fractions=(0.01, 0.1, 1.0)):
    fractions = [f for f in fractions if f in df["fraction"].values]
    fig, axes = plt.subplots(1, len(fractions), figsize=(3.9 * len(fractions), 4.1), sharey=True)
    grid = np.linspace(0, 1, 501)
    for ax, frac in zip(np.atleast_1d(axes), fractions, strict=True):
        ax.plot([0, 1], [0, 1], color=theme["axis"], linewidth=1)
        for m in MAIN_MODELS:
            runs = df[(df["name"] == m) & (df["fraction"] == frac)]
            if runs.empty:
                continue
            tprs = []
            for run_dir in runs["run_dir"]:
                z = np.load(Path(run_dir) / "test_scores.npz")
                fpr, tpr, _ = roc_curve(z["y"], z["score"])
                tprs.append(np.interp(grid, fpr, tpr))
            ax.plot(
                grid, np.mean(tprs, axis=0), color=theme["series"][m],
                label=f"{DISPLAY_NAMES[m]}  (AUC {runs['test_auc'].mean():.3f})",
            )  # fmt: skip
        ax.set_title(f"Trained on {frac:.0%} ({int(runs['n_train'].iloc[0]):,} patches)")
        ax.set_xlabel("False positive rate (1 − specificity)")
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.005)
        ax.legend(loc="lower right", fontsize=8.5)
    np.atleast_1d(axes)[0].set_ylabel("True positive rate (sensitivity)")
    fig.tight_layout()
    return fig


def fig_training_curves(df: pd.DataFrame, theme: dict, fractions=(0.01, 0.1, 1.0)):
    fractions = [f for f in fractions if f in df["fraction"].values]
    fig, axes = plt.subplots(1, len(fractions), figsize=(3.9 * len(fractions), 3.8), sharey=True)
    for ax, frac in zip(np.atleast_1d(axes), fractions, strict=True):
        for m in MAIN_MODELS:
            runs = df[(df["name"] == m) & (df["fraction"] == frac)]
            if runs.empty:
                continue
            hist = [pd.read_csv(Path(r) / "history.csv") for r in runs["run_dir"]]
            auc = np.stack([h["val_auc"].to_numpy() for h in hist])
            ax.plot(hist[0]["step"], auc.mean(0), color=theme["series"][m], label=DISPLAY_NAMES[m], linewidth=1.8)
        ax.set_title(f"{frac:.0%} of training data")
        ax.set_xlabel("Optimiser step")
        ax.set_ylim(0.75, 1.0)
    np.atleast_1d(axes)[0].set_ylabel("Validation ROC-AUC")
    np.atleast_1d(axes)[-1].legend(loc="lower right")
    fig.tight_layout()
    return fig


def fig_robustness(rob: pd.DataFrame, theme: dict, fraction: float = 1.0):
    rob = rob[(rob["variant"].fillna("") == "") & (rob["fraction"] == fraction)]
    fig, axes = plt.subplots(1, len(PERTURBATIONS), figsize=(13, 3.7), sharey=True)
    for ax, (name, (xlabel, severities, _)) in zip(axes, PERTURBATIONS.items(), strict=True):
        for m in MAIN_MODELS:
            d = rob[(rob["model"] == m) & (rob["perturbation"] == name)].groupby("severity")["auc"]
            if len(d):
                _line(ax, d.mean().index, d.mean().values, d.std().fillna(0).values, m, theme)
        ax.set_xticks(severities, [f"{s:g}" for s in severities])
        ax.set_xlabel(xlabel)
        ax.set_title(name.capitalize())
    axes[0].set_ylabel("Test ROC-AUC")
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    return fig


def fig_reconstructions(run_dir: Path, data_dir: Path, theme: dict, n: int = 6):
    import torch

    from .data import load_pcam
    from .robustness import load_run

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, model = load_run(run_dir, device)
    test = load_pcam(data_dir, fraction=0.01, device=device).test
    labels = test.labels.cpu().numpy()
    rng = np.random.default_rng(7)
    idx = np.concatenate([rng.choice(np.flatnonzero(labels == c), n, replace=False) for c in (0, 1)])
    x, y = test.get(torch.from_numpy(idx))
    with torch.no_grad():
        out = model(x, reconstruct=True)
    images = x.permute(0, 2, 3, 1).cpu().numpy()
    recons = out["recon"].float().permute(0, 2, 3, 1).cpu().numpy()
    scores = out["score"].float().cpu().numpy()

    fig, axes = plt.subplots(4, n, figsize=(1.55 * n, 6.9))
    for block, cls in enumerate(("Normal", "Metastatic tumour")):
        for j in range(n):
            k = block * n + j
            for row, img in ((2 * block, images[k]), (2 * block + 1, recons[k])):
                ax = axes[row, j]
                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                for spine in ax.spines.values():
                    spine.set_visible(False)
            axes[2 * block, j].set_title(
                f"p(tumour) {scores[k]:.2f}", fontsize=8, color=theme["secondary"], loc="center"
            )
        axes[2 * block, 0].set_ylabel(f"{cls}\ninput", fontsize=9, color=theme["secondary"])
        axes[2 * block + 1, 0].set_ylabel("reconstruction", fontsize=9, color=theme["secondary"])
    fig.tight_layout(h_pad=0.6, w_pad=0.3)
    return fig


# --------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="runs", type=Path)
    parser.add_argument("--data-dir", default="data/pcam", type=Path)
    parser.add_argument("--out", default="results", type=Path)
    parser.add_argument("--no-reconstructions", action="store_true", help="skip the figure that needs the dataset")
    args = parser.parse_args()

    fig_dir = args.out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = load_runs(args.runs)
    summary = summarise(df)
    df.drop(columns=["run_dir"]).to_csv(args.out / "runs.csv", index=False)
    summary.to_csv(args.out / "summary.csv", index=False)
    rob_path = args.out / "robustness.csv"
    rob = pd.read_csv(rob_path) if rob_path.exists() else None
    (args.out / "summary.md").write_text(markdown_tables(summary, rob), encoding="utf-8")

    main_runs = df[df["name"].isin(MAIN_MODELS)]
    caps_run = args.runs / "capsnet" / "frac1" / "seed0"
    for theme_name, theme in THEMES.items():
        with plt.rc_context(_style(theme)):
            _save(fig_data_efficiency(summary, theme), fig_dir, "data_efficiency", theme_name)
            _save(fig_roc(main_runs, theme), fig_dir, "roc_curves", theme_name)
            _save(fig_training_curves(main_runs, theme), fig_dir, "training_curves", theme_name)
            if rob is not None:
                _save(fig_robustness(rob, theme), fig_dir, "robustness", theme_name)
            if not args.no_reconstructions and (caps_run / "best.pt").exists():
                _save(fig_reconstructions(caps_run, args.data_dir, theme), fig_dir, "reconstructions", theme_name)
    print(f"wrote tables to {args.out} and figures to {fig_dir}")


if __name__ == "__main__":
    main()
