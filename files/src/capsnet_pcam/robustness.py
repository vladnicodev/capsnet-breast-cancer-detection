"""Evaluate trained checkpoints under test-time distribution shifts.

    python -m capsnet_pcam.robustness --fractions 1.0 0.1

Writes one row per (run, perturbation, severity) to ``results/robustness.csv``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from .data import load_pcam
from .metrics import binary_metrics
from .models import build_model
from .perturbations import PERTURBATIONS
from .train import TrainConfig, predict


def load_run(run_dir: Path, device: torch.device) -> tuple[TrainConfig, torch.nn.Module]:
    cfg = TrainConfig(**json.loads((run_dir / "config.json").read_text()))
    model = build_model(cfg.model, cfg.routings, cfg.recon_weight)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True))
    return cfg, model.to(device, memory_format=torch.channels_last).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--data-dir", default="data/pcam")
    parser.add_argument("--fractions", nargs="+", type=float, default=[1.0])
    parser.add_argument("--out", default="results/robustness.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dirs = sorted(p.parent for p in Path(args.runs).glob("*/*/*/metrics.json"))
    data = load_pcam(args.data_dir, fraction=0.01, device=device)  # only the test split is used
    y = data.test.labels.cpu().numpy()

    rows = []
    for run_dir in run_dirs:
        cfg, model = load_run(run_dir, device)
        if cfg.fraction not in args.fractions:
            continue
        for name, (_, severities, fn) in PERTURBATIONS.items():
            for severity in severities:
                scores = predict(model, data.test, cfg.amp, transform=lambda x, fn=fn, s=severity: fn(x, s))
                m = binary_metrics(y, scores)
                rows.append(
                    {"model": cfg.model, "variant": cfg.variant, "fraction": cfg.fraction, "seed": cfg.seed,
                     "perturbation": name, "severity": severity, "auc": m["auc"], "accuracy": m["accuracy"]}
                )  # fmt: skip
        print(f"{run_dir}: done")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
