"""Run the full grid of models x training-set fractions x seeds.

    python -m capsnet_pcam.sweep
    python -m capsnet_pcam.sweep --models capsnet --fractions 0.1 --seeds 0 1 2

Runs that already have a ``metrics.json`` are skipped, so an interrupted sweep can simply be
restarted.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import torch

from .models import MODEL_NAMES
from .train import TrainConfig, add_config_args, train

FRACTIONS = (0.01, 0.1, 0.2, 0.3, 1.0)
SEEDS = (0, 1, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--fractions", nargs="+", type=float, default=list(FRACTIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    add_config_args(parser, skip=("model", "fraction", "seed"))
    args = vars(parser.parse_args())
    models, fractions, seeds = args.pop("models"), args.pop("fractions"), args.pop("seeds")
    base = TrainConfig(**args)

    grid = [replace(base, model=m, fraction=f, seed=s) for f in fractions for m in models for s in seeds]
    todo = [cfg for cfg in grid if not (cfg.run_dir / "metrics.json").exists()]
    print(f"{len(grid)} runs in grid, {len(grid) - len(todo)} already done, {len(todo)} to go")

    start = time.perf_counter()
    for i, cfg in enumerate(todo, 1):
        print(f"\n=== run {i}/{len(todo)} ({(time.perf_counter() - start) / 60:.0f} min elapsed)")
        train(cfg)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
