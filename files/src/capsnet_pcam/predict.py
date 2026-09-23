"""Predict tumour probabilities for a folder of patches (e.g. the Kaggle test set).

    python -m capsnet_pcam.predict --run runs/capsnet/frac1/seed0 --images /path/to/test --out submission.csv

Produces an ``id,label`` CSV in the format of the Kaggle "Histopathologic Cancer Detection"
competition. `--tta` averages the scores over the 8 flips/rotations of each patch.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import read_rgb
from .robustness import load_run
from .train import autocast


@torch.no_grad()
def predict_folder(model, paths: list[Path], device: torch.device, amp: bool, tta: bool, batch_size: int = 512):
    scores = []
    with ThreadPoolExecutor(16) as pool:
        for start in range(0, len(paths), batch_size):
            batch = np.stack(list(pool.map(read_rgb, paths[start : start + batch_size])))
            x = torch.from_numpy(batch).to(device).permute(0, 3, 1, 2).float().div_(255.0)
            views = [torch.rot90(v, k, dims=(-2, -1)) for v in (x, x.flip(-1)) for k in range(4)] if tta else [x]
            with autocast(device, amp):
                scores.append(torch.stack([model(v)["score"].float() for v in views]).mean(0).cpu())
            print(f"\r{min(start + batch_size, len(paths)):,}/{len(paths):,}", end="")
    print()
    return torch.cat(scores).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, type=Path, help="run directory containing config.json and best.pt")
    parser.add_argument("--images", required=True, type=Path, help="folder of .tif/.jpg patches")
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--tta", action="store_true", help="average over the 8 dihedral transforms")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, model = load_run(args.run, device)
    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {".tif", ".jpg", ".jpeg", ".png"})
    scores = predict_folder(model, paths, device, cfg.amp, args.tta)
    pd.DataFrame({"id": [p.stem for p in paths], "label": scores}).to_csv(args.out, index=False)
    print(f"wrote {len(paths):,} predictions to {args.out}")


if __name__ == "__main__":
    main()
