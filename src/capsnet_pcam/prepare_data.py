"""Decode the PCam JPEG folders once into uint8 .npy arrays.

    python -m capsnet_pcam.prepare_data --src /path/to/pcam --out data/pcam

`--src` must contain ``train+val/train/{0,1}`` and ``train+val/valid/{0,1}`` with one image
per patch. Takes about a minute and needs ~4.4 GB of disk space.
"""

from __future__ import annotations

import argparse
import time

from .data import SPLITS, cache_split, write_meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="root of the raw PCam dataset")
    parser.add_argument("--out", default="data/pcam", help="output directory for the cached arrays")
    parser.add_argument("--workers", type=int, default=16, help="decoding threads")
    args = parser.parse_args()

    counts = {}
    for split in SPLITS:
        start = time.perf_counter()
        counts[split] = cache_split(args.src, split, args.out, args.workers)
        print(f"{split}: {counts[split]:,} patches cached in {time.perf_counter() - start:.0f}s")
    write_meta(args.out, source=str(args.src), counts=counts)


if __name__ == "__main__":
    main()
