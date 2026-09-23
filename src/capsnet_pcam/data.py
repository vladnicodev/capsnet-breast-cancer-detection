"""PatchCamelyon (PCam) data pipeline.

The raw data is ~160k JPEG patches in an ImageFolder-style layout::

    <root>/train+val/train/{0,1}/*.jpg   144,000 patches (balanced)
    <root>/train+val/valid/{0,1}/*.jpg    16,000 patches (balanced)

Label 1 means the centre 32x32 px region of the patch contains metastatic tumour tissue.

Decoding JPEGs on every epoch would dominate the runtime of these small models, so
`cache_split` decodes each split once into a uint8 ``.npy`` array (4.4 GB in total). Training
keeps the arrays in (GPU) memory, draws mini-batches by indexing, and augments on the GPU.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMAGE_SIZE = 96
CLASS_NAMES = ("normal", "tumor")
SPLITS = {"train": "train+val/train", "valid": "train+val/valid"}


# --------------------------------------------------------------------------- caching


def list_split(root: str | Path, split: str) -> tuple[list[Path], np.ndarray]:
    """Image paths and labels of a raw split, in a deterministic (sorted) order."""
    split_dir = Path(root) / SPLITS[split]
    paths, labels = [], []
    for label in (0, 1):
        files = sorted(p for p in (split_dir / str(label)).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".tif"})
        paths += files
        labels += [label] * len(files)
    if not paths:
        raise FileNotFoundError(f"no images found under {split_dir}")
    return paths, np.asarray(labels, dtype=np.uint8)


def read_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def cache_split(root: str | Path, split: str, out_dir: str | Path, workers: int = 16) -> int:
    """Decode every image of `split` into ``<out_dir>/<split>_x.npy`` (+ labels and ids)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths, labels = list_split(root, split)

    images = np.lib.format.open_memmap(
        out_dir / f"{split}_x.npy", mode="w+", dtype=np.uint8, shape=(len(paths), IMAGE_SIZE, IMAGE_SIZE, 3)
    )
    with ThreadPoolExecutor(workers) as pool:  # PIL releases the GIL while decoding
        for i, img in enumerate(pool.map(read_rgb, paths)):
            images[i] = img
    images.flush()
    del images

    np.save(out_dir / f"{split}_y.npy", labels)
    (out_dir / f"{split}_ids.txt").write_text("\n".join(p.stem for p in paths))
    return len(paths)


# --------------------------------------------------------------------------- splits


def stratified_subset(labels: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Sorted indices of a class-balanced random subset containing `fraction` of each class.

    For a fixed seed the subsets are nested (the 10% subset is contained in the 20% subset,
    and so on), so differences between training-set sizes are not confounded by which
    patches happened to be drawn.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(labels):
        members = rng.permutation(np.flatnonzero(labels == c))
        keep.append(members[: max(1, round(fraction * len(members)))])
    return np.sort(np.concatenate(keep))


def holdout_split(labels: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Split the official validation set 50/50 (stratified) into a validation and a test set.

    The validation half is used for checkpoint selection; the test half is only touched once,
    after training, so reported numbers are not biased by model selection.
    """
    test = stratified_subset(labels, 0.5, seed)
    val = np.setdiff1d(np.arange(len(labels)), test)
    return val, test


# --------------------------------------------------------------------------- in-memory splits


class ArraySplit:
    """A labelled split held as a uint8 NHWC tensor, yielding float NCHW batches in [0, 1]."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, storage: torch.device, device: torch.device):
        self.images = torch.from_numpy(np.ascontiguousarray(images)).to(storage)
        self.labels = torch.from_numpy(labels.astype(np.int64)).to(storage)
        self.device = device

    def __len__(self) -> int:
        return len(self.labels)

    def get(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        idx = idx.to(self.images.device)
        x = self.images[idx].to(self.device, non_blocking=True)
        y = self.labels[idx].to(self.device, non_blocking=True)
        # NHWC uint8 -> NCHW float view. The permute keeps the memory channels-last, which is
        # also the fastest layout for cuDNN convolutions in reduced precision.
        return x.permute(0, 3, 1, 2).float().div_(255.0), y

    def batches(self, batch_size: int, shuffle: bool = False, generator: torch.Generator | None = None) -> Iterator:
        order = torch.randperm(len(self), generator=generator) if shuffle else torch.arange(len(self))
        for start in range(0, len(self), batch_size):
            yield self.get(order[start : start + batch_size])

    def forever(self, batch_size: int, generator: torch.Generator) -> Iterator:
        """Endless stream of shuffled full batches (reshuffled every epoch)."""
        while True:
            order = torch.randperm(len(self), generator=generator)
            for start in range(0, len(self) - batch_size + 1, batch_size):
                yield self.get(order[start : start + batch_size])


@dataclass
class PCam:
    train: ArraySplit
    val: ArraySplit
    test: ArraySplit


def load_pcam(data_dir: str | Path, fraction: float = 1.0, seed: int = 0, device: str | torch.device = "cpu") -> PCam:
    """Load the cached arrays: a `fraction` of the training set, plus the fixed val/test halves."""
    data_dir = Path(data_dir)
    device = torch.device(device)
    if not (data_dir / "train_x.npy").exists():
        raise FileNotFoundError(f"{data_dir} has no cached arrays; run `python -m capsnet_pcam.prepare_data` first")

    train_x = np.load(data_dir / "train_x.npy", mmap_mode="r")
    train_y = np.load(data_dir / "train_y.npy")
    train_idx = stratified_subset(train_y, fraction, seed) if fraction < 1 else np.arange(len(train_y))
    valid_x = np.load(data_dir / "valid_x.npy", mmap_mode="r")
    valid_y = np.load(data_dir / "valid_y.npy")
    val_idx, test_idx = holdout_split(valid_y)

    # Keep everything on the GPU when it comfortably fits (4.4 GB for the full dataset).
    n_bytes = (len(train_idx) + len(valid_y)) * IMAGE_SIZE * IMAGE_SIZE * 3
    fits_on_gpu = device.type == "cuda" and n_bytes < 0.5 * torch.cuda.mem_get_info(device)[0]
    storage = device if fits_on_gpu else torch.device("cpu")

    def split(x: np.ndarray, y: np.ndarray, idx: np.ndarray) -> ArraySplit:
        return ArraySplit(x[idx], y[idx], storage, device)

    return PCam(split(train_x, train_y, train_idx), split(valid_x, valid_y, val_idx), split(valid_x, valid_y, test_idx))


def write_meta(out_dir: str | Path, **meta) -> None:
    (Path(out_dir) / "meta.json").write_text(json.dumps(meta, indent=2))


# --------------------------------------------------------------------------- augmentation


def random_dihedral(x: torch.Tensor) -> torch.Tensor:
    """Apply an independent random element of D4 (4 rotations x optional flip) to every image.

    Histopathology patches have no canonical orientation, so these 8 transforms are exact
    label-preserving symmetries of the data.
    """
    k = torch.randint(0, 8, (x.shape[0],), device=x.device)
    out = torch.empty_like(x)
    for t in range(8):
        sel = (k == t).nonzero().squeeze(1)
        xt = x[sel]
        if t >= 4:
            xt = xt.flip(-1)
        out[sel] = torch.rot90(xt, t % 4, dims=(-2, -1))
    return out
