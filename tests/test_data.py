import numpy as np
import pytest
import torch
from PIL import Image

from capsnet_pcam.data import ArraySplit, cache_split, holdout_split, load_pcam, random_dihedral, stratified_subset
from capsnet_pcam.metrics import binary_metrics
from capsnet_pcam.perturbations import PERTURBATIONS

LABELS = np.array([0] * 600 + [1] * 400)


def test_stratified_subset_is_balanced_and_deterministic():
    idx = stratified_subset(LABELS, 0.1, seed=3)
    assert len(idx) == 100
    assert np.sum(LABELS[idx] == 0) == 60 and np.sum(LABELS[idx] == 1) == 40
    assert np.array_equal(idx, stratified_subset(LABELS, 0.1, seed=3))
    assert not np.array_equal(idx, stratified_subset(LABELS, 0.1, seed=4))


def test_stratified_subsets_are_nested():
    small, large = stratified_subset(LABELS, 0.1, seed=0), stratified_subset(LABELS, 0.3, seed=0)
    assert set(small) <= set(large)


def test_stratified_subset_rejects_bad_fraction():
    with pytest.raises(ValueError):
        stratified_subset(LABELS, 0, seed=0)


def test_holdout_split_is_a_disjoint_stratified_partition():
    val, test = holdout_split(LABELS)
    assert len(np.intersect1d(val, test)) == 0
    assert len(val) + len(test) == len(LABELS)
    assert LABELS[test].mean() == pytest.approx(LABELS.mean())


def test_array_split_batches():
    images = np.random.default_rng(0).integers(0, 256, (10, 96, 96, 3), dtype=np.uint8)
    split = ArraySplit(images, np.arange(10) % 2, torch.device("cpu"), torch.device("cpu"))
    batches = list(split.batches(4))
    assert [len(y) for _, y in batches] == [4, 4, 2]
    x, _ = batches[0]
    assert x.shape == (4, 3, 96, 96) and x.dtype == torch.float32
    assert torch.allclose(x[0], torch.from_numpy(images[0]).permute(2, 0, 1).float() / 255)


def test_random_dihedral_only_permutes_pixels():
    x = torch.rand(32, 3, 96, 96)
    out = random_dihedral(x)
    assert out.shape == x.shape
    for a, b in zip(x, out, strict=True):
        variants = [torch.rot90(v, k, dims=(-2, -1)) for v in (a, a.flip(-1)) for k in range(4)]
        assert any(torch.equal(b, v) for v in variants)


@pytest.mark.parametrize("name", PERTURBATIONS)
def test_perturbations_preserve_shape_and_range(name):
    _, severities, fn = PERTURBATIONS[name]
    x = torch.rand(4, 3, 96, 96)
    assert torch.equal(fn(x, severities[0]), x)  # first severity is the identity
    out = fn(x, severities[-1])
    assert out.shape == x.shape and out.min() >= -1e-5 and out.max() <= 1 + 1e-5
    assert not torch.allclose(out, x)


def test_cache_and_load_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    for split, n in (("train", 20), ("valid", 8)):
        for label in (0, 1):
            folder = tmp_path / "raw" / "train+val" / split / str(label)
            folder.mkdir(parents=True)
            for i in range(n // 2):
                Image.fromarray(rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)).save(folder / f"{label}_{i}.png")
    # the loader only picks up .jpg/.jpeg/.tif; save lossless copies as .tif
    for p in (tmp_path / "raw").rglob("*.png"):
        Image.open(p).save(p.with_suffix(".tif"))
        p.unlink()

    assert cache_split(tmp_path / "raw", "train", tmp_path / "cache") == 20
    assert cache_split(tmp_path / "raw", "valid", tmp_path / "cache") == 8
    data = load_pcam(tmp_path / "cache", fraction=0.5, seed=0)
    assert (len(data.train), len(data.val), len(data.test)) == (10, 4, 4)
    assert data.train.labels.float().mean() == 0.5

    raw = np.asarray(Image.open(sorted((tmp_path / "raw" / "train+val" / "train" / "0").iterdir())[0]))
    cached = np.load(tmp_path / "cache" / "train_x.npy")[0]
    assert np.array_equal(raw, cached)


def test_binary_metrics():
    y = np.array([0, 0, 1, 1])
    m = binary_metrics(y, np.array([0.1, 0.6, 0.4, 0.9]))
    assert m["auc"] == 0.75
    assert m["accuracy"] == 0.5 and m["sensitivity"] == 0.5 and m["specificity"] == 0.5
    assert binary_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))["f1"] == 1.0
