"""Train and evaluate a single model on a fraction of PCam.

    python -m capsnet_pcam.train --model capsnet --fraction 0.1 --seed 0

Every run trains for the same number of optimiser steps regardless of training-set size,
evaluates validation AUC on a fixed schedule (densely early on, when models trained on small
subsets peak, then every `--eval-every` steps), keeps the best checkpoint, and scores it once
on the held-out test split. Artifacts are written to
``<out-dir>/<model>/frac<fraction>/seed<seed>/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import ArraySplit, load_pcam, random_dihedral
from .metrics import binary_metrics
from .models import MODEL_NAMES, build_model, count_parameters


@dataclass
class TrainConfig:
    model: str = "capsnet"
    variant: str = ""  # optional suffix for ablations, e.g. "no-routing"
    fraction: float = 1.0
    seed: int = 0
    steps: int = 8000
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 250
    eval_every: int = 200
    eval_every_early: int = 50  # denser checks until `early_until`: small subsets overfit fast
    early_until: int = 1000
    augment: bool = True
    amp: bool = True
    routings: int = 3
    recon_weight: float = 0.392
    data_dir: str = "data/pcam"
    out_dir: str = "runs"

    def is_eval_step(self, step: int) -> bool:
        interval = self.eval_every_early if step <= self.early_until else self.eval_every
        return step % interval == 0 or step == self.steps

    @property
    def name(self) -> str:
        return f"{self.model}-{self.variant}" if self.variant else self.model

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.name / f"frac{self.fraction:g}" / f"seed{self.seed}"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def autocast(device: torch.device, enabled: bool):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=enabled and device.type == "cuda")


@torch.no_grad()
def predict(model: nn.Module, split: ArraySplit, amp: bool = True, batch_size: int = 512, transform=None) -> np.ndarray:
    """Tumour probability for every image of `split` (optionally perturbed by `transform`)."""
    model.eval()
    device = split.device
    scores = []
    for x, _ in split.batches(batch_size):
        if transform is not None:
            x = transform(x)
        with autocast(device, amp):
            scores.append(model(x)["score"].float())
    return torch.cat(scores).cpu().numpy()


def measure_throughput(model: nn.Module, split: ArraySplit, amp: bool, batch_size: int = 512, repeats: int = 3):
    """Inference images/second on the given split, after one warm-up pass."""
    predict(model, split, amp, batch_size)
    sync = torch.cuda.synchronize if split.device.type == "cuda" else lambda: None
    sync()
    start = time.perf_counter()
    for _ in range(repeats):
        predict(model, split, amp, batch_size)
    sync()
    return repeats * len(split) / (time.perf_counter() - start)


def warmup_cosine(optimizer: torch.optim.Optimizer, warmup: int, total: int) -> torch.optim.lr_scheduler.LambdaLR:
    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train(cfg: TrainConfig, verbose: bool = True) -> dict:
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    log = print if verbose else (lambda *a, **k: None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    seed_everything(cfg.seed)

    data = load_pcam(cfg.data_dir, cfg.fraction, cfg.seed, device)
    model = build_model(cfg.model, cfg.routings, cfg.recon_weight).to(device, memory_format=torch.channels_last)
    n_params = count_parameters(model)
    log(f"[{cfg.name} | {len(data.train):,} training patches ({cfg.fraction:g}) | seed {cfg.seed}] {n_params:,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = warmup_cosine(optimizer, cfg.warmup_steps, cfg.steps)
    batches = data.train.forever(cfg.batch_size, torch.Generator().manual_seed(cfg.seed))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history, best_auc, best_state, best_step = [], -1.0, None, 0
    running: dict[str, torch.Tensor] = {}
    train_time = 0.0
    tick = time.perf_counter()

    for step in range(1, cfg.steps + 1):
        model.train()
        x, y = next(batches)
        if cfg.augment:
            x = random_dihedral(x)
        with autocast(device, cfg.amp):
            out = model(x, y)
            loss, parts = model.loss(out, x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        for k, v in {"loss": loss.detach(), **parts}.items():
            running[k] = running.get(k, 0) + v

        if cfg.is_eval_step(step):
            if device.type == "cuda":
                torch.cuda.synchronize()
            train_time += time.perf_counter() - tick
            n = step - (history[-1]["step"] if history else 0)
            val = binary_metrics(data.val.labels.cpu().numpy(), predict(model, data.val, cfg.amp))
            row = {
                "step": step,
                "epoch": step * cfg.batch_size / len(data.train),
                "lr": scheduler.get_last_lr()[0],
                **{f"train_{k}": float(v) / n for k, v in running.items()},
                "val_auc": val["auc"],
                "val_accuracy": val["accuracy"],
            }
            history.append(row)
            running = {}
            if val["auc"] > best_auc:
                best_auc, best_step = val["auc"], step
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            log(
                f"  step {step:>5}/{cfg.steps}  epoch {row['epoch']:6.1f}  loss {row['train_loss']:.4f}  "
                f"val AUC {val['auc']:.4f}  val acc {val['accuracy']:.4f}{'  *' if best_step == step else ''}"
            )
            tick = time.perf_counter()

    model.load_state_dict(best_state)
    torch.save(best_state, run_dir / "best.pt")

    y_test = data.test.labels.cpu().numpy()
    test_scores = predict(model, data.test, cfg.amp)
    np.savez_compressed(run_dir / "test_scores.npz", y=y_test.astype(np.uint8), score=test_scores.astype(np.float32))

    metrics = {
        "model": cfg.model,
        "variant": cfg.variant,
        "fraction": cfg.fraction,
        "seed": cfg.seed,
        "n_train": len(data.train),
        "params": n_params,
        "params_with_decoder": count_parameters(model, inference_only=False),
        "best_step": best_step,
        "val_auc": best_auc,
        **{f"test_{k}": v for k, v in binary_metrics(y_test, test_scores).items()},
        "train_time_s": train_time,
        "train_steps_per_s": cfg.steps / train_time,
        "inference_img_per_s": measure_throughput(model, data.test, cfg.amp),
        "peak_gpu_mem_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with open(run_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    log(
        f"  -> best val AUC {best_auc:.4f} @ step {best_step}; test AUC {metrics['test_auc']:.4f}, "
        f"acc {metrics['test_accuracy']:.4f}  ({train_time / 60:.1f} min)"
    )
    return metrics


def add_config_args(parser: argparse.ArgumentParser, skip: tuple[str, ...] = ()) -> None:
    """Expose every TrainConfig field as a --flag (booleans get --flag / --no-flag)."""
    for f in fields(TrainConfig):
        if f.name in skip:
            continue
        flag = "--" + f.name.replace("_", "-")
        if f.type == "bool":
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default)
        elif f.name == "model":
            parser.add_argument(flag, choices=MODEL_NAMES, default=f.default)
        else:
            parser.add_argument(flag, type={"int": int, "float": float}.get(f.type, str), default=f.default)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser)
    train(TrainConfig(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
