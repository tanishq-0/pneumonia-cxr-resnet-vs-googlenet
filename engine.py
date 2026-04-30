"""Training / evaluation loops, EarlyStopping, CSV logging."""
from __future__ import annotations
import csv
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import DEVICE, RESULTS_CSV
from metrics import compute_metrics

CSV_FIELDS = [
    "run_id", "arch", "aug", "stage",
    "lr", "momentum", "weight_decay", "batch_size", "epochs_run",
    "val_auroc", "val_mcc", "val_sensitivity", "val_specificity", "val_f1",
    "test_auroc", "test_mcc", "test_sensitivity", "test_specificity", "test_f1",
    "train_time_s", "peak_gpu_mb", "model_params_mb",
    "notes",
]


class EarlyStopping:
    """Stop when monitored loss does not improve for `patience` epochs.

    Saves the best model weights to `save_path`.
    """

    def __init__(self, patience: int, save_path: Path, mode: str = "min"):
        self.patience = patience
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.best = float("inf") if mode == "min" else -float("inf")
        self.bad_epochs = 0
        self.stop = False

    def _is_better(self, x: float) -> bool:
        return x < self.best if self.mode == "min" else x > self.best

    def __call__(self, score: float, model: nn.Module) -> bool:
        if self._is_better(score):
            self.best = score
            self.bad_epochs = 0
            torch.save(model.state_dict(), self.save_path)
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.stop = True
        return self.stop


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    grad_clip: Optional[float] = 1.0,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / max(n, 1), correct / max(n, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: Optional[nn.Module] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Forward pass over loader; returns metrics + mean loss + raw arrays."""
    model.eval()
    all_probs, all_labels, total_loss, n = [], [], 0.0, 0
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        logits = model(x)
        if criterion is not None:
            total_loss += criterion(logits, y).item() * x.size(0)
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.cpu().numpy())
        n += x.size(0)
    y_prob = np.concatenate(all_probs) if all_probs else np.array([])
    y_true = np.concatenate(all_labels) if all_labels else np.array([])
    metrics = compute_metrics(y_true, y_prob, threshold=threshold)
    metrics["loss"] = total_loss / max(n, 1) if criterion is not None else float("nan")
    metrics["_y_true"] = y_true
    metrics["_y_prob"] = y_prob
    return metrics


def reset_gpu_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_gpu_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def append_csv_row(row: Dict, path: Path = RESULTS_CSV) -> None:
    path = Path(path)
    write_header = not path.exists()
    safe = {k: row.get(k, "") for k in CSV_FIELDS}
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(safe)


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.time() - self.t0
