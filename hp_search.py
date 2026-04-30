"""Algorithm 3 — Coordinate hyperparameter search (one axis at a time)."""
from __future__ import annotations
from typing import Dict, List

import torch
import torch.nn as nn

from config import (
    BASELINE_HP, CKPT_DIR, DEVICE, HP_SEARCH_EPOCHS, HP_SEARCH_PATIENCE,
    HP_SWEEP, set_seed,
)
from data import build_loaders
from engine import (
    EarlyStopping, Timer, append_csv_row, evaluate,
    peak_gpu_mb, reset_gpu_stats, train_one_epoch,
)
from models import build_model, model_size_mb


def _candidate_configs() -> List[Dict]:
    """Coordinate sweep: vary one axis at a time around BASELINE_HP."""
    configs: List[Dict] = []
    seen = set()

    def add(cfg):
        key = (cfg["lr"], cfg["momentum"], cfg["weight_decay"], cfg["batch_size"])
        if key not in seen:
            seen.add(key)
            configs.append(cfg)

    # Always include baseline
    add(dict(BASELINE_HP))
    # LR sweep
    for lr in HP_SWEEP["lr"]:
        c = dict(BASELINE_HP); c["lr"] = lr; add(c)
    # Momentum sweep
    for m in HP_SWEEP["momentum"]:
        c = dict(BASELINE_HP); c["momentum"] = m; add(c)
    # Batch sweep
    for b in HP_SWEEP["batch_size"]:
        c = dict(BASELINE_HP); c["batch_size"] = b; add(c)
    return configs


def _train_head_only(
    model: nn.Module, train_loader, val_loader, hp: Dict, ckpt_path,
    class_weights: torch.Tensor, max_epochs: int = HP_SEARCH_EPOCHS,
    patience: int = HP_SEARCH_PATIENCE,
):
    """Stage-1-style head training (backbone frozen) used for HP screening."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    optimizer = torch.optim.SGD(
        model.fc.parameters(),
        lr=hp["lr"], momentum=hp["momentum"], weight_decay=hp["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    early = EarlyStopping(patience=patience, save_path=ckpt_path, mode="min")

    epochs_run = 0
    for epoch in range(1, max_epochs + 1):
        epochs_run = epoch
        train_one_epoch(model, train_loader, optimizer, criterion)
        val_metrics = evaluate(model, val_loader, criterion=criterion)
        if early(val_metrics["loss"], model):
            break
    # Reload best
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    final_val = evaluate(model, val_loader, criterion=criterion)
    return final_val, epochs_run


def run_hp_search(
    train_df, val_df, test_df, arch: str = "resnet18", aug: bool = False,
) -> Dict:
    """Run coordinate sweep on (arch, aug); return best HP config by val_auroc."""
    set_seed()
    configs = _candidate_configs()
    print(f"\n[HP SEARCH] {len(configs)} configs on arch={arch} aug={aug}")

    best_row = None
    for i, hp in enumerate(configs):
        run_id = f"hp_{arch}_{i:02d}"
        print(f"  [{run_id}] hp={hp}")
        set_seed()  # re-seed per config for fair comparison

        train_loader, val_loader, _, class_weights = build_loaders(
            train_df, val_df, test_df, batch_size=hp["batch_size"], augment=aug,
        )

        model, _ = build_model(arch)
        model.to(DEVICE)
        params_mb = model_size_mb(model)

        ckpt_path = CKPT_DIR / f"{run_id}.pth"
        reset_gpu_stats()
        with Timer() as t:
            val_metrics, epochs_run = _train_head_only(
                model, train_loader, val_loader, hp, ckpt_path, class_weights,
            )
        row = {
            "run_id": run_id, "arch": arch, "aug": int(aug), "stage": 1,
            "lr": hp["lr"], "momentum": hp["momentum"],
            "weight_decay": hp["weight_decay"], "batch_size": hp["batch_size"],
            "epochs_run": epochs_run,
            "val_auroc": val_metrics["auroc"], "val_mcc": val_metrics["mcc"],
            "val_sensitivity": val_metrics["sensitivity"],
            "val_specificity": val_metrics["specificity"],
            "val_f1": val_metrics["f1"],
            "train_time_s": round(t.elapsed, 2),
            "peak_gpu_mb": round(peak_gpu_mb(), 1),
            "model_params_mb": round(params_mb, 2),
            "notes": "hp_search",
        }
        append_csv_row(row)
        if (best_row is None) or (row["val_auroc"] > best_row["val_auroc"]):
            best_row = row

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(
        f"[HP SEARCH] best val_auroc={best_row['val_auroc']:.4f} "
        f"with lr={best_row['lr']} momentum={best_row['momentum']} "
        f"batch_size={best_row['batch_size']}"
    )
    best_hp = {
        "lr": best_row["lr"],
        "momentum": best_row["momentum"],
        "weight_decay": best_row["weight_decay"],
        "batch_size": best_row["batch_size"],
    }
    return best_hp
