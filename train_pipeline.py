"""Two-stage transfer learning: Stage-1 head training and Stage-2 full fine-tuning."""
from __future__ import annotations
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from config import (
    CKPT_DIR, DEVICE, STAGE1_EPOCHS, STAGE1_PATIENCE,
    STAGE2_EPOCHS, STAGE2_PATIENCE, set_seed,
)
from data import build_loaders
from engine import (
    EarlyStopping, Timer, append_csv_row, evaluate,
    peak_gpu_mb, reset_gpu_stats, train_one_epoch,
)
from models import build_model, differential_lr_param_groups, model_size_mb


def _ckpt_path(stage: int, arch: str, aug: bool) -> Path:
    return CKPT_DIR / f"s{stage}_{arch}_aug{int(aug)}_best.pth"


def stage1_head_training(
    arch: str, aug: bool, hp: Dict,
    train_df, val_df, test_df,
) -> Dict:
    """Stage 1: freeze the backbone and train only the fc head."""
    set_seed()
    print(f"\n[STAGE 1] arch={arch} aug={aug} hp={hp}")
    train_loader, val_loader, _, class_weights = build_loaders(
        train_df, val_df, test_df, batch_size=hp["batch_size"], augment=aug,
    )

    model, _ = build_model(arch)
    model.to(DEVICE)
    params_mb = model_size_mb(model)

    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    optimizer = torch.optim.SGD(
        model.fc.parameters(),
        lr=hp["lr"], momentum=hp["momentum"], weight_decay=hp["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    ckpt = _ckpt_path(1, arch, aug)
    early = EarlyStopping(patience=STAGE1_PATIENCE, save_path=ckpt, mode="min")

    reset_gpu_stats()
    epochs_run = 0
    with Timer() as t:
        for epoch in range(1, STAGE1_EPOCHS + 1):
            epochs_run = epoch
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
            val_m = evaluate(model, val_loader, criterion=criterion)
            scheduler.step(val_m["auroc"])
            print(
                f"  ep{epoch:02d} tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} "
                f"val_loss={val_m['loss']:.4f} val_acc={val_m['accuracy']:.3f} "
                f"val_auroc={val_m['auroc']:.4f} val_sens={val_m['sensitivity']:.3f} "
                f"val_spec={val_m['specificity']:.3f}"
            )
            if early(val_m["loss"], model):
                print(f"  early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    final_val = evaluate(model, val_loader, criterion=criterion)

    row = {
        "run_id": f"s1_{arch}_aug{int(aug)}",
        "arch": arch, "aug": int(aug), "stage": 1,
        "lr": hp["lr"], "momentum": hp["momentum"],
        "weight_decay": hp["weight_decay"], "batch_size": hp["batch_size"],
        "epochs_run": epochs_run,
        "val_auroc": final_val["auroc"], "val_mcc": final_val["mcc"],
        "val_sensitivity": final_val["sensitivity"],
        "val_specificity": final_val["specificity"],
        "val_f1": final_val["f1"],
        "train_time_s": round(t.elapsed, 2),
        "peak_gpu_mb": round(peak_gpu_mb(), 1),
        "model_params_mb": round(params_mb, 2),
        "notes": "stage1",
    }
    append_csv_row(row)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def stage2_full_finetune(
    arch: str, aug: bool, hp: Dict,
    train_df, val_df, test_df,
) -> Dict:
    """Stage 2: load Stage-1 weights, unfreeze all layers, differential LR + cosine schedule."""
    set_seed()
    print(f"\n[STAGE 2] arch={arch} aug={aug} hp={hp}")
    train_loader, val_loader, _, class_weights = build_loaders(
        train_df, val_df, test_df, batch_size=hp["batch_size"], augment=aug,
    )

    model, _ = build_model(arch)
    s1_ckpt = _ckpt_path(1, arch, aug)
    if not s1_ckpt.exists():
        raise FileNotFoundError(f"Missing Stage-1 checkpoint at {s1_ckpt}")
    model.load_state_dict(torch.load(s1_ckpt, map_location="cpu"))
    model.to(DEVICE)
    params_mb = model_size_mb(model)

    for p in model.parameters():
        p.requires_grad = True

    param_groups = differential_lr_param_groups(model, arch, base_lr=hp["lr"])
    for g in param_groups:
        g["momentum"] = hp["momentum"]
        g["weight_decay"] = hp["weight_decay"]
    optimizer = torch.optim.SGD(param_groups)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    scheduler = CosineAnnealingLR(optimizer, T_max=STAGE2_EPOCHS, eta_min=1e-7)
    ckpt = _ckpt_path(2, arch, aug)
    early = EarlyStopping(patience=STAGE2_PATIENCE, save_path=ckpt, mode="min")

    reset_gpu_stats()
    epochs_run = 0
    with Timer() as t:
        for epoch in range(1, STAGE2_EPOCHS + 1):
            epochs_run = epoch
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
            val_m = evaluate(model, val_loader, criterion=criterion)
            scheduler.step()
            print(
                f"  ep{epoch:02d} tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} "
                f"val_loss={val_m['loss']:.4f} val_acc={val_m['accuracy']:.3f} "
                f"val_auroc={val_m['auroc']:.4f} val_sens={val_m['sensitivity']:.3f} "
                f"val_spec={val_m['specificity']:.3f}"
            )
            if early(val_m["loss"], model):
                print(f"  early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    final_val = evaluate(model, val_loader, criterion=criterion)

    row = {
        "run_id": f"s2_{arch}_aug{int(aug)}",
        "arch": arch, "aug": int(aug), "stage": 2,
        "lr": hp["lr"], "momentum": hp["momentum"],
        "weight_decay": hp["weight_decay"], "batch_size": hp["batch_size"],
        "epochs_run": epochs_run,
        "val_auroc": final_val["auroc"], "val_mcc": final_val["mcc"],
        "val_sensitivity": final_val["sensitivity"],
        "val_specificity": final_val["specificity"],
        "val_f1": final_val["f1"],
        "train_time_s": round(t.elapsed, 2),
        "peak_gpu_mb": round(peak_gpu_mb(), 1),
        "model_params_mb": round(params_mb, 2),
        "notes": "stage2",
    }
    append_csv_row(row)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row
