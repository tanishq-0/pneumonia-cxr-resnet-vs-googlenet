"""Algorithm 6 — Test-set evaluation with threshold sweep + GradCAM++ figures."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

from config import (
    CKPT_DIR, DEVICE, FIG_DIR, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, set_seed,
)
from data import build_loaders, build_transforms
from engine import evaluate, append_csv_row, peak_gpu_mb
from metrics import compute_metrics, find_threshold_for_sensitivity
from models import build_model, model_size_mb


def _ckpt_path(stage: int, arch: str, aug: bool) -> Path:
    return CKPT_DIR / f"s{stage}_{arch}_aug{int(aug)}_best.pth"


def _denormalize(img_t: torch.Tensor) -> np.ndarray:
    """Tensor (C,H,W) normalised → HxWx3 float [0,1]."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = img_t.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def _try_gradcam_plus_plus(model, target_layer, x_batch, target_classes):
    """Run GradCAM++ on a batch; returns numpy array of CAMs (B,H,W) or None."""
    try:
        from pytorch_grad_cam import GradCAMPlusPlus
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        print("  pytorch-grad-cam not installed; skipping GradCAM figures.")
        return None

    targets = [ClassifierOutputTarget(int(c)) for c in target_classes]
    with GradCAMPlusPlus(model=model, target_layers=[target_layer]) as cam:
        grayscale = cam(input_tensor=x_batch, targets=targets)
    return grayscale  # (B,H,W) in [0,1]


def _save_gradcam_figure(
    arch: str, aug: bool, examples: List[Dict],
) -> None:
    """Save a 1x4 grid: TP, TN, FN, FP with overlaid heatmaps."""
    if not examples:
        return
    fig, axes = plt.subplots(1, len(examples), figsize=(4 * len(examples), 4))
    if len(examples) == 1:
        axes = [axes]
    for ax, ex in zip(axes, examples):
        img = ex["image"]
        cam = ex["cam"]
        if cam is not None:
            heatmap = plt.cm.jet(cam)[..., :3]
            overlay = 0.5 * img + 0.5 * heatmap
            overlay = np.clip(overlay, 0, 1)
        else:
            overlay = img
        ax.imshow(overlay)
        ax.set_title(f"{ex['kind']} (p={ex['prob']:.2f})", fontsize=10)
        ax.axis("off")
    fig.suptitle(f"GradCAM++  {arch}  aug={aug}")
    fig.tight_layout()
    out = FIG_DIR / f"gradcam_{arch}_aug{int(aug)}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def _save_threshold_curve(
    arch: str, aug: bool, y_true: np.ndarray, y_prob: np.ndarray,
) -> None:
    thresholds = np.linspace(0.05, 0.95, 19)
    sens, spec = [], []
    for t in thresholds:
        m = compute_metrics(y_true, y_prob, threshold=float(t))
        sens.append(m["sensitivity"])
        spec.append(m["specificity"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thresholds, sens, "-o", label="sensitivity")
    ax.plot(thresholds, spec, "-o", label="specificity")
    ax.axhline(0.95, color="grey", linestyle="--", linewidth=0.8,
               label="sens target 0.95")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Metric value")
    ax.set_title(f"Threshold sweep  {arch}  aug={aug}")
    ax.legend()
    ax.grid(alpha=0.3)
    out = FIG_DIR / f"threshold_{arch}_aug{int(aug)}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def _pick_examples(y_true, y_prob, threshold=0.5):
    """Indices for one TP, TN, FN, FP — most confident in each category."""
    y_pred = (y_prob >= threshold).astype(int)
    out = {}
    masks = {
        "TP": (y_pred == 1) & (y_true == 1),
        "TN": (y_pred == 0) & (y_true == 0),
        "FN": (y_pred == 0) & (y_true == 1),
        "FP": (y_pred == 1) & (y_true == 0),
    }
    for kind, mask in masks.items():
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        # choose max-confidence-in-prediction example
        if kind in ("TP", "FP"):
            i = idx[np.argmax(y_prob[idx])]
        else:
            i = idx[np.argmin(y_prob[idx])]
        out[kind] = int(i)
    return out


def evaluate_variant(
    arch: str, aug: bool, hp: Dict, train_df, val_df, test_df,
) -> Dict:
    """Algorithm 6 for one (arch, aug). Loads s2 ckpt, computes metrics + figures."""
    set_seed()
    print(f"\n[TEST EVAL] arch={arch} aug={aug}")
    train_loader, val_loader, test_loader, _ = build_loaders(
        train_df, val_df, test_df, batch_size=hp["batch_size"], augment=aug,
    )

    model, target_layer = build_model(arch)
    s2 = _ckpt_path(2, arch, aug)
    if not s2.exists():
        raise FileNotFoundError(f"Missing Stage-2 checkpoint at {s2}")
    model.load_state_dict(torch.load(s2, map_location="cpu"))
    model.to(DEVICE)
    params_mb = model_size_mb(model)

    test_m = evaluate(model, test_loader, criterion=nn.CrossEntropyLoss())
    y_true, y_prob = test_m["_y_true"], test_m["_y_prob"]

    print(
        f"  test  AUROC={test_m['auroc']:.4f}  MCC={test_m['mcc']:.4f}  "
        f"sens={test_m['sensitivity']:.3f}  spec={test_m['specificity']:.3f}  "
        f"F1={test_m['f1']:.3f}"
    )
    print(
        f"  CM    TP={test_m['tp']} TN={test_m['tn']} FP={test_m['fp']} "
        f"FN={test_m['fn']}"
    )

    # Threshold sweep for report
    threshold_table = []
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        m = compute_metrics(y_true, y_prob, threshold=t)
        threshold_table.append({
            "threshold": t,
            "sensitivity": round(m["sensitivity"], 4),
            "specificity": round(m["specificity"], 4),
        })
    df_t = pd.DataFrame(threshold_table)
    df_t.to_csv(FIG_DIR / f"threshold_table_{arch}_aug{int(aug)}.csv", index=False)
    print(f"  threshold sweep:\n{df_t.to_string(index=False)}")

    clinical_t = find_threshold_for_sensitivity(y_true, y_prob, target_sens=0.95)
    clinical_m = compute_metrics(y_true, y_prob, threshold=clinical_t)
    print(
        f"  clinical threshold (sens>=0.95): t={clinical_t:.3f} "
        f"sens={clinical_m['sensitivity']:.3f} spec={clinical_m['specificity']:.3f}"
    )

    _save_threshold_curve(arch, aug, y_true, y_prob)

    # GradCAM++ on selected examples
    pick = _pick_examples(y_true, y_prob)
    examples = []
    if pick:
        # Reload eval-transform versions of the picked indices
        _, eval_tf = build_transforms(augment=False)
        kinds, idxs = list(pick.keys()), list(pick.values())
        imgs, labels = [], []
        for i in idxs:
            fp = test_df.iloc[i]["filepath"]
            pil = Image.open(fp).convert("RGB")
            t = eval_tf(pil)
            imgs.append(t)
            labels.append(int(test_df.iloc[i]["label"]))
        x = torch.stack(imgs).to(DEVICE)
        # GradCAM target = predicted class direction (PNEUMONIA=1)
        cams = _try_gradcam_plus_plus(model, target_layer, x, target_classes=[1] * len(idxs))
        for j, (kind, i) in enumerate(zip(kinds, idxs)):
            img_np = _denormalize(imgs[j])
            cam_j = cams[j] if cams is not None else None
            examples.append({
                "kind": kind, "image": img_np, "cam": cam_j,
                "prob": float(y_prob[i]),
            })
    _save_gradcam_figure(arch, aug, examples)

    val_m = evaluate(model, val_loader, criterion=nn.CrossEntropyLoss())
    row = {
        "run_id": f"eval_{arch}_aug{int(aug)}",
        "arch": arch, "aug": int(aug), "stage": 2,
        "lr": hp["lr"], "momentum": hp["momentum"],
        "weight_decay": hp["weight_decay"], "batch_size": hp["batch_size"],
        "epochs_run": "",
        "val_auroc": val_m["auroc"], "val_mcc": val_m["mcc"],
        "val_sensitivity": val_m["sensitivity"],
        "val_specificity": val_m["specificity"],
        "val_f1": val_m["f1"],
        "test_auroc": test_m["auroc"], "test_mcc": test_m["mcc"],
        "test_sensitivity": test_m["sensitivity"],
        "test_specificity": test_m["specificity"],
        "test_f1": test_m["f1"],
        "train_time_s": "", "peak_gpu_mb": round(peak_gpu_mb(), 1),
        "model_params_mb": round(params_mb, 2),
        "notes": (
            f"clin_t={clinical_t:.3f} "
            f"TP={test_m['tp']} TN={test_m['tn']} "
            f"FP={test_m['fp']} FN={test_m['fn']}"
        ),
    }
    append_csv_row(row)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row
