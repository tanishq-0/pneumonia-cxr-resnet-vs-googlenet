"""Global configuration: seeds, paths, device, experiment matrix."""
from __future__ import annotations
import os
import random
from pathlib import Path

import numpy as np
import torch

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "chest_xray"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
FIG_DIR = OUTPUT_DIR / "figures"
RESULTS_CSV = OUTPUT_DIR / "results.csv"

for d in (OUTPUT_DIR, CKPT_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4-model comparison matrix: 2 architectures x 2 augmentation settings
ARCHS = ["resnet18", "googlenet"]
AUGS = [False, True]

LABELS = {"NORMAL": 0, "PNEUMONIA": 1}
INV_LABELS = {v: k for k, v in LABELS.items()}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224

BASELINE_HP = {
    "lr": 1e-3,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "batch_size": 32,
}

HP_SWEEP = {
    "lr": [1e-2, 1e-3, 1e-4],
    "momentum": [0.85, 0.9, 0.95],
    "batch_size": [16, 32],
}

STAGE1_EPOCHS = 15
STAGE1_PATIENCE = 5
STAGE2_EPOCHS = 20
STAGE2_PATIENCE = 7
HP_SEARCH_EPOCHS = 10
HP_SEARCH_PATIENCE = 5

NUM_WORKERS = 0  # Windows-safe; bump to 2-4 on Linux


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) deterministically."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
