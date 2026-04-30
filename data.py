"""Algorithm 1 — Patient-level data preparation, splits, transforms, loaders."""
from __future__ import annotations
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from config import (
    DATA_ROOT,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_SIZE,
    LABELS,
    NUM_WORKERS,
    SEED,
)

# NORMAL filename patterns:
#   IM-XXXX-YYYY[-ZZZZ].jpeg          -> patient = IM-XXXX
#   NORMAL2-IM-XXXX-YYYY.jpeg         -> patient = NORMAL2-IM-XXXX
# PNEUMONIA: personNNNN_(bacteria|virus)_NNNN.jpeg -> patient = personNNNN
_NORMAL_RE = re.compile(r"(NORMAL2-IM-\d+|IM-\d+)", re.IGNORECASE)
_PNEU_RE = re.compile(r"(person\d+)", re.IGNORECASE)


def extract_patient_id(filename: str, label: int) -> str:
    """Return a label-prefixed unique patient ID for grouped splitting."""
    name = Path(filename).stem
    if label == LABELS["NORMAL"]:
        m = _NORMAL_RE.search(name)
        pid = m.group(1) if m else name
        return f"NORMAL_{pid}"
    m = _PNEU_RE.search(name)
    pid = m.group(1) if m else name
    return f"PNEUMONIA_{pid}"


def collect_records(root: Path = DATA_ROOT) -> pd.DataFrame:
    """Pool all images from train/, val/, test/ into a single DataFrame."""
    rows = []
    for split in ("train", "val", "test"):
        split_dir = root / split
        for cls_name, label in LABELS.items():
            cls_dir = split_dir / cls_name
            if not cls_dir.is_dir():
                continue
            for p in cls_dir.iterdir():
                if p.suffix.lower() not in {".jpeg", ".jpg", ".png"}:
                    continue
                rows.append(
                    {
                        "filepath": str(p),
                        "label": label,
                        "patient_id": extract_patient_id(p.name, label),
                        "orig_split": split,
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No images found under {root}")
    return df


def patient_level_split(
    df: pd.DataFrame, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Two-stage GroupShuffleSplit on patient_id: 72/13/15 train/val/test."""
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    trainval_idx, test_idx = next(gss1.split(df, groups=df["patient_id"]))
    trainval = df.iloc[trainval_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    train_idx, val_idx = next(gss2.split(trainval, groups=trainval["patient_id"]))
    train = trainval.iloc[train_idx].reset_index(drop=True)
    val = trainval.iloc[val_idx].reset_index(drop=True)

    _assert_no_leakage(train, val, test)
    _print_split_summary(train, val, test)
    return train, val, test


def _assert_no_leakage(*splits: pd.DataFrame) -> None:
    sets = [set(s["patient_id"]) for s in splits]
    names = ["train", "val", "test"]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            inter = sets[i] & sets[j]
            assert not inter, f"Patient leakage between {names[i]} and {names[j]}: {len(inter)} ids"


def _print_split_summary(train, val, test) -> None:
    for name, df in [("train", train), ("val", val), ("test", test)]:
        n = len(df)
        n_pat = df["patient_id"].nunique()
        n_pneu = int((df["label"] == 1).sum())
        n_norm = n - n_pneu
        print(
            f"  [{name:5s}] images={n:5d}  patients={n_pat:5d}  "
            f"NORMAL={n_norm:5d}  PNEUMONIA={n_pneu:5d}  "
            f"%PNEU={n_pneu / max(n, 1):.3f}"
        )


def build_transforms(augment: bool) -> tuple[transforms.Compose, transforms.Compose]:
    """Return (train_tf, eval_tf). Augmentation only applied to train."""
    eval_tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    if not augment:
        return eval_tf, eval_tf

    train_tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, eval_tf


class CXRDataset(Dataset):
    """Chest-X-ray dataset reading from a DataFrame of (filepath, label)."""

    def __init__(self, df: pd.DataFrame, transform):
        self.filepaths = df["filepath"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int):
        img = Image.open(self.filepaths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]


def class_weights_from_labels(labels: Iterable[int], n_classes: int = 2) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss."""
    labels = np.asarray(list(labels))
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


def make_weighted_sampler(labels: Iterable[int]) -> WeightedRandomSampler:
    """WeightedRandomSampler with sample weights = inverse class frequency."""
    labels = np.asarray(list(labels))
    counts = np.bincount(labels).astype(np.float64)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    sample_w = inv[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=len(sample_w),
        replacement=True,
    )


def build_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int,
    augment: bool,
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """Build train/val/test DataLoaders + class_weights tensor."""
    train_tf, eval_tf = build_transforms(augment)

    train_ds = CXRDataset(train_df, train_tf)
    val_ds = CXRDataset(val_df, eval_tf)
    test_ds = CXRDataset(test_df, eval_tf)

    sampler = make_weighted_sampler(train_df["label"].tolist())
    class_weights = class_weights_from_labels(train_df["label"].tolist())

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=32,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=32,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader, class_weights


def get_data_splits(seed: int = SEED):
    """Convenience: collect + split. Cached on the function via Python defaults? No, just call once."""
    df = collect_records()
    return patient_level_split(df, seed=seed)
