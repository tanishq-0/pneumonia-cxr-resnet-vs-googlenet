"""Algorithm 2 — Build ResNet-18 / GoogLeNet with 2-class head; expose GradCAM target."""
from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models


def build_model(arch: str, num_classes: int = 2) -> Tuple[nn.Module, nn.Module]:
    """Return (model, target_layer_for_gradcam)."""
    arch = arch.lower()
    if arch == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        in_feat = model.fc.in_features  # 512
        model.fc = nn.Linear(in_feat, num_classes)
        target_layer = model.layer4[-1]
    elif arch == "googlenet":
        weights = models.GoogLeNet_Weights.IMAGENET1K_V1
        model = models.googlenet(weights=weights, aux_logits=True)
        # Disable aux classifiers post-load to avoid dual-head loss
        model.aux_logits = False
        model.aux1 = None
        model.aux2 = None
        in_feat = model.fc.in_features  # 1024
        model.fc = nn.Linear(in_feat, num_classes)
        target_layer = model.inception5b
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return model, target_layer


def model_size_mb(model: nn.Module) -> float:
    """Memory footprint (parameters only) in MiB."""
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def differential_lr_param_groups(
    model: nn.Module, arch: str, base_lr: float
) -> list[dict]:
    """Stage-2 differential-LR groups (early ×0.01, mid ×0.1, head ×1.0)."""
    arch = arch.lower()
    groups: list[dict] = []
    if arch == "resnet18":
        early = list(model.conv1.parameters()) + list(model.bn1.parameters()) \
                + list(model.layer1.parameters()) + list(model.layer2.parameters())
        mid = list(model.layer3.parameters()) + list(model.layer4.parameters())
        head = list(model.fc.parameters())
    elif arch == "googlenet":
        # Early: stem + inception3a/b + inception4a-e
        early_modules = [
            model.conv1, model.conv2, model.conv3,
            model.inception3a, model.inception3b,
            model.inception4a, model.inception4b, model.inception4c,
            model.inception4d, model.inception4e,
        ]
        early = [p for m in early_modules for p in m.parameters()]
        mid_modules = [model.inception5a, model.inception5b]
        mid = [p for m in mid_modules for p in m.parameters()]
        head = list(model.fc.parameters())
    else:
        raise ValueError(arch)

    groups.append({"params": early, "lr": base_lr * 0.01})
    groups.append({"params": mid, "lr": base_lr * 0.1})
    groups.append({"params": head, "lr": base_lr})
    return groups
