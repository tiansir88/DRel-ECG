"""Core DRel-ECG encoder and downstream classifier."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet1d import resnet18


class DRelECGModel(nn.Module):
    """ResNet-18 encoder with contrastive and multilabel heads.

    The encoder is shared by label-informed contrastive pretraining and all
    downstream protocols.  Strict linear probing freezes the complete encoder
    and trains a new linear classifier outside this module.
    """

    def __init__(self, num_classes: int = 5, projection_dim: int = 128):
        super().__init__()
        self.backbone = resnet18(num_classes=num_classes)
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, projection_dim),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return features[0] if isinstance(features, tuple) else features

    def forward_contrastive(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection_head(self.forward_backbone(x)), dim=1)

    def forward_classifier(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_backbone(x))

    def forward(self, x: torch.Tensor, mode: str = "classifier") -> torch.Tensor:
        if mode == "backbone":
            return self.forward_backbone(x)
        if mode == "contrastive":
            return self.forward_contrastive(x)
        if mode in {"classifier", "cls"}:
            return self.forward_classifier(x)
        raise ValueError("mode must be 'backbone', 'contrastive', or 'classifier'")


# Backward-compatible class alias for checkpoints and scripts created before
# the public method name changed from MCKI-ECG to DRel-ECG.
MCKIECGModel = DRelECGModel
