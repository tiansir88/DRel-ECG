from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .model import DRelECGModel


SUPPORTED_BACKBONES = {
    "resnet18",
    "default",
    "drel_ecg_resnet18",
    "two_stage_resnet18",
    "mcki_ecg_resnet18",  # Legacy configuration alias.
}


def _infer_backbone_dim(model: nn.Module, device: torch.device) -> int:
    with torch.no_grad():
        dummy_in = torch.randn(1, 12, 1000, device=device)
        dummy_out = model.backbone(dummy_in)
        feat = dummy_out[0] if isinstance(dummy_out, tuple) else dummy_out
    return int(feat.shape[1])


def build_drel_ecg_backbone(
    backbone_name: str,
    num_classes: int,
    device: torch.device,
    cfg: Optional[Dict] = None,
) -> Tuple[nn.Module, int]:
    cfg = cfg or {}
    normalized_name = str(backbone_name).lower().strip()

    if normalized_name not in SUPPORTED_BACKBONES:
        raise ValueError(
            f'DRel-ECG supports {sorted(SUPPORTED_BACKBONES)}. '
            f'Got: {backbone_name}'
        )

    projection_dim = int(cfg.get('proj_dim', 128))
    model = DRelECGModel(num_classes=num_classes, projection_dim=projection_dim).to(device)
    backbone_dim = _infer_backbone_dim(model, device)
    return model, backbone_dim


# Compatibility aliases used by archived experiment configurations.
build_DRel_backbone = build_drel_ecg_backbone
build_mcki_ecg_backbone = build_drel_ecg_backbone
build_MCKI_backbone = build_drel_ecg_backbone
