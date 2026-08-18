"""DRel-ECG reference implementation."""

from .losses import GHNMLoss
from .model import DRelECGModel, MCKIECGModel

METHOD_NAME = "DRel-ECG"
METHOD_FULL_NAME = (
    "Diagnostic-Relation-Guided Contrastive Pretraining for Multi-Label ECG Diagnosis"
)

__all__ = [
    "DRelECGModel",
    "GHNMLoss",
    "METHOD_FULL_NAME",
    "METHOD_NAME",
    "MCKIECGModel",
]
