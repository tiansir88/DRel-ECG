import numpy as np
import torch

from mcki_ecg.losses import GHNMLoss
from mcki_ecg.relation_graph import blend_relation_matrices, estimate_confusion_matrix_from_probs


def test_confusion_normalization_and_fusion():
    targets = np.asarray([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ], dtype=np.float32)
    probs = np.full_like(targets, 0.1)
    probs[:, 1] += np.asarray([0.4, 0.0, 0.2, 0.1, 0.3])
    confusion = estimate_confusion_matrix_from_probs(probs, targets)
    off_diagonal = confusion.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    assert np.isclose(off_diagonal.max(), 1.0)
    fused = blend_relation_matrices(np.eye(5, dtype=np.float32), confusion, 0.5, 0.5)
    assert np.allclose(fused, fused.T)
    assert np.allclose(np.diag(fused), 1.0)


def test_ghnm_loss_is_finite():
    features = torch.randn(8, 16)
    labels = torch.tensor([
        [1, 0, 0, 0, 0], [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0], [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0], [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0], [0, 0, 0, 1, 0],
    ], dtype=torch.float32)
    loss = GHNMLoss()(features, labels)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
