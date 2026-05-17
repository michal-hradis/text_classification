"""JEPA pretraining loss functions.

Implements spec §8:

- §8.1  Segment JEPA loss (weighted cosine)
- §8.2  Document-level consistency loss
- §8.3  Variance regularization (collapse prevention)
- §8.3  Covariance regularization (collapse prevention)
- §8.4  Total loss = weighted sum of the above
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Loss weight configuration
# ---------------------------------------------------------------------------

@dataclass
class LossWeights:
    """Weights applied to each loss term.

    Defaults match spec §8.4::

        L_total = 1.0 * L_seg + 0.2 * L_doc + 0.05 * L_var + 0.01 * L_cov
    """
    segment: float = 1.0
    document: float = 0.2
    variance: float = 0.05
    covariance: float = 0.01


# ---------------------------------------------------------------------------
# Individual loss functions
# ---------------------------------------------------------------------------

def segment_jepa_loss(
    predicted: torch.Tensor,   # (B, N, pred_dim)  L2-normalised
    targets: torch.Tensor,     # (B, N, pred_dim)  L2-normalised, stop-gradient
    weights: torch.Tensor,     # (B, N)  per-segment loss weight ∈ [0, 1]
    mask: torch.Tensor,        # (B, N)  bool  True = valid canonical segment
) -> torch.Tensor:
    """Weighted mean cosine JEPA loss over valid canonical segments (spec §8.1).

    ``L_seg = weighted_mean_i [1 - cosine(p_i, stopgrad(z_i))]``
    """
    cos = (predicted * targets.detach()).sum(dim=-1)   # (B, N)
    loss = 1.0 - cos                                    # (B, N)
    valid_weights = weights * mask.float()              # (B, N)
    total_w = valid_weights.sum().clamp(min=1e-8)
    return (loss * valid_weights).sum() / total_w


def document_consistency_loss(
    student_doc: torch.Tensor,   # (B, pred_dim)  L2-normalised
    teacher_doc: torch.Tensor,   # (B, pred_dim)  L2-normalised, stop-gradient
) -> torch.Tensor:
    """Document-level JEPA consistency loss (spec §8.2).

    ``L_doc = mean_batch [1 - cosine(s_doc, stopgrad(z_doc))]``
    """
    cos = (student_doc * teacher_doc.detach()).sum(dim=-1)  # (B,)
    return (1.0 - cos).mean()


def variance_regularization(
    *embeddings: torch.Tensor,
    gamma: float = 0.1,
) -> torch.Tensor:
    """Variance regularisation to prevent representational collapse (spec §8.3).

    Penalises embedding dimensions whose standard deviation across the
    batch is below ``gamma``.  Each argument may be shape ``(B, D)`` or
    ``(B, N, D)``; 3-D tensors are flattened on the first two dimensions.

    ``L_var = mean_j max(0, gamma - std_j)``
    """
    device = embeddings[0].device
    total = torch.zeros((), device=device)
    for emb in embeddings:
        if emb.dim() == 3:
            emb = emb.reshape(-1, emb.shape[-1])
        if emb.shape[0] < 2:
            continue
        std = emb.std(dim=0)           # (D,)
        total = total + F.relu(gamma - std).mean()
    return total / max(len(embeddings), 1)


def covariance_regularization(
    *embeddings: torch.Tensor,
) -> torch.Tensor:
    """Covariance regularisation to reduce embedding anisotropy (spec §8.3).

    Penalises the sum of squared off-diagonal covariance entries (scaled by D).
    3-D tensors are flattened on the first two dimensions.
    """
    device = embeddings[0].device
    total = torch.zeros((), device=device)
    count = 0
    for emb in embeddings:
        if emb.dim() == 3:
            emb = emb.reshape(-1, emb.shape[-1])
        N, D = emb.shape
        if N < 2:
            continue
        emb = emb - emb.mean(dim=0, keepdim=True)
        cov = (emb.T @ emb) / (N - 1)   # (D, D)
        off_diag = cov.pow(2)
        off_diag.fill_diagonal_(0.0)
        total = total + off_diag.sum() / D
        count += 1
    if count == 0:
        return total
    return total / count


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

def compute_total_loss(
    predicted_segments: torch.Tensor,    # (B, N, pred_dim)
    teacher_seg_targets: torch.Tensor,   # (B, N, pred_dim)
    predicted_doc: torch.Tensor,         # (B, pred_dim)
    teacher_doc_targets: torch.Tensor,   # (B, pred_dim)
    segment_loss_weights: torch.Tensor,  # (B, N)
    canonical_mask: torch.Tensor,        # (B, N) bool
    weights: LossWeights,
) -> dict[str, torch.Tensor]:
    """Compute all JEPA losses and return them in a dict.

    Returns:
        Dict with keys ``"loss"`` (total), ``"loss/segment"``,
        ``"loss/document"``, ``"loss/variance"``, ``"loss/covariance"``.
    """
    l_seg = segment_jepa_loss(
        predicted_segments, teacher_seg_targets, segment_loss_weights, canonical_mask
    )
    l_doc = document_consistency_loss(predicted_doc, teacher_doc_targets)
    l_var = variance_regularization(predicted_segments, predicted_doc, gamma=0.1)
    l_cov = covariance_regularization(predicted_segments, predicted_doc)

    total = (
        weights.segment * l_seg
        + weights.document * l_doc
        + weights.variance * l_var
        + weights.covariance * l_cov
    )

    return {
        "loss": total,
        "loss/segment": l_seg,
        "loss/document": l_doc,
        "loss/variance": l_var,
        "loss/covariance": l_cov,
    }
