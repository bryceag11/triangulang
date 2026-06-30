"""Shared helpers for the batched, sequential, and cross-view forward passes.

These utilities capture logic that is identical across
``forward_passes`` (batched), ``forward_passes_cross_view`` and
``forward_passes_seq`` so the three paths stay in sync without copy-paste.
"""
import torch.nn.functional as F


# Auxiliary head outputs that must be connected to the loss graph (with a
# zero multiplier) so DDP gradient sync never trips over unused parameters.
_AUX_HEAD_KEYS = (
    'presence_logit', 'iou_pred', 'per_query_centroids', 'text_scores', 'joint_scores',
)


def connect_aux_heads_to_graph(loss, outputs):
    """Add 0-weighted terms for every auxiliary head present in ``outputs``.

    Keeps the heads connected to the autograd graph for DDP without changing
    the gradient (each term is multiplied by 0).
    """
    for key in _AUX_HEAD_KEYS:
        tensor = outputs.get(key)
        if tensor is not None:
            loss = loss + tensor.sum() * 0.0
    return loss


def connect_trainable_params_to_graph(loss, model, include_query_proj=True):
    """Connect all trainable GASA decoder (and optionally query_proj) params.

    The 0-weighted sum ensures DDP gradient sync never fails on unused params.
    """
    base = model.module if hasattr(model, 'module') else model
    for p in base.gasa_decoder.parameters():
        if p.requires_grad:
            loss = loss + p.sum() * 0.0
    if include_query_proj:
        for p in base.query_proj.parameters():
            if p.requires_grad:
                loss = loss + p.sum() * 0.0
    return loss


def smooth_mask_logits(pred, kernel):
    """Apply 2D average-pool smoothing to mask logits (matches eval LangSplat).

    ``kernel <= 0`` returns ``pred`` unchanged.
    """
    if kernel <= 0:
        return pred
    pad = kernel // 2
    return F.avg_pool2d(
        pred.unsqueeze(1), kernel_size=kernel, stride=1, padding=pad,
        count_include_pad=False,
    ).squeeze(1)


def get_norm_scale(outputs):
    """Return the pointmap normalization scale (meters), or 1.0 if absent."""
    norm_params = outputs.get('norm_params', None)
    if norm_params and 'scale' in norm_params:
        return norm_params['scale'].item()
    return 1.0
