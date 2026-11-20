"""FP32 reconstruction for the pinned PatchTST-FM-R2 checkpoint."""

from types import MethodType

import torch


def _overlap_add(module, q_pred: torch.Tensor, original_length: int, stride: int) -> torch.Tensor:
    width = module.config.d_patch
    window = module.patch_loss_window
    # Retain the upstream path for other geometries, BF16 and unexpected layouts.
    if (stride <= 0 or width != 2 * stride or original_length < width or original_length % stride
            or q_pred.ndim != 3 or q_pred.shape[1] != (original_length // stride - 1) * width
            or q_pred.dtype != torch.float32 or window is None
            or window.shape != (1, 1, width) or window.requires_grad
            or window.dtype != q_pred.dtype or window.device != q_pred.device):
        return type(module)._overlap_add(
            module, q_pred, original_length=original_length, stride=stride,
        )

    batch, _, quantiles = q_pred.shape
    patches = original_length // stride - 1
    weights = window.reshape(width)
    weighted = q_pred.reshape(batch, patches, width, quantiles) * weights[None, None, :, None]
    left, right = weighted.split(stride, dim=2)
    # At each interior step the previous patch contributes its right half first.
    # Keep that addition order without building a long chain of in-place slices.
    total = torch.cat((left[:, :1], right[:, :-1] + left[:, 1:], right[:, -1:]), dim=1)
    wl, wr = weights.split(stride)
    coverage = torch.cat((wl[None], (wr + wl).expand(patches - 1, -1), wr[None]), dim=0)
    return total.flatten(1, 2) / coverage.reshape(1, original_length, 1).clamp_min(1e-6)


def install_overlap_add(backbone) -> None:
    """Bind to a single module; deepcopy rebinds the method to the fresh copy."""
    backbone._overlap_add = MethodType(_overlap_add, backbone)
