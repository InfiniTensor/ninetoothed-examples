import torch.nn as nn
import torch.nn.functional as F

import ops.ninetoothed.torch
import ops.triton.torch
from modules._utils import _make_backend_manager


def _torch_fused_rms_norm(x, w, eps):
    return F.rms_norm(x, x.shape[-1:], w, eps)


class RMSNorm(nn.Module):
    fused_rms_norm = None

    def __init__(self, other):
        super().__init__()
        self.__dict__ = other.__dict__

    def forward(self, x):
        return type(self).fused_rms_norm(x, self.weight, self.variance_epsilon)


rms_norm_backend = _make_backend_manager(
    RMSNorm,
    "fused_rms_norm",
    {
        "ninetoothed": ops.ninetoothed.torch.fused_rms_norm,
        "triton": ops.triton.torch.fused_rms_norm,
        "torch": _torch_fused_rms_norm,
    },
)
