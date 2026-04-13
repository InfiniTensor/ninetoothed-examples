import torch.nn as nn
import torch.nn.functional as F

import ops.ninetoothed.torch
import ops.triton.torch
from modules._utils import _make_backend_manager


class SiLU(nn.Module):
    silu = None

    def __init__(self, other):
        super().__init__()
        self.__dict__ = other.__dict__

    def forward(self, input):
        return type(self).silu(input)


silu_backend = _make_backend_manager(
    SiLU,
    "silu",
    {
        "ninetoothed": ops.ninetoothed.torch.silu,
        "triton": ops.triton.torch.silu,
        "torch": F.silu,
    },
)
