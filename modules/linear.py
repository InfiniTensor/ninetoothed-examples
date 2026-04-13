import torch
import torch.nn as nn

import ops.ninetoothed.torch
import ops.triton.torch
from modules._utils import _make_backend_manager


class Linear(nn.Module):
    bmm = None

    def __init__(self, other):
        super().__init__()
        self.__dict__ = other.__dict__

    def forward(self, input):
        return type(self).bmm(
            input, self.weight.T.unsqueeze(0).expand(input.shape[0], -1, -1)
        )


bmm_backend = _make_backend_manager(
    Linear,
    "bmm",
    {
        "ninetoothed": ops.ninetoothed.torch.bmm,
        "triton": ops.triton.torch.bmm,
        "torch": torch.bmm,
    },
)
