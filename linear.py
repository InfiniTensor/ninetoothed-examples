import torch.nn as nn

import ops.ninetoothed.torch


class Linear(nn.Module):
    def __init__(self, other):
        super().__init__()

        self.__dict__ = other.__dict__

    def forward(self, input):
        return ops.ninetoothed.torch.bmm(
            input, self.weight.T.unsqueeze(0).expand(input.shape[0], -1, -1)
        )
