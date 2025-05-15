from contextlib import contextmanager

import torch
import torch.nn as nn

import ops.ninetoothed.torch


class Linear(nn.Module):
    bmm = None

    def __init__(self, other):
        super().__init__()

        self.__dict__ = other.__dict__

    def forward(self, input):
        return type(self).bmm(
            input, self.weight.T.unsqueeze(0).expand(input.shape[0], -1, -1)
        )


@contextmanager
def bmm_backend(backend_name):
    _prev_impl = Linear.bmm

    if backend_name == "ninetoothed":
        impl = ops.ninetoothed.torch.bmm
    elif backend_name == "triton":
        impl = ops.triton.torch.bmm
    elif backend_name == "torch":
        impl = torch.bmm
    else:
        raise ValueError(f"unknown backend: `{backend_name}`")

    Linear.bmm = impl

    try:
        yield
    finally:
        Linear.bmm = _prev_impl
