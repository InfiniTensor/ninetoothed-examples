import ninetoothed
import ninetoothed.language as ntl
import torch
import torch.nn as nn
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


@ninetoothed.jit
def fused_rms_norm_kernel(
    x: Tensor(2).tile((1, BLOCK_SIZE)),
    w: Tensor(2).tile((1, BLOCK_SIZE)),
    y: Tensor(2).tile((1, BLOCK_SIZE)),
    eps: Tensor(0),
):
    x_fp32 = ntl.cast(x, ntl.float32)
    y = x_fp32 * ntl.rsqrt(ntl.sum(x_fp32 * x_fp32) / x.shape[-1] + eps) * w  # noqa: F841


def fused_rms_norm(x, w, eps=None):
    if eps is None:
        eps = torch.finfo(x.dtype).eps()

    x_2d = x.view(-1, x.shape[-1])
    w_2d = w.expand_as(x_2d)
    y_2d = torch.empty_like(x_2d)

    fused_rms_norm_kernel(x_2d, w_2d, y_2d, eps, BLOCK_SIZE=x.shape[-1])

    return y_2d.view(x.shape)


class RMSNorm(nn.Module):
    def __init__(self, other):
        super().__init__()

        self.weight = other.weight
        self.variance_epsilon = other.variance_epsilon

    def forward(self, x):
        return fused_rms_norm(x, self.weight, self.variance_epsilon)
