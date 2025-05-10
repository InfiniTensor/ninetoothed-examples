import ninetoothed
import ninetoothed.language as ntl
import torch
import torch.nn as nn
import torch.nn.functional as F
from ninetoothed import Symbol, Tensor

BLOCK_SIZE_M = Symbol("BLOCK_SIZE_M", meta=True)
BLOCK_SIZE_N = Symbol("BLOCK_SIZE_N", meta=True)
BLOCK_SIZE_K = Symbol("BLOCK_SIZE_K", meta=True)


def arrangement(
    input,
    output,
    BLOCK_SIZE_M=BLOCK_SIZE_M,
    BLOCK_SIZE_N=BLOCK_SIZE_N,
    BLOCK_SIZE_K=BLOCK_SIZE_K,
):
    tile_shape = (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)

    return input.tile(tile_shape), output.tile(tile_shape)


def application(input, output):
    output = input * ntl.sigmoid(input)  # noqa: F841


silu_kernel = ninetoothed.make(arrangement, application, (Tensor(3), Tensor(3)))


def silu(input):
    output = torch.empty_like(input)

    silu_kernel(input, output)

    return output


class SiLU(nn.Module):
    def __init__(self, other):
        super().__init__()

        self.__dict__ = other.__dict__

    def forward(self, input):
        return silu(input)


if __name__ == "__main__":
    torch.manual_seed(0)

    shape = (8, 256, 512)
    dtype = torch.float32
    device = "cuda"
    input = torch.randn(shape, dtype=dtype, device=device)

    ninetoothed_output = silu(input)
    torch_output = F.silu(input)

    print(ninetoothed_output)
    print(torch_output)

    if torch.allclose(ninetoothed_output, torch_output):
        print("✅ NineToothed and PyTorch match.")
    else:
        print("❌ NineToothed and PyTorch differ.")
