import ninetoothed
import torch
from ninetoothed import Symbol, Tensor

import mm

BLOCK_SIZE_M = Symbol("BLOCK_SIZE_M", meta=True)
BLOCK_SIZE_N = Symbol("BLOCK_SIZE_N", meta=True)
BLOCK_SIZE_K = Symbol("BLOCK_SIZE_K", meta=True)


def arrangement(
    lhs,
    rhs,
    output,
    BLOCK_SIZE_M=BLOCK_SIZE_M,
    BLOCK_SIZE_N=BLOCK_SIZE_N,
    BLOCK_SIZE_K=BLOCK_SIZE_K,
):
    output_arranged = output.tile((1, BLOCK_SIZE_M, BLOCK_SIZE_N))
    output_arranged.dtype = output_arranged.dtype.squeeze(0)

    lhs_arranged = lhs.tile((1, BLOCK_SIZE_M, BLOCK_SIZE_K))
    lhs_arranged = lhs_arranged.tile((1, 1, -1))
    lhs_arranged = lhs_arranged.expand((-1, -1, output_arranged.shape[-1]))
    lhs_arranged.dtype = lhs_arranged.dtype.squeeze((0, 1))
    lhs_arranged.dtype.dtype = lhs_arranged.dtype.dtype.squeeze(0)

    rhs_arranged = rhs.tile((1, BLOCK_SIZE_K, BLOCK_SIZE_N))
    rhs_arranged = rhs_arranged.tile((1, -1, 1))
    rhs_arranged = rhs_arranged.expand((-1, output_arranged.shape[-2], -1))
    rhs_arranged.dtype = rhs_arranged.dtype.squeeze((0, 2))
    rhs_arranged.dtype.dtype = rhs_arranged.dtype.dtype.squeeze(0)

    return lhs_arranged, rhs_arranged, output_arranged


tensors = (Tensor(3), Tensor(3), Tensor(3))
bmm_kernel = ninetoothed.make(arrangement, mm.application, tensors)


def bmm(lhs, rhs):
    output_shape = (lhs.shape[0], lhs.shape[-2], rhs.shape[-1])
    output = torch.empty(output_shape, dtype=lhs.dtype, device=lhs.device)

    bmm_kernel(lhs, rhs, output)

    return output


if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size, m, n, k = 4, 512, 2028, 1024
    dtype = torch.float16
    device = "cuda"

    lhs = torch.randn(batch_size, m, k, dtype=dtype, device=device)
    rhs = torch.randn(batch_size, k, n, dtype=dtype, device=device)

    ninetoothed_output = bmm(lhs, rhs)
    torch_output = torch.bmm(lhs, rhs)

    print(ninetoothed_output)
    print(torch_output)

    if torch.allclose(ninetoothed_output, torch_output):
        print("✅ NineToothed and PyTorch match.")
    else:
        print("❌ NineToothed and PyTorch differ.")
