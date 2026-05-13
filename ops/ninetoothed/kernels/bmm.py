import functools

from ninetoothed import Tensor

from ops.ninetoothed.kernels._common import build
from ops.ninetoothed.kernels.mm import application


def arrangement(
    input,
    other,
    output,
    block_size_m,
    block_size_n,
    block_size_k,
):
    output_arranged = output.tile((1, block_size_m, block_size_n))
    output_arranged.dtype = output_arranged.dtype.squeeze(0)

    input_arranged = input.tile((1, block_size_m, block_size_k))
    input_arranged = input_arranged.tile((1, 1, -1))
    input_arranged = input_arranged.expand((-1, -1, output_arranged.shape[-1]))
    input_arranged.dtype = input_arranged.dtype.squeeze((0, 1))
    input_arranged.dtype.dtype = input_arranged.dtype.dtype.squeeze(0)

    other_arranged = other.tile((1, block_size_k, block_size_n))
    other_arranged = other_arranged.tile((1, -1, 1))
    other_arranged = other_arranged.expand((-1, output_arranged.shape[-2], -1))
    other_arranged.dtype = other_arranged.dtype.squeeze((0, 2))
    other_arranged.dtype.dtype = other_arranged.dtype.dtype.squeeze(0)

    return input_arranged, other_arranged, output_arranged


def premake(batch, m, k, n, dtype, block_size_m, block_size_n, block_size_k):
    arrangement_ = functools.partial(
        arrangement,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        block_size_k=block_size_k,
    )
    tensors = (
        Tensor(shape=(batch, m, k), dtype=dtype),
        Tensor(shape=(batch, k, n), dtype=dtype),
        Tensor(shape=(batch, m, n), dtype=dtype),
    )

    return arrangement_, application, tensors


def _configs(batch, m, k, n, dtype):
    return (
        (
            (),
            {
                "batch": batch,
                "m": m,
                "k": k,
                "n": n,
                "dtype": dtype,
                "block_size_m": 16,
                "block_size_n": 64,
                "block_size_k": 32,
            },
            {"num_warps": 4, "num_stages": 3},
        ),
    )


@functools.cache
def _kernel(batch, m, k, n, dtype):
    return build(
        premake,
        _configs(batch, m, k, n, dtype),
        kernel_name=f"bmm_{batch}_{m}_{k}_{n}",
    )


def kernel(lhs, rhs, output, batch, m, k, n, dtype):
    return _kernel(batch, m, k, n, dtype)(
        lhs, rhs, output, batch, m, k, n, dtype, 16, 64, 32, 4, 3
    )
