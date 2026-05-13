import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ops.ninetoothed.kernels._common import build


def arrangement(
    input,
    other,
    output,
    block_size_m,
    block_size_n,
    block_size_k,
):
    output_arranged = output.tile((block_size_m, block_size_n))

    input_arranged = input.tile((block_size_m, block_size_k))
    input_arranged = input_arranged.tile((1, -1))
    input_arranged = input_arranged.expand((-1, output_arranged.shape[1]))
    input_arranged.dtype = input_arranged.dtype.squeeze(0)

    other_arranged = other.tile((block_size_k, block_size_n))
    other_arranged = other_arranged.tile((-1, 1))
    other_arranged = other_arranged.expand((output_arranged.shape[0], -1))
    other_arranged.dtype = other_arranged.dtype.squeeze(1)

    return input_arranged, other_arranged, output_arranged


def application(input, other, output):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)

    for k in range(input.shape[0]):
        accumulator += ntl.dot(input[k], other[k])

    output = accumulator  # noqa: F841


def premake(m, n, k, dtype, block_size_m, block_size_n, block_size_k):
    arrangement_ = functools.partial(
        arrangement,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        block_size_k=block_size_k,
    )
    tensors = (
        Tensor(shape=(m, k), dtype=dtype),
        Tensor(shape=(k, n), dtype=dtype),
        Tensor(shape=(m, n), dtype=dtype),
    )

    return arrangement_, application, tensors


def _configs(m, n, k, dtype):
    return tuple(
        (
            (),
            {
                "m": m,
                "n": n,
                "k": k,
                "dtype": dtype,
                "block_size_m": bm,
                "block_size_n": bn,
                "block_size_k": bk,
            },
            {"num_warps": nw, "num_stages": 3},
        )
        for bm in (64, 128)
        for bn in (64, 128)
        for bk in (32, 64)
        for nw in (4, 8)
    )


@functools.cache
def _kernel(m, n, k, dtype):
    return build(
        premake,
        _configs(m, n, k, dtype),
        meta_parameters=("block_size_m", "block_size_n", "block_size_k"),
        kernel_name=f"mm_{m}_{n}_{k}",
    )


def kernel(input, other, output, m, n, k, dtype):
    return _kernel(m, n, k, dtype)(input, other, output, m, n, k, dtype)
