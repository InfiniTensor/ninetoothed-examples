import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

from ops.ninetoothed.kernels._common import DTYPES, build

BLOCK_SIZE_M = block_size()
BLOCK_SIZE_N = block_size()
BLOCK_SIZE_K = block_size()


def arrangement(
    input,
    other,
    output,
    block_size_m=BLOCK_SIZE_M,
    block_size_n=BLOCK_SIZE_N,
    block_size_k=BLOCK_SIZE_K,
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


_SHAPES = tuple((s, s, s) for s in (2**i for i in range(3, 13)))

configs = tuple(
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
        {"num_warps": nw, "num_stages": ns},
    )
    for m, n, k in _SHAPES
    for dtype in DTYPES
    for bm in (64, 128)
    for bn in (64, 128)
    for bk in (32, 64)
    for nw in (4, 8)
    for ns in (3,)
)

kernel = build(
    premake,
    configs,
    meta_parameters=("block_size_m", "block_size_n", "block_size_k"),
    kernel_name="mm",
)
