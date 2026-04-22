import functools

import ninetoothed
from ninetoothed import Tensor, block_size

from ops.ninetoothed.kernels._common import DTYPES, build
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


def premake(k, n, dtype, block_size_m, block_size_n, block_size_k):
    arrangement_ = functools.partial(
        arrangement,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        block_size_k=block_size_k,
    )
    shape_options = ({"upper_bound": 4}, None, None)
    tensors = (
        Tensor(shape=(None, None, k), shape_options=shape_options, dtype=dtype),
        Tensor(shape=(None, k, n), shape_options=shape_options, dtype=dtype),
        Tensor(shape=(None, None, n), shape_options=shape_options, dtype=dtype),
    )

    return arrangement_, application, tensors


_SHAPES = (
    (4096, 4096),
    (4096, 1024),
    (4096, 14336),
    (14336, 4096),
    (4096, 128256),
)

configs = tuple(
    (
        (),
        {
            "k": k,
            "n": n,
            "dtype": dtype,
            "block_size_m": bm,
            "block_size_n": bn,
            "block_size_k": bk,
        },
        {"num_warps": nw, "num_stages": ns},
    )
    for k, n in _SHAPES
    for dtype in DTYPES
    for bm in (16, 64)
    for bn in (64, 128)
    for bk in (32, 64)
    for nw in (4, 8)
    for ns in (3, 4)
)

_build_kernel = build(
    premake,
    configs,
    meta_parameters=("block_size_m", "block_size_n", "block_size_k"),
    kernel_name="bmm",
)


_BUILD_KN = frozenset(_SHAPES)


_BLOCK_SIZE_M = block_size()
_BLOCK_SIZE_N = block_size()
_BLOCK_SIZE_K = block_size()


def _fallback_arrangement(
    input,
    other,
    output,
    BLOCK_SIZE_M=_BLOCK_SIZE_M,
    BLOCK_SIZE_N=_BLOCK_SIZE_N,
    BLOCK_SIZE_K=_BLOCK_SIZE_K,
):
    return arrangement(input, other, output, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)


_fallback_kernel = ninetoothed.make(
    _fallback_arrangement, application, (Tensor(3), Tensor(3), Tensor(3))
)


def kernel(lhs, rhs, output, k, n, dtype):
    if (k, n) in _BUILD_KN:
        return _build_kernel(lhs, rhs, output, k, n, dtype)

    return _fallback_kernel(lhs, rhs, output)
