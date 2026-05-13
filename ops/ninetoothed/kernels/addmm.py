import functools

import ninetoothed
from ninetoothed import Tensor

import ops.ninetoothed.kernels.mm as mm
from ops.ninetoothed.kernels._common import build


def arrangement(input, mat1, mat2, beta, alpha, output, **block_sizes):
    _, _, input_arranged = mm.arrangement(mat1, mat2, input, **block_sizes)
    mat1_arranged, mat2_arranged, output_arranged = mm.arrangement(
        mat1, mat2, output, **block_sizes
    )

    return input_arranged, mat1_arranged, mat2_arranged, beta, alpha, output_arranged


def application(input, mat1, mat2, beta, alpha, output):
    mm.application(mat1, mat2, output)
    output = beta * input + alpha * output


def premake(m, n, k, dtype, block_size_m, block_size_n, block_size_k):
    arrangement_ = functools.partial(
        arrangement,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        block_size_k=block_size_k,
    )
    tensors = (
        Tensor(shape=(m, n), dtype=dtype),
        Tensor(shape=(m, k), dtype=dtype),
        Tensor(shape=(k, n), dtype=dtype),
        Tensor(0, dtype=ninetoothed.float32),
        Tensor(0, dtype=ninetoothed.float32),
        Tensor(shape=(m, n), dtype=dtype),
    )

    return arrangement_, application, tensors


# Compile the square shapes used by the benchmark; other shapes use the generic
# make-based fallback below.
configs = tuple(
    (
        (),
        {
            "m": s,
            "n": s,
            "k": s,
            "dtype": ninetoothed.float16,
            "block_size_m": bm,
            "block_size_n": bn,
            "block_size_k": bk,
        },
        {"num_warps": nw, "num_stages": 3},
    )
    for s in (128 * i for i in range(2, 33))
    for bm in (64, 128)
    for bn in (64, 128)
    for bk in (32, 64)
    for nw in (4, 8)
)

_build_kernel = build(
    premake,
    configs,
    meta_parameters=("block_size_m", "block_size_n", "block_size_k"),
    kernel_name="addmm",
)

_BUILD_CONFIGS = frozenset(
    (kwargs["m"], kwargs["n"], kwargs["k"], kwargs["dtype"])
    for _, kwargs, _ in configs
)

_fallback_kernel = ninetoothed.make(
    arrangement,
    application,
    (
        Tensor(2),
        Tensor(2),
        Tensor(2),
        Tensor(0),
        Tensor(0),
        Tensor(2),
    ),
)


def kernel(input, mat1, mat2, beta, alpha, output, m, n, k, dtype):
    if (m, n, k, dtype) in _BUILD_CONFIGS:
        return _build_kernel(input, mat1, mat2, beta, alpha, output, m, n, k, dtype)

    return _fallback_kernel(input, mat1, mat2, beta, alpha, output)
