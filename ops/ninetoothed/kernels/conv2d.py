import functools

import ninetoothed
from ninetoothed import Tensor

import ops.ninetoothed.kernels.mm as mm
from ops.ninetoothed.kernels._common import build


def arrangement(input, filter, output, **block_sizes):
    input_arranged = input.tile((1, *filter.shape[1:]), strides=(-1, -1, 1, 1))
    input_arranged = input_arranged.squeeze(1)
    input_arranged.dtype = input_arranged.dtype.squeeze(0)
    input_arranged = input_arranged.ravel()
    input_arranged = input_arranged.flatten(end_dim=3).flatten(start_dim=1)

    filter_arranged = filter.flatten(start_dim=1)
    filter_arranged = filter_arranged.permute((1, 0))

    output_arranged = output.permute((0, 2, 3, 1)).flatten(end_dim=3)

    return mm.arrangement(
        input_arranged, filter_arranged, output_arranged, **block_sizes
    )


def premake(n, c, h, w, k, r, s, dtype, block_size_m, block_size_n, block_size_k):
    arrangement_ = functools.partial(
        arrangement,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        block_size_k=block_size_k,
    )

    p = h - r + 1
    q = w - s + 1

    tensors = (
        Tensor(shape=(n, c, h, w), dtype=dtype),
        Tensor(shape=(k, c, r, s), dtype=dtype),
        Tensor(shape=(n, k, p, q), dtype=dtype),
    )

    return arrangement_, mm.application, tensors


# Block sweep approximating the JIT auto-tuner default range. Conv2d's im2col
# arrangement produces a tall-skinny matmul, so wider bn (up to 256) and
# longer-pipelined num_stages help match the JIT-tuned kernel.
configs = tuple(
    (
        (),
        {
            "n": n,
            "c": 512,
            "h": 14,
            "w": 14,
            "k": 512,
            "r": 3,
            "s": 3,
            "dtype": ninetoothed.float16,
            "block_size_m": bm,
            "block_size_n": bn,
            "block_size_k": bk,
        },
        {"num_warps": 8, "num_stages": ns},
    )
    for n in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
    for bm in (64, 128)
    for bn in (128, 256)
    for bk in (32, 64)
    for ns in (3, 5)
    if bm * bn <= 32768 and bm * bk <= 32768 and bn * bk <= 32768
)

_build_kernel = build(
    premake,
    configs,
    meta_parameters=("block_size_m", "block_size_n", "block_size_k"),
    kernel_name="conv2d",
)

_BUILD_CONFIGS = frozenset(
    (
        kwargs["n"],
        kwargs["c"],
        kwargs["h"],
        kwargs["w"],
        kwargs["k"],
        kwargs["r"],
        kwargs["s"],
        kwargs["dtype"],
    )
    for _, kwargs, _ in configs
)

_fallback_kernel = ninetoothed.make(
    arrangement,
    mm.application,
    tuple(Tensor(4, shape_options={"constexpr": True}) for _ in range(3)),
)


def kernel(input, filter, output, n, c, h, w, k, r, s, dtype):
    if (n, c, h, w, k, r, s, dtype) in _BUILD_CONFIGS:
        return _build_kernel(input, filter, output, n, c, h, w, k, r, s, dtype)

    return _fallback_kernel(input, filter, output)
