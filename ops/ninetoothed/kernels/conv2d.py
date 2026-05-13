import functools

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


def _configs(n, c, h, w, k, r, s, dtype):
    return tuple(
        (
            (),
            {
                "n": n,
                "c": c,
                "h": h,
                "w": w,
                "k": k,
                "r": r,
                "s": s,
                "dtype": dtype,
                "block_size_m": bm,
                "block_size_n": bn,
                "block_size_k": bk,
            },
            {"num_warps": 8, "num_stages": ns},
        )
        for bm in (64, 128)
        for bn in (128, 256)
        for bk in (32, 64)
        for ns in (3, 5)
        if bm * bn <= 32768 and bm * bk <= 32768 and bn * bk <= 32768
    )


@functools.cache
def _kernel(n, c, h, w, k, r, s, dtype):
    return build(
        premake,
        _configs(n, c, h, w, k, r, s, dtype),
        meta_parameters=("block_size_m", "block_size_n", "block_size_k"),
        kernel_name=f"conv2d_{n}_{c}_{h}_{w}_{k}_{r}_{s}",
    )


def kernel(input, filter, output, n, c, h, w, k, r, s, dtype):
    return _kernel(n, c, h, w, k, r, s, dtype)(
        input, filter, output, n, c, h, w, k, r, s, dtype
    )
