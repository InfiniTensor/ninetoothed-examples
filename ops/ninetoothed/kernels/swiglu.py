import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ops.ninetoothed.kernels._common import DTYPES, build


def arrangement(a, b, c, block_size):
    return a.tile((block_size,)), b.tile((block_size,)), c.tile((block_size,))


def application(a, b, c):
    b_loaded = b
    gate = b_loaded * ntl.sigmoid(ntl.cast(b_loaded, ntl.float32))
    c = a * gate  # noqa: F841


def premake(dtype, block_size):
    arrangement_ = functools.partial(arrangement, block_size=block_size)
    tensors = tuple(Tensor(1, dtype=dtype) for _ in range(3))

    return arrangement_, application, tensors


configs = tuple(
    ((), {"dtype": dtype, "block_size": block_size}, {})
    for dtype in DTYPES
    for block_size in (512, 1024, 2048)
)

kernel = build(premake, configs, meta_parameters=("block_size",), kernel_name="swiglu")
