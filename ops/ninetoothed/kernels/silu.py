import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ops.ninetoothed.kernels._common import DTYPES, build


def arrangement(input, output, block_size):
    return input.tile((block_size,)), output.tile((block_size,))


def application(input, output):
    input_loaded = input
    output = input_loaded * ntl.sigmoid(ntl.cast(input_loaded, ntl.float32))  # noqa: F841


def premake(dtype, block_size):
    arrangement_ = functools.partial(arrangement, block_size=block_size)
    tensors = tuple(Tensor(1, dtype=dtype) for _ in range(2))

    return arrangement_, application, tensors


configs = tuple(
    ((), {"dtype": dtype, "block_size": block_size}, {})
    for dtype in DTYPES
    for block_size in (512, 1024, 2048)
)

kernel = build(premake, configs, meta_parameters=("block_size",), kernel_name="silu")
