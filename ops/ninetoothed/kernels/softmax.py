import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ops.ninetoothed.kernels._common import DTYPES, build


def arrangement(input, output, block_size):
    return input.tile((1, block_size)), output.tile((1, block_size))


def application(input, output):
    input_loaded = input

    row_minus_max = input_loaded - ntl.max(input_loaded)
    numerator = ntl.exp(row_minus_max)
    denominator = ntl.sum(numerator)

    output = numerator / denominator  # noqa: F841


def premake(dtype, n, block_size):
    arrangement_ = functools.partial(arrangement, block_size=block_size)
    tensors = (
        Tensor(2, dtype=dtype, other=float("-inf")),
        Tensor(2, dtype=dtype),
    )

    return arrangement_, application, tensors


configs = tuple(
    ((), {"dtype": dtype, "n": n, "block_size": n}, {})
    for dtype in DTYPES
    for n in (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
)

kernel = build(premake, configs, meta_parameters=("block_size",), kernel_name="softmax")
