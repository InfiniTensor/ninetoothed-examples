import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ops.ninetoothed.kernels._common import DTYPES, build


def arrangement(input, eps, output, block_size):
    return input.tile((1, block_size)), eps, output.tile((1, block_size))


def application(input, eps, output):
    input_fp32 = ntl.cast(input, ntl.float32)
    output = input_fp32 * ntl.rsqrt(  # noqa: F841
        ntl.sum(input_fp32 * input_fp32) / input.shape[-1] + eps
    )


def premake(dtype, n, block_size):
    arrangement_ = functools.partial(arrangement, block_size=block_size)
    tensors = (
        Tensor(2, dtype=dtype),
        Tensor(0, dtype=ninetoothed.float32),
        Tensor(2, dtype=dtype),
    )

    return arrangement_, application, tensors


configs = tuple(
    ((), {"dtype": dtype, "n": n, "block_size": n}, {})
    for dtype in DTYPES
    for n in (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
)

kernel = build(
    premake, configs, meta_parameters=("block_size",), kernel_name="rms_norm"
)
