import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ops.ninetoothed.kernels._common import DTYPES, build


def arrangement(x, w, eps, y, block_size):
    def arrange(tensor):
        return tensor.tile((1, block_size))

    return arrange(x), arrange(w), eps, arrange(y)


def application(x, w, eps, y):
    x_fp32 = ntl.cast(x, ntl.float32)
    y = x_fp32 * ntl.rsqrt(ntl.sum(x_fp32 * x_fp32) / x.shape[-1] + eps) * w  # noqa: F841


def premake(dtype, n, block_size):
    arrangement_ = functools.partial(arrangement, block_size=block_size)
    tensors = (
        Tensor(2, dtype=dtype),
        Tensor(2, dtype=dtype),
        Tensor(0, dtype=ninetoothed.float64),
        Tensor(2, dtype=dtype),
    )

    return arrangement_, application, tensors


configs = tuple(
    ((), {"dtype": dtype, "n": n, "block_size": n}, {})
    for dtype in DTYPES
    for n in (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
)

kernel = build(
    premake, configs, meta_parameters=("block_size",), kernel_name="fused_rms_norm"
)
