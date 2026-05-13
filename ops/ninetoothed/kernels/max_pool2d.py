import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor, block_size

from ops.ninetoothed.kernels._common import build

BLOCK_SIZE = block_size()


def arrangement(
    input,
    output,
    window_height,
    window_width,
    BLOCK_SIZE=BLOCK_SIZE,
):
    input_arranged = input.tile((1, 1, window_height, window_width))
    input_arranged = input_arranged.ravel()
    input_arranged = input_arranged.flatten(end_dim=4).flatten(start_dim=1)
    input_arranged = input_arranged.tile((BLOCK_SIZE, -1))

    output_arranged = output.tile((1, 1, 1, 1))
    output_arranged = output_arranged.ravel()
    output_arranged = output_arranged.flatten(end_dim=4).flatten(start_dim=1)
    output_arranged = output_arranged.tile((BLOCK_SIZE, -1))
    output_arranged.dtype = output_arranged.dtype.squeeze(1)

    return input_arranged, output_arranged


def application(input, output):
    output = ntl.max(input, axis=1)  # noqa: F841


def premake(window_height, window_width, dtype, block_size_):
    arrangement_ = functools.partial(
        arrangement,
        window_height=window_height,
        window_width=window_width,
        BLOCK_SIZE=block_size_,
    )
    tensors = (
        Tensor(4, dtype=dtype, other=float("-inf")),
        Tensor(4, dtype=dtype),
    )

    return arrangement_, application, tensors


# Compile the benchmarked 3x3 fp16 case; other windows and dtypes use the
# generic make-based fallback below.
configs = (
    (
        (),
        {
            "window_height": 3,
            "window_width": 3,
            "dtype": ninetoothed.float16,
            "block_size_": 256,
        },
        {},
    ),
)

_build_kernel = build(
    premake,
    configs,
    meta_parameters=("block_size_",),
    kernel_name="max_pool2d",
)

_BUILD_CONFIGS = frozenset(
    (kwargs["window_height"], kwargs["window_width"], kwargs["dtype"])
    for _, kwargs, _ in configs
)

_FALLBACK_BLOCK_SIZE = block_size()
_FALLBACK_WINDOW_HEIGHT = Symbol(
    "FALLBACK_WINDOW_HEIGHT", constexpr=True, upper_bound=16
)
_FALLBACK_WINDOW_WIDTH = Symbol("FALLBACK_WINDOW_WIDTH", constexpr=True, upper_bound=16)


def _fallback_arrangement(
    input,
    output,
    BLOCK_SIZE=_FALLBACK_BLOCK_SIZE,
    WINDOW_HEIGHT=_FALLBACK_WINDOW_HEIGHT,
    WINDOW_WIDTH=_FALLBACK_WINDOW_WIDTH,
):
    return arrangement(input, output, WINDOW_HEIGHT, WINDOW_WIDTH, BLOCK_SIZE)


_fallback_kernel = ninetoothed.make(
    _fallback_arrangement,
    application,
    (Tensor(4, other=float("-inf")), Tensor(4)),
)


def kernel(input, output, window_height, window_width, dtype):
    if (window_height, window_width, dtype) in _BUILD_CONFIGS:
        return _build_kernel(input, output, window_height, window_width, dtype)

    return _fallback_kernel(
        input, output, WINDOW_HEIGHT=window_height, WINDOW_WIDTH=window_width
    )
