import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ops.ninetoothed.kernels._common import build


def _next_power_of_2(value):
    return 1 << (value - 1).bit_length()


def arrangement(input, output, window_height, window_width, block_size_):
    input_arranged = input.tile((1, 1, window_height, window_width))
    padded_window_height = _next_power_of_2(window_height)
    padded_window_width = _next_power_of_2(window_width)
    input_arranged.dtype = input_arranged.dtype.pad(
        (
            (0, 0),
            (0, 0),
            (0, padded_window_height - window_height),
            (0, padded_window_width - window_width),
        )
    )
    input_arranged = input_arranged.ravel()
    input_arranged = input_arranged.flatten(end_dim=4).flatten(start_dim=1)
    input_arranged = input_arranged.tile((block_size_, -1))

    output_arranged = output.tile((1, 1, 1, 1))
    output_arranged = output_arranged.ravel()
    output_arranged = output_arranged.flatten(end_dim=4).flatten(start_dim=1)
    output_arranged = output_arranged.tile((block_size_, -1))
    output_arranged.dtype = output_arranged.dtype.squeeze(1)

    return input_arranged, output_arranged


def application(input, output):
    output = ntl.max(input, axis=1)  # noqa: F841


def premake(window_height, window_width, dtype, block_size_):
    arrangement_ = functools.partial(
        arrangement,
        window_height=window_height,
        window_width=window_width,
        block_size_=block_size_,
    )
    tensors = (
        Tensor(4, dtype=dtype, other=float("-inf")),
        Tensor(4, dtype=dtype),
    )

    return arrangement_, application, tensors


def _configs(window_height, window_width, dtype):
    return (
        (
            (),
            {
                "window_height": window_height,
                "window_width": window_width,
                "dtype": dtype,
                "block_size_": 256,
            },
            {},
        ),
    )


@functools.cache
def _kernel(window_height, window_width, dtype):
    return build(
        premake,
        _configs(window_height, window_width, dtype),
        meta_parameters=("block_size_",),
        kernel_name=f"max_pool2d_{window_height}_{window_width}",
    )


def kernel(input, output, window_height, window_width, dtype):
    return _kernel(window_height, window_width, dtype)(
        input, output, window_height, window_width, dtype
    )
