import functools

import ninetoothed
from ninetoothed import Tensor

from ops.ninetoothed.kernels._common import build


def arrangement(input, sin_table, cos_table, head_dim, interleaved):
    tile_shape = (1, 1, 1, head_dim // 2)

    if interleaved:
        strides = (-1, -1, -1, 1)
        dilation = (1, 1, 1, 2)
    else:
        strides = None
        dilation = None

    input_arranged = input.tile(tile_shape, strides=strides, dilation=dilation)
    input_arranged = input_arranged.tile((1, 1, 1, 2))
    input_arranged.dtype = input_arranged.dtype.squeeze((0, 1, 2))
    input_arranged.dtype.dtype = input_arranged.dtype.dtype.squeeze((0, 1, 2))

    sin_table_arranged = sin_table.tile(tile_shape)
    sin_table_arranged.dtype = sin_table_arranged.dtype.squeeze((0, 1, 2))

    cos_table_arranged = cos_table.tile(tile_shape)
    cos_table_arranged.dtype = cos_table_arranged.dtype.squeeze((0, 1, 2))

    return input_arranged, sin_table_arranged, cos_table_arranged


def application(input, sin_table, cos_table):
    sin_table_loaded = sin_table
    cos_table_loaded = cos_table

    input_0 = input[0]
    input_1 = input[1]

    input[0] = input_0 * cos_table_loaded - input_1 * sin_table_loaded
    input[1] = input_0 * sin_table_loaded + input_1 * cos_table_loaded


def premake(head_dim, dtype, interleaved):
    arrangement_ = functools.partial(
        arrangement, head_dim=head_dim, interleaved=interleaved
    )
    input_tensor = Tensor(shape=(None, None, None, head_dim), dtype=dtype)
    sin_cos_tensors = tuple(
        Tensor(shape=(None, None, None, head_dim // 2), dtype=dtype) for _ in range(2)
    )
    tensors = (input_tensor,) + sin_cos_tensors

    return arrangement_, application, tensors


def _configs(head_dim, dtype, interleaved):
    return (
        ((), {"head_dim": head_dim, "dtype": dtype, "interleaved": interleaved}, {}),
    )


@functools.cache
def _kernel(head_dim, dtype, interleaved):
    return build(
        premake,
        _configs(head_dim, dtype, interleaved),
        kernel_name=f"rotary_position_embedding_{head_dim}_{int(interleaved)}",
    )


_TORCH_TO_NT_DTYPE = {}


def kernel(input, sin_table, cos_table, interleaved=True):
    import torch

    if not _TORCH_TO_NT_DTYPE:
        _TORCH_TO_NT_DTYPE.update(
            {
                torch.float16: ninetoothed.float16,
                torch.bfloat16: ninetoothed.bfloat16,
                torch.float32: ninetoothed.float32,
            }
        )

    head_dim = input.shape[-1]
    dtype = _TORCH_TO_NT_DTYPE[input.dtype]

    return _kernel(head_dim, dtype, bool(interleaved))(
        input, sin_table, cos_table, head_dim, dtype, bool(interleaved)
    )
