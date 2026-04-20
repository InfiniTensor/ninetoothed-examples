import functools

from ninetoothed import Tensor

from ops.ninetoothed.kernels._common import DTYPES, build


def arrangement(input, other, output, block_size):
    return (
        input.tile((block_size,)),
        other.tile((block_size,)),
        output.tile((block_size,)),
    )


def application(input, other, output):
    output = input + other  # noqa: F841


def premake(dtype, block_size):
    arrangement_ = functools.partial(arrangement, block_size=block_size)
    tensors = tuple(Tensor(1, dtype=dtype) for _ in range(3))

    return arrangement_, application, tensors


configs = tuple(
    ((), {"dtype": dtype, "block_size": block_size}, {})
    for dtype in DTYPES
    for block_size in (512, 1024, 2048)
)

kernel = build(premake, configs, meta_parameters=("block_size",), kernel_name="add")
