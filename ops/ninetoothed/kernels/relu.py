import ninetoothed
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


def arrangement(input, output, BLOCK_SIZE=BLOCK_SIZE):
    return input.tile((BLOCK_SIZE,)), output.tile((BLOCK_SIZE,))


def application(input, output):
    output = max(0.0, input)   # noqa: F841

tensors = (Tensor(1), Tensor(1))
kernel = ninetoothed.make(arrangement, application, tensors)