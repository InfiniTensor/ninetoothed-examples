import ninetoothed
from ninetoothed import Symbol, Tensor

BLOCK_SIZE_M = Symbol("BLOCK_SIZE_M", constexpr=True)
BLOCK_SIZE_N = Symbol("BLOCK_SIZE_N", constexpr=True)


def arrangement(input, other, output, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N):
    input_arranged = input.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))
    other_arranged = other.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))
    output_arranged = output.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))

    return input_arranged, other_arranged, output_arranged


def application(input, other, output):
    output = input + other  # noqa: F841


tensors = tuple(Tensor(2) for _ in range(3))

kernel = ninetoothed.make(arrangement, application, tensors)
