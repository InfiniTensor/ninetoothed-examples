import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


def arrangement(input, output, BLOCK_SIZE=BLOCK_SIZE):
    return input.tile((1, BLOCK_SIZE)), output.tile((1, BLOCK_SIZE))


def application(input, output):
    input_loaded = input

    row_minus_max = input_loaded - ntl.max(input_loaded)
    numerator = ntl.exp(row_minus_max)
    denominator = ntl.sum(numerator)

    output = numerator / denominator  # noqa: F841


tensors = (Tensor(2, other=float("-inf")), Tensor(2))

kernel = ninetoothed.make(arrangement, application, tensors)
