import ninetoothed
from ninetoothed import Symbol, Tensor
import ninetoothed.language as ntl
import math

def arrangement(input, output):
    BLOCK_SIZE = Symbol("BLOCK_SIZE", meta=True)

    WINDOW_HEIGHT = Symbol("WINDOW_HEIGHT", constexpr=True, upper_bound=16)
    WINDOW_WIDTH = Symbol("WINDOW_WIDTH", constexpr=True, upper_bound=16)

    input_arranged = input.tile((1, 1, WINDOW_HEIGHT, WINDOW_WIDTH))
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
    output = ntl.sum(input)  # noqa: F841


kernel = ninetoothed.make(
    arrangement, application, (Tensor(4), Tensor(4))
)