import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


def arrangement(input, output, BLOCK_SIZE=BLOCK_SIZE):
    return input.tile((BLOCK_SIZE,)), output.tile((BLOCK_SIZE,))


def application(input, output):
    input_loaded = input
    output = input_loaded * ntl.sigmoid(ntl.cast(input_loaded, ntl.float32))  # noqa: F841


tensors = (Tensor(1), Tensor(1))

kernel = ninetoothed.make(arrangement, application, tensors)
