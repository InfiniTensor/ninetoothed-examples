import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


def arrangement(input, eps, output, BLOCK_SIZE=BLOCK_SIZE):
    return input.tile((1, BLOCK_SIZE)), eps, output.tile((1, BLOCK_SIZE))


def application(input, eps, output):
    input_fp32 = ntl.cast(input, ntl.float32)
    output = input_fp32 * ntl.rsqrt(  # noqa: F841
        ntl.sum(input_fp32 * input_fp32) / input.shape[-1] + eps
    )


tensors = (Tensor(2), Tensor(0), Tensor(2))

kernel = ninetoothed.make(arrangement, application, tensors)
