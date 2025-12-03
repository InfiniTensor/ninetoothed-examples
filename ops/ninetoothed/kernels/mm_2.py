import ninetoothed
import ninetoothed.language as ntl

import ops.ninetoothed.kernels.mm as mm
from ops.ninetoothed.kernels.mm_1 import BLOCK_SIZE_K, BLOCK_SIZE_M, BLOCK_SIZE_N
from ops.ninetoothed.kernels.utils import MAX_NUM_CONFIGS, NUM_STAGES, NUM_WARPS


def arrangement(
    input,
    other,
    output,
    BLOCK_SIZE_M=BLOCK_SIZE_M,
    BLOCK_SIZE_N=BLOCK_SIZE_N,
    BLOCK_SIZE_K=BLOCK_SIZE_K,
):
    input_arranged, other_arranged, output_arranged = mm.arrangement(
        input,
        other,
        output,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    input_arranged = input_arranged.tile((1, -1))
    input_arranged.dtype = input_arranged.dtype.squeeze(0)

    other_arranged = other_arranged.tile((1, -1))
    other_arranged.dtype = other_arranged.dtype.squeeze(0)

    output_arranged = output_arranged.tile((1, -1))
    output_arranged.dtype = output_arranged.dtype.squeeze(0)

    return input_arranged, other_arranged, output_arranged


def application(input, other, output):
    for i in range(input.shape[0]):
        accumulator = ntl.zeros(output.dtype.shape, dtype=ntl.float32)

        for k in range(input.dtype.shape[0]):
            accumulator += ntl.dot(input[i, k], other[i, k])

        output[i] = accumulator


tensors = mm.tensors

kernel = ninetoothed.make(
    arrangement,
    application,
    tensors,
    num_warps=NUM_WARPS,
    num_stages=NUM_STAGES,
    max_num_configs=MAX_NUM_CONFIGS,
)
