import functools

import ninetoothed

from ops.ninetoothed.kernels.mm import application, arrangement, tensors
from ops.ninetoothed.kernels.utils import MAX_NUM_CONFIGS, NUM_STAGES, NUM_WARPS

BLOCK_SIZE_M = ninetoothed.block_size(lower_bound=1)
BLOCK_SIZE_N = ninetoothed.block_size(lower_bound=1)
BLOCK_SIZE_K = ninetoothed.block_size(lower_bound=16)

kernel = ninetoothed.make(
    functools.partial(
        arrangement,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    ),
    application,
    tensors,
    num_warps=NUM_WARPS,
    num_stages=NUM_STAGES,
    max_num_configs=MAX_NUM_CONFIGS,
)
