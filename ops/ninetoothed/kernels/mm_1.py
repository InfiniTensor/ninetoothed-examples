import ninetoothed

from ops.ninetoothed.kernels.mm import application, arrangement, tensors
from ops.ninetoothed.kernels.utils import MAX_NUM_CONFIGS, NUM_STAGES, NUM_WARPS

kernel = ninetoothed.make(
    arrangement,
    application,
    tensors,
    num_warps=NUM_WARPS,
    num_stages=NUM_STAGES,
    max_num_configs=MAX_NUM_CONFIGS,
)
