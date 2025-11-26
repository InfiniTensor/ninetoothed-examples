import ninetoothed

MIN_NDIM = 1
MAX_NDIM = 5

LOWER_BOUND = 1
UPPER_BOUND = 65536

NUM_WARPS = (4, 8, 16, 32)
NUM_STAGES = (1, 3, 5, 7)
MAX_NUM_CONFIGS = 32


def block_size():
    return ninetoothed.block_size(lower_bound=LOWER_BOUND, upper_bound=UPPER_BOUND)
