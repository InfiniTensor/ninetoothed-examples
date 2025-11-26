import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ops.ninetoothed.kernels.reduction import arrangement
from ops.ninetoothed.kernels.utils import MAX_NDIM, MAX_NUM_CONFIGS, MIN_NDIM


def application(input, output):
    prev_max = ntl.full((output.dtype.shape[-2],), float("-inf"), dtype=ntl.float32)
    denominator = ntl.zeros((output.dtype.shape[-2],), dtype=ntl.float32)

    for i in range(input.shape[0]):
        input_i = input[i]
        curr_max = ntl.maximum(prev_max, ntl.max(input_i, axis=-1))
        input_max_diff_exp = ntl.exp(input_i - curr_max[:, None])
        prev_curr_max_diff_exp = ntl.exp(prev_max - curr_max)
        denominator = denominator * prev_curr_max_diff_exp + ntl.sum(
            input_max_diff_exp, axis=-1
        )
        prev_max = curr_max

    for i in range(input.shape[0]):
        numerator = ntl.exp(input[i] - prev_max[:, None])
        output[i] = numerator / denominator[:, None]


def _make(ndim, dim):
    arrangement_ = functools.partial(arrangement, dim=dim)

    tensors = (Tensor(ndim, other=float("-inf")), Tensor(ndim))

    return ninetoothed.make(
        arrangement_, application, tensors, max_num_configs=MAX_NUM_CONFIGS
    )


kernels = {(ndim, (-1,)): _make(ndim, -1) for ndim in range(MIN_NDIM, MAX_NDIM)}

# Uncomment the following lines to achieve more coverage.
# kernels = {
#     (ndim, dim): _make(ndim, dim)
#     for ndim in range(MIN_NDIM, MAX_NDIM)
#     for num_target_dims in range(1, ndim + 1)
#     for dim in itertools.combinations(range(ndim), num_target_dims)
# }
