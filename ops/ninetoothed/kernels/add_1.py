import ninetoothed
from ninetoothed import Tensor

from ops.ninetoothed.kernels.add import application
from ops.ninetoothed.kernels.element_wise import arrangement
from ops.ninetoothed.kernels.utils import MAX_NDIM, MAX_NUM_CONFIGS, MIN_NDIM


def _make(ndim):
    tensors = (Tensor(ndim), Tensor(ndim), Tensor(ndim))

    return ninetoothed.make(
        arrangement, application, tensors, max_num_configs=MAX_NUM_CONFIGS
    )


kernels = tuple(_make(ndim) if ndim >= MIN_NDIM else None for ndim in range(MAX_NDIM))
