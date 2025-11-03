from ops.ninetoothed.kernels import utils


def arrangement(*tensors, block_sizes=None):
    ndim = max(tensor.ndim for tensor in tensors)

    assert all(tensor.ndim == ndim or tensor.ndim == 0 for tensor in tensors)

    if block_sizes is None:
        block_sizes = tuple(utils.block_size() for _ in range(ndim))

    return tuple(
        tensor.tile(block_sizes) if tensor.ndim != 0 else tensor for tensor in tensors
    )
