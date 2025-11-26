from ops.ninetoothed.kernels import utils


def arrangement(*tensors, dim, block_sizes=None):
    dims = dim

    if isinstance(dims, int):
        dims = (dims,)

    ndim = max(tensor.ndim for tensor in tensors)

    assert all(tensor.ndim == ndim or tensor.ndim == 0 for tensor in tensors)

    if block_sizes is None:
        block_sizes = tuple(utils.block_size() for _ in range(ndim))

    dims = tuple(dim if dim >= 0 else dim + ndim for dim in dims)

    non_target_dims = tuple(i for i in range(ndim) if i not in dims)

    def _arrange(tensor):
        arranged = tensor.tile(block_sizes)
        arranged = arranged.tile(tuple(-1 if dim in dims else 1 for dim in range(ndim)))
        arranged.dtype = arranged.dtype.squeeze(non_target_dims)

        return arranged

    return tuple(_arrange(tensor) if tensor.ndim != 0 else tensor for tensor in tensors)
