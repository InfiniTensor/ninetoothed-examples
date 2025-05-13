import ninetoothed
from ninetoothed import Tensor

import ops.ninetoothed.kernels.mm as mm


def arrangement(input, filter, output):
    input_arranged = input.tile((1, *filter.shape[1:]), strides=(-1, -1, 1, 1))
    input_arranged = input_arranged.squeeze(1)
    input_arranged.dtype = input_arranged.dtype.squeeze(0)
    input_arranged = input_arranged.ravel()
    input_arranged = input_arranged.flatten(end_dim=3).flatten(start_dim=1)

    filter_arranged = filter.flatten(start_dim=1)
    filter_arranged = filter_arranged.permute((1, 0))

    output_arranged = output.permute((0, 2, 3, 1)).flatten(end_dim=3)

    return mm.arrangement(input_arranged, filter_arranged, output_arranged)


shape_options = {"constexpr": True, "upper_bound": 16}
tensors = tuple(Tensor(4, shape_options=shape_options) for _ in range(3))

kernel = ninetoothed.make(arrangement, mm.application, tensors)
