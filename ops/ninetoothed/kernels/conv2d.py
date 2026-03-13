import functools
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor, block_size
# import ntops.kernels.mm as mm
import ops.ninetoothed.kernels.mm as mm


STRIDE_H = Symbol("stride_h", constexpr=True)
STRIDE_W = Symbol("stride_w", constexpr=True)
DILATION_H = Symbol("dilation_h", constexpr=True)
DILATION_W = Symbol("dilation_w", constexpr=True)
BLOCK_SIZE_M = block_size()
BLOCK_SIZE_N = block_size()
BLOCK_SIZE_K = block_size()


def arrangement(
    input,
    weight,
    bias,
    output,
    stride_h=None,
    stride_w=None,
    dilation_h=None,
    dilation_w=None,
    block_size_m=None,
    block_size_n=None,
    block_size_k=None,
):
    if stride_h is None:
        stride_h = STRIDE_H

    if stride_w is None:
        stride_w = STRIDE_W

    if dilation_h is None:
        dilation_h = DILATION_H

    if dilation_w is None:
        dilation_w = DILATION_W

    if block_size_m is None:
        block_size_m = mm.BLOCK_SIZE_M

    if block_size_n is None:
        block_size_n = mm.BLOCK_SIZE_N

    if block_size_k is None:
        block_size_k = mm.BLOCK_SIZE_K

    mm_arrangement = functools.partial(
        mm.arrangement,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        block_size_k=block_size_k,
    )

    input_arranged = input.tile(
        (1, *weight.shape[1:]),
        strides=(-1, -1, stride_h, stride_w),
        dilation=(1, 1, dilation_h, dilation_w),
        floor_mode=True,
    )
    input_arranged = input_arranged.squeeze(1)
    input_arranged.dtype = input_arranged.dtype.squeeze(0)
    input_arranged = input_arranged.ravel()
    input_arranged = input_arranged.flatten(end_dim=3).flatten(start_dim=1)

    weight_arranged = weight.flatten(start_dim=1)
    weight_arranged = weight_arranged.permute((1, 0))

    bias_arranged = bias.permute((0, 2, 3, 1)).flatten(end_dim=3)

    # _, _, bias_arranged = mm_arrangement(input_arranged, weight_arranged, bias_arranged)
    _, _, bias_arranged = mm_arrangement(Tensor(2), Tensor(2), bias_arranged)
    

    output_arranged = output.permute((0, 2, 3, 1)).flatten(end_dim=3)

    input_arranged, weight_arranged, output_arranged = mm_arrangement(
        input_arranged, weight_arranged, output_arranged
    )

    return input_arranged, weight_arranged, bias_arranged, output_arranged


def application(input, weight, bias, output):
    mm_output = ntl.zeros(output.shape, dtype=ntl.float32)
    mm.application(input, weight, mm_output)
    output = mm_output + bias
    
shape_options = {"constexpr": True}
tensors = tuple(Tensor(4, shape_options=shape_options) for _ in range(4))

# kernel = ninetoothed.make(arrangement, application, tensors)

def _make(
    stride_h=None,
    stride_w=None,
    dilation_h=None,
    dilation_w=None,
    dtype=None,
    block_size_m=None,
    block_size_n=None,
    block_size_k=None,
):
    arrangement_ = functools.partial(
        arrangement,
        stride_h=stride_h,
        stride_w=stride_w,
        dilation_h=dilation_h,
        dilation_w=dilation_w,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        block_size_k=block_size_k,
    )

    tensors = tuple(Tensor(4, dtype=dtype) for _ in range(4))
    return ninetoothed.make(arrangement_, application, tensors)

kernels = {(stride, dilation): _make(stride_h=stride, stride_w=stride, dilation_h=dilation, dilation_w=dilation, block_size_m=BLOCK_SIZE_M, block_size_n=BLOCK_SIZE_N, block_size_k=BLOCK_SIZE_K) 
           for stride in [1, 3] 
           for dilation in [1,2]}
# print(kernels)