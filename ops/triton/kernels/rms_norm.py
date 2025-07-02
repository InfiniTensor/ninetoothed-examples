import triton
import triton.language as tl


@triton.jit
def kernel(
    input_ptr,
    output_ptr,
    num_cols,
    input_stride,
    output_stride,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = input_ptr + row_idx * input_stride + col_offsets
    mask = col_offsets < num_cols

    input = tl.load(input_ptrs, mask=mask).to(tl.float32)

    output = input * tl.rsqrt(tl.sum(input * input) / num_cols + eps)

    output_ptrs = output_ptr + row_idx * output_stride + col_offsets

    tl.store(output_ptrs, output, mask=mask)
