import triton
import triton.language as tl


@triton.jit
def kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    num_cols,
    x_stride,
    w_stride,
    y_stride,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < num_cols

    x_ptrs = x_ptr + row_idx * x_stride + col_offsets
    w_ptrs = w_ptr + row_idx * w_stride + col_offsets

    x = tl.load(x_ptrs, mask=mask).to(tl.float32)
    w = tl.load(w_ptrs, mask=mask).to(tl.float32)

    y = x * tl.rsqrt(tl.sum(x * x) / num_cols + eps) * w

    y_ptrs = y_ptr + row_idx * y_stride + col_offsets

    tl.store(y_ptrs, y, mask=mask)
