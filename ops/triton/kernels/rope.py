import triton
import triton.language as tl


@triton.jit
def kernel(
    tensor_ptr,
    sin_table_ptr,
    cos_table_ptr,
    tensor_stride_n,
    tensor_stride_l,
    tensor_stride_h,
    tensor_stride_e,
    sin_table_stride_l,
    cos_table_stride_l,
    emb_dim,
    INTERLEAVED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    off_n = tl.program_id(0)
    off_l = tl.program_id(1)
    off_h = tl.program_id(2)

    offs = tl.arange(0, BLOCK_SIZE)

    half_emb_dim = emb_dim // 2
    mask = offs < half_emb_dim

    sin_table = tl.load(sin_table_ptr + off_l * sin_table_stride_l + offs, mask=mask)
    cos_table = tl.load(cos_table_ptr + off_l * cos_table_stride_l + offs, mask=mask)

    even_offs = (
        off_n * tensor_stride_n + off_l * tensor_stride_l + off_h * tensor_stride_h
    )
    odd_offs = (
        off_n * tensor_stride_n + off_l * tensor_stride_l + off_h * tensor_stride_h
    )

    if INTERLEAVED:
        even_offs += (2 * offs) * tensor_stride_e
        odd_offs += (2 * offs + 1) * tensor_stride_e
    else:
        even_offs += offs * tensor_stride_e
        odd_offs += (offs + half_emb_dim) * tensor_stride_e

    even_ptrs = tensor_ptr + even_offs
    odd_ptrs = tensor_ptr + odd_offs

    even = tl.load(even_ptrs, mask=mask)
    odd = tl.load(odd_ptrs, mask=mask)

    tl.store(even_ptrs, even * cos_table - odd * sin_table, mask=mask)
    tl.store(odd_ptrs, even * sin_table + odd * cos_table, mask=mask)
