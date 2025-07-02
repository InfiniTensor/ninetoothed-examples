import triton
import triton.language as tl


@triton.jit
def kernel(a_ptr, b_ptr, c_ptr, num_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    silu_b = b * tl.sigmoid(tl.cast(b, tl.float32))
    c = a * silu_b

    tl.store(c_ptr + offsets, c, mask=mask)
