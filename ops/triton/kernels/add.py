import triton
import triton.language as tl


@triton.jit
def kernel(input_ptr, other_ptr, output_ptr, height, width, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < height * width

    input = tl.load(input_ptr + offsets, mask=mask)
    other = tl.load(other_ptr + offsets, mask=mask)
    output = input + other

    tl.store(output_ptr + offsets, output, mask=mask)
