import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128}, num_stages=4, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64}, num_stages=4, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64}, num_stages=4, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32}, num_stages=4, num_warps=8
        ),
    ],
    key=["EMB_DIM"],
)
@triton.jit
def kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    q_stride_z,
    q_stride_h,
    q_stride_m,
    q_stride_k,
    k_stride_z,
    k_stride_h,
    k_stride_n,
    k_stride_k,
    v_stride_z,
    v_stride_h,
    v_stride_k,
    v_stride_n,
    o_stride_z,
    o_stride_h,
    o_stride_m,
    o_stride_n,
    scale,
    seq_len,
    EMB_DIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    off_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)

    offs_m_start = off_m * BLOCK_SIZE_M

    q_off = off_z * q_stride_z + off_h * q_stride_h
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + q_off,
        shape=(seq_len, EMB_DIM),
        strides=(q_stride_m, q_stride_k),
        offsets=(offs_m_start, 0),
        block_shape=(BLOCK_SIZE_M, EMB_DIM),
        order=(1, 0),
    )
    k_off = off_z * k_stride_z + off_h * k_stride_h
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + k_off,
        shape=(EMB_DIM, seq_len),
        strides=(k_stride_k, k_stride_n),
        offsets=(0, 0),
        block_shape=(EMB_DIM, BLOCK_SIZE_N),
        order=(0, 1),
    )
    v_off = off_z * v_stride_z + off_h * v_stride_h
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr + v_off,
        shape=(seq_len, EMB_DIM),
        strides=(v_stride_k, v_stride_n),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_N, EMB_DIM),
        order=(1, 0),
    )
    o_off = off_z * o_stride_z + off_h * o_stride_h
    o_block_ptr = tl.make_block_ptr(
        base=o_ptr + o_off,
        shape=(seq_len, EMB_DIM),
        strides=(o_stride_m, o_stride_n),
        offsets=(offs_m_start, 0),
        block_shape=(BLOCK_SIZE_M, EMB_DIM),
        order=(1, 0),
    )

    q = tl.load(q_block_ptr, boundary_check=(0, 1))
    q = (q * scale * 1.44269504089).to(q_block_ptr.type.element_ty)

    acc = tl.zeros((BLOCK_SIZE_M, EMB_DIM), dtype=tl.float32)
    l_i = tl.full((BLOCK_SIZE_M,), 1, dtype=tl.float32)
    m_i = tl.full((BLOCK_SIZE_M,), float("-inf"), dtype=tl.float32)

    for i in range(0, tl.cdiv(seq_len, BLOCK_SIZE_N)):
        k = tl.load(k_block_ptr, boundary_check=(0, 1))

        mask = i * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) < seq_len
        qk = tl.where(mask, tl.dot(q, k), float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        v = tl.load(v_block_ptr, boundary_check=(0, 1))
        alpha = tl.exp2(m_i - m_ij)
        acc = acc * alpha[:, None] + tl.dot(p.to(v_block_ptr.type.element_ty), v)
        m_i = m_ij
        l_i = l_i * alpha + l_ij

        v_block_ptr = tl.advance(v_block_ptr, (BLOCK_SIZE_N, 0))
        k_block_ptr = tl.advance(k_block_ptr, (0, BLOCK_SIZE_N))

    acc /= l_i[:, None]

    tl.store(o_block_ptr, acc.to(o_ptr.type.element_ty), boundary_check=(0, 1))
