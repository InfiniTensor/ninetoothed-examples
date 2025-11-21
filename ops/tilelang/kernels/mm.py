import tilelang
import tilelang.language as T


@tilelang.jit
def mm(M, N, K, block_M, block_N, block_K):
    @T.prim_func
    def mm_kernel(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared((block_M, block_K), "float16")
            B_shared = T.alloc_shared((block_K, block_N), "float16")
            C_local = T.alloc_fragment((block_M, block_N), "float")

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)

                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return mm_kernel


M = T.dynamic("m")
N = T.dynamic("n")
K = T.dynamic("k")
# TODO: Do not use constant values.
block_M = 128
block_N = 128
block_K = 32

kernel = mm(M, N, K, block_M, block_N, block_K)
