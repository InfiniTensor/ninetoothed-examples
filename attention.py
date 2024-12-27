import ninetoothed
import ninetoothed.language as ntl
import torch
import torch.nn.functional as F
import triton
from ninetoothed import Symbol, Tensor

BLOCK_SIZE_M = Symbol("BLOCK_SIZE_M", meta=True)
BLOCK_SIZE_N = Symbol("BLOCK_SIZE_N", meta=True)

q = Tensor(2, constexpr_shape=True)
k = Tensor(2, constexpr_shape=True)
v = Tensor(2, constexpr_shape=True)
o = Tensor(2, constexpr_shape=True)

q_tiled = q.tile((BLOCK_SIZE_M, -1))
k_tiled = k.tile((BLOCK_SIZE_N, -1)).tile((-1, -1)).expand((q_tiled.shape[0], -1))
v_tiled = v.tile((BLOCK_SIZE_N, -1)).tile((-1, -1)).expand((q_tiled.shape[0], -1))
o_tiled = o.tile((BLOCK_SIZE_M, -1))


@ninetoothed.jit
def attention_kernel(q: q_tiled, k: k_tiled, v: v_tiled, o: o_tiled):
    acc = ntl.zeros((q.shape[-2], q.shape[-1]), dtype=ntl.float32)
    l_i = ntl.full((q.shape[-2],), 1, dtype=ntl.float32)
    m_i = ntl.full((q.shape[-2],), float("-inf"), dtype=ntl.float32)

    for i in range(k.shape[0]):
        qk = ntl.dot((q * 1.44269504089).to(ntl.float16), ntl.trans(k[i, 0]))

        m_ij = ntl.maximum(m_i, ntl.max(qk, 1))
        p = ntl.exp2(qk - m_ij[:, None])
        l_ij = ntl.sum(p, 1)

        alpha = ntl.exp2(m_i - m_ij)
        acc = acc * alpha[:, None] + ntl.dot(p.to(ntl.float16), v[i, 0])
        m_i = m_ij
        l_i = l_i * alpha + l_ij

    acc /= l_i[:, None]
    o = acc.to(ntl.float32)  # noqa: F841


def attention(q, k, v):
    o = torch.empty_like(q, dtype=v.dtype)

    attention_kernel(q, k, v, o)

    return o


if __name__ == "__main__":
    torch.manual_seed(0)
    shape = (1, 1, 1024, 64)
    dtype = torch.float16
    q = torch.randn(shape, dtype=dtype, device="cuda")
    k = torch.randn(shape, dtype=dtype, device="cuda")
    v = torch.randn(shape, dtype=dtype, device="cuda")

    ninetoothed_output = attention(
        q.view(q.shape[-2:]), k.view(k.shape[-2:]), v.view(v.shape[-2:])
    )
    torch_output = F.scaled_dot_product_attention(q, k, v, scale=1)
    print(ninetoothed_output)
    print(torch_output)
    if torch.allclose(ninetoothed_output, torch_output, atol=0.01, rtol=0.01):
        print("✅ NineToothed and PyTorch match.")
    else:
        print("❌ NineToothed and PyTorch differ.")

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["n"],
            x_vals=[2**i for i in range(10, 15)],
            line_arg="provider",
            line_vals=["ninetoothed", "torch"],
            line_names=["NineToothed", "PyTorch"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="TFLOPS",
            plot_name="attention-performance",
            args={},
        )
    )
    def benchmark(n, provider):
        d = 64
        shape = (n, d)
        dtype = torch.float16
        q = torch.randn(shape, dtype=dtype, device="cuda")
        k = torch.randn(shape, dtype=dtype, device="cuda")
        v = torch.randn(shape, dtype=dtype, device="cuda")

        if provider == "ninetoothed":
            ms = triton.testing.do_bench(
                lambda: attention(
                    q.view(q.shape[-2:]), k.view(k.shape[-2:]), v.view(v.shape[-2:])
                )
            )
        elif provider == "torch":
            ms = triton.testing.do_bench(
                lambda: F.scaled_dot_product_attention(q, k, v, scale=1)
            )

        def perf(ms):
            flops_per_matmul = 2 * n * n * d
            total_flops = 2 * flops_per_matmul

            return total_flops * 1e-12 / (ms * 1e-3)

        return perf(ms)

    benchmark.run(show_plots=True, print_data=True, save_path=".")
