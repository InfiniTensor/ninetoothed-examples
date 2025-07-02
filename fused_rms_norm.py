from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton

import ops.ninetoothed.torch
import ops.triton.torch


class RMSNorm(nn.Module):
    fused_rms_norm = None

    def __init__(self, other):
        super().__init__()

        self.__dict__ = other.__dict__

    def forward(self, x):
        return type(self).fused_rms_norm(x, self.weight, self.variance_epsilon)


@contextmanager
def rms_norm_backend(backend_name):
    def _torch_fused_rms_norm(x, w, eps):
        return F.rms_norm(x, x.shape[-1:], w, eps)

    _prev_impl = RMSNorm.fused_rms_norm

    if backend_name == "ninetoothed":
        impl = ops.ninetoothed.torch.fused_rms_norm
    elif backend_name == "triton":
        impl = ops.triton.torch.fused_rms_norm
    elif backend_name == "torch":
        impl = _torch_fused_rms_norm
    else:
        raise ValueError(f"unknown backend: `{backend_name}`")

    RMSNorm.fused_rms_norm = impl

    try:
        yield
    finally:
        RMSNorm.fused_rms_norm = _prev_impl


if __name__ == "__main__":
    torch.manual_seed(0)

    dtype = torch.float16
    device = "cuda"

    x = torch.randn(1151, 8192, dtype=dtype, device=device)
    w = torch.randn(8192, dtype=dtype, device=device)
    eps = 1e-5

    ninetoothed_output = ops.ninetoothed.torch.fused_rms_norm(x, w, eps)
    torch_output = F.rms_norm(x, x.shape[-1:], w, eps)
    triton_output = ops.triton.torch.fused_rms_norm(x, w, eps)

    print(ninetoothed_output)
    print(torch_output)
    print(triton_output)

    if torch.allclose(ninetoothed_output, torch_output, atol=0.001, rtol=0.005):
        print("✅ NineToothed and PyTorch match.")
    else:
        print("❌ NineToothed and PyTorch differ.")
    if torch.allclose(ninetoothed_output, triton_output, atol=0.001, rtol=0.005):
        print("✅ NineToothed and Triton match.")
    else:
        print("❌ NineToothed and Triton differ.")

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["n"],
            x_vals=[2**i for i in range(5, 15)],
            x_log=True,
            line_arg="provider",
            line_vals=["ninetoothed", "torch", "triton"],
            line_names=["NineToothed", "PyTorch", "Triton"],
            styles=[("blue", "-"), ("green", "-"), ("orange", "-")],
            ylabel="ms",
            plot_name="fused-rms-norm-performance",
            args={"m": 4096},
        )
    )
    def benchmark(m, n, provider):
        x = torch.randn(m, n, dtype=dtype, device=device)
        w = torch.randn(n, dtype=dtype, device=device)
        eps = 1e-5

        ninetoothed_output = ops.ninetoothed.torch.fused_rms_norm(x, w, eps)
        torch_output = F.rms_norm(x, x.shape[-1:], w, eps)
        triton_output = ops.triton.torch.fused_rms_norm(x, w, eps)

        assert torch.allclose(ninetoothed_output, torch_output, atol=0.001, rtol=0.005)
        assert torch.allclose(ninetoothed_output, triton_output, atol=0.001, rtol=0.005)

        if provider == "ninetoothed":
            ms = triton.testing.do_bench(
                lambda: ops.ninetoothed.torch.fused_rms_norm(x, w, eps)
            )
        elif provider == "torch":
            ms = triton.testing.do_bench(lambda: F.rms_norm(x, x.shape[-1:], w, eps))
        elif provider == "triton":
            ms = triton.testing.do_bench(
                lambda: ops.triton.torch.fused_rms_norm(x, w, eps)
            )

        return ms

    benchmark.run(show_plots=True, print_data=True, save_path=".")
