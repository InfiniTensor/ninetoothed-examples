import random

import pytest
import torch
import torch.nn.functional as F

import bench
import ops.ninetoothed.torch
import ops.triton.torch
from modules import generate_sin_and_cos_tables, torch_rotary_position_embedding

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.benchmark,
]

DTYPE = torch.float16
DEVICE = "cuda"


class TestAddBenchmark:
    def test_benchmark(self):
        impls = {
            "ninetoothed": ops.ninetoothed.torch.add,
            "torch": torch.add,
            "triton": ops.triton.torch.add,
        }

        def make_inputs(size):
            return (
                torch.randn(size, dtype=DTYPE, device=DEVICE),
                torch.randn(size, dtype=DTYPE, device=DEVICE),
            ), {}

        bench.benchmark(
            impls,
            make_inputs,
            x_names=["size"],
            x_vals=[2**i for i in range(18, 28)],
            tolerances={"triton": {"atol": 0, "rtol": 0}},
            name="add",
        )


class TestMMBenchmark:
    def test_benchmark(self):
        impls = {
            "ninetoothed": ops.ninetoothed.torch.mm,
            "torch": torch.mm,
            "triton": ops.triton.torch.mm,
        }

        def make_inputs(m, n, k):
            return (
                torch.randn((m, k), dtype=DTYPE, device=DEVICE),
                torch.randn((k, n), dtype=DTYPE, device=DEVICE),
            ), {}

        bench.benchmark(
            impls,
            make_inputs,
            x_names=["m", "n", "k"],
            x_vals=[2**i for i in range(3, 13)],
            tolerances={
                "torch": {"atol": 0.025, "rtol": 0.025},
                "triton": {"atol": 0, "rtol": 0},
            },
            name="mm",
        )


class TestBMMBenchmark:
    def test_benchmark(self):
        impls = {
            "ninetoothed": ops.ninetoothed.torch.bmm,
            "torch": torch.bmm,
            "triton": ops.triton.torch.bmm,
        }

        def make_inputs(b, m, n, k):
            return (
                torch.randn((b, m, k), dtype=DTYPE, device=DEVICE),
                torch.randn((b, k, n), dtype=DTYPE, device=DEVICE),
            ), {}

        bench.benchmark(
            impls,
            make_inputs,
            x_names=["m", "n", "k"],
            x_vals=[2**i for i in range(3, 13)],
            benchmark_args={"b": 4},
            tolerances={"triton": {"atol": 0, "rtol": 0}},
            name="bmm",
        )


class TestAddMMBenchmark:
    def test_benchmark(self):
        random.seed(0)

        impls = {
            "ninetoothed": ops.ninetoothed.torch.addmm,
            "torch": torch.addmm,
            "triton": ops.triton.torch.addmm,
        }

        def make_inputs(m, n, k):
            return (
                torch.randn((m, n), dtype=DTYPE, device=DEVICE),
                torch.randn((m, k), dtype=DTYPE, device=DEVICE),
                torch.randn((k, n), dtype=DTYPE, device=DEVICE),
            ), {"beta": random.uniform(0, 1), "alpha": random.uniform(0, 1)}

        bench.benchmark(
            impls,
            make_inputs,
            x_names=["m", "n", "k"],
            x_vals=[128 * i for i in range(2, 33)],
            x_log=False,
            assert_correctness=False,
            name="addmm",
        )


class TestConv2DBenchmark:
    def test_benchmark(self):
        impls = {
            "ninetoothed": ops.ninetoothed.torch.conv2d,
            "torch": F.conv2d,
            "triton": ops.triton.torch.conv2d,
        }

        def make_inputs(n):
            c, h, w = 512, 14, 14
            k, r, s = 512, 3, 3

            return (
                torch.randn((n, c, h, w), dtype=DTYPE, device=DEVICE),
                torch.randn((k, c, r, s), dtype=DTYPE, device=DEVICE),
            ), {}

        bench.benchmark(
            impls,
            make_inputs,
            x_names=["n"],
            x_vals=[2**i for i in range(1, 11)],
            tolerances={
                "torch": {"atol": 0.01, "rtol": 0.01},
                "triton": {"atol": 0, "rtol": 0},
            },
            name="conv2d",
        )


class TestSoftmaxBenchmark:
    def test_benchmark(self):
        impls = {
            "ninetoothed": ops.ninetoothed.torch.softmax,
            "torch": lambda input: torch.softmax(input, dim=-1),
            "triton": ops.triton.torch.softmax,
        }

        def make_inputs(m, n):
            return (torch.randn(m, n, dtype=DTYPE, device=DEVICE),), {}

        bench.benchmark(
            impls,
            make_inputs,
            x_names=["n"],
            x_vals=[2**i for i in range(5, 15)],
            benchmark_args={"m": 4096},
            tolerances={
                "torch": {"atol": 0.001},
                "triton": {"atol": 0, "rtol": 0},
            },
            name="softmax",
        )


class TestRMSNormBenchmark:
    def test_benchmark(self):
        impls = {
            "ninetoothed": ops.ninetoothed.torch.rms_norm,
            "torch": lambda input: F.rms_norm(input, input.shape[-1:]),
            "triton": ops.triton.torch.rms_norm,
        }

        def make_inputs(m, n):
            return (torch.randn(m, n, dtype=DTYPE, device=DEVICE),), {}

        bench.benchmark(
            impls,
            make_inputs,
            x_names=["n"],
            x_vals=[2**i for i in range(5, 15)],
            benchmark_args={"m": 4096},
            tolerances={
                "torch": {"atol": 0.001, "rtol": 0.005},
                "triton": {"atol": 0, "rtol": 0},
            },
            name="rms-norm",
        )


class TestFusedRMSNormBenchmark:
    def test_benchmark(self):
        impls = {
            "ninetoothed": ops.ninetoothed.torch.fused_rms_norm,
            "torch": lambda x, w, eps: F.rms_norm(x, x.shape[-1:], w, eps),
            "triton": ops.triton.torch.fused_rms_norm,
        }

        def make_inputs(m, n):
            return (
                torch.randn(m, n, dtype=DTYPE, device=DEVICE),
                torch.randn(n, dtype=DTYPE, device=DEVICE),
                1e-5,
            ), {}

        bench.benchmark(
            impls,
            make_inputs,
            x_names=["n"],
            x_vals=[2**i for i in range(5, 15)],
            benchmark_args={"m": 4096},
            tolerances={
                "torch": {"atol": 0.001, "rtol": 0.005},
                "triton": {"atol": 0.001, "rtol": 0.005},
            },
            name="fused-rms-norm",
        )


class TestSiLUBenchmark:
    def test_benchmark(self):
        impls = {
            "ninetoothed": ops.ninetoothed.torch.silu,
            "torch": F.silu,
            "triton": ops.triton.torch.silu,
        }

        def make_inputs(m, n, k):
            return (torch.randn(m, n, k, dtype=DTYPE, device=DEVICE),), {}

        bench.benchmark(
            impls,
            make_inputs,
            x_names=["m", "n", "k"],
            x_vals=[2**i for i in range(3, 10)],
            tolerances={
                "torch": {"atol": 0.001},
                "triton": {"atol": 0, "rtol": 0},
            },
            name="silu",
        )


class TestSwiGLUBenchmark:
    def test_benchmark(self):
        impls = {
            "ninetoothed": ops.ninetoothed.torch.swiglu,
            "torch": lambda a, b: a * F.silu(b),
            "triton": ops.triton.torch.swiglu,
        }

        def make_inputs(m, n):
            return (
                torch.randn((m, n), dtype=DTYPE, device=DEVICE),
                torch.randn((m, n), dtype=DTYPE, device=DEVICE),
            ), {}

        bench.benchmark(
            impls,
            make_inputs,
            x_names=["m", "n"],
            x_vals=[128 * i for i in range(2, 50)],
            assert_correctness=False,
            name="swiglu",
        )


class TestRotaryPositionEmbeddingBenchmark:
    def test_benchmark(self):
        impls = {
            "ninetoothed": ops.ninetoothed.torch.rotary_position_embedding,
            "torch": torch_rotary_position_embedding,
            "triton": ops.triton.torch.rotary_position_embedding,
        }

        def make_inputs(seq_len):
            batch_size, num_heads, emb_dim = 4, 32, 64
            sin_table, cos_table = generate_sin_and_cos_tables(seq_len, emb_dim)

            return (
                torch.randn(
                    (batch_size, seq_len, num_heads, emb_dim),
                    dtype=torch.float16,
                    device=DEVICE,
                ),
                sin_table,
                cos_table,
            ), {}

        bench.benchmark(
            impls,
            make_inputs,
            x_names=["seq_len"],
            x_vals=[2**i for i in range(5, 15)],
            x_log=False,
            assert_correctness=False,
            name="rotary-position-embedding",
        )


class TestScaledDotProductAttentionBenchmark:
    def test_benchmark(self):
        impls = {
            "ninetoothed": ops.ninetoothed.torch.scaled_dot_product_attention,
            "torch": F.scaled_dot_product_attention,
            "triton": ops.triton.torch.scaled_dot_product_attention,
        }

        def make_inputs(seq_len):
            batch_size, num_heads, emb_dim = 4, 32, 64
            shape = (batch_size, num_heads, seq_len, emb_dim)

            return (
                torch.randn(shape, dtype=DTYPE, device=DEVICE),
                torch.randn(shape, dtype=DTYPE, device=DEVICE),
                torch.randn(shape, dtype=DTYPE, device=DEVICE),
            ), {}

        bench.benchmark(
            impls,
            make_inputs,
            x_names=["seq_len"],
            x_vals=[2**i for i in range(7, 17)],
            tolerances={
                "torch": {"atol": 0.025, "rtol": 0.025},
                "triton": {"atol": 0.001, "rtol": 0.001},
            },
            name="scaled-dot-product-attention",
        )


class TestMaxPool2DBenchmark:
    def test_benchmark(self):
        impls = {
            "ninetoothed": ops.ninetoothed.torch.max_pool2d,
            "torch": lambda input, window_shape: F.max_pool2d(
                input, window_shape, ceil_mode=True
            ),
        }

        def make_inputs(h, w):
            return (
                torch.randn((64, 64, h, w), dtype=DTYPE, device=DEVICE),
                (3, 3),
            ), {}

        bench.benchmark(
            impls,
            make_inputs,
            x_names=["h", "w"],
            x_vals=[8 * i for i in range(2, 33)],
            x_log=False,
            assert_correctness=False,
            name="max-pool2d",
        )
