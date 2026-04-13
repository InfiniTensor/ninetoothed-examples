import random

import pytest
import torch
import torch.nn.functional as F

import ops.ninetoothed.torch
import ops.triton.torch
from bench import assert_match
from modules import generate_sin_and_cos_tables, torch_rotary_position_embedding

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

DTYPE = torch.float16
DEVICE = "cuda"


class TestAdd:
    def test_correctness(self):
        assert_match(
            {
                "ninetoothed": ops.ninetoothed.torch.add,
                "torch": torch.add,
                "triton": ops.triton.torch.add,
            },
            args=(
                torch.randn(98432, dtype=DTYPE, device=DEVICE),
                torch.randn(98432, dtype=DTYPE, device=DEVICE),
            ),
            tolerances={"triton": {"atol": 0, "rtol": 0}},
        )


class TestMM:
    def test_correctness(self):
        assert_match(
            {
                "ninetoothed": ops.ninetoothed.torch.mm,
                "torch": torch.mm,
                "triton": ops.triton.torch.mm,
            },
            args=(
                torch.randn((512, 512), dtype=DTYPE, device=DEVICE),
                torch.randn((512, 512), dtype=DTYPE, device=DEVICE),
            ),
            tolerances={"triton": {"atol": 0, "rtol": 0}},
        )


class TestBMM:
    def test_correctness(self):
        assert_match(
            {
                "ninetoothed": ops.ninetoothed.torch.bmm,
                "torch": torch.bmm,
                "triton": ops.triton.torch.bmm,
            },
            args=(
                torch.randn((4, 512, 1024), dtype=DTYPE, device=DEVICE),
                torch.randn((4, 1024, 2028), dtype=DTYPE, device=DEVICE),
            ),
            tolerances={"triton": {"atol": 0, "rtol": 0}},
        )


class TestAddMM:
    def test_correctness(self):
        random.seed(0)
        shape = (512, 512)
        beta = random.uniform(0, 1)
        alpha = random.uniform(0, 1)

        assert_match(
            {
                "ninetoothed": ops.ninetoothed.torch.addmm,
                "torch": torch.addmm,
                "triton": ops.triton.torch.addmm,
            },
            args=(
                torch.randn(shape, dtype=DTYPE, device=DEVICE),
                torch.randn(shape, dtype=DTYPE, device=DEVICE),
                torch.randn(shape, dtype=DTYPE, device=DEVICE),
            ),
            kwargs={"beta": beta, "alpha": alpha},
            tolerances={"torch": {"atol": 0.01, "rtol": 0.01}},
        )


class TestConv2D:
    def test_correctness(self):
        n, c, h, w = 4, 3, 224, 224
        k, r, s = 8, 3, 3

        assert_match(
            {
                "ninetoothed": ops.ninetoothed.torch.conv2d,
                "torch": F.conv2d,
                "triton": ops.triton.torch.conv2d,
            },
            args=(
                torch.randn(n, c, h, w, dtype=DTYPE, device=DEVICE),
                torch.randn(k, c, r, s, dtype=DTYPE, device=DEVICE),
            ),
            tolerances={
                "torch": {"atol": 0.01, "rtol": 0.01},
                "triton": {"atol": 0, "rtol": 0},
            },
        )


class TestSoftmax:
    def test_correctness(self):
        assert_match(
            {
                "ninetoothed": ops.ninetoothed.torch.softmax,
                "torch": lambda input: torch.softmax(input, dim=-1),
                "triton": ops.triton.torch.softmax,
            },
            args=(torch.randn(1823, 781, dtype=DTYPE, device=DEVICE),),
            tolerances={
                "torch": {"atol": 0.001},
                "triton": {"atol": 0, "rtol": 0},
            },
        )


class TestRMSNorm:
    def test_correctness(self):
        assert_match(
            {
                "ninetoothed": ops.ninetoothed.torch.rms_norm,
                "torch": lambda input: F.rms_norm(input, input.shape[-1:]),
                "triton": ops.triton.torch.rms_norm,
            },
            args=(torch.randn(1151, 8192, dtype=DTYPE, device=DEVICE),),
            tolerances={
                "torch": {"atol": 0.001, "rtol": 0.005},
                "triton": {"atol": 0, "rtol": 0},
            },
        )


class TestFusedRMSNorm:
    def test_correctness(self):
        assert_match(
            {
                "ninetoothed": ops.ninetoothed.torch.fused_rms_norm,
                "torch": lambda x, w, eps: F.rms_norm(x, x.shape[-1:], w, eps),
                "triton": ops.triton.torch.fused_rms_norm,
            },
            args=(
                torch.randn(1151, 8192, dtype=DTYPE, device=DEVICE),
                torch.randn(8192, dtype=DTYPE, device=DEVICE),
                1e-5,
            ),
            tolerances={
                "torch": {"atol": 0.001, "rtol": 0.005},
                "triton": {"atol": 0.001, "rtol": 0.005},
            },
        )


class TestSiLU:
    def test_correctness(self):
        assert_match(
            {
                "ninetoothed": ops.ninetoothed.torch.silu,
                "torch": F.silu,
                "triton": ops.triton.torch.silu,
            },
            args=(torch.randn((8, 256, 512), dtype=DTYPE, device=DEVICE),),
            tolerances={
                "torch": {"atol": 1e-3, "rtol": 1e-3},
                "triton": {"atol": 0, "rtol": 0},
            },
        )


class TestSwiGLU:
    def test_correctness(self):
        assert_match(
            {
                "ninetoothed": ops.ninetoothed.torch.swiglu,
                "torch": lambda a, b: a * F.silu(b),
                "triton": ops.triton.torch.swiglu,
            },
            args=(
                torch.randn((13, 3), dtype=DTYPE, device=DEVICE),
                torch.randn((13, 3), dtype=DTYPE, device=DEVICE),
            ),
            tolerances={"torch": {"atol": 0, "rtol": 1e-3}},
        )


class TestRotaryPositionEmbedding:
    def test_correctness(self):
        batch_size, seq_len, num_heads, emb_dim = 4, 128, 8, 64
        sin_table, cos_table = generate_sin_and_cos_tables(seq_len, emb_dim)

        assert_match(
            {
                "ninetoothed": ops.ninetoothed.torch.rotary_position_embedding,
                "torch": torch_rotary_position_embedding,
                "triton": ops.triton.torch.rotary_position_embedding,
            },
            args=(
                torch.randn(
                    batch_size,
                    seq_len,
                    num_heads,
                    emb_dim,
                    dtype=torch.float32,
                    device=DEVICE,
                ),
                sin_table,
                cos_table,
            ),
            kwargs={"interleaved": False},
            tolerances={
                "torch": {"atol": 0.001},
                "triton": {"atol": 0, "rtol": 0},
            },
        )


class TestScaledDotProductAttention:
    def test_correctness(self):
        q_o_shape = (2, 8, 1024, 64)
        k_v_shape = (2, 8, 1024, 64)

        assert_match(
            {
                "ninetoothed": ops.ninetoothed.torch.scaled_dot_product_attention,
                "torch": F.scaled_dot_product_attention,
                "triton": ops.triton.torch.scaled_dot_product_attention,
            },
            args=(
                torch.randn(q_o_shape, dtype=DTYPE, device=DEVICE),
                torch.randn(k_v_shape, dtype=DTYPE, device=DEVICE),
                torch.randn(k_v_shape, dtype=DTYPE, device=DEVICE),
            ),
            tolerances={
                "torch": {"atol": 0.01},
                "triton": {"atol": 1e-3, "rtol": 0},
            },
        )


class TestMaxPool2D:
    def test_correctness(self):
        input = torch.randn((32, 3, 64, 64), dtype=DTYPE, device=DEVICE)
        window_shape = (3, 3)

        ninetoothed_output = ops.ninetoothed.torch.max_pool2d(input, window_shape)
        torch_output = F.max_pool2d(input, window_shape, ceil_mode=True)

        assert torch.allclose(ninetoothed_output, torch_output)
