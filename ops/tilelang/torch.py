from typing import Optional, Callable
import torch
import tilelang.language as T
import os

_TILELANG_EXAMPLES_CACHE = {}
_TILELANG_EXAMPLES_PATH = os.path.join(os.path.dirname(__file__), "../../examples/tilelang")


def _get_tilelang_example(op_name: str, example_path: str):
    
    if example_path not in _TILELANG_EXAMPLES_CACHE:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            f"examples.tilelang.{example_path}",
            f"{_TILELANG_EXAMPLES_PATH}/{example_path}"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _TILELANG_EXAMPLES_CACHE[example_path] = module

    return _TILELANG_EXAMPLES_CACHE[example_path]


def tune_mm(m: int, n: int, k: int, profile_backend: str = "event", dtype: torch.dtype = torch.float16):
    
    if dtype == torch.float16:
        tilelang_dtype = T.float16
    elif dtype == torch.float32:
        tilelang_dtype = T.float32
    elif dtype == torch.bfloat16:
        tilelang_dtype = T.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}. Only float16 and bfloat16 are supported.")

    example_module = _get_tilelang_example("gemm", "gemm/example_gemm_autotune.py")
    result = example_module.get_best_config(m, n, k, with_roller=True, profile_backend=profile_backend, dtype=tilelang_dtype)
    print(f"Best config: {result.config}")

    return result.kernel

# ==================== Gemm  ====================

def mm(input: torch.Tensor,
        other: torch.Tensor,
        kernel: Optional[Callable] = None) -> torch.Tensor:
    
    m, k = input.shape
    k2, n = other.shape
    assert k == k2, f"Matrix dimensions mismatch: ({m}, {k}) x ({k2}, {n})"

    if kernel is None:
        raise ValueError(
            "kernel is None. Please call tune_mm(m, n, k) first to get the optimal kernel, "
            "or pass the kernel directly."
        )

    return kernel(input, other.T.contiguous())


# ==================== RMS Norm  ====================


def rms_norm(input: torch.Tensor, eps: Optional[float] = None) -> torch.Tensor:
    
    example_module = _get_tilelang_example("rms_norm", "rms_norm/rms_norm.py")
    kernel = example_module.rms_norm(input.shape[0], input.shape[1], blk_m = 1)
    
    return kernel(input)


# ==================== Add  ====================


def add(input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    
    input_dtype = input.dtype
    if input_dtype == torch.float16:
        tilelang_dtype = T.float16
    elif input_dtype == torch.float32:
        tilelang_dtype = T.float32
    elif input_dtype == torch.bfloat16:
        tilelang_dtype = T.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {input_dtype}")

    example_module = _get_tilelang_example("add", "add/example_elementwise_add.py")
    kernel = example_module.elementwise_add(input.shape[0],
        input.shape[1],
        block_M=32,
        block_N=32,
        in_dtype=tilelang_dtype,
        out_dtype=tilelang_dtype,
        threads=128
    )

    return kernel(input, other)


# ==================== Softmax  ====================


def softmax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    input_dtype = input.dtype
    if input_dtype == torch.float16:
        tilelang_dtype = T.float16
    elif input_dtype == torch.float32:
        tilelang_dtype = T.float32
    elif input_dtype == torch.bfloat16:
        tilelang_dtype = T.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {input_dtype}")
   
    example_module = _get_tilelang_example("softmax", "softmax/online_softmax.py")
    return example_module.softmax_kernel(input.shape[0], input.shape[1], tilelang_dtype)(input)



def benchmark_tilelang_op(op_func: Callable,
                           warmup: int = 25,
                           rep: int = 100,
                           **kwargs) -> float:
    
    import triton.testing
    return triton.testing.do_bench(lambda: op_func(**kwargs), warmup=warmup, rep=rep)


def get_kernel_source(op_func: Callable, **kwargs) -> str:
    
    result = op_func(**kwargs)
    if hasattr(result, 'get_kernel_source'):
        return result.get_kernel_source()
    return "<source not available>"


# ==================== FlashAttention  ====================

def tune_scaled_dot_product_attention(
    batch: int,
    heads: int,
    seq_len: int,
    dim: int,
    causal: bool = False,
    profile_backend: str = "event",
    dtype: torch.dtype = torch.float16
):
    
    if dtype == torch.float16:
        tilelang_dtype = T.float16
    elif dtype == torch.bfloat16:
        tilelang_dtype = T.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}. Only float16 and bfloat16 are supported.")

    example_module = _get_tilelang_example(
        "flash_attention",
        "flash_attention/example_mha_fwd_bhsd.py"
    )
    result = example_module.flashattn(batch, heads, seq_len, seq_len, dim, causal)
    
    return result


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    kernel: Optional[Callable] = None
) -> torch.Tensor:
    
    if kernel is None:
        raise ValueError(
            "kernel is None. Please call tune_scaled_dot_product_attention() first "
            "to get optimal kernel, or pass kernel directly."
        )

    if q.dim() == 4:
        # [batch, heads, seq, dim]
        batch_val = q.shape[0]
        heads_val = q.shape[1]
        seq_val = q.shape[2]
        dim_val = q.shape[3]
    else:
        raise ValueError(f"Unsupported shape: {q.shape}")

    return kernel(q, k, v)
