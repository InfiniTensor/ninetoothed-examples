import functools
import random

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional
import triton

import ops.ninetoothed.torch
import ops.triton.torch
import rotary_position_embedding
from compare_code_metrics import _BACKSLASH_CHAR


def _run_task(op_name, dtype, device, *arg_shapes, **kwarg_shapes):
    ninetoothed_op = getattr(ops.ninetoothed.torch, op_name)
    triton_op = getattr(ops.triton.torch, op_name)

    if op_name == "rotary_position_embedding":
        torch_op = rotary_position_embedding.torch_rotary_position_embedding
    else:
        torch_op = (
            getattr(torch, op_name)
            if hasattr(torch, op_name)
            else getattr(torch.nn.functional, op_name)
        )

    if op_name == "rms_norm":
        torch_op = functools.partial(torch_op, normalized_shape=arg_shapes[0][-1:])
    elif op_name == "softmax":
        torch_op = functools.partial(torch_op, dim=-1)

    args = tuple(
        torch.randn(shape, dtype=dtype, device=device) if shape else random.gauss(0, 1)
        for shape in arg_shapes
    )
    kwargs = {
        key: torch.randn(shape, dtype=dtype, device=device)
        if shape
        else random.gauss(0, 1)
        for key, shape in kwarg_shapes.items()
    }

    arg_shape_string = ", ".join(str(shape) for shape in arg_shapes)
    kwarg_shape_string = ", ".join(
        f"{key}={shape}" for key, shape in kwarg_shapes.items()
    )
    shape_string = (
        f"{arg_shape_string}, {kwarg_shape_string}"
        if kwarg_shape_string
        else arg_shape_string
    )

    task_description = f"{op_name}({shape_string})"

    return task_description, _benchmark_ops(
        (ninetoothed_op, triton_op, torch_op), *args, **kwargs
    )


def _benchmark_ops(ops, *args, **kwargs):
    assert all(
        torch.allclose(
            op(*args, **kwargs), ops[0](*args, **kwargs), rtol=0.01, atol=0.01
        )
        for op in ops[1:]
    )

    return tuple(triton.testing.do_bench(lambda: op(*args, **kwargs)) for op in ops)


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)

    plt.rcParams["figure.dpi"] = 600
    plt.rcParams["font.family"] = "Linux Biolinum"

    dtype = torch.float16
    device = "cuda"

    tasks = (
        ("add", ((4096 * 4096,), (4096 * 4096,)), {}),
        (
            "addmm",
            ((4096, 4096), (4096, 4096), (4096, 4096)),
            {"beta": (), "alpha": ()},
        ),
        ("bmm", ((4, 2048, 2048), (4, 2048, 2048)), {}),
        ("conv2d", ((4, 512, 14, 14), (512, 512, 3, 3)), {}),
        ("mm", ((4096, 4096), (4096, 4096)), {}),
        ("rms_norm", ((4096, 4096),), {}),
        ("rotary_position_embedding", ((4, 1024, 48, 64), (1024, 32), (1024, 32)), {}),
        (
            "scaled_dot_product_attention",
            ((4, 48, 1024, 64), (4, 48, 1024, 64), (4, 48, 1024, 64)),
            {},
        ),
        ("silu", ((4096 * 4096,),), {}),
        ("softmax", ((4096, 4096),), {}),
    )

    data = {"Task": [], "NineToothed": [], "Triton": [], "PyTorch": []}

    for name, args, kwargs in tasks:
        description, results = _run_task(name, dtype, device, *args, **kwargs)

        latex_item = f"\item {_BACKSLASH_CHAR}texttt{{{description.replace('scaled_dot_product_attention', 'sdpa').replace('rotary_position_embedding', 'rope').replace('_', f'{_BACKSLASH_CHAR}_')}}}"

        print(latex_item)

        data["Task"].append(description)

        for i, provider in enumerate(("NineToothed", "Triton", "PyTorch")):
            data[provider].append(results[i])

    df = pd.DataFrame(data)
    df.index += 1

    df.set_index("Task").to_csv("performance-metrics.csv")

    df.plot(kind="bar", rot=0)
    plt.ylabel("Execution Time (ms)")
    plt.xlabel("Task")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("performance-metrics.png")
