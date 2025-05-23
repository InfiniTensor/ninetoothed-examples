import argparse
import functools
import random
import subprocess

import pandas as pd
import torch
import torch.nn.functional
import triton

import ops.ninetoothed.torch
import ops.triton.torch
import rotary_position_embedding

PROMPTS = (
    "The emergence of deep learning domain-specific languages (DSLs) has substantially reduced the obstacles in developing high-performance, cross-platform compute kernels, but current DSLs",
    "Driven by recent advancements in the AI industry, the AI accelerator sector has increasingly diversified, with vendors developing their own hardware architectures and programming models, such as NVIDIA",
)

NUM_WARMUP_ITERATIONS = 1

NUM_PROFILING_ITERATIONS = 3

BACKENDS = ("ninetoothed", "triton", "torch")

ALL_MAX_NEW_TOKENS = (128, 512, 2048)


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
    parser = argparse.ArgumentParser(description="Run experiments.")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model or model identifier from Hugging Face.",
    )

    args = parser.parse_args()

    model_name_or_path = args.model

    random.seed(0)
    torch.manual_seed(0)

    radon_commands = (
        (
            "radon",
            "cc",
            "--show-complexity",
            "--json",
            "--output-file",
            "cc.json",
            "ops/",
        ),
        ("radon", "mi", "--show", "--json", "--output-file", "mi.json", "ops/"),
        ("radon", "raw", "--json", "--output-file", "raw.json", "ops/"),
        ("radon", "hal", "--json", "--output-file", "hal.json", "ops/"),
    )

    for command in radon_commands:
        subprocess.run(command, check=True)

    with open("code_evaluation.tex", "w") as f:
        subprocess.run(("python", "evaluate_code.py"), stdout=f, check=True)

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

        data["Task"].append(description)

        for i, provider in enumerate(("NineToothed", "Triton", "PyTorch")):
            data[provider].append(results[i])

        pd.DataFrame(data).set_index("Task").to_csv("microbenchmark_data.csv")

    for max_new_tokens in ALL_MAX_NEW_TOKENS:
        for backend in BACKENDS:
            with open(f"infer_{max_new_tokens}_{backend}.json", "w") as f:
                subprocess.run(
                    (
                        "python",
                        "infer.py",
                        "--model",
                        model_name_or_path,
                        "--prompts",
                        *PROMPTS,
                        "--max-new-tokens",
                        str(max_new_tokens),
                        "--device",
                        "cuda",
                        "--backend",
                        "ninetoothed",
                        "--num-warmup-iterations",
                        str(NUM_WARMUP_ITERATIONS),
                        "--num-profiling-iterations",
                        str(NUM_PROFILING_ITERATIONS),
                    ),
                    stdout=f,
                    check=True,
                )

    with open("performance_evaluation.tex", "w") as f:
        subprocess.run(("python", "evaluate_performance.py"), stdout=f, check=True)
