import argparse
import subprocess

PROMPTS = (
    "The emergence of deep learning domain-specific languages (DSLs) has substantially reduced the obstacles in developing high-performance, cross-platform compute kernels, but current DSLs",
    "Driven by recent advancements in the AI industry, the AI accelerator sector has increasingly diversified, with vendors developing their own hardware architectures and programming models, such as NVIDIA",
)

NUM_WARMUP_ITERATIONS = 1

NUM_PROFILING_ITERATIONS = 3

BACKENDS = ("ninetoothed", "triton", "torch")

ALL_MAX_NEW_TOKENS = (128, 512, 2048)


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

    with open("code_metrics.tex", "w") as f:
        subprocess.run(("python", "compare_code_metrics.py"), stdout=f, check=True)

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

    with open("performance_metrics.tex", "w") as f:
        subprocess.run(
            ("python", "compare_performance_metrics.py"), stdout=f, check=True
        )
