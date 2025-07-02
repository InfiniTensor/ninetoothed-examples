import json

import matplotlib.pyplot as plt
import pandas as pd

from evaluate_code import _BACKSLASH_CHAR
from run_experiments import ALL_MAX_NEW_TOKENS, BACKENDS

if __name__ == "__main__":
    plt.rcParams["figure.dpi"] = 600
    plt.rcParams["font.family"] = "Linux Biolinum"

    df = pd.read_csv("microbenchmark_data.csv")

    for task in df["Task"]:
        latex_item = f"\item {_BACKSLASH_CHAR}texttt{{{task.replace('scaled_dot_product_attention', 'sdpa').replace('rotary_position_embedding', 'rope').replace('_', f'{_BACKSLASH_CHAR}_')}}}"
        print(latex_item)

    df.index += 1
    df.plot(kind="bar", rot=0)
    plt.ylabel("Execution Time (ms)")
    plt.xlabel("Task")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("microbenchmark-results.png")

    data = {"Output Length": [], "NineToothed": [], "Triton": [], "PyTorch": []}

    for max_new_tokens in ALL_MAX_NEW_TOKENS:
        data["Output Length"].append(max_new_tokens)

        for backend in BACKENDS:
            with open(f"infer_{max_new_tokens}_{backend}.json") as f:
                num_tokens_per_second = json.load(f)["num_tokens_per_second"]

            if backend == "ninetoothed":
                data["NineToothed"].append(num_tokens_per_second)
            elif backend == "triton":
                data["Triton"].append(num_tokens_per_second)
            elif backend == "torch":
                data["PyTorch"].append(num_tokens_per_second)

    df = pd.DataFrame(data)

    df.set_index("Output Length").plot(kind="bar", rot=0)
    plt.ylabel("Throughput (TPS)")
    plt.xlabel("Output Length")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("benchmark-results.png")
