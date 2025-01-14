from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["figure.figsize"] = [12, 6]
plt.rcParams["figure.dpi"] = 600
plt.rcParams["font.family"] = "JetBrains Mono"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"


@dataclass
class KernelInformation:
    name: str
    perf_report_path: str
    independent_variable: str


kernels = (
    KernelInformation("add", "vector-addition-performance.csv", "Length"),
    KernelInformation("softmax", "softmax-performance.csv", "Number of Columns"),
    KernelInformation("rms_norm", "rms-norm-performance.csv", "Number of Columns"),
    KernelInformation("matmul", "matrix-multiplication-performance.csv", "Sizes"),
    KernelInformation("conv2d", "2d-convolution-performance.csv", "Batch Size"),
    KernelInformation("attention", "attention-performance.csv", "Sequence Length"),
)

providers = ("Triton", "NineToothed", "PyTorch")

num_rows = 2
num_cols = 3

fig, axs = plt.subplots(num_rows, num_cols)

performance_changes = []

for i, kernel in enumerate(kernels):
    df = pd.read_csv(kernel.perf_report_path)
    ax = axs[i // num_cols, i % num_cols]

    x = df.iloc[:, 0]

    performance_changes.append((kernel, []))

    for provider in providers:
        y = df[provider]

        ax.plot(x, y, label=provider)

        if provider == "NineToothed":
            y_triton = df["Triton"]
            change = (y - y_triton) / y_triton * 100
            performance_changes[-1][-1].append(change)

        ax.set_title(kernel.name)
        ax.set_xlabel(kernel.independent_variable)
        ax.set_ylabel("Execution Time (ms)")
        ax.set_xscale("log", base=2)

fig.legend(providers, loc="upper center", ncols=len(providers))
fig.tight_layout()
fig.subplots_adjust(top=0.9)

plt.show()
plt.savefig("performance-comparison.png")

all_changes = []
stats_data = []

for kernel, changes in performance_changes:
    all_changes.extend(changes)

    kernel_stats = {
        "Kernel": kernel.name,
        "Mean": np.mean(changes),
        "Median": np.median(changes),
    }

    stats_data.append(kernel_stats)

overall_stats = {
    "Kernel": "Overall",
    "Mean": np.mean(all_changes),
    "Median": np.median(all_changes),
}

stats_data.append(overall_stats)

print("Relative Performance Change (%):")
print(pd.DataFrame(stats_data))
