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
    memory_bound: bool
    compute_bound: bool
    perf_report_path: str
    independent_variable: str


@dataclass
class CategoryInformation:
    kernels: tuple
    y_label: str


kernels = (
    KernelInformation("add", True, False, "vector-addition-performance.csv", "Length"),
    KernelInformation(
        "softmax", True, False, "softmax-performance.csv", "Number of Columns"
    ),
    KernelInformation(
        "rms_norm", True, False, "rms-norm-performance.csv", "Number of Columns"
    ),
    KernelInformation(
        "matmul", False, True, "matrix-multiplication-performance.csv", "Sizes"
    ),
    KernelInformation(
        "conv2d", False, True, "2d-convolution-performance.csv", "Batch Size"
    ),
    KernelInformation(
        "attention", False, True, "attention-performance.csv", "Sequence Length"
    ),
)

providers = ("Triton", "NineToothed")

categories = (
    CategoryInformation(
        tuple(kernel for kernel in kernels if kernel.memory_bound), "GB/s"
    ),
    CategoryInformation(
        tuple(kernel for kernel in kernels if kernel.compute_bound), "TFLOPS"
    ),
)

num_rows = len(categories)
num_cols = max(len(category.kernels) for category in categories)

fig, axs = plt.subplots(num_rows, num_cols)

performance_differences = []

for row, category in enumerate(categories):
    axs[row, 0].set_ylabel(category.y_label)

    for col, kernel in enumerate(category.kernels):
        df = pd.read_csv(kernel.perf_report_path)
        ax = axs[row, col]

        x = df.iloc[:, 0]

        performance_differences.append((kernel, []))

        for provider in providers:
            y = df[provider]

            ax.plot(x, y, label=provider)

            if provider == "NineToothed":
                y_triton = df["Triton"]
                diff = (y - y_triton) / y_triton * 100
                performance_differences[-1][-1].append(diff)

            ax.set_title(kernel.name)
            ax.set_xlabel(kernel.independent_variable)
            ax.set_xscale("log", base=2)

fig.legend(providers, loc="upper center", ncols=len(providers))
fig.tight_layout()
fig.subplots_adjust(top=0.9)

plt.show()
plt.savefig("performance-comparison.png")

all_differences = []
stats_data = []

for kernel, diffs in performance_differences:
    all_differences.extend(diffs)

    kernel_stats = {
        "Kernel": kernel.name,
        "Mean": np.mean(diffs),
        "Median": np.median(diffs),
    }

    stats_data.append(kernel_stats)

overall_stats = {
    "Kernel": "Overall",
    "Mean": np.mean(all_differences),
    "Median": np.median(all_differences),
}

stats_data.append(overall_stats)

print("Relative Performance Change (%):")
print(pd.DataFrame(stats_data))
