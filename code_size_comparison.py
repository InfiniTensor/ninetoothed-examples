import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.dpi"] = 600
plt.rcParams["font.family"] = "JetBrains Mono"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

kernels = ("add", "softmax", "matmul", "conv2d", "attention")
lines_of_code = {"Triton": (19, 26, 57, 110, 98), "NineToothed": (10, 12, 34, 17, 51)}

x = np.arange(len(kernels))
width = 0.4
multiplier = 0

fig, ax = plt.subplots()

for provider, lines in lines_of_code.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, lines, width, label=provider)
    ax.bar_label(rects, fontsize=16)
    multiplier += 1

ax.set_ylabel("Lines of Code", fontsize=16)
ax.tick_params(axis="y", labelsize=12, labelcolor="gray")
ax.set_xticks(x + width / 2, kernels, fontsize=12)
ax.xaxis.set_ticks_position("none")
ax.yaxis.set_ticks_position("none")
ax.legend(fontsize=10)
ax.spines[["top", "left", "right"]].set_visible(False)
ax.spines["bottom"].set_linewidth(1.5)
ax.grid(axis="y", linewidth=1.5)
ax.set_axisbelow(True)

plt.show()
plt.savefig("code-size-comparison.png")
