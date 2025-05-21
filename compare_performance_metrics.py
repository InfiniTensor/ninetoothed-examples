import matplotlib.pyplot as plt
import pandas as pd

from compare_code_metrics import _BACKSLASH_CHAR

if __name__ == "__main__":
    plt.rcParams["figure.dpi"] = 600
    plt.rcParams["font.family"] = "Linux Biolinum"

    df = pd.read_csv("performance-metrics.csv")

    for task in df["Task"]:
        latex_item = f"\item {_BACKSLASH_CHAR}texttt{{{task.replace('scaled_dot_product_attention', 'sdpa').replace('rotary_position_embedding', 'rope').replace('_', f'{_BACKSLASH_CHAR}_')}}}"
        print(latex_item)

    df.index += 1
    df.plot(kind="bar", rot=0)
    plt.ylabel("Execution Time (ms)")
    plt.xlabel("Task")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("performance-metrics.png")
