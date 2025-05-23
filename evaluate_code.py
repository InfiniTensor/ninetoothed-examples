import functools
import json
import os.path
from pathlib import Path

import pandas as pd

_PARENT_PATH = Path(__file__).parent

_OPS_PATH = _PARENT_PATH / "ops"

_NINETOOTHED_KERNELS_PATH = _OPS_PATH / "ninetoothed" / "kernels"

_TRITON_KERNELS_PATH = _OPS_PATH / "triton" / "kernels"

_BACKSLASH_CHAR = "\\"


def _generate_cc_table():
    path = _PARENT_PATH / "cc.json"

    metric_names = {"complexity": "$G$"}

    data = json.loads(path.read_text())

    data = {
        kernel: {
            metric_names["complexity"]: sum(block["complexity"] for block in blocks)
        }
        for kernel, blocks in data.items()
        if "torch" not in kernel
    }

    df = _generate_table(data, metric_names.values())

    return df


def _generate_mi_table():
    path = _PARENT_PATH / "mi.json"

    metric_names = {"mi": "$MI$"}

    data = json.loads(path.read_text())

    data = {
        kernel: {
            latex_name: metrics[raw_name]
            for raw_name, latex_name in metric_names.items()
        }
        for kernel, metrics in data.items()
        if "torch" not in kernel
    }

    df = _generate_table(data, metric_names.values())

    return df


def _generate_raw_table():
    path = _PARENT_PATH / "raw.json"

    metric_names = {"loc": "LOC", "lloc": "LLOC", "sloc": "SLOC"}

    data = json.loads(path.read_text())

    data = {
        kernel: {
            latex_name: metrics[raw_name]
            for raw_name, latex_name in metric_names.items()
        }
        for kernel, metrics in data.items()
        if "torch" not in kernel
    }

    df = _generate_table(data, metric_names.values())

    return df


def _generate_hal_table():
    path = _PARENT_PATH / "hal.json"

    metric_names = {
        "vocabulary": "$\\eta$",
        "length": "$N$",
        "volume": "$V$",
        "difficulty": "$D$",
    }

    data = json.loads(path.read_text())

    data = {
        kernel: {
            latex_name: metrics["total"][raw_name]
            for raw_name, latex_name in metric_names.items()
        }
        for kernel, metrics in data.items()
        if "torch" not in kernel
    }

    df = _generate_table(data, metric_names.values())

    return df


def _generate_table(data, metric_names):
    kernel_names = sorted(
        set(
            os.path.splitext(os.path.basename(kernel_name))[0]
            for kernel_name in data.keys()
        )
    )

    def _key_from_kernel_name(path, kernel_name):
        return str(path / f"{kernel_name}.py").removeprefix(str(_PARENT_PATH))[1:]

    data = {
        f"{_BACKSLASH_CHAR}texttt{{{kernel_name.replace('scaled_dot_product_attention', 'sdpa').replace('rotary_position_embedding', 'rope').replace('_', f'{_BACKSLASH_CHAR}_')}}}": {
            "Triton": {
                metric_name: data[
                    _key_from_kernel_name(_TRITON_KERNELS_PATH, kernel_name)
                ][metric_name]
                for metric_name in metric_names
            },
            "NineToothed": {
                metric_name: data[
                    _key_from_kernel_name(_NINETOOTHED_KERNELS_PATH, kernel_name)
                ][metric_name]
                for metric_name in metric_names
            },
        }
        for kernel_name in kernel_names
    }

    df = pd.DataFrame.from_dict(
        {
            (outer_key, inner_key): value
            for outer_key, inner_dict in data.items()
            for inner_key, value in inner_dict.items()
        },
        orient="index",
    )

    df.index = pd.MultiIndex.from_tuples(df.index)

    return df


def _highlight(df):
    new_df = pd.DataFrame("", index=df.index, columns=df.columns)

    for _, group in df[
        ["LOC", "LLOC", "SLOC", "$G$", "$\\eta$", "$N$", "$V$", "$D$"]
    ].groupby(level=0):
        mask = group == group.min()

        new_df.update(
            mask.replace(True, "background-color: green!20").replace(False, "")
        )

    for _, group in df[["$MI$"]].groupby(level=0):
        mask = group == group.max()

        new_df.update(
            mask.replace(True, "background-color: green!20").replace(False, "")
        )

    return new_df


if __name__ == "__main__":
    raw_table = _generate_raw_table()
    cc_table = _generate_cc_table()
    hal_table = _generate_hal_table()
    mi_table = _generate_mi_table()

    df = functools.reduce(
        lambda left, right: pd.merge(left, right, left_index=True, right_index=True),
        (raw_table, cc_table, hal_table, mi_table),
    )

    styler = df.style.apply(_highlight, axis=None).format(precision=2)

    print(styler.to_latex(hrules=True, multicol_align="c", convert_css=True))
