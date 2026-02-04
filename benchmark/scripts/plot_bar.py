# Copyright 2025 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _plot_bar(
    df: pd.DataFrame,
    output_path: Path,
    name_col: str,
    value_col: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (8, 4),
    dpi: int = 150,
):
    df_use = df[[name_col, value_col]].copy()
    df_use[name_col] = df_use[name_col].astype(str)
    df_use[value_col] = df_use[value_col].astype(float)

    df_use = df_use.sort_values(value_col, ascending=False)

    _fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.bar(df_use[name_col], df_use[value_col])

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    plt.xticks(rotation=45, ha="right")

    for i, v in enumerate(df_use[value_col].to_numpy()):
        ax.text(i, v, f"{v:g}", ha="center", va="bottom", fontsize=8)

    if ylim:
        ax.set_ylim(ylim[0], ylim[1])
    plt.tight_layout()
    plt.savefig(output_path)


def plot_bar_from_csv(
    csv_path: Path,
    output_path: Path,
    name_col: str,
    value_col: str,
    eval_type: str,
    eval_dataset: str,
    subset: str,
    ranker: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (8, 4),
    dpi: int = 150,
):
    """
    Plot bar chart from csv file.

    Args:
        csv_path: Path to a result csv file.
        output_path: Path to output figure.
        name_col: Column name to use as x-axis.
        value_col: Column name to use as y-axis.
        eval_type: Value of eval_type column.
        eval_dataset: Value of eval_dataset column.
        subset:  Value of subset column.
        ranker: Value of ranker column.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        title: Figure title.
        ylim: Y-axis limit.
        dpi: Figure dpi.
    """
    df = pd.read_csv(csv_path)
    df = df[
        (df["ranker"] == ranker)
        & (df["eval_type"] == eval_type)
        & (df["eval_dataset"] == eval_dataset)
        & (df["subset"] == subset)
    ]

    _plot_bar(
        df,
        output_path,
        name_col,
        value_col,
        xlabel,
        ylabel,
        title,
        ylim,
        figsize,
        dpi,
    )
    logging.info("Save figure to %s", output_path)


if __name__ == "__main__":
    """
    Example:
    python3 benchmark/scripts/plot_bar.py -c pxm_results_monomer/LDDT_results.csv \
    -o best.png -n name -r best -v lddt -e Intra-Protein
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--csv_path", type=Path, required=True)
    parser.add_argument("-o", "--output_path", type=Path, required=True)
    parser.add_argument("-n", "--name_col", type=str, required=True)
    parser.add_argument("-v", "--value_col", type=str, required=True)
    parser.add_argument("-r", "--ranker", type=str, required=True)
    parser.add_argument("-e", "--eval_type", type=str, required=True)
    parser.add_argument("--eval_dataset", type=str, default="RecentPDB")
    parser.add_argument("--subset", type=str, default="All")
    parser.add_argument("--xlabel", type=str, default=None)
    parser.add_argument("--ylabel", type=str, default=None)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--ylim", type=float, nargs=2, default=None)
    parser.add_argument("--figsize", type=float, nargs=2, default=(8, 4))
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    plot_bar_from_csv(
        csv_path=args.csv_path,
        output_path=args.output_path,
        name_col=args.name_col,
        value_col=args.value_col,
        eval_type=args.eval_type,
        eval_dataset=args.eval_dataset,
        subset=args.subset,
        ranker=args.ranker,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        title=args.title,
        ylim=args.ylim,
        figsize=args.figsize,
        dpi=args.dpi,
    )
