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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark.scripts.analysis_diff import _get_diff_for_sub_df


def plot_diff_distribution(diff: pd.Series, output_fig: Path):
    """
    Create a bar plot of per-sample differences and save to file.

    Args:
        diff (pd.Series): Series of difference values for each sample.
        output_fig (Path): Path to save the generated figure (e.g., .png or .pdf).
    """
    x = np.arange(len(diff))

    plt.figure(figsize=(8, 4))
    plt.bar(x, diff, color="skyblue", edgecolor="black", alpha=0.7)
    plt.axhline(0, color="red", linestyle="--", label="Zero line")

    plt.xlabel("Sample Index")
    plt.ylabel("Difference")
    plt.title("Per-Sample Differences (Bar)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_fig)


def get_all_diff_for_each_type(
    details_csv,
    output_dir: Path,
    name1: str,
    name2: str,
    ranker: str = "best",
    metric_col: str = "lddt",
    subset_include: list[str] | None = None,
    eval_type_include: list[str] | None = None,
):
    """
    Reads a CSV containing evaluation results and computes the difference of `metric_col`
    between `name1` and `name2` within each group defined by
    ['eval_dataset', 'subset', 'eval_type']. Outputs CSVs and bar plots per group.

    Args:
        details_csv (str | Path): Path to the input CSV file containing evaluation details.
        output_dir (Path): Directory where output CSVs and figures will be saved.
        name1 (str): The first entity name to compare.
        name2 (str): The second entity name to compare.
        ranker (str, optional): Ranker column filter. Defaults to "best".
        metric_col (str, optional): Metric column to compute differences on. Defaults to "lddt".
        subset_include (list[str] | None, optional): Only include these subset values.
            If None, all subsets are included. Defaults to None.
        eval_type_include (list[str] | None, optional): Only include these evaluation types.
            If None, all eval_types are included. Defaults to None.
    """
    details_df = pd.read_csv(
        details_csv,
        dtype={
            "name": str,
            "entry_id": str,
            "chain_id_1": str,
            "chain_id_2": str,
            "cluster_id": str,
        },
    )

    if "subset" not in details_df.columns:
        details_df["subset"] = "All"

    details_df = details_df[details_df["ranker"] == ranker]

    group_cols = ["eval_dataset", "subset", "eval_type"]

    for group, sub_df in details_df.groupby(group_cols):
        eval_dataset, subset, eval_type = group
        if (subset_include and subset not in subset_include) or (
            eval_type_include and eval_type not in eval_type_include
        ):
            continue

        sub_df_w_diff = _get_diff_for_sub_df(
            sub_df, name1=name1, name2=name2, metric_col=metric_col
        )

        output_file_name = (
            f"{eval_dataset}_{subset}_{eval_type}_{ranker}_{metric_col}_diff.csv"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_f = output_dir / output_file_name
        sub_df_w_diff.to_csv(output_f, index=False)

        plot_diff_distribution(
            sub_df_w_diff[f"{metric_col}_diff"], output_fig=output_f.with_suffix(".png")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--details_csv", type=Path, required=True)
    parser.add_argument("-o", "--output_dir", type=Path, required=True)
    parser.add_argument("-m", "--metric_col", type=str, required=True)
    parser.add_argument("--name1", type=str, required=True)
    parser.add_argument("--name2", type=str, required=True)
    parser.add_argument("-r", "--ranker", type=str, default="best")

    args = parser.parse_args()

    get_all_diff_for_each_type(
        details_csv=args.details_csv,
        output_dir=args.output_dir,
        name1=args.name1,
        name2=args.name2,
        ranker=args.ranker,
        metric_col=args.metric_col,
    )
