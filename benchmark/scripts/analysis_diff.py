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
import csv
import logging
from pathlib import Path

import pandas as pd


def get_diff_for_two_csvs(
    details_csv: Path,
    metric_col: str,
    name1: str,
    name2: str,
    output_csv: Path | None = None,
    rankers: list[str] = None,
    eval_dataset: list[str] = None,
    eval_type: list[str] = None,
):
    """
    Calculate the difference between two subsets of data from a CSV file based on specified names.

    Args:
        details_csv (Path): Path to the input CSV file containing detailed data.
        output_csv (Path): Path to the output CSV file where the results will be saved.
        metric_col (str): Name of the column containing the metric to calculate the difference.
        name1 (str): Name used to filter the first subset of data.
        name2 (str): Name used to filter the second subset of data.
        rankers (list[str], optional): List of rankers to filter the data. Defaults to None.
        eval_dataset (list[str], optional): List of evaluation datasets to filter the data. Defaults to None.
        eval_type (list[str], optional): List of evaluation types to filter the data. Defaults to None.
    """
    details_df = pd.read_csv(
        details_csv,
        dtype={
            "entry_id": str,
            "chain_id_1": str,
            "chain_id_2": str,
            "cluster_id": str,
        },
    )
    if rankers:
        details_df = details_df[details_df["ranker"].isin(rankers)]
    if eval_dataset:
        details_df = details_df[details_df["eval_dataset"].isin(eval_dataset)]
    if eval_type:
        details_df = details_df[details_df["eval_type"].isin(eval_type)]

    details_df[metric_col] = details_df[metric_col].astype(float)

    df1 = details_df[details_df["name"] == name1].copy().drop(columns=["name"])
    df2 = details_df[details_df["name"] == name2].copy().drop(columns=["name"])
    df1.rename(
        columns={
            metric_col: f"{name1}_{metric_col}",
            "seed": f"{name1}_seed",
            "sample": f"{name1}_sample",
        },
        inplace=True,
    )
    df2.rename(
        columns={
            metric_col: f"{name2}_{metric_col}",
            "seed": f"{name2}_seed",
            "sample": f"{name2}_sample",
        },
        inplace=True,
    )

    df = pd.merge(
        df1,
        df2,
        on=[
            "eval_dataset",
            "ranker",
            "eval_type",
            "entry_id",
            "chain_id_1",
            "chain_id_2",
            "cluster_id",
        ],
        how="inner",
    )

    df[f"{metric_col}_diff"] = df[f"{name1}_{metric_col}"] - df[f"{name2}_{metric_col}"]
    df = df.round(4)

    if output_csv is None:
        ranker_str = "_" + "-".join(rankers) if rankers else ""
        eval_dataset_str = "_" + "-".join(eval_dataset) if eval_dataset else ""
        output_csv = (
            f"{name1}_vs_{name2}{eval_dataset_str}{ranker_str}_{metric_col}.csv"
        )
    df.sort_values(by=f"{metric_col}_diff", ascending=False, inplace=True)
    df.to_csv(output_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)
    logging.info("Save diff csv to %s", output_csv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--details_csv", type=Path)
    parser.add_argument("-m", "--metric_col", type=str)
    parser.add_argument("--name1", type=str)
    parser.add_argument("--name2", type=str)
    parser.add_argument("-o", "--output_csv", type=Path, default=None)
    parser.add_argument("-r", "--rankers", type=str, default=None)
    parser.add_argument("--eval_dataset", type=str, default=None)
    parser.add_argument("--eval_type", type=str, default=None)

    args = parser.parse_args()

    intput_rankers = args.rankers.split(",") if args.rankers else None
    intput_eval_dataset = args.eval_dataset.split(",") if args.eval_dataset else None
    intput_eval_type = args.eval_type.split(",") if args.eval_type else None

    get_diff_for_two_csvs(
        details_csv=args.details_csv,
        metric_col=args.metric_col,
        name1=args.name1,
        name2=args.name2,
        output_csv=args.output_csv,
        rankers=intput_rankers,
        eval_dataset=intput_eval_dataset,
        eval_type=intput_eval_type,
    )
