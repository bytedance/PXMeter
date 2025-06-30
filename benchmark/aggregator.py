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
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from pxmeter.constants import POLYMER

COMPLEX_METRICS = ["lddt"]
CHAIN_METRICS = ["lddt", "ref_pocket_chain", "lig_rmsd_wo_refl", "pocket_rmsd_wo_refl"]
INTERFACE_METRICS = ["lddt", "dockq"]


class ResultJsonToDataFrame:
    """
    Convert result json to dataframe.
    """

    def __init__(
        self, metrics_json: Path, confidences_json: Path | None = None
    ) -> None:
        self.metrics_json = metrics_json
        self.confidences_json = confidences_json
        self.metrics = self._read_json(metrics_json)

        (
            self.ref_chain_id_to_entity_id,
            self.ref_chain_id_to_entity_type,
            self.ref_to_model_chain_id,
        ) = self._get_mapping_info()

        self.complex_metrics = COMPLEX_METRICS
        self.chain_metrics = CHAIN_METRICS
        self.interface_metrics = INTERFACE_METRICS

    @staticmethod
    def _read_json(json_f: Path | str) -> dict[str, Any]:
        with open(json_f) as f:
            content = json.load(f)
        return content

    def _get_mapping_info(self) -> tuple[dict[str, str], ...]:
        """
        Get mapping information between reference chain IDs and entity IDs/types,
        as well as the mapping between reference and model chain IDs.

        Returns:
            tuple[dict[str, str]]: A tuple containing three dictionaries:
                - ref_chain_id_to_entity_id: Maps reference chain IDs to entity IDs.
                - ref_chain_id_to_entity_type: Maps reference chain IDs to entity types.
                - ref_to_model_chain_id: Maps reference chain IDs to model chain IDs.
        """
        ref_chain_id_to_entity_id = {}
        ref_chain_id_to_entity_type = {}
        for ref_chain_id, v in self.metrics["ref_chain_info"].items():
            ref_chain_id_to_entity_id[ref_chain_id] = v["label_entity_id"]
            ref_chain_id_to_entity_type[ref_chain_id] = v["entity_type"]

        ref_to_model_chain_id = self.metrics["ref_to_model_chain_mapping"]
        return (
            ref_chain_id_to_entity_id,
            ref_chain_id_to_entity_type,
            ref_to_model_chain_id,
        )

    def _get_complex_dict(self) -> dict[str, Any]:
        complex_dict = {"type": "complex"}
        for k, v in self.metrics["complex"].items():
            if k in self.complex_metrics:
                complex_dict[k] = v
        return complex_dict

    def _get_chain_list(self) -> list[dict[str, Any]]:
        chain_list = []
        for chain_id, metric_dict in self.metrics["chain"].items():
            chain_dict = {
                "type": "chain",
                "chain_id_1": chain_id,
                "entity_id_1": self.ref_chain_id_to_entity_id[chain_id],
                "entity_type_1": self.ref_chain_id_to_entity_type[chain_id],
                "model_chain_id_1": self.ref_to_model_chain_id[chain_id],
            }
            for k, v in metric_dict.items():
                if k in self.chain_metrics:
                    chain_dict[k] = v
            chain_list.append(chain_dict)
        return chain_list

    def _get_interface_list(self) -> list[dict[str, Any]]:
        interface_list = []
        for chain_id, metric_dict in self.metrics["interface"].items():
            chain_id_1, chain_id_2 = chain_id.split(",")
            interface_dict = {"type": "interface"}
            for idx, each_chain_id in enumerate([chain_id_1, chain_id_2]):
                num = idx + 1
                interface_dict[f"chain_id_{num}"] = each_chain_id
                interface_dict[f"entity_id_{num}"] = self.ref_chain_id_to_entity_id[
                    each_chain_id
                ]
                interface_dict[f"entity_type_{num}"] = self.ref_chain_id_to_entity_type[
                    each_chain_id
                ]
                interface_dict[f"model_chain_id_{num}"] = self.ref_to_model_chain_id[
                    each_chain_id
                ]

            for k, v in metric_dict.items():
                if k in self.interface_metrics:
                    interface_dict[k] = v
            interface_list.append(interface_dict)
        return interface_list

    def _get_metrics_df(self) -> pd.DataFrame:
        """
        Get a DataFrame containing the metrics for the current entry (only for one sample).

        This method calls three methods to collect the metrics for the complex, chains, and interfaces,
        and then combines them into a single DataFrame. It also adds the entry ID to the DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the metrics for the current entry.
        """
        complex_dict = self._get_complex_dict()
        chain_list = self._get_chain_list()
        interface_list = self._get_interface_list()
        metrics_df = pd.DataFrame([complex_dict] + chain_list + interface_list)
        metrics_df["entry_id"] = self.metrics["entry_id"]
        return metrics_df

    def _add_rankers_to_metric_df(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds rankers to the metrics DataFrame based on the confidences JSON file.

        Args:
            metrics_df (pd.DataFrame): The DataFrame containing the metrics data of one sample.

        Returns:
            pd.DataFrame: The updated DataFrame with rankers added.
        """
        if self.confidences_json is None:
            # No confidences json
            return metrics_df

        confidences = self._read_json(self.confidences_json)

        for ranker, score in confidences.get("complex", {}).items():
            metrics_df[ranker] = score

        for ranker, chain_id_to_score in confidences.get("chain", {}).items():
            metrics_df[ranker] = metrics_df.apply(
                lambda row, mapping=chain_id_to_score: (
                    mapping[row["model_chain_id_1"]] if row["type"] == "chain" else None
                ),
                axis=1,
            )

        for ranker, interface_id_to_score in confidences.get("interface", {}).items():
            metrics_df[ranker] = metrics_df.apply(
                lambda row, mapping=interface_id_to_score: (
                    mapping[
                        ",".join(
                            sorted([row["model_chain_id_1"], row["model_chain_id_2"]])
                        )
                    ]
                    if row["type"] == "interface"
                    else None
                ),
                axis=1,
            )

        if "ref_pocket_chain" in metrics_df.columns:
            # Add rankers for ligand-pocket interfaces
            def _add_lig_pocket_ranker(row, ranker_key, mapping):
                if row["type"] != "chain" or pd.isna(row["ref_pocket_chain"]):
                    return row[ranker_key]
                else:
                    return mapping[
                        ",".join(
                            sorted(
                                [
                                    self.ref_to_model_chain_id[row["chain_id_1"]],
                                    self.ref_to_model_chain_id[row["ref_pocket_chain"]],
                                ]
                            )
                        )
                    ]

            for ranker, interface_id_to_score in confidences["interface"].items():
                metrics_df[ranker] = metrics_df.apply(
                    lambda row, ranker_key=ranker, mapping=interface_id_to_score: _add_lig_pocket_ranker(
                        row, ranker_key, mapping
                    ),
                    axis=1,
                )

        return metrics_df

    def get_summary_dataframe(self) -> pd.DataFrame:
        """
        Retrieves a summary DataFrame containing metrics and rankers.

        Returns:
            pd.DataFrame: A DataFrame containing metrics and, if applicable, rankers

        """
        metrics_df = self._get_metrics_df()

        # No rankers if confidences json is not provided
        metrics_df = self._add_rankers_to_metric_df(metrics_df)
        return metrics_df

    def get_pb_valid_dataframe(self) -> pd.DataFrame | None:
        """
        Generates a DataFrame containing the pb_valid metrics for each ligand chain.

        Returns:
            pd.DataFrame or None: A DataFrame containing the pb_valid metrics for each ligand chain.
                                    If no pb_valid metrics are found, None is returned.
        """
        # Check if pb_valid metrics are present in the metrics dictionary
        if self.metrics.get("pb_valid") is None:
            return

        pb_valid_list = []
        for lig_chain_id, valid_dict in self.metrics["pb_valid"].items():
            lig_dict = {
                "chain_id_1": lig_chain_id,
                "model_chain_id_1": self.ref_to_model_chain_id[lig_chain_id],
            }
            for k, v in valid_dict.items():
                lig_dict[k] = v
            pb_valid_list.append(lig_dict)
        pb_valid_df = pd.DataFrame(pb_valid_list)
        pb_valid_df["entry_id"] = self.metrics["entry_id"]
        pb_valid_df["type"] = "chain"
        return pb_valid_df


def agg_a_single_dir(pdb_dir: Path | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregates evaluation results from a single directory into a tuple of DataFrames.

    Args:
        pdb_dir (Path or str): The directory containing the evaluation results.

    Returns:
        tuple[pd.DataFrame]: A tuple containing two DataFrames:
            - The first DataFrame contains the summary metrics for each sample.
            - The second DataFrame contains the pb_valid metrics for each sample.
              If no pb_valid metrics are found, an empty DataFrame is returned.
    """
    pdb_dir = Path(pdb_dir)

    if pdb_dir.name == "ERR":
        # Skip error log dir
        return pd.DataFrame(), pd.DataFrame()

    all_metrics_df_list = []
    all_pb_valid_df_list = []
    for seed_dir in pdb_dir.iterdir():
        seed = seed_dir.name
        for sample_json in seed_dir.glob("sample_*_metrics.json"):
            sample = sample_json.stem.split("_")[1]

            confidence_json = Path(str(sample_json).replace("_metrics", "_confidences"))

            json_to_df = ResultJsonToDataFrame(sample_json, confidence_json)
            metrics_df = json_to_df.get_summary_dataframe()
            metrics_df["seed"] = seed
            metrics_df["sample"] = sample
            all_metrics_df_list.append(metrics_df)

            pb_valid_df = json_to_df.get_pb_valid_dataframe()
            if pb_valid_df is None or pb_valid_df.empty:
                continue
            pb_valid_df.dropna(axis=1, how="all", inplace=True)
            pb_valid_df["seed"] = seed
            pb_valid_df["sample"] = sample
            all_pb_valid_df_list.append(pb_valid_df)

    if len(all_metrics_df_list) == 0:
        logging.warning(f"No metrics found in {pdb_dir}")
        return pd.DataFrame(), pd.DataFrame()

    all_metrics_df = pd.concat(all_metrics_df_list)
    if len(all_pb_valid_df_list) == 0:
        all_pb_valid_df = pd.DataFrame()
    else:
        all_pb_valid_df = pd.concat(all_pb_valid_df_list)

    return all_metrics_df, all_pb_valid_df


def add_cluster_id_to_metrics_df(
    cluster_csv: Path | str,
    metrics_df: pd.DataFrame,
    interface_only_use_polymer_cluster: bool = False,
) -> pd.DataFrame:
    """
    Adds cluster IDs to the metrics DataFrame based on the cluster information in the provided CSV file.

    Args:
        cluster_csv (Path or str): The path to the CSV file containing cluster information.
        metrics_df (pd.DataFrame): The DataFrame containing the metrics data.
        interface_only_use_polymer_cluster (bool, optional): Whether to only use polymer
                                           cluster for interface evaluation. Defaults to False.

    Returns:
        pd.DataFrame: The updated DataFrame with cluster IDs added.
    """
    # Convert the cluster_csv to a Path object
    cluster_csv = Path(cluster_csv)
    cluster_df = pd.read_csv(cluster_csv, dtype=str)

    # Drop rows with NaN values in the "cluster_id" column
    cluster_df.dropna(subset=["cluster_id"], inplace=True, how="all", axis=0)
    cluster_key = cluster_df["entry_id"] + "_" + cluster_df["label_entity_id"]
    entry_entity_to_cluster_id = dict(zip(cluster_key, cluster_df["cluster_id"]))

    def gen_cluster_id(row) -> dict[str, str | None]:
        """
        Generates cluster IDs for a given row in the metrics DataFrame.

        Args:
            row (pd.Series): A row from the metrics DataFrame.

        Returns:
            dict[str, str | None]: A dictionary containing cluster IDs for the row.
        """
        cluster = {}
        cluster_id_1 = entry_entity_to_cluster_id.get(
            row["entry_id"] + "_" + str(row["entity_id_1"])
        )
        cluster_id_2 = entry_entity_to_cluster_id.get(
            row["entry_id"] + "_" + str(row["entity_id_2"])
        )

        if row["type"] == "complex":
            cluster["cluster_id_1"] = None
            cluster["cluster_id_2"] = None
            cluster["cluster_id"] = None

        elif row["type"] == "chain":
            cluster["cluster_id_1"] = cluster_id_1
            cluster["cluster_id_2"] = None
            cluster["cluster_id"] = cluster_id_1

        elif row["type"] == "interface":
            cluster["cluster_id_1"] = cluster_id_1
            cluster["cluster_id_2"] = cluster_id_2

            if interface_only_use_polymer_cluster:
                is_polymer_1 = row["entity_type_1"] in POLYMER
                is_polymer_2 = row["entity_type_2"] in POLYMER
                if (is_polymer_1 and is_polymer_2) or (
                    not is_polymer_1 and not is_polymer_2
                ):
                    if cluster_id_1 and cluster_id_2:
                        cluster["cluster_id"] = ":".join(
                            sorted([cluster_id_1, cluster_id_2])
                        )
                    else:
                        cluster["cluster_id"] = None

                elif is_polymer_1:
                    cluster["cluster_id"] = cluster_id_1
                elif is_polymer_2:
                    cluster["cluster_id"] = cluster_id_2
                else:
                    cluster["cluster_id"] = None
            else:
                if cluster_id_1 and cluster_id_2:
                    # If both cluster IDs are present, join them with a colon
                    cluster["cluster_id"] = ":".join(
                        sorted([cluster_id_1, cluster_id_2])
                    )
                else:
                    cluster["cluster_id"] = None
        return cluster

    metrics_df[["cluster_id_1", "cluster_id_2", "cluster_id"]] = metrics_df.apply(
        gen_cluster_id, axis=1, result_type="expand"
    )[["cluster_id_1", "cluster_id_2", "cluster_id"]]
    return metrics_df


def run_aggregator(
    eval_result_dir: Path | str,
    cluster_csv: Path | str | None = None,
    interface_only_use_polymer_cluster: bool = False,
    num_cpu: int = -1,
):
    """
    Aggregates evaluation results from multiple directories into a single DataFrame.

    Save the results into two separate CSV files:
        - *_metrics.csv: eval_result_dir.parent / f"{eval_result_dir.name}_metrics.csv"
        - *_pb_valid.csv (optional): eval_result_dir.parent / f"{eval_result_dir.name}_pb_valid.csv"

    Args:
        eval_result_dir (Path or str): The directory containing the evaluation results.
                        For example: eval_result_dir/[pdb_id]/[seed]/*.json
        cluster_csv (Path or str, optional): The csv file containing cluster information.
                    There are 3 columns in csv:
                    "entry_id", "label_entity_id", "cluster_id"
                    Defaults to None.
        interface_only_use_polymer_cluster (bool, optional): Whether to only use polymer
                                           cluster for interface evaluation. Defaults to False.
        num_cpu (int, optional): The number of CPU cores to use for parallel processing. Defaults to -1.
    """
    eval_result_dir = Path(eval_result_dir)
    all_pdb_dirs = list(eval_result_dir.iterdir())

    results = [
        r
        for r in (
            tqdm(
                Parallel(n_jobs=num_cpu, return_as="generator_unordered")(
                    delayed(agg_a_single_dir)(
                        pdb_dir,
                    )
                    for pdb_dir in all_pdb_dirs
                ),
                total=len(all_pdb_dirs),
                desc="Aggregating results",
            )
        )
    ]

    all_metrics_df_list = []
    all_pb_valid_df_list = []
    for metrics_df, pb_valid_df in results:
        if not metrics_df.empty:
            metrics_df.dropna(axis=1, how="all", inplace=True)
            all_metrics_df_list.append(metrics_df)
        if not pb_valid_df.empty:
            pb_valid_df.dropna(axis=1, how="all", inplace=True)
            all_pb_valid_df_list.append(pb_valid_df)

    if len(all_metrics_df_list) == 0:
        logging.warning("All metrics DataFrame are empty in %s", eval_result_dir)
        return

    all_metrics_df = pd.concat(all_metrics_df_list)

    if cluster_csv:
        all_metrics_df = add_cluster_id_to_metrics_df(
            cluster_csv, all_metrics_df, interface_only_use_polymer_cluster
        )

    output_metrics_csv = eval_result_dir.parent / f"{eval_result_dir.name}_metrics.csv"
    all_metrics_df.to_csv(output_metrics_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)
    logging.info("Output metrics csv to %s", output_metrics_csv)

    if len(all_pb_valid_df_list) > 0:
        all_pb_valid_df = pd.concat(all_pb_valid_df_list)
        output_pb_valid_csv = (
            eval_result_dir.parent / f"{eval_result_dir.name}_pb_valid.csv"
        )
        all_pb_valid_df.to_csv(
            output_pb_valid_csv, index=False, quoting=csv.QUOTE_NONNUMERIC
        )
        logging.info("Output pb valid csv to %s", output_pb_valid_csv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--eval_result_dir",
        type=str,
        required=True,
        help="Path to the evaluation result directory.",
    )
    parser.add_argument(
        "-c",
        "--cluster_csv",
        type=str,
        default=None,
        help="Path to the cluster csv file.",
    )
    parser.add_argument(
        "-n",
        "--num_cpu",
        type=int,
        default=-1,
        help="Number of CPU cores to use for parallel processing.",
    )

    args = parser.parse_args()

    run_aggregator(
        args.eval_result_dir, cluster_csv=args.cluster_csv, num_cpu=args.num_cpu
    )
