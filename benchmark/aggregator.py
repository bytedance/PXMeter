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
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from joblib import delayed, Parallel
from tqdm import tqdm

from benchmark.utils import add_comp_chain_iface_id, shrink_dataframe

COMPLEX_METRICS = ["lddt", "bb_lddt"]
CHAIN_METRICS = [
    "lddt",
    "bb_lddt",
    "ref_pocket_chain",
    "lig_rmsd",
    "pocket_rmsd",
    "lig_rmsd_wo_refl",  # legacy
    "pocket_rmsd_wo_refl",  # legacy
]
INTERFACE_METRICS = ["lddt", "bb_lddt", "dockq"]


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
        try:
            with open(json_f) as f:
                content = json.load(f)
        except json.JSONDecodeError:
            logging.error("Error decoding JSON from %s", json_f)
            return {}

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
            # Add ref_pocket_entity
            metrics_df["ref_pocket_entity"] = metrics_df.apply(
                lambda row: (
                    self.ref_chain_id_to_entity_id[row["ref_pocket_chain"]]
                    if pd.notna(row["ref_pocket_chain"])
                    else None
                ),
                axis=1,
            )

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

            for ranker, interface_id_to_score in confidences.get(
                "interface", {}
            ).items():
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

            if not confidence_json.exists():
                # Skip if confidence json is not found
                continue

            json_to_df = ResultJsonToDataFrame(sample_json, confidence_json)
            metrics_df = json_to_df.get_summary_dataframe()
            if metrics_df.empty:
                continue
            metrics_df["seed"] = seed
            metrics_df["sample"] = sample
            all_metrics_df_list.append(metrics_df)

            pb_valid_df = json_to_df.get_pb_valid_dataframe()
            if pb_valid_df is None or pb_valid_df.empty:
                continue
            pb_valid_df.dropna(axis=1, how="all", inplace=True)
            if pb_valid_df.empty:
                continue
            pb_valid_df["seed"] = seed
            pb_valid_df["sample"] = sample
            all_pb_valid_df_list.append(pb_valid_df)

    if len(all_metrics_df_list) == 0:
        logging.warning("No metrics found in %s", pdb_dir)
        return pd.DataFrame(), pd.DataFrame()

    all_metrics_df = pd.concat(all_metrics_df_list)
    if len(all_pb_valid_df_list) == 0:
        all_pb_valid_df = pd.DataFrame()
    else:
        all_pb_valid_df = pd.concat(all_pb_valid_df_list)

    return all_metrics_df, all_pb_valid_df


def run_aggregator(
    eval_result_dir: Path | str,
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
            if not metrics_df.empty:
                all_metrics_df_list.append(metrics_df)
        if not pb_valid_df.empty:
            pb_valid_df.dropna(axis=1, how="all", inplace=True)
            if not pb_valid_df.empty:
                all_pb_valid_df_list.append(pb_valid_df)

    if len(all_metrics_df_list) == 0:
        logging.warning("All metrics DataFrame are empty in %s", eval_result_dir)
        return

    all_metrics_df = pd.concat(all_metrics_df_list)

    # Add columns "2" if there are only chains
    for col_name in ["chain_id_2", "entity_id_2", "entity_type_2"]:
        if col_name not in all_metrics_df.columns:
            all_metrics_df[col_name] = None

    def strfloat_to_strint(x):
        if pd.isna(x):
            return x
        try:
            return str(int(float(x)))
        except ValueError:
            return x

    all_metrics_df["entity_id_1"] = all_metrics_df["entity_id_1"].apply(
        strfloat_to_strint
    )
    all_metrics_df["entity_id_2"] = all_metrics_df["entity_id_2"].apply(
        strfloat_to_strint
    )

    output_parquet = eval_result_dir.parent / f"{eval_result_dir.name}_metrics.parquet"
    all_metrics_df = add_comp_chain_iface_id(all_metrics_df)
    all_metrics_df, _report = shrink_dataframe(all_metrics_df)
    all_metrics_df.to_parquet(
        output_parquet,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )
    logging.info("Output metrics parquet to %s", output_parquet)

    if len(all_pb_valid_df_list) > 0:
        all_pb_valid_df = pd.concat(all_pb_valid_df_list)
        output_pb_valid_parquet = (
            eval_result_dir.parent / f"{eval_result_dir.name}_pb_valid.parquet"
        )
        all_pb_valid_df = add_comp_chain_iface_id(all_pb_valid_df)
        all_pb_valid_df, _report = shrink_dataframe(all_pb_valid_df)
        all_pb_valid_df.to_parquet(
            output_pb_valid_parquet,
            engine="pyarrow",
            compression="zstd",
            index=False,
        )
        logging.info("Output pb valid parquet to %s", output_pb_valid_parquet)


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
        "-n",
        "--num_cpu",
        type=int,
        default=-1,
        help="Number of CPU cores to use for parallel processing.",
    )

    args = parser.parse_args()

    run_aggregator(args.eval_result_dir, num_cpu=args.num_cpu)
