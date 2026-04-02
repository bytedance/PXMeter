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
    "cdr_h3_bb_rmsd",
    "lddt_pli",
]
INTERFACE_METRICS = ["lddt", "bb_lddt", "dockq"]


class ResultJsonToDataFrame:
    """
    Convert result json to dictionary lists for aggregation.

    Args:
        metrics_json (Path): Path to the metrics JSON file.
        confidences_json (Path | None): Path to the confidences JSON file. Defaults to None.
    """

    def __init__(
        self, metrics_json: Path, confidences_json: Path | None = None
    ) -> None:
        self.metrics_json = metrics_json
        self.confidences_json = confidences_json
        self.metrics = self._read_json(metrics_json)

        if not self.metrics:
            self.valid = False
            return
        else:
            self.valid = True

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
        """
        Read JSON file content.

        Args:
            json_f (Path | str): Path to the JSON file.

        Returns:
            dict[str, Any]: Parsed JSON content, or empty dict if decoding fails.
        """
        try:
            with open(json_f) as f:
                content = json.load(f)
        except json.JSONDecodeError:
            logging.error("Error decoding JSON from %s", json_f)
            return {}

        return content

    def _get_mapping_info(
        self,
    ) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
        """
        Get mapping information between reference chain IDs and entity IDs/types,
        as well as the mapping between reference and model chain IDs.

        Returns:
            tuple[dict[str, str], dict[str, str], dict[str, str]]: A tuple containing three dictionaries:
                - ref_chain_id_to_entity_id: Maps reference chain IDs to entity IDs.
                - ref_chain_id_to_entity_type: Maps reference chain IDs to entity types.
                - ref_to_model_chain_id: Maps reference chain IDs to model chain IDs.
        """
        ref_chain_id_to_entity_id = {}
        ref_chain_id_to_entity_type = {}
        for ref_chain_id, v in self.metrics.get("ref_chain_info", {}).items():
            ref_chain_id_to_entity_id[ref_chain_id] = v["label_entity_id"]
            ref_chain_id_to_entity_type[ref_chain_id] = v["entity_type"]

        ref_to_model_chain_id = self.metrics.get("ref_to_model_chain_mapping", {})
        return (
            ref_chain_id_to_entity_id,
            ref_chain_id_to_entity_type,
            ref_to_model_chain_id,
        )

    def _get_complex_dict(self) -> dict[str, Any]:
        """
        Extract complex level metrics into a dictionary.

        Returns:
            dict[str, Any]: Dictionary containing complex level metrics.
        """
        complex_dict = {"type": "complex"}
        for k, v in self.metrics.get("complex", {}).items():
            if k in self.complex_metrics:
                complex_dict[k] = v
        return complex_dict

    def _get_chain_list(self) -> list[dict[str, Any]]:
        """
        Extract chain level metrics into a list of dictionaries.

        Returns:
            list[dict[str, Any]]: List of dictionaries, each containing chain level metrics.
        """
        chain_list = []
        for chain_id, metric_dict in self.metrics.get("chain", {}).items():
            chain_dict = {
                "type": "chain",
                "chain_id_1": chain_id,
                "entity_id_1": self.ref_chain_id_to_entity_id.get(chain_id),
                "entity_type_1": self.ref_chain_id_to_entity_type.get(chain_id),
                "model_chain_id_1": self.ref_to_model_chain_id.get(chain_id),
            }
            for k, v in metric_dict.items():
                if k in self.chain_metrics:
                    chain_dict[k] = v
            chain_list.append(chain_dict)
        return chain_list

    def _get_interface_list(self) -> list[dict[str, Any]]:
        """
        Extract interface level metrics into a list of dictionaries.

        Returns:
            list[dict[str, Any]]: List of dictionaries, each containing interface level metrics.
        """
        interface_list = []
        for chain_id, metric_dict in self.metrics.get("interface", {}).items():
            chain_id_1, chain_id_2 = chain_id.split(",")
            interface_dict = {"type": "interface"}
            for idx, each_chain_id in enumerate([chain_id_1, chain_id_2]):
                num = idx + 1
                interface_dict[f"chain_id_{num}"] = each_chain_id
                interface_dict[f"entity_id_{num}"] = self.ref_chain_id_to_entity_id.get(
                    each_chain_id
                )
                interface_dict[
                    f"entity_type_{num}"
                ] = self.ref_chain_id_to_entity_type.get(each_chain_id)
                interface_dict[
                    f"model_chain_id_{num}"
                ] = self.ref_to_model_chain_id.get(each_chain_id)

            for k, v in metric_dict.items():
                if k in self.interface_metrics:
                    interface_dict[k] = v
            interface_list.append(interface_dict)
        return interface_list

    def _add_rankers_to_dicts(
        self, dicts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Add ranker scores to the metric dictionaries.

        Args:
            dicts (list[dict[str, Any]]): List of metric dictionaries.

        Returns:
            list[dict[str, Any]]: Updated list of metric dictionaries with ranker scores added.
        """
        if self.confidences_json is None:
            return dicts

        confidences = self._read_json(self.confidences_json)

        # Get complex rankers once to broadcast
        complex_rankers = confidences.get("complex", {})

        for d in dicts:
            if d["type"] == "complex":
                for ranker, score in complex_rankers.items():
                    d[ranker] = score

            elif d["type"] == "chain":
                # broadcast complex rankers first
                for ranker, score in complex_rankers.items():
                    if ranker not in d:
                        d[ranker] = score

                for ranker, mapping in confidences.get("chain", {}).items():
                    model_chain_id = d.get("model_chain_id_1")
                    if model_chain_id is not None and model_chain_id in mapping:
                        d[ranker] = mapping.get(model_chain_id)

                ref_pocket = d.get("ref_pocket_chain")
                if ref_pocket is not None and pd.notna(ref_pocket):
                    d["ref_pocket_entity"] = self.ref_chain_id_to_entity_id.get(
                        ref_pocket
                    )

                    for ranker, mapping in confidences.get("interface", {}).items():
                        c1 = self.ref_to_model_chain_id.get(d.get("chain_id_1", ""))
                        c2 = self.ref_to_model_chain_id.get(ref_pocket)
                        if c1 is not None and c2 is not None:
                            key = ",".join(sorted([c1, c2]))
                            if key in mapping:
                                d[ranker] = mapping[key]

            elif d["type"] == "interface":
                # broadcast complex rankers first
                for ranker, score in complex_rankers.items():
                    if ranker not in d:
                        d[ranker] = score

                for ranker, mapping in confidences.get("interface", {}).items():
                    c1 = d.get("model_chain_id_1")
                    c2 = d.get("model_chain_id_2")
                    if c1 is not None and c2 is not None:
                        key = ",".join(sorted([c1, c2]))
                        if key in mapping:
                            d[ranker] = mapping[key]

        return dicts

    def get_summary_dicts(self) -> list[dict[str, Any]]:
        """
        Get a list of dictionaries containing metrics and rankers for the current entry.

        This method collects metrics for complex, chains, and interfaces,
        and adds rankers if confidence information is available.

        Returns:
            list[dict[str, Any]]: List of dictionaries containing metrics and rankers.
        """
        if not self.valid:
            return []

        complex_dict = self._get_complex_dict()
        chain_list = self._get_chain_list()
        interface_list = self._get_interface_list()

        dicts = [complex_dict] + chain_list + interface_list
        entry_id = self.metrics.get("entry_id")
        for d in dicts:
            d["entry_id"] = entry_id

        return self._add_rankers_to_dicts(dicts)

    def get_pb_valid_dicts(self) -> list[dict[str, Any]]:
        """
        Get a list of dictionaries containing pb_valid metrics for each ligand chain.

        Returns:
            list[dict[str, Any]]: List of dictionaries containing pb_valid metrics.
        """
        if not self.valid or self.metrics.get("pb_valid") is None:
            return []

        pb_valid_list = []
        entry_id = self.metrics.get("entry_id")
        for lig_chain_id, valid_dict in self.metrics["pb_valid"].items():
            lig_dict = {
                "chain_id_1": lig_chain_id,
                "model_chain_id_1": self.ref_to_model_chain_id.get(lig_chain_id),
                "entry_id": entry_id,
                "type": "chain",
            }
            lig_dict.update(valid_dict)
            pb_valid_list.append(lig_dict)

        return pb_valid_list


def discover_seed_dirs(pdb_dir: Path | str) -> list[tuple[Path, str]]:
    """
    Discover all seed directories within a given PDB directory.

    Args:
        pdb_dir (Path | str): The path to the PDB directory.

    Returns:
        list[tuple[Path, str]]: A list of tuples, where each tuple contains
        the path to a seed directory and the seed name (directory name).
    """
    pdb_dir = Path(pdb_dir)
    tasks = []
    if pdb_dir.name == "ERR" or not pdb_dir.is_dir():
        return tasks

    # Just list directories, no globbing here
    for seed_dir in pdb_dir.iterdir():
        if seed_dir.is_dir():
            tasks.append((seed_dir, seed_dir.name))
    return tasks


def process_seed_task(
    task: tuple[Path, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Process a single seed directory to extract metrics and pb_valid data.

    This function iterates through all sample JSON files in the seed directory,
    parses them using ResultJsonToDataFrame, and collects the metrics and
    pb_valid data into lists of dictionaries.

    Args:
        task (tuple[Path, str]): A tuple containing the seed directory path and the seed name.

    Returns:
        tuple[list[dict[str, Any]], list[dict[str, Any]]]: A tuple containing two lists:
            - metrics_dicts_all: A list of dictionaries containing the summary metrics.
            - pb_valid_dicts_all: A list of dictionaries containing the pb_valid metrics.
    """
    seed_dir, seed = task
    metrics_dicts_all = []
    pb_valid_dicts_all = []

    # Globbing happens here in the worker process
    for sample_json in seed_dir.glob("sample_*_metrics.json"):
        sample = sample_json.stem.split("_")[1]
        confidence_json = sample_json.with_name(
            str(sample_json.name).replace("_metrics", "_confidences")
        )

        if not confidence_json.exists():
            continue

        json_to_df = ResultJsonToDataFrame(sample_json, confidence_json)

        m_dicts = json_to_df.get_summary_dicts()
        for d in m_dicts:
            d["seed"] = seed
            d["sample"] = sample
        metrics_dicts_all.extend(m_dicts)

        p_dicts = json_to_df.get_pb_valid_dicts()
        for d in p_dicts:
            d["seed"] = seed
            d["sample"] = sample
        pb_valid_dicts_all.extend(p_dicts)

    return metrics_dicts_all, pb_valid_dicts_all


def run_aggregator(
    eval_result_dir: Path | str,
    num_cpu: int = -1,
):
    """
    Aggregates evaluation results from multiple directories into parquet files.

    Save the results into two separate parquet files:
        - *_metrics.parquet: eval_result_dir.parent / f"{eval_result_dir.name}_metrics.parquet"
        - *_pb_valid.parquet (optional): eval_result_dir.parent / f"{eval_result_dir.name}_pb_valid.parquet"

    Args:
        eval_result_dir (Path | str): The directory containing the evaluation results.
                        For example: eval_result_dir/[pdb_id]/[seed]/*.json
        num_cpu (int, optional): The number of CPU cores to use for parallel processing. Defaults to -1.
    """
    eval_result_dir = Path(eval_result_dir)
    all_pdb_dirs = [p for p in eval_result_dir.iterdir() if p.is_dir()]

    # Discover all seed directories in parallel
    all_tasks = []
    discovery_results = Parallel(n_jobs=num_cpu)(
        delayed(discover_seed_dirs)(pdb_dir)
        for pdb_dir in tqdm(all_pdb_dirs, desc="Discovering seeds")
    )
    for tasks in discovery_results:
        all_tasks.extend(tasks)

    if not all_tasks:
        logging.warning("No seeds found in %s", eval_result_dir)
        return

    # Process seed tasks in parallel to get lists of dicts
    process_results = Parallel(n_jobs=num_cpu, return_as="generator_unordered")(
        delayed(process_seed_task)(task)
        for task in tqdm(all_tasks, desc="Aggregating results")
    )

    all_metrics_dicts = []
    all_pb_valid_dicts = []
    for metrics_dicts, pb_valid_dicts in process_results:
        all_metrics_dicts.extend(metrics_dicts)
        all_pb_valid_dicts.extend(pb_valid_dicts)

    if not all_metrics_dicts:
        logging.warning("All metrics dictionaries are empty in %s", eval_result_dir)
        return

    # Phase 3: Create DataFrames directly from aggregated lists of dictionaries
    all_metrics_df = pd.DataFrame(all_metrics_dicts)
    all_metrics_df.dropna(axis=1, how="all", inplace=True)

    if all_metrics_df.empty:
        logging.warning("All metrics DataFrame are empty in %s", eval_result_dir)
        return

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

    if all_pb_valid_dicts:
        all_pb_valid_df = pd.DataFrame(all_pb_valid_dicts)
        all_pb_valid_df.dropna(axis=1, how="all", inplace=True)
        if not all_pb_valid_df.empty:
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
