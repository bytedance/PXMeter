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

import json
import logging
import random
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from benchmark.utils import divide_list_into_chunks, nested_dict_to_sorted_list
from pxmeter.configs.run_config import RUN_CONFIG
from pxmeter.eval import evaluate


class BaseEvaluator:
    """
    A class designed to evaluate predicted structures against true structures.

    Args:
        true_dir (Path or str): Directory containing the true structures.
        pred_dir (Path or str): Directory containing the predicted structures.
        output_dir (Path or str): Directory where the evaluation results will be saved.
        num_cpu (int, optional): Number of CPU cores to use for parallel processing. Defaults to 1.
        overwrite (bool, optional): Whether to overwrite existing results. Defaults to False.
        ref_assembly_id (str, optional): Reference assembly ID. Defaults to None.
        pdb_id_to_lig_label_asym_id (dict[str, str or list], optional): Mapping of PDB IDs to ligand label asym IDs.
                                                                                       Defaults to None.
        pdb_id_to_altloc (dict[str, str], optional): Mapping of PDB IDs to alternative locations. Defaults to None.
        pdb_ids_list (list, optional): List of PDB IDs to evaluate. Defaults to None.
        chunk_str (str, optional): Chunk string to identify the evaluation chunk. Defaults to None.
                                   For example, "1of5" means this evaluator is evaluating the first chunk out of 5.
    """

    def __init__(
        self,
        true_dir: Path | str,
        pred_dir: Path | str,
        output_dir: Path | str,
        num_cpu: int = 1,
        overwrite: bool = False,
        ref_assembly_id: str | None = None,
        pdb_id_to_lig_label_asym_id: dict[str, str | list] | None = None,
        pdb_id_to_altloc: dict[str, str] | None = None,
        pdb_ids_list: list | None = None,
        chunk_str: str | None = None,
    ):

        self.true_dir = Path(true_dir)
        self.pred_dir = Path(pred_dir)
        self.output_dir = Path(output_dir)
        self.num_cpu = num_cpu
        self.overwrite = overwrite

        self.ref_assembly_id = ref_assembly_id
        self.pdb_id_to_lig_label_asym_id = pdb_id_to_lig_label_asym_id
        self.pdb_id_to_altloc = pdb_id_to_altloc
        self.pdb_ids_list = pdb_ids_list
        self.chunk_str = chunk_str
        self.ranker = {
            "complex": [],
            "chain": [],
            "interface": [],
        }
        self.eval_run_config = RUN_CONFIG

    def _filter_each_data(self, each_data: list):
        (
            name,
            pdb_id,
            seed,
            sample,
        ) = each_data[:4]

        # Skip if not overwrite and output already exists
        output_metric_json, output_confidence_json = self._get_output_path(
            name, seed, sample
        )
        if (
            (not self.overwrite)
            and output_metric_json.exists()
            and output_confidence_json.exists()
        ):
            return None
        return each_data

    def _filter_data(self, data):
        """
        Filters the input data based on certain criteria.

        There are three conditions:
        1. If chunk_str is not None, the data is divided into chunks
           and the chunk corresponding to chunk_str is selected.
        2. If pdb_ids_list is not None, the data is filtered to include
           only those with pdb_id in pdb_ids_list.
        3. If overwrite is False, the data is filtered to include only those for
           which the output metric and confidence JSON files do not exist.

        Args:
            data (list): A list of tuples containing data to be filtered.

        Returns:
            list: A list of filtered data tuples.
        """

        results = [
            r
            for r in tqdm(
                Parallel(n_jobs=self.num_cpu, return_as="generator_unordered")(
                    delayed(self._filter_each_data)(each_data) for each_data in data
                ),
                total=len(data),
                desc="Filter data",
            )
        ]

        filtered_data = []
        for r in results:
            if r is not None:
                filtered_data.append(r)
        return filtered_data

    def _get_info_from_each_pdb_dir(self, pdb_dir: Path) -> list:
        """
        Extract evaluation task information from a single PDB prediction directory.

        This method must be implemented by subclasses to handle different model output structures.

        Args:
            pdb_dir (Path): The directory containing predictions for a specific PDB.

        Returns:
            list[tuple]: A list of tuples containing (name, pdb_id, seed, sample, pred_cif,
                         confidence_json, model_chain_id_to_lig_mol).
        """
        raise NotImplementedError(
            "_get_info_from_each_pdb_dir must be implemented by subclasses"
        )

    def load_all_cif_and_confidence(self) -> list[tuple[Any]]:
        """
        Load all CIF and confidence JSON files from the prediction directories.

        This method iterates through the prediction directories and collects
        tuples containing the name, pdb_id, seed, sample, path to the predicted
        CIF file, and path to the confidence JSON file.

        Returns:
            list[tuple[str]]: A list of tuples where each tuple contains:
                - name (str): The name of the prediction directory.
                - pdb_id (str): The PDB ID extracted from the directory name.
                - seed (str): The seed value extracted from the seed directory name.
                - sample (str): The sample identifier extracted from the file name.
                - pred_cif (Path): The path to the predicted CIF file.
                - confidence_json (Path): The path to the confidence JSON file.
                - model_chain_id_to_lig_mol (dict[str, Chem.Mol], optional): A dictionary
                                      mapping ligand chain IDs to their corresponding molecules.
        """
        # Skip if pdb_id not in pdb_ids_list
        if self.pdb_ids_list is not None:
            pdb_dir_list = [
                self.pred_dir / pdb_id.strip() for pdb_id in self.pdb_ids_list
            ]
        else:
            pdb_dir_list = list(self.pred_dir.iterdir())

        if self.chunk_str is not None:
            # Shuffle data to prevent OutOfMemory from large structures.
            random.seed(42)
            random.shuffle(pdb_dir_list)

            # chunk_id start from "1"
            chunk_id, chunk_num = self.chunk_str.split("of")
            chunk_id = int(chunk_id)
            chunk_num = int(chunk_num)
            pdb_dir_list = divide_list_into_chunks(pdb_dir_list, chunk_num)[
                chunk_id - 1
            ]

        results = [
            r
            for r in tqdm(
                Parallel(n_jobs=self.num_cpu, return_as="generator_unordered")(
                    delayed(self._get_info_from_each_pdb_dir)(pdb_dir)
                    for pdb_dir in pdb_dir_list
                ),
                total=len(pdb_dir_list),
                desc="Looking for CIF and confidence JSON files",
            )
        ]

        # List of tuple (name, pdb_id, seed, sample, pred_cif, confidence_json)
        data = []
        for sub_data in results:
            data.extend(sub_data)

        logging.info("Found %s data", len(data))

        data_after_filter = self._filter_data(data)
        logging.info("Found %s data after filtering", len(data_after_filter))
        return data_after_filter

    @staticmethod
    def save_mapped_confidence_json(
        rankers: dict[str, list[tuple[str, bool]]],
        ori_confidence_json: str | Path,
        output_confidence_json: str | Path,
        ori_model_chain_ids: list[str],
    ):
        """
        Save mapped confidence scores to a JSON file.

        This function reads the original confidence scores from a JSON file,
        maps the chain IDs to their corresponding scores, and saves
        the mapped confidence scores to a new JSON file.

        Args:
            rankers (dict[str, list[tuple[str, bool]]]): Dictionary containing ranker information.
                                                         {ranker level: [(ranker key, ascending)]}
            ori_confidence_json (str or Path): Path to the original confidence JSON file.
            ori_model_cif (str or Path): Path to the original model CIF file.
            output_confidence_json (str or Path): Path to the output confidence JSON file.
            ori_model_chain_ids (list[str]): A list of original model chain IDs.
        """
        with open(ori_confidence_json, "r") as f:
            ori_confidence = json.load(f)

        ranker_results = {}
        # complex: {key: score}
        complex_ranker_results = {}
        for ranker_key, _ascending in rankers["complex"]:
            if ranker_key not in ori_confidence:
                continue

            # Some model save the complex ranking score as a 1D list
            confidence_for_ranker = ori_confidence[ranker_key]
            complex_score = np.array([confidence_for_ranker]).reshape(-1)[0]
            try:
                complex_score = complex_score.item()
            except Exception:
                pass
            complex_ranker_results[ranker_key] = complex_score
        ranker_results["complex"] = complex_ranker_results

        # chain: {key: {chain_id: score}}
        chain_ranker_results = defaultdict(dict)
        for ranker_key, _ascending in rankers["chain"]:
            if ranker_key not in ori_confidence:
                continue
            confidence_for_ranker = ori_confidence[ranker_key]

            if isinstance(confidence_for_ranker, dict):
                confidence_for_ranker = nested_dict_to_sorted_list(
                    confidence_for_ranker
                )

            chain_score = np.array(confidence_for_ranker).reshape(-1)
            for idx, chain_id in enumerate(ori_model_chain_ids):
                this_chain_score = chain_score[idx]
                try:
                    this_chain_score = this_chain_score.item()
                except Exception:
                    pass
                chain_ranker_results[ranker_key][chain_id] = this_chain_score
        ranker_results["chain"] = chain_ranker_results

        # interface: {key: {f"{chain_id_1},{chain_id_2}": score}}
        interface_ranker_results = defaultdict(dict)
        for ranker_key, _ascending in rankers["interface"]:
            if ranker_key not in ori_confidence:
                continue
            confidence_for_ranker = ori_confidence[ranker_key]

            if isinstance(confidence_for_ranker, dict):
                confidence_for_ranker = nested_dict_to_sorted_list(
                    confidence_for_ranker
                )

            interface_array = np.array(confidence_for_ranker)
            n_chains = interface_array.shape[-1]
            interface_score = interface_array.reshape(n_chains, n_chains)

            for idx_i, chain_id_i in enumerate(ori_model_chain_ids):
                for idx_j, chain_id_j in enumerate(ori_model_chain_ids[idx_i + 1 :]):
                    interface_id = ",".join(sorted((chain_id_i, chain_id_j)))
                    this_interface_score = interface_score[idx_i][idx_i + idx_j + 1]
                    try:
                        this_interface_score = this_interface_score.item()
                    except Exception:
                        pass
                    interface_ranker_results[ranker_key][
                        interface_id
                    ] = this_interface_score
        ranker_results["interface"] = interface_ranker_results

        output_confidence_tmp_json = Path(output_confidence_json).with_suffix(
            ".json.tmp"
        )
        with open(output_confidence_tmp_json, "w") as f:
            json.dump(ranker_results, f, indent=4)
        output_confidence_tmp_json.rename(output_confidence_json)

    def _get_output_path(self, name, seed, sample) -> tuple[Path, Path]:
        """
        Generates the output paths for metric and confidence JSON files.

        Args:
            name (str): The name of the evaluation.
            seed (str): The seed value used for the evaluation.
            sample (int): The sample number.

        Returns:
            tuple[Path, Path]: A tuple containing the paths to the metric JSON file and the confidence JSON file.
        """
        metric_json = self.output_dir / name / seed / f"sample_{sample}_metrics.json"
        confidence_json = (
            self.output_dir / name / seed / f"sample_{sample}_confidences.json"
        )
        return metric_json, confidence_json

    def run_eval_for_one_pdb_dir(self, pdb_dir: Path):
        """
        Run evaluation for a single PDB prediction directory.

        This method extracts the necessary information from the PDB directory
        and calls the `run_eval` method to perform the evaluation.

        Args:
            pdb_dir (Path): The directory containing predictions for a specific PDB.
        """
        tasks = self._get_info_from_each_pdb_dir(pdb_dir)
        for task in tasks:
            self.run_eval(task)

    def run_eval(self, task: tuple[str, ...]):
        """
        Run evaluation for a given task.

        Args:
            task (tuple[str]): A tuple containing the following elements:
                - name (str): The name of the task.
                - pdb_id (str): The PDB ID of the structure.
                - seed (int): The seed value for the evaluation.
                - sample (str): The sample identifier.
                - pred_cif (Path): The path to the predicted CIF file.
                - confidence_json (Path): The path to the confidence JSON file.
                - lig_chain_id_to_mol (dict[str, Chem.Mol]): mapping of ligand chain IDs
        """
        (
            name,
            pdb_id,
            seed,
            sample,
            pred_cif,
            confidence_json,
            lig_chain_id_to_mol,
        ) = task

        true_cif = self.true_dir / f"{pdb_id}.cif"
        output_metric_json, output_confidence_json = self._get_output_path(
            name, seed, sample
        )

        if self.pdb_id_to_lig_label_asym_id and pdb_id == "8f4j":
            # For PoseBusters of PXM-Legacy only
            true_cif = self.true_dir / f"{pdb_id}_cropped.cif"

        if self.pdb_id_to_lig_label_asym_id:
            interested_lig_label_asym_id = self.pdb_id_to_lig_label_asym_id.get(pdb_id)
            if isinstance(interested_lig_label_asym_id, str):
                interested_lig_label_asym_id = interested_lig_label_asym_id.split(",")
        else:
            interested_lig_label_asym_id = None

        if self.pdb_id_to_altloc:
            ref_altloc = self.pdb_id_to_altloc.get(pdb_id, "first")
        else:
            ref_altloc = "first"

        try:
            metric_result = evaluate(
                ref_cif=true_cif,
                model_cif=pred_cif,
                ref_assembly_id=self.ref_assembly_id,
                ref_altloc=ref_altloc,
                model_chain_id_to_lig_mol=lig_chain_id_to_mol,
                interested_lig_label_asym_id=interested_lig_label_asym_id,
                run_config=self.eval_run_config,
            )

            output_metric_json.parent.mkdir(parents=True, exist_ok=True)

            metric_result.to_json(json_file=output_metric_json)

            self.save_mapped_confidence_json(
                rankers=self.ranker,
                ori_confidence_json=confidence_json,
                output_confidence_json=output_confidence_json,
                ori_model_chain_ids=metric_result.ori_model_chain_ids,
            )
        except Exception:
            logging.error("Error evaluating %s, Error CIF: %s", pdb_id, pred_cif)
            error_info = f"ref:{true_cif}\nmodel:{pred_cif}\n{traceback.format_exc()}"
            logging.error(error_info)

            output_err_log = (
                self.output_dir
                / "ERR"
                / name
                / f"error_seed_{seed}_sample_{sample}.log"
            )
            output_err_log.parent.mkdir(parents=True, exist_ok=True)
            with open(output_err_log, "w") as f:
                f.write(error_info)

    def run_eval_batch(self):
        """
        Executes evaluation on a batch of data.

        This method performs the following steps:
        1. Loads all CIF and confidence JSON files.
        2. Shuffles the data to prevent OutOfMemory errors from large structures.
        3. Runs the evaluation in parallel using multiple CPU cores.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # List of tuple (name, pdb_id, seed, sample, pred_cif, confidence_json)
        data = self.load_all_cif_and_confidence()

        if len(data) == 0:
            logging.info(
                "No CIF files available for evaluation were found in %s, exit",
                self.pred_dir,
            )
            return

        # Shuffle data to prevent OutOfMemory from large structures.
        random.seed(42)
        random.shuffle(data)

        if self.num_cpu > 1:
            _results = [
                r
                for r in tqdm(
                    Parallel(n_jobs=self.num_cpu, return_as="generator_unordered")(
                        delayed(self.run_eval)(task) for task in data
                    ),
                    total=len(data),
                    desc="Evaluating",
                )
            ]
        else:
            for task in tqdm(data, total=len(data), desc="Evaluating"):
                self.run_eval(task)
