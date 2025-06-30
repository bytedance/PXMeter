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
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

from benchmark.configs.rankers_config import MODEL_TO_RANKER_KEYS
from benchmark.utils import (
    divide_list_into_chunks,
    int_to_letters,
    nested_dict_to_sorted_list,
)
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
        self.ranker = MODEL_TO_RANKER_KEYS.nan
        self.eval_run_config = RUN_CONFIG

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
        filtered_data = []

        if self.chunk_str is not None:
            # chunk_id start from "1"
            chunk_id, chunk_num = self.chunk_str.split("of")
            chunk_id = int(chunk_id)
            chunk_num = int(chunk_num)
            data = divide_list_into_chunks(data, chunk_num)[chunk_id - 1]

        for each_data in data:
            (
                name,
                pdb_id,
                seed,
                sample,
            ) = each_data[:4]
            # Skip if pdb_id not in pdb_ids_list
            if self.pdb_ids_list is not None and pdb_id not in self.pdb_ids_list:
                continue
            # Skip if not overwrite and output already exists
            output_metric_json, output_confidence_json = self._get_output_path(
                name, seed, sample
            )
            if (
                (not self.overwrite)
                and output_metric_json.exists()
                and output_confidence_json.exists()
            ):
                continue
            filtered_data.append(each_data)
        return filtered_data

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
        # List of tuple (name, pdb_id, seed, sample, pred_cif, confidence_json)
        data = []
        for pdb_dir in self.pred_dir.iterdir():
            name = pdb_dir.stem
            pdb_id = name.split("_")[0]

            if (pdb_dir / pdb_id).exists():
                pdb_dir = pdb_dir / pdb_id

            for seed_dir in pdb_dir.iterdir():
                seed = seed_dir.name.replace("seed_", "")

                # e.g. 5sak_summary_confidence_confidences_sample_0.json
                sample_to_confidence_path = {
                    json_f.stem.split("_")[-1]: json_f
                    for json_f in (seed_dir / "predictions").glob("*sample_*.json")
                }

                # e.g. 5sak_sample_0.cif
                for pred_cif in (seed_dir / "predictions").glob("*sample_*.cif"):
                    sample = pred_cif.stem.split("_")[-1]
                    confidence_json = sample_to_confidence_path.get(sample)
                    if confidence_json is None:
                        logging.warning(
                            "Cannot find confidence json for %s, skip.", pred_cif
                        )
                        continue
                    model_chain_id_to_lig_mol = None

                    data.append(
                        (
                            name,
                            pdb_id,
                            seed,
                            sample,
                            pred_cif,
                            confidence_json,
                            model_chain_id_to_lig_mol,
                        )
                    )
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

        with open(output_confidence_json, "w") as f:
            json.dump(ranker_results, f, indent=4)

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
            # For PoseBusters only
            true_cif = self.true_dir / f"{pdb_id}_cropped.cif"

        # Make directory if not exist
        output_metric_json.parent.mkdir(parents=True, exist_ok=True)

        if self.pdb_id_to_lig_label_asym_id:
            interested_lig_label_asym_id = self.pdb_id_to_lig_label_asym_id[pdb_id]
            if isinstance(interested_lig_label_asym_id, str):
                interested_lig_label_asym_id = interested_lig_label_asym_id.split(",")
        else:
            interested_lig_label_asym_id = None

        if self.pdb_id_to_altloc:
            ref_altloc = self.pdb_id_to_altloc[pdb_id]
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

        assert (
            len(data) > 0
        ), f"No CIF files available for evaluation were found in {self.pred_dir}"

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


class ProtenixEvaluator(BaseEvaluator):
    """
    A class for evaluating protein structures using the Protenix model.
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
        super().__init__(
            true_dir,
            pred_dir,
            output_dir,
            num_cpu,
            overwrite,
            ref_assembly_id,
            pdb_id_to_lig_label_asym_id,
            pdb_id_to_altloc,
            pdb_ids_list,
            chunk_str,
        )
        self.ranker = MODEL_TO_RANKER_KEYS.protenix


class BoltzEvaluator(BaseEvaluator):
    """
    A class for evaluating protein structures using the Boltz model.
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
        super().__init__(
            true_dir,
            pred_dir,
            output_dir,
            num_cpu,
            overwrite,
            ref_assembly_id,
            pdb_id_to_lig_label_asym_id,
            pdb_id_to_altloc,
            pdb_ids_list,
            chunk_str,
        )
        self.ranker = MODEL_TO_RANKER_KEYS.boltz

    def load_all_cif_and_confidence(self) -> list[tuple[str]]:
        """
        Derived from BaseEvaluator.

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
        # List of tuple (name, pdb_id, seed, sample, pred_cif, confidence_json)
        data = []
        for pdb_dir in self.pred_dir.iterdir():
            name = pdb_dir.stem
            pdb_id = name.split("_")[0]

            for seed_dir in pdb_dir.iterdir():
                seed = seed_dir.name.replace("seed_", "")

                # e.g. confidence_6st5_model_0.json
                sample_to_confidence_path = {
                    json_f.stem.split("_")[-1]: json_f
                    for json_f in (
                        seed_dir / f"boltz_results_{pdb_id}" / "predictions" / pdb_id
                    ).glob("*model_*.json")
                }

                # e.g. 6st5_model_0.cif
                for pred_cif in (
                    seed_dir / f"boltz_results_{pdb_id}" / "predictions" / pdb_id
                ).glob("*model_*.cif"):
                    sample = pred_cif.stem.split("_")[-1]
                    confidence_json = sample_to_confidence_path.get(sample)
                    if confidence_json is None:
                        logging.warning(
                            "Cannot find confidence json for %s, skip.", pred_cif
                        )
                        continue
                    model_chain_id_to_lig_mol = None

                    data.append(
                        (
                            name,
                            pdb_id,
                            seed,
                            sample,
                            pred_cif,
                            confidence_json,
                            model_chain_id_to_lig_mol,
                        )
                    )
        logging.info("Found %s data from %s", len(data), self.pred_dir)

        data_after_filter = self._filter_data(data)
        logging.info("Found %s data after filtering", len(data_after_filter))
        return data_after_filter


class ChaiEvaluator(BaseEvaluator):
    """
    A class for evaluating protein structures using the Chai model.
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
        input_fasta_dir: Path | str | None = None,
    ):
        super().__init__(
            true_dir,
            pred_dir,
            output_dir,
            num_cpu,
            overwrite,
            ref_assembly_id,
            pdb_id_to_lig_label_asym_id,
            pdb_id_to_altloc,
            pdb_ids_list,
            chunk_str,
        )
        self.input_fasta_dir = Path(input_fasta_dir) if input_fasta_dir else None
        self.ranker = MODEL_TO_RANKER_KEYS.chai

    @staticmethod
    def _get_mols_from_chai_fasta(fasta_file: Path | str) -> dict[str, Chem.Mol]:
        """
        Reads a FASTA file and extracts ligand information to create a
        dictionary of ligand chain IDs to RDKit molecule objects.

        Args:
            fasta_file (str): The path to the FASTA file.

        Returns:
            dict: A dictionary mapping ligand chain IDs to RDKit molecule objects.
        """
        with open(fasta_file, "r") as f:
            lines = f.readlines()

        lig_chain_id_to_mol = {}
        for idx, line in enumerate(lines):
            if line.startswith(">ligand"):
                smi = lines[idx + 1].strip()
                chain_id = int_to_letters(idx // 2 + 1)
                mol = Chem.MolFromSmiles(smi)

                # remove all Hs from mol (e.g. ZRY in 5sak)
                mol = AllChem.RemoveAllHs(mol)

                lig_chain_id_to_mol[chain_id] = mol
        return lig_chain_id_to_mol

    def load_all_cif_and_confidence(self) -> list[tuple[str]]:
        """
        Derived from BaseEvaluator.

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
        # List of tuple (name, pdb_id, seed, sample, pred_cif, confidence_json)
        data = []
        for pdb_dir in self.pred_dir.iterdir():
            name = pdb_dir.stem
            pdb_id = name.split("_")[0]

            for seed_dir in pdb_dir.iterdir():
                seed = seed_dir.name.replace("seed_", "")

                # e.g. scores.model_idx_0.json
                sample_to_confidence_path = {
                    json_f.stem.split("_")[-1]: json_f
                    for json_f in seed_dir.glob("scores.model_idx_*.json")
                }

                # e.g. pred.model_idx_0.cif
                for pred_cif in (seed_dir).glob("pred.model_idx_*.cif"):
                    sample = pred_cif.stem.split("_")[-1]
                    confidence_json = sample_to_confidence_path.get(sample)
                    if confidence_json is None:
                        logging.warning(
                            "Can not find confidence file for %s, skip.", pred_cif
                        )
                        continue

                    if self.input_fasta_dir is None:
                        model_chain_id_to_lig_mol = None
                    else:
                        fasta_file = self.input_fasta_dir / f"{pdb_id}.fasta"
                        if not fasta_file.exists():
                            logging.warning(
                                "Fasta file %s does not exist",
                                fasta_file,
                            )
                            model_chain_id_to_lig_mol = None
                        else:
                            model_chain_id_to_lig_mol = self._get_mols_from_chai_fasta(
                                fasta_file
                            )

                    data.append(
                        (
                            name,
                            pdb_id,
                            seed,
                            sample,
                            pred_cif,
                            confidence_json,
                            model_chain_id_to_lig_mol,
                        )
                    )
        logging.info("Found %s data", len(data))

        data_after_filter = self._filter_data(data)
        logging.info("Found %s data after filtering", len(data_after_filter))
        return data_after_filter


class AF2MultimerEvaluator(BaseEvaluator):
    """
    A class for evaluating protein structures using the AF2-Multimer model.
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
        super().__init__(
            true_dir,
            pred_dir,
            output_dir,
            num_cpu,
            overwrite,
            ref_assembly_id,
            pdb_id_to_lig_label_asym_id,
            pdb_id_to_altloc,
            pdb_ids_list,
            chunk_str,
        )
        self.ranker = MODEL_TO_RANKER_KEYS.af2m

    def load_all_cif_and_confidence(self) -> list[tuple[str]]:
        """
        Derived from BaseEvaluator.

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
        # List of tuple (name, pdb_id, seed, sample, pred_cif, confidence_json)
        data = []
        for pdb_dir in self.pred_dir.iterdir():
            name = pdb_dir.stem
            pdb_id = name.split("_")[0]

            debug_file = pdb_dir / "ranking_debug.json"
            with open(debug_file) as f:
                confidence_dict = json.load(f)

            # e.g. unrelaxed_model_1_multimer_v3_pred_0.cif
            for pred_cif in pdb_dir.glob("unrelaxed_model_*.cif"):
                split_filename = pred_cif.stem.split("_")
                seed, sample = split_filename[2], split_filename[-1]

                model_name = pred_cif.stem.replace("unrelaxed_", "")
                confidence_score = confidence_dict["iptm+ptm"][model_name]
                model_chain_id_to_lig_mol = None

                data.append(
                    (
                        name,
                        pdb_id,
                        seed,
                        sample,
                        pred_cif,
                        confidence_score,  # return confidence score instead of confidence json
                        model_chain_id_to_lig_mol,
                    )
                )
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
        Derived from BaseEvaluator. Only save "iptm+ptm" score.
        The "ori_confidence_json" is "iptm+ptm" score read from the ranking_debug.json file.
        """
        # complex: {key: score}
        ranker_results = {}
        complex_ranker_results = {}
        complex_ranker_results["iptm+ptm"] = ori_confidence_json
        ranker_results["complex"] = complex_ranker_results
        with open(output_confidence_json, "w") as f:
            json.dump(ranker_results, f, indent=4)
