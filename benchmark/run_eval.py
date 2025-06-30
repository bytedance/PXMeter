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

import pandas as pd

from benchmark.configs.data_config import PXM_EVAL_DATA_ROOT_PATH, SUPPORTED_DATA
from benchmark.evaluators import (
    AF2MultimerEvaluator,
    BaseEvaluator,
    BoltzEvaluator,
    ChaiEvaluator,
    ProtenixEvaluator,
)


def get_pb_lig_info() -> tuple[dict[str, str], dict[str, str]]:
    """
    Reads a CSV file containing PDB information and returns two dictionaries:
    one mapping PDB IDs to ligand asym IDs and another mapping PDB IDs to altloc IDs.

    Returns:
        tuple: A tuple containing:
            - dict: A dictionary mapping PDB IDs to ligand asym IDs.
            - dict: A dictionary mapping PDB IDs to altloc IDs.
    """
    df = pd.read_csv(SUPPORTED_DATA.pb_info_csv)

    # Use lower case PDB ID
    pdb_ids = df["pdb_id"].str.lower().tolist()
    altlocs = df["altloc_id"].tolist()
    lig_asym_ids = df["pb_select_asym_id"].tolist()

    pdb_id_to_lig_label_asym_id = dict(zip(pdb_ids, lig_asym_ids))
    pdb_id_to_altloc = dict(zip(pdb_ids, altlocs))
    return pdb_id_to_lig_label_asym_id, pdb_id_to_altloc


def run_batch_eval(
    eval_dir: Path | str,
    model: str,
    dataset: str,
    output_dir: Path | str,
    num_cpu: int = 1,
    pdb_ids_list: list[str] | None = None,
    chunk_str: str | None = None,
):
    """
    Run batch evaluation for a given model and dataset.

    Args:
        eval_dir (Path or str): Directory containing the evaluation data.
        model (str): Name of the model to evaluate.
        dataset (str): Name of the dataset to evaluate on.
        output_dir (Path or str]): Directory to save the evaluation results.
        num_cpu (int, optional): Number of CPUs to use for parallel processing. Defaults to 1.
        pdb_ids_list (list[str], optional): List of PDB IDs to evaluate. Defaults to None.
        chunk_str (str, optional): Chunk string for processing. Defaults to None.
    """
    logging.info("Run batch eval for %s on %s", model, dataset)

    if dataset == "PoseBusters":
        true_dir = SUPPORTED_DATA.pb_true_dir
        pdb_id_to_lig_label_asym_id, pdb_id_to_altloc = get_pb_lig_info()
    else:
        true_dir = SUPPORTED_DATA.true_dir
        pdb_id_to_lig_label_asym_id, pdb_id_to_altloc = None, None

    if dataset in [
        "RecentPDB",
        "RecentPDB-NA",
        "dsDNA-Protein",
        "RNA-Protein",
        "AF3-AB",
    ]:
        ref_assembly_id = "1"
    else:
        ref_assembly_id = None

    if model == "protenix":
        evaluator = ProtenixEvaluator(
            true_dir=true_dir,
            pred_dir=eval_dir,
            output_dir=output_dir,
            num_cpu=num_cpu,
            overwrite=False,
            ref_assembly_id=ref_assembly_id,
            pdb_id_to_lig_label_asym_id=pdb_id_to_lig_label_asym_id,
            pdb_id_to_altloc=pdb_id_to_altloc,
            pdb_ids_list=pdb_ids_list,
            chunk_str=chunk_str,
        )
    elif model == "chai":
        # Directory containing input fasta
        input_fasta_dir = (
            Path(PXM_EVAL_DATA_ROOT_PATH)
            / "input"
            / "chai-1"
            / dataset.replace("_", "-")
        )
        if not input_fasta_dir.exists():
            input_fasta_dir = None
        else:
            logging.info("Using input fasta dir for Chai: %s", input_fasta_dir)

        evaluator = ChaiEvaluator(
            true_dir=true_dir,
            pred_dir=eval_dir,
            output_dir=output_dir,
            num_cpu=num_cpu,
            overwrite=False,
            ref_assembly_id=ref_assembly_id,
            pdb_id_to_lig_label_asym_id=pdb_id_to_lig_label_asym_id,
            pdb_id_to_altloc=pdb_id_to_altloc,
            pdb_ids_list=pdb_ids_list,
            chunk_str=chunk_str,
            input_fasta_dir=input_fasta_dir,
        )

    elif model == "boltz":
        evaluator = BoltzEvaluator(
            true_dir=true_dir,
            pred_dir=eval_dir,
            output_dir=output_dir,
            num_cpu=num_cpu,
            overwrite=False,
            ref_assembly_id=ref_assembly_id,
            pdb_id_to_lig_label_asym_id=pdb_id_to_lig_label_asym_id,
            pdb_id_to_altloc=pdb_id_to_altloc,
            pdb_ids_list=pdb_ids_list,
            chunk_str=chunk_str,
        )

    elif model == "af2m":
        evaluator = AF2MultimerEvaluator(
            true_dir=true_dir,
            pred_dir=eval_dir,
            output_dir=output_dir,
            num_cpu=num_cpu,
            overwrite=False,
            ref_assembly_id=ref_assembly_id,
            pdb_id_to_lig_label_asym_id=pdb_id_to_lig_label_asym_id,
            pdb_id_to_altloc=pdb_id_to_altloc,
            pdb_ids_list=pdb_ids_list,
            chunk_str=chunk_str,
        )

    else:
        evaluator = BaseEvaluator(
            true_dir=true_dir,
            pred_dir=eval_dir,
            output_dir=output_dir,
            num_cpu=num_cpu,
            overwrite=False,
            ref_assembly_id=ref_assembly_id,
            pdb_id_to_lig_label_asym_id=pdb_id_to_lig_label_asym_id,
            pdb_id_to_altloc=pdb_id_to_altloc,
            pdb_ids_list=pdb_ids_list,
            chunk_str=chunk_str,
        )
    evaluator.run_eval_batch()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=Path, required=True)
    parser.add_argument("-o", "--output_dir", type=Path, required=True)
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,  # protenix, chai, boltz, af2m
    )
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-c", "--chunk_str", type=str, default=None)

    parser.add_argument("-n", "--num_cpu", type=int, default=-1)
    args = parser.parse_args()

    run_batch_eval(
        eval_dir=args.input_dir,
        model=args.model,
        dataset=args.dataset,
        output_dir=args.output_dir,
        num_cpu=args.num_cpu,
        chunk_str=args.chunk_str,
    )
