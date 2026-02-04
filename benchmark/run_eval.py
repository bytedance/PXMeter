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

from benchmark.configs.data_config import PXM_MMCIF_DIR, SUPPORT_DATA_DIR
from benchmark.evaluators import MODEL_TO_EVALUATOR
from pxmeter.configs.run_config import apply_run_config_overrides
from pxmeter.utils import str_to_none


def get_lig_info(
    lig_info_csv: Path | str,
    entry_id_col: str = "entry_id",
    chain_id_col: str = "label_asym_id",
    altloc_id_col: str = "label_alt_id",
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Reads a CSV file containing PDB information and returns two dictionaries:
    one mapping PDB IDs to ligand asym IDs and another mapping PDB IDs to altloc IDs.

    Args:
        lig_info_csv (Path or str): Path to the CSV file containing PDB information.
        entry_id_col (str, optional): Column name for PDB ID. Defaults to "entry_id".
        chain_id_col (str, optional): Column name for ligand asym ID. Defaults to "label_asym_id".
        altloc_id_col (str, optional): Column name for altloc ID. Defaults to "label_alt_id".

    Returns:
        tuple: A tuple containing:
            - dict: A dictionary mapping PDB IDs to ligand asym IDs.
            - dict: A dictionary mapping PDB IDs to altloc IDs.
    """
    df = pd.read_csv(lig_info_csv)

    df = df.dropna(subset=[entry_id_col, chain_id_col])
    df = df.drop_duplicates(subset=[entry_id_col, chain_id_col])

    has_altloc_col = altloc_id_col in df.columns
    pdb_id_to_lig_label_asym_id = {}
    pdb_id_to_altloc = {}
    for _, row in df.iterrows():
        # Use lower case PDB ID
        pdb_id = row[entry_id_col].lower()
        chain_id = row[chain_id_col]
        if pdb_id not in pdb_id_to_lig_label_asym_id:
            pdb_id_to_lig_label_asym_id[pdb_id] = chain_id
        else:
            pdb_id_to_lig_label_asym_id[pdb_id] += "," + chain_id
        if has_altloc_col:
            altloc = row[altloc_id_col]
            if pdb_id in pdb_id_to_altloc:
                existing_altloc = pdb_id_to_altloc[pdb_id]
                assert (existing_altloc == altloc) or (
                    pd.isna(existing_altloc) and pd.isna(altloc)
                ), (
                    f"Conflicting altloc values found for {entry_id_col} {pdb_id}: '{existing_altloc}' and '{altloc}'. "
                    f"Please ensure all rows for the same {entry_id_col} have the same altloc ({altloc_id_col})."
                )
            pdb_id_to_altloc[pdb_id] = altloc

    return pdb_id_to_lig_label_asym_id, pdb_id_to_altloc


def _get_evaluator_cls(
    eval_dir: Path | str,
    model: str,
    output_dir: Path | str,
    dataset: str = None,
    lig_info_csv: Path | str = None,
    ref_assembly_id: str | None = None,
    num_cpu: int = 1,
    pdb_ids_list: list[str] | None = None,
    chunk_str: str | None = None,
    chai_input_dir: Path | None = None,
):
    if dataset == "PoseBusters":
        true_dir = SUPPORT_DATA_DIR / "posebusters_mmcif"
        pdb_id_to_lig_label_asym_id, pdb_id_to_altloc = get_lig_info(
            SUPPORT_DATA_DIR / "posebusters_lig_info.csv",
            entry_id_col="pdb_id",
            chain_id_col="pb_select_asym_id",
            altloc_id_col="altloc_id",
        )

    else:
        true_dir = PXM_MMCIF_DIR
        if lig_info_csv and Path(str(lig_info_csv)).exists():
            pdb_id_to_lig_label_asym_id, pdb_id_to_altloc = get_lig_info(
                lig_info_csv,
                entry_id_col="entry_id",
                chain_id_col="label_asym_id",
                altloc_id_col="label_alt_id",
            )
        else:
            pdb_id_to_lig_label_asym_id, pdb_id_to_altloc = {}, {}

    if dataset in [
        "RecentPDB",
        "dsDNA-Protein",
        "RNA-Protein",
        "AF3-AB",
    ]:
        ref_assembly_id = "1"

    # Dictionary mapping model names to their respective Evaluator classes
    evaluator_cls = MODEL_TO_EVALUATOR.get(model)
    assert (
        evaluator_cls is not None
    ), f"Model {model} not found in benchmark/evaluators/"

    # Common arguments for all evaluators
    evaluator_args = {
        "true_dir": true_dir,
        "pred_dir": eval_dir,
        "output_dir": output_dir,
        "num_cpu": num_cpu,
        "overwrite": False,
        "ref_assembly_id": ref_assembly_id,
        "pdb_id_to_lig_label_asym_id": pdb_id_to_lig_label_asym_id,
        "pdb_id_to_altloc": pdb_id_to_altloc,
        "pdb_ids_list": pdb_ids_list,
        "chunk_str": chunk_str,
    }

    if model == "chai":
        input_fasta_dir = None
        if chai_input_dir and chai_input_dir.exists():
            input_fasta_dir = chai_input_dir
            logging.info("Using input fasta dir for Chai: %s", input_fasta_dir)
        evaluator_args["input_fasta_dir"] = input_fasta_dir

    evaluator = evaluator_cls(**evaluator_args)
    return evaluator


def run_eval_for_one_pdb_dir(
    pdb_dir: Path | str,
    model: str,
    output_dir: Path | str,
    dataset: str = None,
    lig_info_csv: Path | str = None,
    ref_assembly_id: str | None = None,
    num_cpu: int = 1,
    pdb_ids_list: list[str] | None = None,
    chunk_str: str | None = None,
    chai_input_dir: Path | None = None,
):
    """
    Run evaluation for a single PDB ID.

    Args:
        pdb_dir (Path or str): Directory containing prediction CIF files of a single PDB ID.
        model (str): Name of the model to evaluate.
        dataset (str): Name of the dataset to evaluate on.
        output_dir (Path or str]): Directory to save the evaluation results.
        lig_info_csv (Path or str, optional): CSV file path containing entry_id, label_asym_id,
                                      and optional altloc_id columns.
        ref_assembly_id (str | None, optional): Assembly ID to use for reference structure.
                                                Defaults to None (use asymmetric unit).
        num_cpu (int, optional): Number of CPUs to use for parallel processing. Defaults to 1.
        pdb_ids_list (list[str], optional): List of PDB IDs to evaluate. Defaults to None.
        chunk_str (str, optional): Chunk string for processing. Defaults to None.
        chai_input_dir (Path or None, optional): Directory containing input fasta files for Chai.
            Defaults to None.
    """
    eval_dir = Path("tmp")  # dummy path
    evaluator = _get_evaluator_cls(
        eval_dir=eval_dir,
        model=model,
        output_dir=output_dir,
        dataset=dataset,
        lig_info_csv=lig_info_csv,
        ref_assembly_id=ref_assembly_id,
        num_cpu=num_cpu,
        pdb_ids_list=pdb_ids_list,
        chunk_str=chunk_str,
        chai_input_dir=chai_input_dir,
    )
    evaluator.run_eval_for_one_pdb_dir(pdb_dir)


def run_batch_eval(
    eval_dir: Path | str,
    model: str,
    output_dir: Path | str,
    dataset: str = None,
    lig_info_csv: Path | str = None,
    ref_assembly_id: str | None = None,
    num_cpu: int = 1,
    pdb_ids_list: list[str] | None = None,
    chunk_str: str | None = None,
    chai_input_dir: Path | None = None,
):
    """
    Run batch evaluation for a given model and dataset.

    Args:
        eval_dir (Path or str): Directory containing the evaluation data.
        model (str): Name of the model to evaluate.
        dataset (str): Name of the dataset to evaluate on.
        output_dir (Path or str]): Directory to save the evaluation results.
        lig_info_csv (Path or str, optional): CSV file path containing entry_id, label_asym_id,
                                      and optional altloc_id columns.
        ref_assembly_id (str | None, optional): Assembly ID to use for reference structure.
                                                Defaults to None (use asymmetric unit).
        num_cpu (int, optional): Number of CPUs to use for parallel processing. Defaults to 1.
        pdb_ids_list (list[str], optional): List of PDB IDs to evaluate. Defaults to None.
        chunk_str (str, optional): Chunk string for processing. Defaults to None.
        chai_input_dir (Path or None, optional): Directory containing input fasta files for Chai.
            Defaults to None.
    """
    evaluator = _get_evaluator_cls(
        eval_dir=eval_dir,
        model=model,
        output_dir=output_dir,
        dataset=dataset,
        lig_info_csv=lig_info_csv,
        ref_assembly_id=ref_assembly_id,
        num_cpu=num_cpu,
        pdb_ids_list=pdb_ids_list,
        chunk_str=chunk_str,
        chai_input_dir=chai_input_dir,
    )
    evaluator.run_eval_batch()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing the prediction structures to be evaluated.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where the evaluation JSON results will be stored.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Model name, corresponding to one of the module names in benchmark/evaluators.",
    )
    parser.add_argument(
        "-r",
        "--ref_assembly_id",
        type=str_to_none,
        default=None,
        help="Assembly ID for the reference structure (e.g., '1'). If None, the asymmetric unit is used.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str_to_none,
        default=None,
        help=(
            "Legacy parameter. Only required to be set as 'PoseBusters' when "
            "evaluating PXM-Legacy PoseBusters. In other cases, it can be "
            "left empty, but ensure ref_assembly_id is set correctly."
        ),
    )
    parser.add_argument(
        "-c",
        "--chunk_str",
        type=str_to_none,
        default=None,
        help="Chunking identifier (e.g., '1of5') to evaluate only a subset of the data. Useful for distributed parallelization.",
    )
    parser.add_argument(
        "-l",
        "--lig_info_csv",
        type=str_to_none,
        default=None,
        help="Path to a CSV file providing ligand specific information like 'entry_id' and 'label_asym_id'",
    )
    parser.add_argument(
        "-n",
        "--num_cpu",
        type=int,
        default=-1,
        help="Number of CPU cores for parallel evaluation. Defaults to -1 (use all available cores).",
    )
    parser.add_argument(
        "--chai_input_dir",
        type=Path,
        default=None,
        help="Directory containing input fasta files, required specifically for the Chai model to parse ligand SMILES.",
    )
    parser.add_argument(
        "-C",
        "--config",
        dest="config_overrides",
        action="append",
        default=[],
        help=(
            "Override run config. Use dotted keys from RUN_CONFIG, e.g. "
            "-C metric.lddt.eps=1e-4 -C mapping.mapping_ligand=false. "
            "This option can be repeated."
        ),
    )
    args = parser.parse_args()

    # Apply -C overrides to RUN_CONFIG before evaluation
    if args.config_overrides:
        apply_run_config_overrides(args.config_overrides)

    run_batch_eval(
        eval_dir=args.input_dir,
        model=args.model,
        output_dir=args.output_dir,
        dataset=args.dataset,
        lig_info_csv=args.lig_info_csv,
        ref_assembly_id=args.ref_assembly_id,
        num_cpu=args.num_cpu,
        chunk_str=args.chunk_str,
        chai_input_dir=args.chai_input_dir,
    )
