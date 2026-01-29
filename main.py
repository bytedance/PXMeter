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
import time
from pathlib import Path

from pxmeter.cli import run_eval_cif
from pxmeter.configs.run_config import apply_run_config_overrides
from pxmeter.utils import str_to_none

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--ref_cif",
        type=Path,
        required=True,
        help="Path to the reference CIF file.",
    )
    parser.add_argument(
        "-m",
        "--model_cif",
        type=Path,
        required=True,
        help="Path to the model CIF file.",
    )
    parser.add_argument(
        "-o",
        "--output_json",
        type=Path,
        default="./pxm_output.json",
        help="Path to the output JSON file. Defaults to 'pxm_output.json'.",
    )
    parser.add_argument(
        "--ref_model",
        type=int,
        default=1,
        help="Model number in the reference CIF file to use. Defaults to 1.",
    )
    parser.add_argument(
        "--ref_assembly_id",
        type=str_to_none,
        default=None,
        help="Assembly ID in the reference CIF file. Defaults to None.",
    )
    parser.add_argument(
        "--ref_altloc",
        type=str_to_none,
        default="first",
        help="Altloc ID in the reference CIF file. Defaults to 'first'.",
    )
    parser.add_argument(
        "-l",
        "--interested_lig_label_asym_id",
        type=str_to_none,
        default=None,
        help="The label_asym_id of the ligand of interest in the reference structure (for ligand RMSD metrics). \
            If multiple ligands are present, separate them by comma. Defaults to None.",
    )
    parser.add_argument(
        "-c",
        "--chain_id_to_mol_json",
        type=str_to_none,
        default=None,
        help="Path to a JSON file containing a mapping of chain IDs to molecular input (SMILES). \
            E.g. {'B': 'c1ccccc1', 'D':'CCCC'}",
    )
    parser.add_argument(
        "--output_mapped_cif",
        action="store_true",
        help="Whether to output the mapped CIF file. Defaults to False.",
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

    time_start = time.time()
    run_eval_cif(
        ref_cif=args.ref_cif,
        model_cif=args.model_cif,
        ref_model=args.ref_model,
        output_json=args.output_json,
        ref_assembly_id=args.ref_assembly_id,
        ref_altloc=args.ref_altloc,
        interested_lig_label_asym_id=args.interested_lig_label_asym_id,
        chain_id_to_mol_json=args.chain_id_to_mol_json,
        output_mapped_cif=args.output_mapped_cif,
    )
    time_end = time.time()
    logging.info("Save results to %s", args.output_json)
    logging.info("Evaluation time: %.2f seconds", (time_end - time_start))
