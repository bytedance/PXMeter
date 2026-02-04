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

import logging
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from benchmark.evaluators.base import BaseEvaluator
from benchmark.evaluators.chai.config import RANKER_KEYS
from benchmark.utils import int_to_letters


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
        self.ranker = RANKER_KEYS

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

    def _get_info_from_each_pdb_dir(self, pdb_dir: Path) -> list:
        if not pdb_dir.is_dir():
            return []

        name = pdb_dir.stem
        pdb_id = name.split("_")[0]

        sub_data = []
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

                sub_data.append(
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
        return sub_data
