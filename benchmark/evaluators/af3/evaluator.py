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

from benchmark.evaluators.af3.config import RANKER_KEYS
from benchmark.evaluators.base import BaseEvaluator


class AF3Evaluator(BaseEvaluator):
    """
    A class for evaluating protein structures using the AF3 model.
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
        self.ranker = RANKER_KEYS

    def _get_info_from_each_pdb_dir(self, pdb_dir: Path) -> list:
        if not pdb_dir.is_dir():
            return []

        name = pdb_dir.stem
        pdb_id = name.split("_")[0]

        if (pdb_dir / pdb_id).exists():
            pdb_dir = pdb_dir / pdb_id

        sub_data = []
        for seed_sample_dir in pdb_dir.iterdir():
            if not seed_sample_dir.is_dir():
                continue

            seed = seed_sample_dir.name.split("-")[1].split("_")[0]
            sample = seed_sample_dir.name.split("-")[-1]

            pred_cif = (
                seed_sample_dir / f"{pdb_id}_seed-{seed}_sample-{sample}_model.cif"
            )
            confidence_json = (
                seed_sample_dir
                / f"{pdb_id}_seed-{seed}_sample-{sample}_summary_confidences.json"
            )
            if not pred_cif.exists() or not confidence_json.exists():
                logging.warning(
                    "Can not find pred_cif or confidence_json for %s, skip.",
                    pred_cif,
                )
                continue
            model_chain_id_to_lig_mol = None

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
