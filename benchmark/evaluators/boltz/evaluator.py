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

from benchmark.evaluators.base import BaseEvaluator
from benchmark.evaluators.boltz.config import RANKER_KEYS


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
        self.ranker = RANKER_KEYS

    def _get_info_from_each_pdb_dir(self, pdb_dir: Path) -> list:
        if not pdb_dir.is_dir():
            return []

        name = pdb_dir.stem
        pdb_id = name.split("_")[0]

        sub_data = []
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
