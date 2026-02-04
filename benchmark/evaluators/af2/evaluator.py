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
from pathlib import Path

from benchmark.evaluators.af2.config import RANKER_KEYS

from benchmark.evaluators.base import BaseEvaluator


class AF2Evaluator(BaseEvaluator):
    """
    A class for evaluating protein structures using the AF2 model.
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

        debug_file = pdb_dir / "ranking_debug.json"
        if not debug_file.exists():
            return []
        with open(debug_file) as f:
            confidence_dict = json.load(f)

        # e.g. unrelaxed_model_1_pred_0.cif
        sub_data = []
        for pred_cif in pdb_dir.glob("unrelaxed_model_*.cif"):
            split_filename = pred_cif.stem.split("_")
            sample, seed = split_filename[2], split_filename[-1]

            model_name = pred_cif.stem.replace("unrelaxed_", "")
            confidence_score = confidence_dict["plddts"][model_name]
            model_chain_id_to_lig_mol = None

            sub_data.append(
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
        return sub_data

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
        complex_ranker_results["plddt"] = ori_confidence_json
        ranker_results["complex"] = complex_ranker_results

        output_confidence_tmp_json = Path(output_confidence_json).with_suffix(
            ".json.tmp"
        )
        with open(output_confidence_tmp_json, "w") as f:
            json.dump(ranker_results, f, indent=4)
        output_confidence_tmp_json.rename(output_confidence_json)
