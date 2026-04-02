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
from collections import defaultdict
from pathlib import Path

from benchmark.evaluators.base import BaseEvaluator

from benchmark.evaluators.openfold3.config import RANKER_KEYS


class OpenFold3Evaluator(BaseEvaluator):
    """
    A class for evaluating protein structures using the OpenFold3 model.
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
        # Try to infer pdb_id. Assuming the name is just the pdb_id if it's 4 letters,
        # but if the name has an underscore, split it.
        pdb_id = name.split("_")[0]

        if (pdb_dir / name).exists():
            pdb_dir = pdb_dir / name
        elif (pdb_dir / pdb_id).exists():
            pdb_dir = pdb_dir / pdb_id

        sub_data = []
        for seed_dir in pdb_dir.iterdir():
            if not seed_dir.is_dir():
                continue

            # OpenFold3 output directories are expected to be like "seed_1"
            if not seed_dir.name.startswith("seed_"):
                continue

            seed_parts = seed_dir.name.split("_")
            if len(seed_parts) < 2:
                continue
            seed = seed_parts[1]

            # Find all model.cif files
            for pred_cif in seed_dir.glob(f"{name}_seed_{seed}_sample_*_model.cif"):
                stem_parts = pred_cif.stem.split("_sample_")
                if len(stem_parts) != 2:
                    continue

                sample = stem_parts[1].split("_")[0]

                confidence_json = (
                    seed_dir
                    / f"{name}_seed_{seed}_sample_{sample}_confidences_aggregated.json"
                )

                if not confidence_json.exists():
                    logging.warning(
                        "Can not find confidence_json for %s, skip.",
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

    @staticmethod
    def save_mapped_confidence_json(
        rankers: dict[str, list[tuple[str, bool]]],
        ori_confidence_json: str | Path,
        output_confidence_json: str | Path,
        ori_model_chain_ids: list[str],
    ):
        """
        Save mapped confidence scores to a JSON file.

        Args:
            rankers (dict[str, list[tuple[str, bool]]]): Dictionary containing ranker information.
            ori_confidence_json (str or Path): Path to the original confidence JSON file.
            output_confidence_json (str or Path): Path to the output confidence JSON file.
            ori_model_chain_ids (list[str]): A list of original model chain IDs.
        """
        with open(ori_confidence_json, "r") as f:
            ori_confidence = json.load(f)

        ranker_results = {}

        # complex: {key: score}
        complex_ranker_results = {}
        for ranker_key, _ascending in rankers.get("complex", []):
            if ranker_key not in ori_confidence:
                continue
            complex_score = ori_confidence[ranker_key]
            complex_ranker_results[ranker_key] = complex_score
        ranker_results["complex"] = complex_ranker_results

        # chain: {key: {chain_id: score}}
        chain_ranker_results = defaultdict(dict)
        for ranker_key, _ascending in rankers.get("chain", []):
            if ranker_key not in ori_confidence:
                continue

            chain_scores = ori_confidence[ranker_key]
            for chain_id in ori_model_chain_ids:
                if chain_id in chain_scores:
                    chain_ranker_results[ranker_key][chain_id] = chain_scores[chain_id]
        ranker_results["chain"] = chain_ranker_results

        # interface: {key: {f"{chain_id_1},{chain_id_2}": score}}
        interface_ranker_results = defaultdict(dict)
        for ranker_key, _ascending in rankers.get("interface", []):
            if ranker_key not in ori_confidence:
                continue

            interface_scores = ori_confidence[ranker_key]

            for idx_i, chain_id_i in enumerate(ori_model_chain_ids):
                for idx_j, chain_id_j in enumerate(ori_model_chain_ids[idx_i + 1 :]):
                    interface_id = ",".join(sorted((chain_id_i, chain_id_j)))

                    k1 = f"({chain_id_i}, {chain_id_j})"
                    k2 = f"({chain_id_j}, {chain_id_i})"

                    val = None
                    if k1 in interface_scores:
                        val = interface_scores[k1]
                    elif k2 in interface_scores:
                        val = interface_scores[k2]

                    if val is not None:
                        interface_ranker_results[ranker_key][interface_id] = val
        ranker_results["interface"] = interface_ranker_results

        output_confidence_tmp_json = Path(output_confidence_json).with_suffix(
            ".json.tmp"
        )
        with open(output_confidence_tmp_json, "w") as f:
            json.dump(ranker_results, f, indent=4)
        output_confidence_tmp_json.rename(output_confidence_json)
