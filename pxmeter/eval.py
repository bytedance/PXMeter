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

import warnings

import numpy as np
from ml_collections.config_dict import ConfigDict
from rdkit import Chem

from pxmeter.calc_metric import MetricResult
from pxmeter.configs.run_config import RUN_CONFIG
from pxmeter.mapping import MappingResult

# Suppress biotite warning message
warnings.filterwarnings(
    "ignore",
    message="Attribute .* not found within 'atom_site' category. The fallback attribute .* will be used instead",
)


def evaluate(
    ref_cif: str,
    model_cif: str,
    ref_model: int = 1,
    ref_assembly_id: str | None = None,
    ref_altloc: str | None = "first",
    model_chain_id_to_lig_mol: dict[str, Chem.Mol] | None = None,
    interested_lig_label_asym_id: list[str] | str | None = None,
    run_config: ConfigDict = RUN_CONFIG,
) -> MetricResult:
    """
    Evaluate the performance of a model CIF file against a reference CIF file.

    Args:
        ref_cif (str): Path to the reference CIF file.
        model_cif (str): Path to the model CIF file.
        ref_model (int, optional): Model number in the reference CIF file to use. Defaults to 1.
        ref_assembly_id (str, optional): Assembly ID in the reference CIF file. Defaults to None.
        ref_altloc (str, optional): Alternate location indicator in the reference CIF file.
                    Defaults to "first".
        model_chain_id_to_lig_mol (dict[str, Chem.Mol], optional): Mapping model chain IDs
                                  to ligand molecules. Defaults to None.
        interested_lig_label_asym_id (list[str] | str, optional): Label asym ID of the ligand of interest
                                      in the reference structure. Defaults to None.
        run_config (ConfigDict, optional): Configuration for the run. Defaults to RUN_CONFIG.

    Returns:
        MetricResult: An object containing the evaluation results.
    """

    map_result = MappingResult.from_cifs(
        ref_cif=ref_cif,
        model_cif=model_cif,
        ref_model=ref_model,
        ref_assembly_id=ref_assembly_id,
        model_chain_id_to_lig_mol=model_chain_id_to_lig_mol,
        ref_altloc=ref_altloc,
        mapping_config=run_config.mapping,
    )
    sele_ref_struct, sele_model_struct = map_result.get_mapped_structures()

    # Get original model chain ID for mapping confidence score
    # "A.1" should not appear in model_struct
    _, chain_start = np.unique(map_result.model_struct.uni_chain_id, return_index=True)
    ori_model_chain_ids = map_result.model_struct.uni_chain_id[chain_start].tolist()

    metric_result = MetricResult.from_struct(
        ref_struct=sele_ref_struct,
        model_struct=sele_model_struct,
        ori_model_chain_ids=ori_model_chain_ids,
        interested_lig_label_asym_id=interested_lig_label_asym_id,
        metric_config=run_config.metric,
    )
    return metric_result
