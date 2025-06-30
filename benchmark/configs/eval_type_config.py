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
from pxmeter.constants import DNA, LIGAND, PROTEIN, RNA

EVAL_TYPE_TO_ENTITIY_TYPES = {
    "Intra-Protein": [PROTEIN],
    "Intra-RNA": [RNA],
    "Intra-DNA": [DNA],
    "Intra-Ligand": [LIGAND],
    "Protein-Protein": [PROTEIN, PROTEIN],
    "DNA-DNA": [DNA, DNA],
    "RNA-RNA": [RNA, RNA],
    "Ligand-Ligand": [LIGAND, LIGAND],
    "Protein-Ligand": [PROTEIN, LIGAND],
    "RNA-Protein": [RNA, PROTEIN],
    "DNA-Protein": [DNA, PROTEIN],
    "DNA-RNA": [DNA, RNA],
    "DNA-Ligand": [DNA, LIGAND],
    "RNA-Ligand": [RNA, LIGAND],
}


PB_VALID_CHECK_COL = [
    "sanitization",
    "molecular_formula",
    "molecular_bonds",
    "tetrahedral_chirality",
    "double_bond_stereochemistry",
    "bond_lengths",
    "bond_angles",
    "aromatic_ring_flatness",
    "double_bond_flatness",
    "internal_steric_clash",
    "internal_energy",
    "minimum_distance_to_protein",
    "minimum_distance_to_organic_cofactors",
    "minimum_distance_to_inorganic_cofactors",
    "volume_overlap_with_protein",
    "volume_overlap_with_organic_cofactors",
    "volume_overlap_with_inorganic_cofactors",
]
