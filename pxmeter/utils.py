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

from rdkit import Chem


def read_chain_id_to_mol_from_json(json_f: Path | str) -> dict[str, Chem.Mol]:
    """
    Reads a JSON file containing chain IDs and their corresponding SMILES representations,
    and returns a dictionary mapping chain IDs to RDKit Mol objects.

    Args:
        json_f (Path | str): The path to the JSON file.

    Returns:
        dict[str, Chem.Mol]: A dictionary mapping chain IDs to RDKit Mol objects.
    """
    with open(json_f, "r") as f:
        chain_id_to_mol_rep = json.load(f)

    chain_id_to_mol = {}
    for k, v in chain_id_to_mol_rep.items():
        chain_id_to_mol[k] = Chem.MolFromSmiles(v)
    return chain_id_to_mol
