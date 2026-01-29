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
from typing import Optional, Union

from rdkit import Chem


def str_to_none(s: Optional[str]) -> Optional[str]:
    """
    Convert string "None" (case-insensitive) to Python None object.
    Otherwise return the original string.

    Args:
        s (str | None): Input string.

    Returns:
        str | None: None if s is "None", else s.
    """
    if s is not None and s.lower() == "none":
        return None
    return s


def read_chain_id_to_mol_from_json(json_f: Union[Path, str]) -> dict[str, Chem.Mol]:
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


def int_to_letters(n: int) -> str:
    """
    Convert int to letters.
    Useful for converting chain index to label_asym_id.

    Args:
        n (int): int number
    Returns:
        str: letters. e.g. 1 -> A, 2 -> B, 27 -> AA, 28 -> AB
    """
    result = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        result = chr(65 + remainder) + result
    return result


def letters_to_int(s: str) -> int:
    """
    Convert letters back to int.
    Inverse of int_to_letters().

    Args:
        s (str): letter sequence, e.g. "A", "B", "AA", "AB"
    Returns:
        int: corresponding integer, e.g. A -> 1, B -> 2, AA -> 27
    """
    s = s.upper()
    result = 0
    for ch in s:
        result = result * 26 + (ord(ch) - 64)  # 'A'→1, 'B'→2 ...
    return result
