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

from pathlib import Path

from biotite.structure import AtomArray
from biotite.structure.io import load_structure

TEST_DATA_DIR = Path(__file__).absolute().parent / "files"


def get_atom_array(cif_path) -> AtomArray:
    """
    Load an AtomArray object from a CIF file with specified extra fields.

    Args:
        cif_path (str or Path): The path to the CIF file to be loaded.

    Returns:
        AtomArray: A Biotite AtomArray object containing structure data from the CIF file.
    """
    extra_fields = ["label_asym_id", "label_entity_id", "auth_asym_id"]  # Chain
    extra_fields += ["label_seq_id", "auth_seq_id"]  # Residue

    # res_id == -1 for HETATM because use_author_fields=False
    atom_array = load_structure(
        cif_path, extra_fields=extra_fields, use_author_fields=False
    )
    return atom_array
