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

from collections import defaultdict
from pathlib import Path

import numpy as np
from biotite.structure import AtomArray, get_chain_starts, get_residue_starts
from biotite.structure.io import pdbx


class CIFWriter:
    """
    Write AtomArray to cif.

    Args:
            atom_array (AtomArray): Biotite AtomArray object.
            entity_poly_type (dict[str, str], optional):  A dict of label_entity_id
                             to entity_poly_type. Defaults to None.
                             If None, "the entity_poly" and
                             "entity_poly_seq" will not be written to the cif.
            atom_array_output_mask (np.ndarray, optional): A mask of atom_array.
                                    Defaults to None. If None, all atoms will be
                                    written to the cif.
    """

    def __init__(
        self,
        atom_array: AtomArray,
        entity_poly_type: dict[str, str] = None,
        atom_array_output_mask: np.ndarray | None = None,
    ):
        self.atom_array = atom_array
        self.entity_poly_type = entity_poly_type
        self.atom_array_output_mask = atom_array_output_mask

    def _get_unresolved_block(self):
        res_starts = get_residue_starts(self.atom_array, add_exclusive_stop=True)
        is_res_starts = np.zeros(len(self.atom_array_output_mask), dtype=bool)
        for start, stop in zip(res_starts[:-1], res_starts[1:]):
            if not any(self.atom_array_output_mask[start:stop]):
                is_res_starts[start] = True

        mask = (~self.atom_array_output_mask) & is_res_starts
        if not np.any(mask):
            # No unresolved atoms
            return
        polymer_flag_bool = np.isin(
            self.atom_array.label_entity_id[mask], list(self.entity_poly_type.keys())
        )
        polymer_flag = ["Y" if i else "N" for i in polymer_flag_bool]

        unresolved_block = defaultdict(list)
        unresolved_block["id"] = np.arange(mask.sum()) + 1
        unresolved_block["PDB_model_num"] = np.ones(mask.sum(), dtype=int)
        unresolved_block["polymer_flag"] = polymer_flag
        unresolved_block["occupancy_flag"] = np.ones(mask.sum(), dtype=int)
        unresolved_block["auth_asym_id"] = self.atom_array.chain_id[mask]
        unresolved_block["auth_comp_id"] = self.atom_array.res_name[mask]
        unresolved_block["auth_seq_id"] = self.atom_array.res_id[mask]
        unresolved_block["PDB_ins_code"] = ["?"] * mask.sum()
        unresolved_block["label_asym_id"] = self.atom_array.chain_id[mask]
        unresolved_block["label_comp_id"] = self.atom_array.res_name[mask]
        unresolved_block["label_seq_id"] = self.atom_array.res_id[mask]
        return pdbx.CIFCategory(unresolved_block)

    def _get_entity_block(self):
        if self.entity_poly_type is None:
            return {}
        entity_ids_in_atom_array = np.sort(np.unique(self.atom_array.label_entity_id))
        entity_block_dict = defaultdict(list)
        for entity_id in entity_ids_in_atom_array:
            if entity_id not in self.entity_poly_type:
                entity_type = "non-polymer"
            else:
                entity_type = "polymer"
            entity_block_dict["id"].append(entity_id)
            entity_block_dict["pdbx_description"].append(".")
            entity_block_dict["type"].append(entity_type)
        return pdbx.CIFCategory(entity_block_dict)

    def _get_entity_poly_and_entity_poly_seq_block(self):
        entity_poly = defaultdict(list)
        for entity_id, entity_type in self.entity_poly_type.items():
            label_asym_ids = np.unique(
                self.atom_array.label_asym_id[
                    self.atom_array.label_entity_id == entity_id
                ]
            )
            label_asym_ids_str = ",".join(label_asym_ids)

            if label_asym_ids_str == "":
                # The entity not in current atom_array
                continue

            entity_poly["entity_id"].append(entity_id)
            entity_poly["pdbx_strand_id"].append(label_asym_ids_str)
            entity_poly["type"].append(entity_type)

        if not entity_poly:
            return {}

        entity_poly_seq = defaultdict(list)
        for entity_id, label_asym_ids_str in zip(
            entity_poly["entity_id"], entity_poly["pdbx_strand_id"]
        ):
            first_label_asym_id = label_asym_ids_str.split(",")[0]
            first_asym_chain = self.atom_array[
                self.atom_array.label_asym_id == first_label_asym_id
            ]
            chain_starts = get_chain_starts(first_asym_chain, add_exclusive_stop=True)
            asym_chain = first_asym_chain[
                chain_starts[0] : chain_starts[1]
            ]  # ensure the asym chain is a single chain

            res_starts = get_residue_starts(asym_chain, add_exclusive_stop=False)
            asym_chain_entity_id = asym_chain[res_starts].label_entity_id.tolist()
            asym_chain_hetero = [
                "n" if not i else "y" for i in asym_chain[res_starts].hetero
            ]
            asym_chain_res_name = asym_chain[res_starts].res_name.tolist()
            asym_chain_res_id = asym_chain[res_starts].res_id.tolist()

            entity_poly_seq["entity_id"].extend(asym_chain_entity_id)
            entity_poly_seq["hetero"].extend(asym_chain_hetero)
            entity_poly_seq["mon_id"].extend(asym_chain_res_name)
            entity_poly_seq["num"].extend(asym_chain_res_id)

        block_dict = {
            "entity_poly": pdbx.CIFCategory(entity_poly),
            "entity_poly_seq": pdbx.CIFCategory(entity_poly_seq),
        }
        return block_dict

    def save_to_cif(
        self, output_path: str, entry_id: str = None, include_bonds: bool = False
    ):
        """
        Save AtomArray to cif.

        Args:
            output_path (str): Output path of cif file.
            entry_id (str, optional): The value of "_entry.id" in cif. Defaults to None.
                                      If None, the entry_id will be the basename
                                      of output_path (without ".cif" extension).
            include_bonds (bool, optional): Whether to include  bonds in the cif.
                          Defaults to False. If set to True and `array`
                          has associated ``bonds`` , the intra-residue
                          bonds will be written into the ``chem_comp_bond``
                          category. Inter-residue bonds will be written
                          into the ``struct_conn`` independent of this parameter.
        """
        output_path = Path(output_path)
        if entry_id is None:
            entry_id = output_path.stem

        block_dict = {"entry": pdbx.CIFCategory({"id": entry_id})}
        if self.entity_poly_type:
            block_dict["entity"] = self._get_entity_block()
            block_dict.update(self._get_entity_poly_and_entity_poly_seq_block())

        if self.atom_array_output_mask is not None:
            unresolved_block = self._get_unresolved_block()
            if unresolved_block is not None:
                block_dict["pdbx_unobs_or_zero_occ_residues"] = unresolved_block

        block = pdbx.CIFBlock(block_dict)
        cif = pdbx.CIFFile({entry_id: block})
        if self.atom_array_output_mask is not None:
            atom_array = self.atom_array[self.atom_array_output_mask]
        else:
            atom_array = self.atom_array

        pdbx.set_structure(cif, atom_array, include_bonds=include_bonds)
        block = cif.block
        atom_site = block.get("atom_site")

        occ = atom_site.get("occupancy")
        if occ is None:
            atom_site["occupancy"] = np.ones(len(atom_array), dtype=float)

        b_factor = atom_site.get("B_iso_or_equiv")
        if b_factor is None:
            atom_site["B_iso_or_equiv"] = np.round(
                np.zeros(len(atom_array), dtype=float), 2
            ).astype(str)

        if "label_entity_id" in atom_array.get_annotation_categories():
            atom_site["label_entity_id"] = atom_array.label_entity_id
        cif.write(output_path)
