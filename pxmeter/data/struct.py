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

import copy
import dataclasses
from pathlib import Path
from typing import Sequence

import numpy as np
from biotite.structure import AtomArray, CellList

from pxmeter.constants import CRYSTALLIZATION_AIDS, CRYSTALLIZATION_METHODS
from pxmeter.data.parser import MMCIFParser
from pxmeter.data.utils import get_unique_atom_id, get_unique_chain_id
from pxmeter.data.writer import CIFWriter


@dataclasses.dataclass
class Structure:
    """
    Structure class.

    Attributes:
        atom_array (AtomArray): Biotite AtomArray object.
        chain_id (np.ndarray): Unique chain ID of atom_array.
        entity_poly_seq (dict[str, str]): A dict of label_entity_id (polymer) → sequence.
        entity_poly_type (dict[str, str]): A dict of label_entity_id (polymer) → entity_poly_type.
        uni_chain_id (np.ndarray): Unique chain ID of atom_array.
        uni_atom_id (np.ndarray): Unique atom ID of atom_array.
                    Composed by {res_id}_{res_name}_{atom_name}

        entry_id (str): Entry ID of atom_array. defaults to "".
        exptl_methods (tuple[str]): Experimental methods of Structure. Defaults to tuple().
        cif_block (dict): Original CIF block of Structure. Defaults to None.
    """

    atom_array: AtomArray
    entity_poly_seq: dict[str, str]
    entity_poly_type: dict[str, str]
    uni_chain_id: np.ndarray
    uni_atom_id: np.ndarray
    entry_id: str = ""
    exptl_methods: tuple[str] = tuple()
    cif_block: dict = None

    @classmethod
    def from_mmcif(
        cls,
        mmcif: Path | str,
        model: int = 1,
        altloc: str = "first",
        assembly_id: str | None = None,
        include_bonds: bool = True,
    ) -> "Structure":
        """
        Create a Structure object from MMCIF.

        Args:
            mmcif (Path or str): Path to MMCIF file.
            model (int): Model number. Defaults to 1.
            altloc (str): It could be one of "all", "first", "occupancy", "A", "B", etc.
                          Defaults to "first".
            assembly_id (str, optional): Assembly ID. Defaults to None.
            include_bonds (bool): Whether to include bonds in the AtomArray. Defaults to True.

        Returns:
            Structure: Structure object.
        """
        cif_parser = MMCIFParser(mmcif)
        atom_array = cif_parser.get_structure(
            model=model,
            altloc=altloc,
            assembly_id=assembly_id,
            include_bonds=include_bonds,
        )

        return cls(
            atom_array=atom_array,
            entity_poly_seq=cif_parser.get_entity_poly_seq(atom_array),
            entity_poly_type=cif_parser.entity_poly_type,
            uni_chain_id=get_unique_chain_id(atom_array),
            uni_atom_id=get_unique_atom_id(atom_array),
            entry_id=cif_parser.entry_id,
            exptl_methods=tuple(cif_parser.exptl_methods),
            cif_block=cif_parser.cif.block,
        )

    def _get_hydrogens_mask(self) -> np.ndarray:
        """
        Get mask of hydrogens.


        Returns:
            np.ndarray: Mask of hydrogens.
        """
        return np.isin(self.atom_array.element, ["H", "D"])

    def _get_water_mask(self) -> np.ndarray:
        """
        Get mask of water.

        Returns:
            np.ndarray: Mask of water atoms.
        """
        return np.isin(self.atom_array.res_name, ["HOH", "DOD"])

    def _get_element_x_mask(self) -> np.ndarray:
        """
        Get mask of element X in UNX and UNL.

        - UNX: Unknown one atom or ion.
        - UNL: Unknown ligand, some atoms are marked as X.

        Returns:
            np.ndarray: Mask of element X in UNX and UNL.
        """
        X_mask = np.zeros(len(self.atom_array), dtype=bool)
        starts = self.get_residue_starts(add_exclusive_stop=True)
        for start, stop in zip(starts[:-1], starts[1:]):
            res_name = self.atom_array.res_name[start]
            if res_name in ["UNX", "UNL"]:
                X_mask[start:stop] = True
        return X_mask

    def _modified_asx_and_glx(self):
        """
        Modify ASX and GLX to ASP and GLU.

        - ASX: ASP/ASN ambiguous, two ambiguous atoms are marked as X.
        - GLX: GLU/GLN ambiguous, two ambiguous atoms are marked as X.
        """
        # map ASX to ASP, as ASP is more symmetric than ASN
        mask = self.atom_array.res_name == "ASX"
        self.atom_array.res_name[mask] = "ASP"
        self.atom_array.atom_name[mask & (self.atom_array.atom_name == "XD1")] = "OD1"
        self.atom_array.atom_name[mask & (self.atom_array.atom_name == "XD2")] = "OD2"
        self.atom_array.element[mask & (self.atom_array.element == "X")] = "O"

        # map GLX to GLU, as GLU is more symmetric than GLN
        mask = self.atom_array.res_name == "GLX"
        self.atom_array.res_name[mask] = "GLU"
        self.atom_array.atom_name[mask & (self.atom_array.atom_name == "XE1")] = "OE1"
        self.atom_array.atom_name[mask & (self.atom_array.atom_name == "XE2")] = "OE2"
        self.atom_array.element[mask & (self.atom_array.element == "X")] = "O"

    def _get_crystallization_aids_mask(self) -> np.ndarray:
        """
        Get a mask of crystallization aids, eg: SO4, GOL, etc.

        Returns:
            np.ndarray: Mask of crystallization aids.
        """
        aids_mask = np.isin(self.atom_array.res_name, CRYSTALLIZATION_AIDS)
        return aids_mask

    def get_polymer_mask(self) -> np.ndarray:
        """
        Get a mask of polymer residues.

        Returns:
            np.ndarray: Mask of polymer atoms.
        """
        polyer_mask = np.isin(
            self.atom_array.label_entity_id, self.entity_poly_seq.keys()
        )
        return polyer_mask

    def get_residue_starts(self, add_exclusive_stop: bool = False) -> np.ndarray:
        """
        Get the indices of the first atom of each residue.
        Use unique chain id, res id, res name to identify a residue.
        The code modified from biotite.structure.get_residue_starts.

        Args:
            add_exclusive_stop (bool): Whether to add the index of the last atom of the last
            residue. Defaults to False.

        Returns:
            np.ndarray: Indices of the first atom of each residue.
        """
        if self.atom_array.array_length() == 0:
            return np.array([], dtype=int)

        # These mask are 'true' at indices where the value changes
        chain_id_changes = self.uni_chain_id[1:] != self.uni_chain_id[:-1]
        res_id_changes = self.atom_array.res_id[1:] != self.atom_array.res_id[:-1]
        res_name_changes = self.atom_array.res_name[1:] != self.atom_array.res_name[:-1]

        # If any of these annotation arrays change, a new residue starts
        residue_change_mask = chain_id_changes | res_id_changes | res_name_changes

        # Convert mask to indices
        # Add 1, to shift the indices from the end of a residue
        # to the start of a new residue
        residue_starts = np.where(residue_change_mask)[0] + 1

        # The first residue is not included yet -> Insert '[0]'
        if add_exclusive_stop:
            return np.concatenate(
                ([0], residue_starts, [self.atom_array.array_length()])
            )
        else:
            return np.concatenate(([0], residue_starts))

    def _modified_arg_atom_naming(self):
        """
        Arginine naming ambiguities are fixed (ensuring NH1 is always closer to CD than NH2).
        """
        starts = self.get_residue_starts(add_exclusive_stop=True)
        for start_i, stop_i in zip(starts[:-1], starts[1:]):
            if self.atom_array.res_name[start_i] != "ARG":
                continue

            atom_indices = {"CD": None, "NH1": None, "NH2": None}
            for idx in range(start_i, stop_i):
                atom_name = self.atom_array.atom_name[idx]
                if atom_name in atom_indices:
                    atom_indices[atom_name] = idx

            cd_idx, nh1_idx, nh2_idx = (
                atom_indices["CD"],
                atom_indices["NH1"],
                atom_indices["NH2"],
            )
            if all([cd_idx, nh1_idx, nh2_idx]):  # all not None
                cd_nh1 = self.atom_array.coord[nh1_idx] - self.atom_array.coord[cd_idx]
                d2_cd_nh1 = np.sum(cd_nh1**2)
                cd_nh2 = self.atom_array.coord[nh2_idx] - self.atom_array.coord[cd_idx]
                d2_cd_nh2 = np.sum(cd_nh2**2)
                if d2_cd_nh2 < d2_cd_nh1:
                    # swap NH1 and NH2
                    self.atom_array.coord[[nh1_idx, nh2_idx]] = self.atom_array.coord[
                        [nh2_idx, nh1_idx]
                    ]

    def _mse_to_met(self):
        """
        MSE residues are converted to MET residues.
        """
        mse = self.atom_array.res_name == "MSE"
        se = mse & (self.atom_array.atom_name == "SE")
        self.atom_array.atom_name[se] = "SD"
        self.atom_array.element[se] = "S"
        self.atom_array.res_name[mse] = "MET"
        self.atom_array.hetero[mse] = False

    def reset_atom_array_annot(self, annot_name: str, annot_value: Sequence):
        """
        Reset an annotation of the atom_array of Structure.

        Args:
            annot_name (str): The name of the annotation.
            annot_value (Sequence): The value of the annotation.
        """
        # if the annot name in AtomArray._annot, the dtype will be set to the old one in set_annotation
        # or np.promote_types for two types
        self.atom_array.del_annotation(annot_name)
        self.atom_array.set_annotation(annot_name, annot_value)

    def reset_entity_poly_by_atom_array(self):
        """
        Resets the entity_poly_seq and entity_poly_type attributes based on the unique entity IDs
        present in the atom_array's label_entity_id.
        """
        entity_id_in_atom_array = np.unique(self.atom_array.label_entity_id)
        polymer_entities = list(self.entity_poly_type.keys())

        entity_poly_seq = {}
        entity_poly_type = {}
        for entity_id in entity_id_in_atom_array:
            if entity_id not in polymer_entities:
                continue
            entity_poly_seq[entity_id] = self.entity_poly_seq[entity_id]
            entity_poly_type[entity_id] = self.entity_poly_type[entity_id]

        # Reset self.entity_poly_seq and self.entity_poly_type
        self.entity_poly_seq = entity_poly_seq
        self.entity_poly_type = entity_poly_type

    def update_entity_poly(self, entity_id_old_and_new: list[tuple[str, str]]):
        """
        Update entity_poly_seq and entity_poly_type of Structure.

        Args:
            entity_id_old_and_new (list[tuple[str, str]]): A list of (old entity_id (polymer), new entity_id).
        """
        new_entity_poly_seq = {}
        new_entity_poly_type = {}
        for entity_old, entity_new in entity_id_old_and_new:
            old_entity_poly_seq = self.entity_poly_seq.get(entity_old)
            old_entity_poly_type = self.entity_poly_type.get(entity_old)

            # Check if the entity is a polymer
            if old_entity_poly_seq and old_entity_poly_type:
                new_entity_poly_seq[entity_new] = old_entity_poly_seq
                new_entity_poly_type[entity_new] = old_entity_poly_type
        self.entity_poly_seq = new_entity_poly_seq
        self.entity_poly_type = new_entity_poly_type

    def select_substructure(
        self, mask: Sequence[int | bool], reset_uni_id: bool = False
    ) -> "Structure":
        """
        Select a substructure from the Structure by a mask.

        Args:
            mask (Sequence[int or bool]): A 1D mask array (int or bool).
                               If dtype == bool, the length should be equal to the length
                               of the atom_array of the Structure.
            reset_uni_id (bool, optial): Whether to reset unique chain and residue IDs.
                                         Defaults to False.

        Returns:
            Structure: A new Structure object.
        """
        substructure_atom_array = self.atom_array[mask]
        if reset_uni_id:
            uni_chain_id = get_unique_chain_id(substructure_atom_array)
            uni_atom_id = get_unique_atom_id(substructure_atom_array)
        else:
            uni_chain_id = self.uni_chain_id[mask]
            uni_atom_id = self.uni_atom_id[mask]

        return dataclasses.replace(
            self,
            atom_array=substructure_atom_array,
            uni_chain_id=uni_chain_id,
            uni_atom_id=uni_atom_id,
        )

    def get_chains_and_interfaces(
        self, interface_radius: int = 5
    ) -> tuple[list[str], list[tuple[str, str]]]:
        """
        Get unique chains and their interfaces within a specified radius.

        Args:
            interface_radius (int): The radius within which to search for
                                    interfaces between chains. Default is 5.

        Returns:
        tuple:
            chains (list[str]): A list of unique chain identifiers.
            interfaces (list[tuple[str, str]]]): A list of tuples, where each tuple contains
                                                 a pair of chain identifiers that
                                                 have interfaces within the specified radius.
        """
        chains = np.unique(self.uni_chain_id).tolist()

        interfaces = []
        cell_list = CellList(self.atom_array, cell_size=interface_radius)
        for chain_i in chains:
            chain_mask = self.uni_chain_id == chain_i
            coord = self.atom_array.coord[chain_mask]
            neighbors_indices_2d = cell_list.get_atoms(
                coord, radius=interface_radius
            )  # Shape = [n_coord, max_n_neighbors], padding with -1
            neighbors_indices = np.unique(neighbors_indices_2d)
            neighbors_indices = neighbors_indices[neighbors_indices != -1]

            chain_j_list = np.unique(self.uni_chain_id[neighbors_indices])
            for chain_j in chain_j_list:
                if chain_i == chain_j:
                    continue

                if (chain_i, chain_j) in interfaces or (chain_j, chain_i) in interfaces:
                    continue
                interfaces.append((chain_i, chain_j))
        return chains, interfaces

    def get_entity_id_to_chain_ids(
        self, use_uni_chain_id: bool = True
    ) -> dict[str, list[str]]:
        """
        Maps each entity ID to a list of chain IDs within the current structure.

        Args:
            use_uni_chain_id (bool, optional): Whether to use unique chain IDs.
                If True, uses `uni_chain_id`; otherwise, uses `atom_array.chain_id`.
                Defaults to True.

        Returns:
            dict[str, list[str]]: A dictionary where keys are entity IDs (as strings)
                and values are lists of corresponding chain IDs.
        """
        entity_id_to_chain_ids = {}
        for entity_id in np.unique(self.atom_array.label_entity_id):
            mask = self.atom_array.label_entity_id == entity_id
            if use_uni_chain_id:
                chain_ids = np.unique(self.uni_chain_id[mask])
            else:
                chain_ids = np.unique(self.atom_array.chain_id[mask])
            entity_id_to_chain_ids[entity_id] = list(chain_ids)
        return entity_id_to_chain_ids

    def get_chain_id_to_entity_id(
        self, use_uni_chain_id: bool = True
    ) -> dict[str, str]:
        """
        Maps each chain ID to the corresponding entity ID within the current structure.

        Args:
            use_uni_chain_id (bool, optional): Whether to use unique chain IDs.
                If True, uses `uni_chain_id`; otherwise, uses `atom_array.chain_id`.
                Defaults to True.

        Returns:
            dict[str, str]: A dictionary where keys are chain IDs (as strings)
                and values are corresponding entity IDs (also as strings).
        """
        if use_uni_chain_id:
            chain_id_array = self.uni_chain_id
        else:
            chain_id_array = self.atom_array.chain_id

        chain_ids, chain_id_index = np.unique(chain_id_array, return_index=True)

        chain_id_to_entity_id = {}
        for chain_id, idx in zip(chain_ids, chain_id_index):
            entity_id = self.atom_array.label_entity_id[idx]
            chain_id_to_entity_id[chain_id] = entity_id
        return chain_id_to_entity_id

    def clean_structure(
        self,
        mse_to_met=True,
        modified_arg_atom_naming=True,
        remove_water=True,
        remove_hydrogens=True,
        remove_element_x=True,
        remove_crystallization_aids=True,
    ) -> "Structure":
        """
        Clean the structure using some filtering and modification operations.

        Args:
            mse_to_met (bool): Whether to convert MSE to MET. Defaults to True.
            modified_arg_atom_naming (bool): Whether to modify the NH1/NH2
                                             atom naming of ARG. Defaults to True.
            remove_water (bool): Whether to remove water. Defaults to True.
            remove_hydrogens (bool): Whether to remove hydrogens. Defaults to True.
            remove_element_x (bool): Whether to remove element X. Defaults to True.
            remove_crystallization_aids (bool): Whether to remove crystallization aids.
                                                Defaults to True.
        Returns:
            Structure: A new Structure object.
        """
        if modified_arg_atom_naming:
            self._modified_arg_atom_naming()
        if mse_to_met:
            self._mse_to_met()

        mask = np.ones(len(self.atom_array), dtype=bool)

        if remove_water:
            mask &= ~self._get_water_mask()
        if remove_hydrogens:
            mask &= ~self._get_hydrogens_mask()
        if remove_element_x:
            self._modified_asx_and_glx()
            mask &= ~self._get_element_x_mask()
        if remove_crystallization_aids and (
            set(self.exptl_methods) & CRYSTALLIZATION_METHODS
        ):
            # only remove aids in non-polymer residues
            non_polymer_mask = ~self.get_polymer_mask()
            crys_aids_mask = self._get_crystallization_aids_mask()
            mask &= ~(non_polymer_mask & crys_aids_mask)

        return self.select_substructure(mask)

    def to_cif(
        self,
        output_cif: str | Path,
        use_uni_chain_id: bool = False,
        include_bonds: bool = False,
    ):
        """
        Write the structure to a CIF file.

        Args:
            output_cif (str or Path): Path to the output CIF file.
            use_uni_chain_id (bool): Whether to use the unique chain IDs. Defaults to False.
            include_bonds (bool, optional): Whether to include  bonds in the cif. Defaults to False.
                                            If set to True and `array` has associated ``bonds`` , the
                                            intra-residue bonds will be written into the ``chem_comp_bond``
                                            category.
                                            Inter-residue bonds will be written into the ``struct_conn``
                                            independent of this parameter.
        """
        atom_array = copy.deepcopy(self.atom_array)

        if use_uni_chain_id:
            atom_array.del_annotation("chain_id")
            atom_array.set_annotation("chain_id", self.uni_chain_id)

        cif_writer = CIFWriter(atom_array, self.entity_poly_type)
        cif_writer.save_to_cif(
            output_cif, entry_id=self.entry_id, include_bonds=include_bonds
        )
