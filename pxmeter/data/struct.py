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
import functools
from pathlib import Path
from typing import Optional, Sequence, Union

import biotite.structure as bs
import numpy as np
from biotite.structure import AtomArray, CellList
from biotite.structure.info import residue
from biotite.structure.info.ccd import get_from_ccd

from pxmeter.constants import (
    CRYSTALLIZATION_AIDS,
    CRYSTALLIZATION_METHODS,
    DNA,
    NUC_BACKBONE,
    NUC_BACKBONE_REP_ATOM,
    POLYMER,
    PROTEIN,
    PROTEIN_BACKBONE,
    PROTEIN_BACKBONE_REP_ATOM,
    RNA,
)
from pxmeter.data.parser import MMCIFParser
from pxmeter.data.utils import get_unique_atom_id, get_unique_chain_id
from pxmeter.data.writer import CIFWriter


@functools.lru_cache(maxsize=None)
def _get_leaving_groups(res_name: str) -> dict:
    try:
        names = get_from_ccd("chem_comp_atom", res_name, "atom_id").as_array()
        flags = get_from_ccd(
            "chem_comp_atom", res_name, "pdbx_leaving_atom_flag"
        ).as_array()
        a1 = get_from_ccd("chem_comp_bond", res_name, "atom_id_1").as_array()
        a2 = get_from_ccd("chem_comp_bond", res_name, "atom_id_2").as_array()
    except KeyError:
        return {}

    is_leave = dict(zip(names, flags == "Y"))
    graph = {str(n): [] for n in names}
    for u, v in zip(a1, a2):
        graph[str(u)].append(str(v))
        graph[str(v)].append(str(u))

    central_to_leaving = {}
    for atom in names:
        if not is_leave.get(str(atom), False):
            continue
        q, visited, central = [str(atom)], {str(atom)}, None
        while q:
            curr = q.pop(0)
            if not is_leave.get(curr, False):
                central = curr
                break
            for nxt in graph.get(curr, []):
                if nxt not in visited:
                    visited.add(nxt)
                    q.append(nxt)
        if central:
            central_to_leaving.setdefault(central, []).append(str(atom))
    return central_to_leaving


@functools.lru_cache(maxsize=None)
def _get_ccd_residue_no_h(res_name: str):
    try:
        ccd_atoms = residue(res_name)
    except KeyError:
        return None
    return ccd_atoms[~np.isin(ccd_atoms.element, ["H", "D"])]


@functools.lru_cache(maxsize=None)
def _get_remove_atom_names_by_pos(res_name, pos_type):
    leaving = _get_leaving_groups(res_name)
    remove = []
    if pos_type in (1, 2):  # not N-term
        remove.extend(leaving.get("N", []) + leaving.get("P", []))
    if pos_type in (0, 1):  # not C-term
        remove.extend(leaving.get("C", []) + leaving.get("O3'", []))
    return remove


def _get_remove_atom_names(res_name, res_id, seq_len):
    # This is not easily cached if res_id varies, but we can cache
    # based on position type: 0=N-term, 1=middle, 2=C-term
    pos_type = 0 if res_id == 1 else (2 if res_id == seq_len else 1)
    return _get_remove_atom_names_by_pos(res_name, pos_type)


def _create_missing(
    ccd_atoms, missing_names, chain_id, entity_id, res_id, res_name, template
):
    mask = np.isin(ccd_atoms.atom_name, missing_names)
    if not np.any(mask):
        return None
    new_arr = ccd_atoms[mask].copy()
    new_arr.coord[:] = 0.0

    # Use faster annotation copy
    new_cats = new_arr.get_annotation_categories()
    temp_cats = template.get_annotation_categories()

    for annot in temp_cats:
        if annot not in new_cats:
            new_arr.add_annotation(annot, dtype=template.get_annotation(annot).dtype)

    if "res_id" in new_cats or "res_id" in temp_cats:
        new_arr.res_id[:] = res_id
    if "res_name" in new_cats or "res_name" in temp_cats:
        new_arr.res_name[:] = res_name
    if "chain_id" in new_cats or "chain_id" in temp_cats:
        new_arr.chain_id[:] = chain_id
    if "label_entity_id" in new_cats or "label_entity_id" in temp_cats:
        new_arr.set_annotation("label_entity_id", np.array([entity_id] * len(new_arr)))

    for annot in list(new_cats):
        if annot not in temp_cats:
            new_arr.del_annotation(annot)
    return new_arr


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
    poly_res_names: dict[str, list[str]] = dataclasses.field(default_factory=dict)
    valid_mask: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.valid_mask is None:
            self.valid_mask = np.ones(len(self.atom_array), dtype=bool)

    def __repr__(self):
        return f"Structure(atom_array={self.atom_array[:30]})"

    @classmethod
    def from_mmcif(
        cls,
        mmcif: Union[Path, str],
        model: int = 1,
        altloc: str = "first",
        assembly_id: Optional[str] = None,
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
            poly_res_names=cif_parser.get_poly_res_names(atom_array),
        )

    @classmethod
    def from_atom_array(
        cls,
        atom_array: AtomArray,
        entity_poly_seq: dict[str, str],
        entity_poly_type: dict[str, str],
        entry_id: str = "",
        exptl_methods: tuple[str] = tuple(),
        cif_block: dict = None,
        poly_res_names: dict[str, list[str]] = None,
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
        return cls(
            atom_array=atom_array,
            entity_poly_seq=entity_poly_seq,
            entity_poly_type=entity_poly_type,
            uni_chain_id=get_unique_chain_id(atom_array),
            uni_atom_id=get_unique_atom_id(atom_array),
            entry_id=entry_id,
            exptl_methods=exptl_methods,
            cif_block=cif_block,
            poly_res_names=poly_res_names or {},
        )

    def get_release_date(self) -> str:
        """
        Get first release date.

        Returns:
            str: yyyy-mm-dd
        """
        assert self.cif_block is not None, "cif_block is None."
        return MMCIFParser.get_release_date(self.cif_block)

    def get_resolution(self) -> float:
        """
        Get resolution for X-ray and cryoEM.
        Some methods don't have resolution, set as -1.0.

        Returns:
            float: resolution (set to -1.0 if not found).
        """
        assert self.cif_block is not None, "cif_block is None."
        return MMCIFParser.get_resolution(self.cif_block)

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

    def get_mask_for_given_entity_types(
        self, entity_types: Union[list[str], str]
    ) -> np.ndarray:
        """
        Get a mask of atoms given entity types.

        Args:
            entity_types (list[str] | str): Entity types.

        Returns:
            np.ndarray: Mask of atoms.
        """
        if isinstance(entity_types, str):
            entity_types = [entity_types]

        entity_ids = []
        for k, v in self.entity_poly_type.items():
            if v in entity_types:
                entity_ids.append(k)

        mask = np.isin(self.atom_array.label_entity_id, entity_ids)
        return mask

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
        self, mask: Sequence[Union[int, bool]], reset_uni_id: bool = False
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

        if self.valid_mask is not None:
            valid_mask = self.valid_mask[mask]
        else:
            valid_mask = np.ones(len(substructure_atom_array), dtype=bool)

        return dataclasses.replace(
            self,
            atom_array=substructure_atom_array,
            uni_chain_id=uni_chain_id,
            uni_atom_id=uni_atom_id,
            valid_mask=valid_mask,
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

    def get_ligand_polymer_bonds(self) -> np.ndarray:
        """
        Get bonds between the bonded ligand and its parent chain.

        Returns:
            np.ndarray: bond records between the bonded ligand and its parent chain.
                        e.g. np.array([[atom1, atom2, bond_order]...])
        """
        atom_array = self.atom_array

        if atom_array.bonds is None:
            return np.empty((0, 3), dtype=int)

        bond_array = atom_array.bonds.as_array()

        polymer_mask = np.isin(
            atom_array.label_entity_id, list(self.entity_poly_type.keys())
        )

        lig_mask = ~polymer_mask

        idx_i = bond_array[:, 0]
        idx_j = bond_array[:, 1]

        lig_polymer_bond_indices = np.where(
            (lig_mask[idx_i] & polymer_mask[idx_j])
            | (lig_mask[idx_j] & polymer_mask[idx_i])
        )[0]
        if lig_polymer_bond_indices.size == 0:
            # no ligand-polymer bonds
            lig_polymer_bonds = np.empty((0, 3), dtype=int)
        else:
            # np.array([[atom1, atom2, bond_order], ...])
            lig_polymer_bonds = bond_array[lig_polymer_bond_indices]
        return lig_polymer_bonds

    def get_backbone_atom_masks(self, only_rep_atom: bool = True) -> np.ndarray:
        """
        Get masks for protein and nucleic acid backbone atoms.

        Args:
            only_rep_atom (bool, optional): Whether to only include
                one representative atom per residue. Defaults to True.

        Returns:
            np.ndarray: Bool masks for protein and nucleic acid backbone atoms.
        """
        protein_mask = self.get_mask_for_given_entity_types(entity_types=PROTEIN)
        nuc_mask = self.get_mask_for_given_entity_types(entity_types=[RNA, DNA])

        if only_rep_atom:
            protein_backbone_mask = protein_mask & (
                self.atom_array.atom_name == PROTEIN_BACKBONE_REP_ATOM
            )
            nuc_backbone_mask = nuc_mask & (
                self.atom_array.atom_name == NUC_BACKBONE_REP_ATOM
            )
        else:
            protein_backbone_mask = protein_mask & np.isin(
                self.atom_array.atom_name, PROTEIN_BACKBONE
            )
            nuc_backbone_mask = nuc_mask & np.isin(
                self.atom_array.atom_name, NUC_BACKBONE
            )
        backbone_mask = protein_backbone_mask | nuc_backbone_mask
        return backbone_mask

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
            non_polymer_mask = ~self.get_mask_for_given_entity_types(
                entity_types=POLYMER
            )
            crys_aids_mask = self._get_crystallization_aids_mask()
            mask &= ~(non_polymer_mask & crys_aids_mask)

        return self.select_substructure(mask)

    def add_missing_atoms(self):
        """
        Add missing residues and atoms for polymers in the Structure based on `entity_poly_seq` and CCD.
        The coordinates of the newly added atoms will be set to (0, 0, 0).
        Original atom order is preserved, and missing atoms are inserted at their correct sequence positions.
        Also correctly maintains atomic bonds and removes terminal leaving atoms from intermediate residues.
        """

        starts = self.get_residue_starts(add_exclusive_stop=True)
        if len(starts) <= 1:
            return self

        # Pre-compute all required CCD residues to avoid repeated lookups
        unique_res_names = set()
        for res_names_list in self.poly_res_names.values():
            unique_res_names.update(res_names_list)

        ccd_cache = {}
        for r_name in unique_res_names:
            ccd_cache[r_name] = _get_ccd_residue_no_h(r_name)

        new_arrays, new_u_chains, new_u_atoms, new_valid_masks = [], [], [], []
        last_processed_res_id, o2n_amap = {}, {}
        new_global_start = 0

        last_block_idx = {
            self.uni_chain_id[starts[i]]: i for i in range(len(starts) - 1)
        }

        for i in range(len(starts) - 1):
            start, stop = starts[i], starts[i + 1]
            res_atoms = self.atom_array[start:stop]
            chain_id, u_chain_id = res_atoms.chain_id[0], self.uni_chain_id[start]
            res_id, entity_id = res_atoms.res_id[0], res_atoms.label_entity_id[0]

            if entity_id in self.poly_res_names:
                seq_list = self.poly_res_names[entity_id]
                expected_id = last_processed_res_id.get(u_chain_id, 0) + 1

                def add_missing_res(start_id, end_id):
                    nonlocal new_global_start
                    for eid in range(start_id, end_id):
                        if eid > len(seq_list):
                            break
                        ename = seq_list[eid - 1]
                        ccd_atoms = ccd_cache.get(ename)
                        if ccd_atoms is not None:
                            rm_names = _get_remove_atom_names(ename, eid, len(seq_list))
                            missing = [
                                n for n in ccd_atoms.atom_name if n not in rm_names
                            ]
                            arr = _create_missing(
                                ccd_atoms,
                                missing,
                                chain_id,
                                entity_id,
                                eid,
                                ename,
                                self.atom_array,
                            )
                            if arr is not None:
                                new_arrays.append(arr)
                                new_u_chains.extend([u_chain_id] * len(arr))
                                new_u_atoms.extend(
                                    [f"{eid}_{ename}_{n}" for n in arr.atom_name]
                                )
                                new_valid_masks.append(np.zeros(len(arr), dtype=bool))
                                new_global_start += len(arr)

                # 1. Fill missing residues before current res_id
                add_missing_res(expected_id, res_id)

                # 2. Append current residue's existing atoms (filtered)
                ename = (
                    seq_list[res_id - 1]
                    if res_id <= len(seq_list)
                    else res_atoms.res_name[0]
                )
                rm_names = _get_remove_atom_names(ename, res_id, len(seq_list))
                keep_mask = ~np.isin(res_atoms.atom_name, rm_names)
                filtered_atoms = res_atoms[keep_mask]

                # 2. Add current residue's existing atoms (filtered) and missing atoms in CCD order
                if res_id <= len(seq_list):
                    ccd_atoms = ccd_cache.get(ename)
                else:
                    ccd_atoms = None

                if ccd_atoms is not None:
                    # CCD defines the target atomic order
                    target_names = [n for n in ccd_atoms.atom_name if n not in rm_names]

                    missing_names = [
                        n for n in target_names if n not in filtered_atoms.atom_name
                    ]
                    missing_arr = _create_missing(
                        ccd_atoms,
                        missing_names,
                        chain_id,
                        entity_id,
                        res_id,
                        ename,
                        self.atom_array,
                    )

                    # Merge existing and missing atoms following target_names order
                    if missing_arr is not None or len(filtered_atoms) > 0:
                        merged_arrays = []
                        merged_u_chains = []
                        merged_u_atoms = []
                        merged_valid_masks = []

                        orig_offsets = np.where(keep_mask)[0]
                        orig_name_to_idx = {
                            n: i for i, n in enumerate(filtered_atoms.atom_name)
                        }
                        missing_name_to_idx = (
                            {n: i for i, n in enumerate(missing_arr.atom_name)}
                            if missing_arr is not None
                            else {}
                        )

                        for name in target_names:
                            if name in orig_name_to_idx:
                                idx = orig_name_to_idx[name]
                                merged_arrays.append(filtered_atoms[idx : idx + 1])
                                merged_u_chains.append(
                                    self.uni_chain_id[start:stop][keep_mask][idx]
                                )
                                merged_u_atoms.append(
                                    self.uni_atom_id[start:stop][keep_mask][idx]
                                )
                                if self.valid_mask is not None:
                                    merged_valid_masks.append(
                                        self.valid_mask[start:stop][keep_mask][idx]
                                    )
                                else:
                                    merged_valid_masks.append(True)

                                o2n_amap[start + orig_offsets[idx]] = new_global_start
                            elif name in missing_name_to_idx:
                                idx = missing_name_to_idx[name]
                                merged_arrays.append(missing_arr[idx : idx + 1])
                                merged_u_chains.append(u_chain_id)
                                merged_u_atoms.append(f"{res_id}_{ename}_{name}")
                                merged_valid_masks.append(False)
                            new_global_start += 1

                        # Note: we might have existing atoms that are NOT in CCD but also NOT removed by rm_names.
                        # Usually this shouldn't happen for standard residues, but just in case, append them at the end.
                        extra_names = [
                            n for n in filtered_atoms.atom_name if n not in target_names
                        ]
                        for name in extra_names:
                            idx = orig_name_to_idx[name]
                            merged_arrays.append(filtered_atoms[idx : idx + 1])
                            merged_u_chains.append(
                                self.uni_chain_id[start:stop][keep_mask][idx]
                            )
                            merged_u_atoms.append(
                                self.uni_atom_id[start:stop][keep_mask][idx]
                            )
                            if self.valid_mask is not None:
                                merged_valid_masks.append(
                                    self.valid_mask[start:stop][keep_mask][idx]
                                )
                            else:
                                merged_valid_masks.append(True)

                            o2n_amap[start + orig_offsets[idx]] = new_global_start
                            new_global_start += 1

                        if merged_arrays:
                            merged_arr_all = merged_arrays[0]
                            for arr in merged_arrays[1:]:
                                merged_arr_all += arr
                            new_arrays.append(merged_arr_all)
                            new_u_chains.extend(merged_u_chains)
                            new_u_atoms.extend(merged_u_atoms)
                            new_valid_masks.append(
                                np.array(merged_valid_masks, dtype=bool)
                            )
                else:
                    # Fallback for non-standard residues or when ccd_atoms is not found
                    if len(filtered_atoms) > 0:
                        new_arrays.append(filtered_atoms)
                        new_u_chains.extend(self.uni_chain_id[start:stop][keep_mask])
                        new_u_atoms.extend(self.uni_atom_id[start:stop][keep_mask])
                        if self.valid_mask is not None:
                            new_valid_masks.append(
                                self.valid_mask[start:stop][keep_mask]
                            )
                        else:
                            new_valid_masks.append(
                                np.ones(len(filtered_atoms), dtype=bool)
                            )
                        for orig_offset, new_offset in zip(
                            np.where(keep_mask)[0], range(len(filtered_atoms))
                        ):
                            o2n_amap[start + orig_offset] = (
                                new_global_start + new_offset
                            )
                        new_global_start += len(filtered_atoms)

                last_processed_res_id[u_chain_id] = res_id

                # 4. Flush trailing missing residues
                if i == last_block_idx[u_chain_id]:
                    add_missing_res(res_id + 1, len(seq_list) + 1)
            else:
                # Non-polymer
                new_arrays.append(res_atoms)
                new_u_chains.extend(self.uni_chain_id[start:stop])
                new_u_atoms.extend(self.uni_atom_id[start:stop])
                if self.valid_mask is not None:
                    new_valid_masks.append(self.valid_mask[start:stop])
                else:
                    new_valid_masks.append(np.ones(len(res_atoms), dtype=bool))
                for orig_offset, new_offset in zip(
                    range(len(res_atoms)), range(len(res_atoms))
                ):
                    o2n_amap[start + orig_offset] = new_global_start + new_offset
                new_global_start += len(res_atoms)

        if new_arrays:
            # Flatten AtomArrays
            new_arr_all = new_arrays[0]
            for arr in new_arrays[1:]:
                new_arr_all += arr

            # Map bonds
            if self.atom_array.bonds is not None:
                old_bonds = self.atom_array.bonds.as_array()
                o2n_arr = np.full(len(self.atom_array), -1, dtype=int)
                o2n_arr[list(o2n_amap.keys())] = list(o2n_amap.values())
                valid_mask = (o2n_arr[old_bonds[:, 0]] != -1) & (
                    o2n_arr[old_bonds[:, 1]] != -1
                )
                filtered_bonds = old_bonds[valid_mask]
                mapped_bonds = np.zeros_like(filtered_bonds)
                mapped_bonds[:, 0], mapped_bonds[:, 1], mapped_bonds[:, 2] = (
                    o2n_arr[filtered_bonds[:, 0]],
                    o2n_arr[filtered_bonds[:, 1]],
                    filtered_bonds[:, 2],
                )
                new_arr_all.bonds = bs.BondList(len(new_arr_all), mapped_bonds)

            self.atom_array = new_arr_all
            self.uni_chain_id = np.array(new_u_chains)
            self.uni_atom_id = np.array(new_u_atoms)
            self.valid_mask = np.concatenate(new_valid_masks)

            # Add inter-residue bonds
            new_starts = self.get_residue_starts(add_exclusive_stop=True)
            new_inter_bonds = set()
            for i in range(len(new_starts) - 2):
                start, stop = new_starts[i], new_starts[i + 1]
                nxt_start, nxt_stop = new_starts[i + 1], new_starts[i + 2]
                if (
                    self.atom_array.chain_id[start]
                    == self.atom_array.chain_id[nxt_start]
                    and self.atom_array.res_id[nxt_start]
                    == self.atom_array.res_id[start] + 1
                ):
                    c_idx = np.where(
                        np.isin(self.atom_array.atom_name[start:stop], ["C", "O3'"])
                    )[0]
                    n_idx = np.where(
                        np.isin(
                            self.atom_array.atom_name[nxt_start:nxt_stop], ["N", "P"]
                        )
                    )[0]
                    if len(c_idx) > 0 and len(n_idx) > 0:
                        abs_c, abs_n = start + c_idx[0], nxt_start + n_idx[0]
                        if (
                            self.atom_array.bonds is not None
                            and abs_n in self.atom_array.bonds.get_bonds(abs_c)[0]
                        ):
                            continue
                        new_inter_bonds.add((abs_c, abs_n, 1))

            if new_inter_bonds:
                bonds_to_merge = bs.BondList(
                    len(self.atom_array),
                    np.array(list(new_inter_bonds), dtype=np.uint32),
                )
                self.atom_array.bonds = (
                    bonds_to_merge
                    if self.atom_array.bonds is None
                    else self.atom_array.bonds.merge(bonds_to_merge)
                )

        return self

    def to_cif(
        self,
        output_cif: Union[str, Path],
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

        cif_writer = CIFWriter(
            atom_array, self.entity_poly_type, atom_array_output_mask=self.valid_mask
        )

        cif_writer.save_to_cif(
            output_cif, entry_id=self.entry_id, include_bonds=include_bonds
        )
