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
import functools
import logging
from typing import Any

import gemmi
import numpy as np
from biotite.structure import AtomArray, get_residue_starts
from pdbeccdutils.core import ccd_reader
from rdkit import Chem
from rdkit.Geometry import Point3D

from pxmeter.configs.data_config import COMPONENTS_FILE
from pxmeter.data.utils import (
    get_inter_residue_bonds,
    get_mol_graph_matches,
    rdkit_mol_to_nx_graph,
)

logging.getLogger("rdkit").setLevel(logging.ERROR)


@functools.lru_cache
def _gemmi_load_ccd_cif() -> gemmi.cif.Document:
    """
    Load the CCD components file using the gemmi library.

    Returns:
        gemmi.cif.Document: gemmi ccd components file
    """
    return gemmi.cif.read(str(COMPONENTS_FILE))


@functools.lru_cache
def get_ccd_mol_from_cif(ccd_code: str) -> Chem.Mol:
    """
    Retrieve a molecular object from a CCD CIF file using the given CCD code.

    Args:
        ccd_code (str): The CCD code of the molecule to retrieve.

    Returns:
        mol (Chem.Mol): The RDKit molecule object corresponding to the given CCD code.
                              Returns None if the CCD code is not found.
    """
    ccd_cif = _gemmi_load_ccd_cif()

    try:
        ccd_block = ccd_cif[ccd_code]
    except KeyError:
        return
    ccd_reader_result = ccd_reader._parse_pdb_mmcif(ccd_block, sanitize=True)
    mol = ccd_reader_result.component.mol
    mol.atom_map = {atom.GetProp("name"): atom.GetIdx() for atom in mol.GetAtoms()}
    return mol


def get_ccd_mol_by_atom_names(ccd_code: str, atom_names: list[str] = None) -> Chem.Mol:
    """
    Get a mol from a CCD code.
    If atom_names is None, return the whole mol, otherwise return a substructure.

    Args:
        ccd_code (str): CCD code
        atom_names (list[str], optional): A list of atom names.
                                          Atoms not in the list will be removed from the mol.
                                          Defaults to None（return the whole mol).

    Returns:
        Chem.Mol: A rdkit mol of the CCD code, with the atom property "atom_name".
    """
    mol = copy.deepcopy(get_ccd_mol_from_cif(ccd_code))
    atom_name_to_idx = mol.atom_map
    idx_to_atom_name = {v: k for k, v in atom_name_to_idx.items()}

    for atom in mol.GetAtoms():
        atom.SetProp("atom_name", idx_to_atom_name[atom.GetIdx()])

    if atom_names is None:
        return mol

    atoms_to_remove = [
        atom_name_to_idx[i] for i in atom_name_to_idx if i not in atom_names
    ]
    atoms_to_remove.sort(reverse=True)

    edit_mol = Chem.EditableMol(mol)
    for atom_idx in atoms_to_remove:
        edit_mol.RemoveAtom(atom_idx)
    new_mol = edit_mol.GetMol()
    return new_mol


def _set_coord_by_chain_atom_array(
    mol: Chem.Mol, chain_atom_array: AtomArray
) -> Chem.Mol:
    """
    Set the coordinates of the atoms in the molecule to the coordinates of the atoms in the chain_atom_array.

    Args:
        mol (Chem.Mol): The molecule whose atom coordinates are to be set.
        chain_atom_array (AtomArray): An array containing the coordinates and properties of atoms.

    Returns:
        Chem.Mol: The molecule with updated atom coordinates.
    """
    conf = mol.GetConformer()
    for atom in mol.GetAtoms():
        atom_name = atom.GetProp("atom_name")
        res_id = atom.GetProp("res_id")
        coord = chain_atom_array.coord[
            (chain_atom_array.res_id == int(res_id))
            & (chain_atom_array.atom_name == atom_name)
        ][0]
        x, y, z = [float(i) for i in coord]
        conf.SetAtomPosition(atom.GetIdx(), Point3D(x, y, z))
    return mol


def get_ccd_mol_from_chain_atom_array(chain_atom_array: AtomArray) -> Chem.Mol:
    """
    Convert from a chain of AtomArrays to an RDKit Mol according to CCD Codes.
    The single bond between the residues will be retained.

    Args:
        chain_atom_array (AtomArray): a chain of AtomArrays.

    Returns:
        Chem.Mol: an RDKit Mol.
    """
    mols = []
    all_atom_names = []
    res_starts = get_residue_starts(chain_atom_array, add_exclusive_stop=True)
    for start, stop in zip(res_starts[:-1], res_starts[1:]):
        res_name = chain_atom_array.res_name[start]
        res_id = chain_atom_array.res_id[start]
        atom_names = chain_atom_array.atom_name[start:stop]
        mol = get_ccd_mol_by_atom_names(res_name, atom_names)
        for a in mol.GetAtoms():
            a.SetProp("res_id", str(res_id))
            a.SetProp("res_name", res_name)

        mols.append(mol)
        all_atom_names.append(atom_names)

    combo_mols = mols[0]
    for mol in mols[1:]:
        combo_mols = Chem.CombineMols(combo_mols, mol)

    inter_res_bonds = get_inter_residue_bonds(chain_atom_array)
    if inter_res_bonds.shape[0] == 0:
        Chem.SanitizeMol(combo_mols)
        combo_mols = _set_coord_by_chain_atom_array(
            mol=combo_mols, chain_atom_array=chain_atom_array
        )
        return combo_mols

    res_id_atom_name_to_idx = {}
    for atom in combo_mols.GetAtoms():
        res_id = atom.GetProp("res_id")
        atom_name = atom.GetProp("atom_name")
        atom_key = f"{res_id}_{atom_name}"
        res_id_atom_name_to_idx[atom_key] = atom.GetIdx()

    edcombo = Chem.EditableMol(combo_mols)
    for bond in inter_res_bonds:
        atom_idx1, atom_idx2 = bond
        res_id1, res_id2 = (
            chain_atom_array.res_id[atom_idx1],
            chain_atom_array.res_id[atom_idx2],
        )
        atom_name1, atom_name2 = (
            chain_atom_array.atom_name[atom_idx1],
            chain_atom_array.atom_name[atom_idx2],
        )
        mol_atom_1 = res_id_atom_name_to_idx[f"{res_id1}_{atom_name1}"]
        mol_atom_2 = res_id_atom_name_to_idx[f"{res_id2}_{atom_name2}"]
        edcombo.AddBond(mol_atom_1, mol_atom_2, order=Chem.rdchem.BondType.SINGLE)
    new_mol = edcombo.GetMol()
    Chem.SanitizeMol(new_mol)
    new_mol = _set_coord_by_chain_atom_array(
        mol=new_mol, chain_atom_array=chain_atom_array
    )
    return new_mol


@functools.lru_cache
def get_ccd_perm_info(ccd_code: str) -> dict[str, Any]:
    """
    Get permutation information for a given CCD code.

    This function retrieves the molecular structure from the CCD code, sanitizes it,
    and computes the atom permutation array. If the molecule is invalid or has no atoms,
    an empty dictionary is returned.

    Args:
        ccd_code (str): The CCD code representing the molecule.

    Returns:
        dict[str, Any]: A dictionary containing the permutation information with the following keys:
            - "atom_map": The dict of atom name to atom index of the molecule.
            - "perm_array": A numpy array of atom permutations, shape (n_atom_wo_h, n_perm).
    """
    perm_info = {}
    mol = copy.deepcopy(get_ccd_mol_from_cif(ccd_code))
    if mol is None:
        return perm_info

    elif mol.GetNumAtoms() == 0:  # eg: "UNL"
        return perm_info
    else:
        mol_graph = rdkit_mol_to_nx_graph(mol)

        # remove H
        removed_nodes = [
            node for node in mol_graph.nodes if mol_graph.nodes[node]["atomic_num"] == 1
        ]
        for node in removed_nodes:
            mol_graph.remove_node(node)

        matches = get_mol_graph_matches(mol_graph, mol_graph)[:1000]

        # re-index after removing H
        reverted_old_atom_map = {v: k for k, v in mol.atom_map.items()}
        old_map_to_new_map = {}
        atom_map = {}
        for new_idx, node in enumerate(mol_graph.nodes):
            old_map_to_new_map[node] = new_idx
            atom_map[reverted_old_atom_map[node]] = new_idx

        perm = []
        for match in matches:
            sorted_result = sorted(
                match.items(), key=lambda x: old_map_to_new_map[x[0]]
            )
            match_values = [old_map_to_new_map[i[1]] for i in sorted_result]
            perm.append(match_values)
        perm = np.array(perm)

        perm_info["atom_map"] = atom_map
        perm_info["perm_array"] = perm.T
        return (
            perm_info  # np.ndarray[int]: atom permutation, shape:(n_atom_wo_h, n_perm)
        )
