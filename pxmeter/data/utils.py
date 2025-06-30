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

from collections import Counter

import biotite.sequence as seq
import biotite.sequence.align as align
import networkx as nx
import numpy as np
from biotite.structure import Atom, AtomArray, get_chain_starts, get_residue_starts
from rdkit import Chem

from pxmeter.constants import PRO_STD_RESIDUES_ONE_LETTER


def get_unique_chain_id(atom_array: AtomArray) -> np.ndarray:
    """
    Generate unique chain IDs for each chain in the AtomArray.

    This function takes an AtomArray and assigns a unique string ID to each chain.
    The ID is based on the original chain ID, with a numerical suffix added to ensure uniqueness.
    Example: [A, B, A, B, C] -> [A, B, A.1, B.1, C]

    Args:
        atom_array (AtomArray): The input AtomArray containing the atomic coordinates and metadata.

    Returns:
        np.ndarray: An array of unique chain IDs, with the same length as the input AtomArray.
    """
    chain_ids = np.zeros(len(atom_array), dtype="<U8")  # <U8 for very large structure
    chain_starts = get_chain_starts(atom_array, add_exclusive_stop=True)

    chain_counter = Counter()
    for start, stop in zip(chain_starts[:-1], chain_starts[1:]):
        ori_chain_id = atom_array.chain_id[start]
        cnt = chain_counter[ori_chain_id]
        if cnt == 0:
            new_chain_id = ori_chain_id
        else:
            new_chain_id = f"{ori_chain_id}.{chain_counter[ori_chain_id]}"

        chain_ids[start:stop] = new_chain_id
        chain_counter[ori_chain_id] += 1

    assert "" not in chain_ids
    return chain_ids


def get_unique_atom_id(atom_array: AtomArray) -> np.ndarray:
    """
    Generate unique atom identifiers by combining residue IDs, residue names, and atom names.

    Args:
        res_id (np.ndarray): An array of residue IDs.
        res_name (np.ndarray): An array of residue names.
        atom_name (np.ndarray): An array of atom names.

    Returns:
        np.ndarray: An array of unique atom identifiers, each formed by concatenating
                    the corresponding residue ID, residue name, and atom name with underscores.
    """
    stacked = np.column_stack(
        (atom_array.res_id, atom_array.res_name, atom_array.atom_name)
    )
    unique_atom_id = np.array(["_".join(map(str, row)) for row in stacked])
    return unique_atom_id


def get_chain_ccd_seq(atom_array: AtomArray) -> dict[str, str]:
    """
    Generate a dictionary mapping chain IDs to CCD sequences.

    This function takes an AtomArray and extracts the CCD sequences for each chain.
    The CCD sequence is a string representation of the residue names in the chain,
    separated by underscores.

    Args:
        atom_array (AtomArray): The input AtomArray containing the atomic coordinates and metadata.

    Returns:
        dict[str, str]: A dictionary where the keys are chain IDs and the values are CCD sequences.
    """
    chain_id_to_ccd_seq = {}
    for chain_id in np.unique(atom_array.chain_id):
        chain_array = atom_array[atom_array.chain_id == chain_id]
        res_starts = get_residue_starts(chain_array)
        ccd_array = chain_array.res_name[res_starts]
        ccd_seq = "_".join(ccd_array)
        chain_id_to_ccd_seq[chain_id] = ccd_seq
    return chain_id_to_ccd_seq


def get_seq_alignment_score(seq1: str, seq2: str, seq_type: str = "protein") -> tuple:
    """
    Calculate the alignment score between two sequences.
    The global alignment based on the Needleman-Wunsch algorithm.

    Args:
        seq1 (str): The first sequence string.
        seq2 (str): The second sequence string.
        seq_type (str): The type of sequences to align.
            One of {"protein", "nuc"} for protein or nucleotide sequences. Default is "protein".

    Returns:
        tuple: A tuple containing:
            - alignments (list): A list of optimal alignments.
            - sequence_identity (float): The sequence identity of the first optimal alignment.

    Raises:
        NotImplementedError: If the seq_type is not "protein" or "nuc".
    """

    def _std_prot_seq(prot_seq):
        std_seq = ""
        for i in prot_seq:
            if i in PRO_STD_RESIDUES_ONE_LETTER:
                std_seq += i
            elif i == "U":
                # U -> SEC -> CYS -> C
                std_seq += "C"
            else:
                std_seq += "X"
        return std_seq

    def _std_nuc_seq(nuc_seq):
        std_seq = ""
        for i in nuc_seq:
            if i in {"A", "T", "C", "G"}:
                std_seq += i
            elif i == "U":
                std_seq += "T"
            else:
                std_seq += "N"
        return std_seq

    if seq_type == "protein":
        seq1 = seq.ProteinSequence(_std_prot_seq(seq1))
        seq2 = seq.ProteinSequence(_std_prot_seq(seq2))
        matrix = align.SubstitutionMatrix.std_protein_matrix()
    elif seq_type == "nuc":
        seq1 = seq.NucleotideSequence(_std_nuc_seq(seq1))
        seq2 = seq.NucleotideSequence(_std_nuc_seq(seq2))
        matrix = align.SubstitutionMatrix.std_nucleotide_matrix()
    else:
        raise NotImplementedError(f"seq_type {seq_type} is not supported")

    alignments = align.align_optimal(seq1, seq2, matrix, gap_penalty=-10, local=False)
    return alignments[0], align.get_sequence_identity(alignments[0])


def get_inter_residue_bonds(atom_array: AtomArray) -> np.ndarray:
    """
    Get inter residue bonds by checking chain_id and res_id

    Args:
        atom_array (AtomArray): Biotite AtomArray, must have chain_id and res_id

    Returns:
        np.ndarray: inter residue bonds, shape = (n,2)
    """
    if atom_array.bonds is None:
        return np.array([])
    idx_i = atom_array.bonds._bonds[:, 0]
    idx_j = atom_array.bonds._bonds[:, 1]
    chain_id_diff = atom_array.chain_id[idx_i] != atom_array.chain_id[idx_j]
    res_id_diff = atom_array.res_id[idx_i] != atom_array.res_id[idx_j]
    diff_mask = chain_id_diff | res_id_diff
    inter_residue_bonds = atom_array.bonds._bonds[diff_mask]
    inter_residue_bonds = inter_residue_bonds[:, :2]  # remove bond type
    return inter_residue_bonds


def create_single_atom_array(atom: Atom) -> AtomArray:
    """
    Create a single atom AtomArray from a single atom.
    The biotite.structure.array function would change the dtype of the annotation categories.
    For example, if the Atom has annotation category "label_entity_id", the dtype of Atom is "<U2",
    but the dtype of AtomArray is "<U1".

    Args:
        atom (Atom): Biotite Atom object

    Returns:
        AtomArray: A AtomArray object with a single atom.
    """
    array = AtomArray(1)
    # Add all (also optional) annotation categories
    for name in atom._annot.keys():
        if name in array._annot:
            array.del_annotation(name)
        array.set_annotation(name, [atom._annot[name]])

    # Add all atoms to AtomArray
    array._coord[0] = atom.coord
    return array


def rdkit_mol_to_nx_graph(mol: Chem.Mol) -> nx.Graph:
    """
    Convert an RDKit molecule to a NetworkX graph representation.

    Args:
        mol (Chem.Mol): Input RDKit molecule object

    Returns:
        nx.Graph: NetworkX graph where nodes represent atoms and edges represent bonds
    """
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(
            atom.GetIdx(),
            atomic_num=atom.GetAtomicNum(),
            is_aromatic=atom.GetIsAromatic(),
            atom_symbol=atom.GetSymbol(),
        )

    for bond in mol.GetBonds():
        G.add_edge(
            bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type=bond.GetBondType()
        )

    return G


def get_mol_graph_matches(mol_graph1: nx.Graph, mol_graph2: nx.Graph) -> list[dict]:
    """
    Find all isomorphisms between subgraph of mol_graph1 and mol_graph2.

    Args:
        mol_graph1 (nx.Graph): Source molecular graph (typically larger subgraph)
        mol_graph2 (nx.Graph): Target molecular graph (typically smaller supergraph)

    Returns:
        list[dict]: List of node mapping dictionaries where each dictionary represents
        a subgraph isomorphism (mapping from mol_graph1 node indices to mol_graph2 node indices)

    Note:
        Uses atomic number comparison for node matching, ignoring other atom properties
    """
    isomatcher = nx.algorithms.isomorphism.GraphMatcher(
        mol_graph1,
        mol_graph2,
        node_match=lambda x, y: x["atomic_num"] == y["atomic_num"],
    )
    return list(isomatcher.subgraph_isomorphisms_iter())
