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


from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
from biotite.interface.rdkit import to_mol
from biotite.structure.info import residue
from biotite.structure.io.pdbx.convert import PDBX_BOND_TYPE_ID_TO_TYPE
from rdkit import Chem

from pxmeter.constants import LIGAND, POLYMER, PROTEIN_D, STD_RESIDUES
from pxmeter.data.struct import Structure


class BaseChainSequence:
    """
    Abstract base class for chain-level sequence descriptors.

    Subclasses must implement `is_polymer()` and `get_num_tokens()` to expose
    whether the chain is a polymer and how many modeling tokens it contributes
    (e.g., residues or grouped atoms).

    This type exists to unify polymer and ligand sequence representations in code
    that operates on `Sequences.sequences`.
    """

    def is_polymer(self) -> bool: ...

    def get_num_tokens(self) -> int: ...


@dataclass
class PolymerChainSequence(BaseChainSequence):
    """
    Sequence descriptor for a single polymer chain.

    Represents the primary sequence and per-residue modifications for one chain that
    belongs to a polymer entity (e.g., protein, DNA, RNA). Multiple chains may share
    the same entity sequence; this object keeps both the original entity ID and the
    originating chain ID to disambiguate instances.

    Attributes:
        sequence (str): Canonical one-letter (protein) or letter-coded (NA) sequence.
        entity_type (str): Polymer type, expected to be one of `POLYMER`.
        modifications (tuple[tuple[int, str]]): Unique set of (res_id, res_name) for
            non-standard residues observed in the structure along this chain.
        ori_entity_id (Optional[str]): Entity ID from the source structure.
        ori_chain_id (Optional[str]): Chain identifier from the source structure.

    Equality/Hashing:
        Two instances compare equal if `sequence`, `entity_type`, and the *set* of
        `modifications` are equal. Hashing is aligned with equality and allows use
        as dictionary keys or set members.
    """

    sequence: str
    entity_type: str
    modifications: tuple[tuple[int, str]] = tuple()
    ori_entity_id: Optional[str] = None
    ori_chain_id: Optional[str] = None

    def __post_init__(self):
        assert len(set(self.modifications)) == len(
            self.modifications
        ), "modifications must be unique"

    def __eq__(self, polymer_seq: "PolymerChainSequence") -> bool:
        if not isinstance(polymer_seq, PolymerChainSequence):
            return False
        return (
            self.sequence == polymer_seq.sequence
            and self.entity_type == polymer_seq.entity_type
            and set(self.modifications) == set(polymer_seq.modifications)
        )

    def __hash__(self):
        return hash((self.sequence, self.entity_type, self.modifications))

    def is_polymer(self) -> bool:
        """
        Polymers always report `True`.

        Returns:
            bool: Always `True`.
        """
        return True

    def get_num_tokens(self) -> int:
        """
        Return the token count for a polymer chain.

        Tokens are computed as the number of residues in `sequence`, with each
        modified residue contributing additional heavy atoms beyond the standard
        residue (excluding H/D).

        Returns:
            int: Total number of modeling tokens for the chain.
        """
        n_tokens = len(self.sequence)
        for _, mod_type in self.modifications:
            ccd_atom_array = residue(mod_type, allow_missing_coord=True)
            ccd_atom_array = ccd_atom_array[
                ~np.isin(ccd_atom_array.element, ["H", "D"])
            ]
            n_tokens += len(ccd_atom_array) - 1
        return n_tokens


@dataclass
class LigandChainSequence(BaseChainSequence):
    """
    Sequence-like descriptor for a ligand (non-polymer) chain.

    Encodes a chain that does not belong to a polymer entity. The chain may be
    described by a tuple of CCD codes (ordered residues/components), an external
    file path, or a SMILES string. Exactly one of these should be provided for a
    well-formed instance.

    Attributes:
        ccd_codes (Optional[tuple[str]]): Ordered component codes along the chain.
        file_path (Optional[Path]): Path to a file describing the ligand (e.g., SDF).
        smiles (Optional[str]): SMILES string describing the ligand or assembly.
        ori_entity_id (Optional[str]): Entity ID from the source structure.
        ori_chain_id (Optional[str]): Chain identifier from the source structure.
        entity_type (str): Entity type, expected to be `LIGAND`.

    Equality/Hashing:
        Equality is defined only when both sides use the same descriptor type:
        - If both have `ccd_codes`, they must be identical tuples.
        - If both have `file_path`, the paths must be identical.
        - If both have `smiles`, the strings must match exactly.
        Otherwise, they are considered unequal. Hashing follows these fields.
    """

    ccd_codes: Optional[tuple[str]] = None
    file_path: Optional[Path] = None
    smiles: Optional[str] = None
    ori_entity_id: Optional[str] = None
    ori_chain_id: Optional[str] = None
    entity_type: str = LIGAND

    def __post_init__(self):
        assert (
            self.ccd_codes or self.file_path or self.smiles
        ), "LigandChainSequence must have ccd_codes, file_path or smiles"

    def __eq__(self, ligand_seq: "LigandChainSequence") -> bool:
        if not isinstance(ligand_seq, LigandChainSequence):
            return False

        if self.ccd_codes is not None and ligand_seq.ccd_codes is not None:
            return self.ccd_codes == ligand_seq.ccd_codes
        elif self.file_path is not None and ligand_seq.file_path is not None:
            return self.file_path == ligand_seq.file_path
        elif self.smiles is not None and ligand_seq.smiles is not None:
            # Just equal if string of smiles are the same
            return self.smiles == ligand_seq.smiles
        else:
            return False

    def __hash__(self):
        return hash((self.ccd_codes, self.file_path, self.smiles))

    @staticmethod
    def _parse_ligand_file(lig_file_path: Path) -> Chem.Mol:
        if lig_file_path.endswith(".mol"):
            mol = Chem.MolFromMolFile(lig_file_path)
        elif lig_file_path.endswith(".sdf"):
            suppl = Chem.SDMolSupplier(lig_file_path)
            mol = next(suppl)
        elif lig_file_path.endswith(".pdb"):
            mol = Chem.MolFromPDBFile(lig_file_path)
        elif lig_file_path.endswith(".mol2"):
            mol = Chem.MolFromMol2File(lig_file_path)
        else:
            raise ValueError(
                f"Invalid ligand file type: .{lig_file_path.split('.')[-1]}"
            )
        return mol

    def is_polymer(self) -> bool:
        """
        Ligands are non-polymers.

        Returns:
            bool: Always `False`.
        """
        return False

    def get_num_tokens(self) -> int:
        """
        Return the non-hydrogen atom count used as ligand token size.

        The token count is computed from one of the available ligand descriptors:
        - `ccd_codes`: Sum of heavy atoms across all CCD components (using Biotite).
        - `smiles`: Heavy atom count parsed via RDKit.
        - `file_path`: Heavy atom count parsed from the ligand structure file.

        Returns:
            int: Number of heavy atoms serving as modeling tokens.

        Raises:
            ValueError: If none of `ccd_codes`, `smiles`, or `file_path` is provided.
        """
        if self.ccd_codes is not None:
            n_tokens = 0
            for ccd_code in self.ccd_codes:
                ccd_atom_array = residue(ccd_code, allow_missing_coord=True)
                ccd_atom_array = ccd_atom_array[
                    ~np.isin(ccd_atom_array.element, ["H", "D"])
                ]
                n_tokens += len(ccd_atom_array)
        elif self.smiles is not None:
            mol = Chem.MolFromSmiles(self.smiles, sanitize=False)
            n_tokens = 0
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == "H":
                    continue
                n_tokens += 1
        elif self.file_path is not None:
            mol = self._parse_ligand_file(self.file_path)
            n_tokens = 0
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == "H":
                    continue
                n_tokens += 1
        else:
            raise ValueError(
                "LigandChainSequence must have ccd_codes, file_path or smiles"
            )

        return n_tokens

    def get_smiles_by_ccd_codes(self) -> str:
        """
        Get SMILES string by CCD code.

        Returns:
            str: SMILES string of the ligand.
        """
        if len(self.ccd_codes) > 1:
            raise NotImplementedError("Only support single CCD code to get smiles")
        ccd_code = self.ccd_codes[0]
        ccd_atom_array = residue(ccd_code, allow_missing_coord=True)
        mol = to_mol(ccd_atom_array)
        return Chem.MolToSmiles(Chem.RemoveHs(mol))

    def get_smiles_by_file_path(self) -> str:
        """
        Get SMILES string by ligand file path.

        Returns:
            str: SMILES string of the ligand.
        """
        mol = self._parse_ligand_file(self.file_path)
        return Chem.MolToSmiles(Chem.RemoveHs(mol))


@dataclass
class Bond:
    """
    Explicit bond between two atoms, identified in chain/residue/atom space.

    This record captures cross- or intra-chain connectivity between atoms in the
    structure after filtering (e.g., excluding trivial same-residue polymer bonds or
    adjacent-residue backbone links). It is intended for graph construction across
    chains and residues.

    Attributes:
        chain_index_1 (int): Index into `Sequences.sequences` for the first atom.
        res_id_1 (int): Residue ID of the first atom within its chain.
        atom_name_1 (str): Atom name (e.g., 'CA', 'C1', 'SG') of the first atom.
        chain_index_2 (int): Index into `Sequences.sequences` for the second atom.
        res_id_2 (int): Residue ID of the second atom within its chain.
        atom_name_2 (str): Atom name of the second atom.
        ori_chain_id_1 (Optional[str]): Original chain ID for the first atom.
        ori_chain_id_2 (Optional[str]): Original chain ID for the second atom.
    """

    chain_index_1: int
    res_id_1: int
    atom_name_1: str

    chain_index_2: int
    res_id_2: int
    atom_name_2: str

    ori_chain_id_1: Optional[str] = None
    ori_chain_id_2: Optional[str] = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Bond):
            return False
        atom1 = (self.chain_index_1, self.res_id_1, self.atom_name_1)
        atom2 = (self.chain_index_2, self.res_id_2, self.atom_name_2)
        other1 = (other.chain_index_1, other.res_id_1, other.atom_name_1)
        other2 = (other.chain_index_2, other.res_id_2, other.atom_name_2)
        return (atom1 == other1 and atom2 == other2) or (
            atom1 == other2 and atom2 == other1
        )

    def __hash__(self) -> int:
        atom1 = (self.chain_index_1, self.res_id_1, self.atom_name_1)
        atom2 = (self.chain_index_2, self.res_id_2, self.atom_name_2)
        return hash(frozenset([atom1, atom2]))

    def is_intra_chain(self) -> bool:
        """
        Return whether both atoms are on the same chain.

        Returns:
            bool: `True` if `chain_index_1 == chain_index_2`; `False` otherwise.
        """
        return self.chain_index_1 == self.chain_index_2

    def is_inter_chain(self) -> bool:
        """
        Return whether the atoms are on different chains.

        Returns:
            bool: `True` if `chain_index_1 != chain_index_2`; `False` otherwise.
        """
        return not self.is_intra_chain()

    def is_intra_res(self) -> bool:
        """
        Return whether both atoms are within the same residue of the same chain.

        Returns:
            bool: `True` if same chain and `res_id_1 == res_id_2`; `False` otherwise.
        """
        return self.is_intra_chain() & (self.res_id_1 == self.res_id_2)

    def is_inter_res(self) -> bool:
        """
        Return whether the atoms span different residues or different chains.

        Returns:
            bool: `True` if inter-chain or (same chain and `res_id_1 != res_id_2`).
        """
        return self.is_inter_chain() | (self.res_id_1 != self.res_id_2)


@dataclass
class Sequences:
    """
    Container for all chain sequences in a structure and a filtered bond set.

    This class aggregates polymer and ligand chain sequence descriptors and a set of
    explicit bonds that connect atoms across residues and chains. It provides class
    methods to construct itself from an mmCIF file and utilities to count tokens.

    Attributes:
        sequences (tuple[Union[PolymerChainSequence, LigandChainSequence]]): Ordered chain
            descriptors. The order defines `chain_index` used by `Bond`.
        bonds (tuple[Bond]): Filtered bonds spanning intra- and inter-chain links,
            excluding trivial same-residue polymer bonds and adjacent polymer
            backbone links.

    Notes:
        The chain order is derived from the underlying structure and entity/chain
        traversal in `_get_polymer_seqs()` and `_get_non_polymer_seqs()`. The
        mapping from original chain IDs to `chain_index` is persisted in `Bond`.
    """

    name: str
    sequences: tuple[Union[PolymerChainSequence, LigandChainSequence]]
    bonds: tuple[Bond] = tuple()

    @staticmethod
    def _get_polymer_seqs(structure: Structure) -> list[PolymerChainSequence]:
        atom_array = structure.atom_array
        res_id_and_name = np.stack((atom_array.res_id, atom_array.res_name), axis=1)
        polymer_seqs = []
        for entity_id, seq in sorted(
            structure.entity_poly_seq.items(), key=lambda x: int(x[0])
        ):
            entity_type = structure.entity_poly_type[entity_id]

            mod_positions = []
            mod_types = []
            ori_chains = []
            for chain_id in np.unique(
                structure.uni_chain_id[atom_array.label_entity_id == entity_id]
            ):
                for res_id, res_name in np.unique(
                    res_id_and_name[structure.uni_chain_id == chain_id], axis=0
                ):
                    if (
                        res_name not in STD_RESIDUES
                        and int(res_id) not in mod_positions
                    ):
                        mod_positions.append(int(res_id))
                        mod_types.append(str(res_name))
                ori_chains.append(chain_id)

            for chain_id in ori_chains:
                polymer_seqs.append(
                    PolymerChainSequence(
                        sequence=seq,
                        entity_type=str(entity_type),
                        modifications=tuple(zip(mod_positions, mod_types)),
                        ori_entity_id=str(entity_id),
                        ori_chain_id=str(chain_id),
                    )
                )
        return polymer_seqs

    @staticmethod
    def _get_non_polymer_seqs(structure: Structure) -> list[LigandChainSequence]:
        ligand_seqs = []
        atom_array = structure.atom_array
        for entity_id in np.unique(atom_array.label_entity_id):
            if entity_id in structure.entity_poly_seq:
                continue

            for chain_id in np.unique(
                structure.uni_chain_id[atom_array.label_entity_id == entity_id]
            ):
                _, res_starts = np.unique(
                    atom_array.res_id[structure.uni_chain_id == chain_id],
                    return_index=True,
                )
                res_names = atom_array.res_name[structure.uni_chain_id == chain_id][
                    res_starts
                ]

                ligand_seqs.append(
                    LigandChainSequence(
                        ccd_codes=tuple(res_names.tolist()),
                        ori_entity_id=str(entity_id),
                        ori_chain_id=str(chain_id),
                    )
                )
        return ligand_seqs

    @staticmethod
    def _get_bonds_from_structure(
        structure: Structure,
        seqs_lst: list[Union[PolymerChainSequence, LigandChainSequence]],
        keep_polymer_crosslinks: bool = False,
        remove_metalc: bool = True,
    ) -> tuple[Bond]:
        # Map ori_chain_id to chain_index
        ori_chain_id_to_index = {seq.ori_chain_id: i for i, seq in enumerate(seqs_lst)}

        bond_array = structure.atom_array.bonds.as_array()

        if remove_metalc:
            biotite_metalc_type = PDBX_BOND_TYPE_ID_TO_TYPE["metalc"]
            bond_mask = bond_array[:, 2] != biotite_metalc_type
            bond_array = bond_array[bond_mask]

        is_polymer = structure.get_mask_for_given_entity_types(
            entity_types=POLYMER + [PROTEIN_D]
        )
        chain_id = structure.uni_chain_id
        res_id = structure.atom_array.res_id
        atoms1 = bond_array[:, 0]
        atoms2 = bond_array[:, 1]

        is_same_chain = chain_id[atoms1] == chain_id[atoms2]
        is_same_res = (res_id[atoms1] == res_id[atoms2]) & is_same_chain

        is_polymer_polymer = is_polymer[atoms1] & is_polymer[atoms2]
        non_pp_bond_mask = (~is_polymer_polymer) & (~is_same_res)

        if keep_polymer_crosslinks:
            is_nb_res = (
                np.abs((res_id[atoms1] - res_id[atoms2])) <= 1
            ) & is_same_chain  # include same res
            pp_bond_mask = is_polymer_polymer & (~is_nb_res)

            bond_mask = pp_bond_mask | non_pp_bond_mask
        else:
            bond_mask = non_pp_bond_mask

        bonds = []
        for bond in bond_array[bond_mask]:
            res_1 = structure.atom_array.res_id[bond[0]]
            res_2 = structure.atom_array.res_id[bond[1]]
            atom_1 = structure.atom_array.atom_name[bond[0]]
            atom_2 = structure.atom_array.atom_name[bond[1]]
            ori_chain_id_1 = structure.uni_chain_id[bond[0]]
            ori_chain_id_2 = structure.uni_chain_id[bond[1]]

            chain_index_1 = ori_chain_id_to_index[ori_chain_id_1]
            chain_index_2 = ori_chain_id_to_index[ori_chain_id_2]

            bonds.append(
                Bond(
                    chain_index_1=int(chain_index_1),
                    res_id_1=int(res_1),
                    atom_name_1=str(atom_1),
                    chain_index_2=int(chain_index_2),
                    res_id_2=int(res_2),
                    atom_name_2=str(atom_2),
                    ori_chain_id_1=ori_chain_id_1,
                    ori_chain_id_2=ori_chain_id_2,
                )
            )
        return tuple(bonds)

    @classmethod
    def from_mmcif(
        cls,
        mmcif: Union[Path, str],
        model: int = 1,
        altloc: str = "first",
        assembly_id: Optional[str] = None,
    ) -> "Sequences":
        """
        Construct a `Sequences` object from an mmCIF file.

        Loads an mmCIF into a `Structure`, optionally expands an assembly, cleans it,
        filters out non-standard polymer entity types (keeps only those in `POLYMER`),
        then gathers polymer and non-polymer chains and derives a filtered bond set.

        Args:
            mmcif (Path | str): Path to the mmCIF file.
            model (int, optional): Model ID to load. Defaults to 1.
            altloc (str, optional): AltLoc handling policy for atom selection
                (e.g., "first"). Defaults to "first".
            assembly_id (str | None, optional): If provided, expand the specified
                biological assembly before extraction. Defaults to `None`.

        Returns:
            Sequences: Container with chain descriptors and filtered bonds.

        Notes:
            After loading, the structure is cleaned and subselected to exclude entities
            whose `entity_poly_type` is not in `POLYMER`. This avoids including
            unsupported polymer types in the sequence set.
        """
        structure = Structure.from_mmcif(
            mmcif, model=model, altloc=altloc, assembly_id=assembly_id
        )
        structure = structure.clean_structure()

        # Only support protein, dna, rna
        non_std_polymer_entities = []
        for entity_id, entity_type in structure.entity_poly_type.items():
            if entity_type not in (POLYMER + [PROTEIN_D]):
                non_std_polymer_entities.append(entity_id)
        non_std_polymer_entity_mask = ~np.isin(
            structure.atom_array.label_entity_id, non_std_polymer_entities
        )
        structure = structure.select_substructure(non_std_polymer_entity_mask)

        polymer_seqs = Sequences._get_polymer_seqs(structure)
        non_polymer_seqs = Sequences._get_non_polymer_seqs(structure)
        all_seqs = polymer_seqs + non_polymer_seqs

        bonds = Sequences._get_bonds_from_structure(
            structure, all_seqs, keep_polymer_crosslinks=False, remove_metalc=True
        )
        return cls(name=structure.entry_id, sequences=tuple(all_seqs), bonds=bonds)

    def get_num_tokens(self) -> int:
        """
        Return the total token count across all chains.

        This aggregates `get_num_tokens()` from each chain descriptor, allowing the
        caller to retrieve the modeling token budget for the entire structure.

        Returns:
            int: Sum of tokens across `sequences`.
        """
        return sum(seq.get_num_tokens() for seq in self.sequences)
