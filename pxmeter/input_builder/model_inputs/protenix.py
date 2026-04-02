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
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from pxmeter.constants import DNA, LIGAND, PROTEIN, PROTEIN_D, RNA
from pxmeter.input_builder.seq import (
    Bond,
    LigandChainSequence,
    PolymerChainSequence,
    Sequences,
)
from pxmeter.utils import int_to_letters


@dataclass
class ProtenixEntity:
    """
    Protenix entity wrapper for a polymer or ligand sequence.

    Each entity groups identical chains (same `sequence`) and tracks:
    - an integer entity ID used by Protenix,
    - the chain-level sequence descriptor,
    - how many copies of this entity appear in the system,
    - optional MSA information for protein entities.

    Attributes:
        entity_id (int): 1-based entity identifier in the Protenix input.
        sequence (Union[PolymerChainSequence, LigandChainSequence]): Chain sequence
            descriptor for this entity.
        count (int): Number of chain copies for this entity.
        msa_path (Optional[Path]): Optional path to a precomputed MSA directory.
        pairing_db (str): Name of the pairing database for MSA (e.g. "uniref100").
        ori_chain_ids (tuple[str]): Original chain IDs (one per copy).
    """

    entity_id: int
    sequence: Union[PolymerChainSequence, LigandChainSequence]
    count: int

    msa_path: Optional[Path] = None
    pairing_db: str = "uniref100"
    ori_chain_ids: tuple[str] = tuple()

    def add_ori_chain_id(self, ori_chain_id: str):
        """
        Append a new original chain ID to this entity.

        Args:
            ori_chain_id (str): Original chain identifier to add.
        """
        self.ori_chain_ids = tuple(list(self.ori_chain_ids) + [ori_chain_id])

    def to_dict(self) -> dict:
        """
        Serialize the entity into the Protenix JSON sequence format.

        Returns:
            dict: A JSON-serializable dict describing a protein, RNA, DNA
            or ligand entity in the Protenix schema.
        """
        if self.sequence.is_polymer():
            entity_dict = {
                "sequence": self.sequence.sequence,
                "count": self.count,
            }
            if self.ori_chain_ids and all(
                isinstance(x, str) for x in self.ori_chain_ids
            ):
                entity_dict["id"] = list(self.ori_chain_ids)

            if self.sequence.entity_type in [PROTEIN, PROTEIN_D]:
                # # U -> SEC -> CYS -> C
                entity_dict["sequence"] = entity_dict["sequence"].replace("U", "C")

                if self.sequence.modifications:
                    entity_dict["modifications"] = [
                        {
                            "ptmType": f"CCD_{mod_type}",
                            "ptmPosition": mod_pos,
                        }
                        for mod_pos, mod_type in self.sequence.modifications
                    ]

                if self.msa_path is not None:
                    entity_dict["msa"] = {
                        "precomputed_msa_dir": str(self.msa_path),
                        "pairing_db": self.pairing_db,
                    }
                return {"proteinChain": entity_dict}

            elif self.sequence.entity_type == RNA:
                if self.sequence.modifications:
                    entity_dict["modifications"] = [
                        {
                            "modificationType": f"CCD_{mod_type}",
                            "basePosition": mod_pos,
                        }
                        for mod_pos, mod_type in self.sequence.modifications
                    ]
                return {"rnaSequence": entity_dict}

            elif self.sequence.entity_type == DNA:
                if self.sequence.modifications:
                    entity_dict["modifications"] = [
                        {
                            "modificationType": f"CCD_{mod_type}",
                            "basePosition": mod_pos,
                        }
                        for mod_pos, mod_type in self.sequence.modifications
                    ]
                return {"dnaSequence": entity_dict}
            else:
                raise ValueError(f"Unknown entity type: {self.sequence.entity_type}")

        else:
            # Ligand
            if self.sequence.ccd_codes:
                lig_seq = "CCD_" + "_".join(self.sequence.ccd_codes)
            elif self.sequence.file_path:
                lig_seq = "FILE_" + str(self.sequence.file_path)
            elif self.sequence.smiles:
                lig_seq = self.sequence.smiles
            else:
                raise ValueError(
                    "LigandChainSequence must have ccd_codes, file_path or smiles"
                )

            return {"ligand": {"ligand": lig_seq, "count": self.count}}


@dataclass
class ProtenixBond:
    """
    Covalent bond between two entities/copies in Protenix schema.

    Attributes:
        entity_id_1 (int): ID of the first entity.
        copy_id_1 (int): Copy index of the first entity (1-based).
        res_id_1 (int): Residue position on the first entity.
        atom_name_1 (str): Atom name on the first entity.
        entity_id_2 (int): ID of the second entity.
        copy_id_2 (int): Copy index of the second entity (1-based).
        res_id_2 (int): Residue position on the second entity.
        atom_name_2 (str): Atom name on the second entity.
    """

    entity_id_1: int
    copy_id_1: int
    res_id_1: int
    atom_name_1: str

    entity_id_2: int
    copy_id_2: int
    res_id_2: int
    atom_name_2: str

    def to_dict(self) -> dict:
        """
        Serialize the bond into the Protenix JSON covalent bond format.

        Returns:
            dict: A JSON-serializable dict describing one covalent bond.
        """
        return {
            "entity1": str(self.entity_id_1),
            "copy1": self.copy_id_1,
            "position1": str(self.res_id_1),
            "atom1": self.atom_name_1,
            "entity2": str(self.entity_id_2),
            "copy2": self.copy_id_2,
            "position2": str(self.res_id_2),
            "atom2": self.atom_name_2,
        }


@dataclass
class ProtenixInput:
    """
    Top-level Protenix job description.

    Wraps a name, seeds, entities, and covalent bonds into a structure that
    can be serialized to or reconstructed from the Protenix JSON schema.

    Attributes:
        name (str): Job or complex name.
        seeds (list[int]): List of random seeds for model runs.
        sequences (tuple[ProtenixEntity]): All entities in the system.
        bonds (tuple[ProtenixBond]): All covalent bonds between entities/copies.
    """

    name: str
    seeds: list[int]
    sequences: tuple[ProtenixEntity]
    bonds: tuple[ProtenixBond] = tuple()

    @classmethod
    def from_sequences(cls, sequences: Sequences, seeds: list[int]) -> "ProtenixInput":
        """
        Build a ProtenixInput from a Sequences object.

        Chains with identical sequence descriptors are grouped into a single
        ProtenixEntity, and covalent bonds are mapped into entity/copy space.

        Args:
            sequences (Sequences): Chain sequences and bonds extracted from
                structural data.
            seeds (list[int]): List of model seeds.

        Returns:
            ProtenixInput: Protenix-ready input structure.
        """
        seqs_to_entities = {}
        chain_to_entity_and_copy = {}

        # Determine unique chain ID for each sequence
        # Prioritize ori_chain_id, generate unique ID if missing.
        existing_chain_ids = {
            seq.ori_chain_id for seq in sequences.sequences if seq.ori_chain_id
        }

        seq_idx_to_chain_id = {}
        next_chain_id_int = 1

        for seq_idx, seq in enumerate(sequences.sequences):
            if seq.ori_chain_id:
                chain_id = seq.ori_chain_id
            else:
                # Generate a unique chain ID
                while True:
                    candidate = int_to_letters(next_chain_id_int)
                    next_chain_id_int += 1
                    if candidate not in existing_chain_ids:
                        chain_id = candidate
                        existing_chain_ids.add(chain_id)
                        break
            seq_idx_to_chain_id[seq_idx] = chain_id

        entity_id = 0
        for seq_idx, seq in enumerate(sequences.sequences):
            chain_id = seq_idx_to_chain_id[seq_idx]

            if seq not in seqs_to_entities:
                entity_id += 1
                seqs_to_entities[seq] = ProtenixEntity(
                    entity_id=entity_id,
                    sequence=seq,
                    count=1,
                    ori_chain_ids=(chain_id,),
                )
                # copy 1
                chain_to_entity_and_copy[seq_idx] = (entity_id, 1)

            else:
                seqs_to_entities[seq].count += 1
                seqs_to_entities[seq].add_ori_chain_id(chain_id)
                copy_id = len(seqs_to_entities[seq].ori_chain_ids)
                chain_to_entity_and_copy[seq_idx] = (entity_id, copy_id)

        bonds = []
        for bond in sequences.bonds:
            entity_id_1, copy_id_1 = chain_to_entity_and_copy[bond.chain_index_1]
            entity_id_2, copy_id_2 = chain_to_entity_and_copy[bond.chain_index_2]

            bonds.append(
                ProtenixBond(
                    entity_id_1=entity_id_1,
                    copy_id_1=copy_id_1,
                    res_id_1=bond.res_id_1,
                    atom_name_1=bond.atom_name_1,
                    entity_id_2=entity_id_2,
                    copy_id_2=copy_id_2,
                    res_id_2=bond.res_id_2,
                    atom_name_2=bond.atom_name_2,
                )
            )
        return cls(
            name=sequences.name,
            seeds=seeds,
            sequences=tuple(seqs_to_entities.values()),
            bonds=tuple(bonds),
        )

    @classmethod
    def from_json(cls, json_dict: dict) -> "ProtenixInput":
        """
        Reconstruct a ProtenixInput from a Protenix-style JSON dict.

        Args:
            json_dict (dict): Parsed JSON matching the Protenix input schema.

        Returns:
            ProtenixInput: Reconstructed input object.
        """
        name = json_dict["name"]
        seeds = json_dict.get("modelSeeds", [])

        entity_type_mapping = {
            "proteinChain": PROTEIN,
            "rnaSequence": RNA,
            "dnaSequence": DNA,
            "ligand": LIGAND,
        }

        all_seqs = []
        chain_id_int = 1
        for entity_id_int, seq_dict in enumerate(json_dict["sequences"]):
            entity_id = str(entity_id_int + 1)

            for json_entity_type, entity_dict in seq_dict.items():
                entity_type = entity_type_mapping[json_entity_type]
                count = entity_dict["count"]

                # Determine chain IDs (from 'id' field or auto-generated)
                json_ids = entity_dict.get("id")
                if json_ids and len(json_ids) == count:
                    ori_chain_ids = tuple(json_ids)
                else:
                    ori_chain_ids = tuple(
                        [int_to_letters(chain_id_int + i) for i in range(count)]
                    )

                if entity_type in [PROTEIN, RNA, DNA]:
                    sequence = entity_dict["sequence"]
                    modifications = []
                    for m in entity_dict.get("modifications", []):
                        if entity_type == PROTEIN:
                            modifications.append(
                                (
                                    int(m["ptmPosition"]),
                                    m["ptmType"].replace("CCD_", ""),
                                )
                            )
                        else:
                            modifications.append(
                                (
                                    int(m["basePosition"]),
                                    m["modificationType"].replace("CCD_", ""),
                                )
                            )

                    seq_obj = PolymerChainSequence(
                        sequence=sequence,
                        entity_type=entity_type,
                        modifications=tuple(modifications),
                        ori_chain_id=ori_chain_ids[0] if ori_chain_ids else None,
                    )

                else:
                    ccd_codes = None
                    file_path = None
                    smiles = None

                    lig_str = entity_dict["ligand"]
                    if lig_str.startswith("CCD_"):
                        ccd_codes = lig_str[4:].split("_")
                    elif lig_str.startswith("FILE_"):
                        file_path = Path(lig_str[5:])

                    else:
                        smiles = lig_str

                    seq_obj = LigandChainSequence(
                        ccd_codes=tuple(ccd_codes) if ccd_codes else None,
                        file_path=file_path,
                        smiles=smiles,
                        ori_chain_id=ori_chain_ids[0] if ori_chain_ids else None,
                        ori_entity_id=entity_id,
                    )

                entity_obj = ProtenixEntity(
                    entity_id=int(entity_id),
                    sequence=seq_obj,
                    count=count,
                    ori_chain_ids=ori_chain_ids,
                )
                chain_id_int += count
                all_seqs.append(entity_obj)

        bonds = []
        for bond_dict in json_dict.get("covalent_bonds", []):
            copy1 = bond_dict.get("copy1", bond_dict.get("left_copy"))
            copy2 = bond_dict.get("copy2", bond_dict.get("right_copy"))

            entity1 = int(bond_dict.get("entity1", bond_dict.get("left_entity")))
            entity2 = int(bond_dict.get("entity2", bond_dict.get("right_entity")))
            position1 = int(bond_dict.get("position1", bond_dict.get("left_position")))
            position2 = int(bond_dict.get("position2", bond_dict.get("right_position")))
            atom1 = bond_dict.get("atom1", bond_dict.get("left_atom"))
            atom2 = bond_dict.get("atom2", bond_dict.get("right_atom"))

            if copy1 is None or copy2 is None:
                count1 = all_seqs[int(entity1) - 1].count
                count2 = all_seqs[int(entity2) - 1].count

                assert count1 == count2

                for c in range(1, count1 + 1):
                    bonds.append(
                        ProtenixBond(
                            entity_id_1=entity1,
                            copy_id_1=c,
                            res_id_1=position1,
                            atom_name_1=atom1,
                            entity_id_2=entity2,
                            copy_id_2=c,
                            res_id_2=position2,
                            atom_name_2=atom2,
                        )
                    )

            else:

                bonds.append(
                    ProtenixBond(
                        entity_id_1=entity1,
                        copy_id_1=copy1,
                        res_id_1=position1,
                        atom_name_1=atom1,
                        entity_id_2=entity2,
                        copy_id_2=copy2,
                        res_id_2=position2,
                        atom_name_2=atom2,
                    )
                )

        return cls(
            name=name,
            seeds=seeds,
            sequences=tuple(all_seqs),
            bonds=tuple(bonds),
        )

    @classmethod
    def from_json_file(cls, json_f: Union[Path, str]) -> "ProtenixInput":
        """
        Load a ProtenixInput from a JSON file.

        The file is expected to contain a list with a single Protenix entry.

        Args:
            json_f (Path | str): Path to a JSON file.

        Returns:
            ProtenixInput: Reconstructed input object.

        Warns:
            UserWarning: If the file contains more than one top-level item.
        """
        with open(json_f, "r", encoding="utf-8") as f:
            json_lst = json.load(f)

        if len(json_lst) != 1:
            logging.warning("ProtenixInput json file %s only load one item.", json_f)

        json_dict = json_lst[0]
        return cls.from_json(json_dict)

    @classmethod
    def from_json_file_multi(cls, json_f: Union[Path, str]) -> list["ProtenixInput"]:
        """
        Load all ProtenixInput entries from a JSON file containing a list of entries.

        Args:
            json_f (Path | str): Path to a JSON file.

        Returns:
            list[ProtenixInput]: List of reconstructed input objects.
        """
        with open(json_f, "r", encoding="utf-8") as f:
            json_lst = json.load(f)

        inputs = []
        for json_dict in json_lst:
            inputs.append(cls.from_json(json_dict))
        return inputs

    @staticmethod
    def _merge_covalent_bonds(
        covalent_bonds: list[dict], all_entity_counts: dict[str, int]
    ) -> list[dict]:
        """
        Merge covalent bonds with same entity and position.

        Args:
            covalent_bonds (list[dict]): A list of covalent bond dicts.
            all_entity_counts (dict[str, int]): A dict of entity id to chain count.

        Returns:
            list[dict]: A list of merged covalent bond dicts.
        """
        bonds_recorder = defaultdict(list)
        bonds_entity_counts = {}
        for bond_dict in covalent_bonds:
            bond_unique_string = []
            entity_counts = (
                all_entity_counts[str(bond_dict["entity1"])],
                all_entity_counts[str(bond_dict["entity2"])],
            )
            for i in [1, 2]:
                for j in ["entity", "position", "atom"]:
                    k = f"{j}{i}"
                    bond_unique_string.append(str(bond_dict[k]))
            bond_unique_string = "_".join(bond_unique_string)
            bonds_recorder[bond_unique_string].append(bond_dict)
            bonds_entity_counts[bond_unique_string] = entity_counts

        merged_covalent_bonds = []
        for k, v in bonds_recorder.items():
            left_counts = bonds_entity_counts[k][0]
            right_counts = bonds_entity_counts[k][1]
            if left_counts == right_counts == len(v):
                bond_dict_copy = copy.deepcopy(v[0])
                del bond_dict_copy["copy1"]
                del bond_dict_copy["copy2"]
                merged_covalent_bonds.append(bond_dict_copy)
            else:
                merged_covalent_bonds.extend(v)
        return merged_covalent_bonds

    def to_json(self) -> dict:
        """
        Serialize the ProtenixInput into a Protenix JSON dict.

        Sequences and covalent bonds are converted to the schema expected by
        the Protenix service, with merged bond entries when applicable.

        Returns:
            dict: JSON-serializable dict representing the full Protenix job.
        """
        all_entity_counts = {str(seq.entity_id): seq.count for seq in self.sequences}
        json_dict = {
            "name": self.name,
            "modelSeeds": list(self.seeds),
            "sequences": [seq.to_dict() for seq in self.sequences],
        }
        if self.bonds:
            json_dict["covalent_bonds"] = self._merge_covalent_bonds(
                [bond.to_dict() for bond in self.bonds], all_entity_counts
            )
        return json_dict

    def to_json_file(self, json_file: Path):
        """
        Write the ProtenixInput to a JSON file.

        The output file will contain a single-element list wrapping the job dict.

        Args:
            json_file (Path): Output JSON file path.
        """
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump([self.to_json()], f, indent=4)

    def to_sequences(self) -> Sequences:
        """
        Convert ProtenixInput back to Sequences and Bond objects.

        Expands entity/copy definitions into a flat list of chain sequences and
        reconstructs chain-level bonds suitable for downstream structural code.

        Returns:
            Sequences: Sequence container with chain descriptors and bonds.
        """
        seqs = []
        id_to_index = {}
        for entity in self.sequences:
            for i, copy_id in enumerate(range(1, entity.count + 1)):
                id_to_index[(entity.entity_id, copy_id)] = len(seqs)

                # Clone sequence to avoid sharing mutable state and to assign distinct ori_chain_id
                new_seq = copy.deepcopy(entity.sequence)
                if i < len(entity.ori_chain_ids):
                    new_seq.ori_chain_id = entity.ori_chain_ids[i]

                seqs.append(new_seq)

        bonds = []
        for px_bond in self.bonds:
            chain1 = id_to_index[(px_bond.entity_id_1, px_bond.copy_id_1)]
            chain2 = id_to_index[(px_bond.entity_id_2, px_bond.copy_id_2)]

            bonds.append(
                Bond(
                    chain_index_1=chain1,
                    res_id_1=px_bond.res_id_1,
                    atom_name_1=px_bond.atom_name_1,
                    chain_index_2=chain2,
                    res_id_2=px_bond.res_id_2,
                    atom_name_2=px_bond.atom_name_2,
                )
            )

        return Sequences(name=self.name, sequences=tuple(seqs), bonds=tuple(bonds))

    def get_num_tokens(self) -> int:
        """
        Return the total token count for all sequences.

        Tokens are computed as the sum of tokens across `sequences`.

        Returns:
            int: Total number of modeling tokens for all sequences.
        """
        return self.to_sequences().get_num_tokens()
