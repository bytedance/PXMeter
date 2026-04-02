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
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from pxmeter.constants import DNA, LIGAND, PROTEIN, PROTEIN_D, RNA
from pxmeter.input_builder.seq import (
    Bond,
    LigandChainSequence,
    PolymerChainSequence,
    Sequences,
)
from pxmeter.input_builder.utils.unstd_res_mapping import AF3_CCD_NAME_TO_ONE_LETTER
from pxmeter.utils import int_to_letters


@dataclass
class AlpahFold3Input:
    """
    AlphaFold3-style model input wrapper.

    This class holds the minimal information needed to describe an AlphaFold3
    job: a name, random seeds, chain-level sequences (polymer or ligand),
    and optional covalent bonds between chains.

    Attributes:
        name (str): Name of the job or complex.
        seeds (list[int]): List of model seed values.
        sequences (tuple[PolymerChainSequence | LigandChainSequence]):
            Chain-level sequence descriptors.
        bonds (tuple[Bond]): Optional covalent bonds between chains.
    """

    name: str
    seeds: list[int]
    sequences: tuple[Union[PolymerChainSequence, LigandChainSequence]]
    bonds: tuple[Bond] = tuple()

    @staticmethod
    def _convert_sequence_to_dict(sequence, chain_ids) -> dict:
        """
        Convert a single sequence object into AlphaFold3 JSON format.

        Args:
            sequence (PolymerChainSequence | LigandChainSequence): Sequence
                object to convert.
            chain_ids (list[str] | tuple[str] | str): One or more chain IDs
                associated with this sequence.

        Returns:
            dict: JSON-serializable entity entry in AlphaFold3 schema.

        Raises:
            ValueError: If the ligand input is missing both CCD codes and SMILES,
                or if the entity type is unknown.
        """
        entity_chain_ids = chain_ids if len(chain_ids) > 1 else chain_ids[0]
        if sequence.is_polymer():
            entity_dict = {
                "sequence": sequence.sequence,
                "id": entity_chain_ids,
            }
            if sequence.entity_type in {PROTEIN, PROTEIN_D}:
                if sequence.modifications:
                    modifications = []
                    entity_seq = copy.deepcopy(sequence.sequence)
                    for mod_pos, mod_type in sequence.modifications:
                        modifications.append(
                            {
                                "ptmType": mod_type,
                                "ptmPosition": mod_pos,
                            }
                        )

                        entity_seq = (
                            entity_seq[: int(mod_pos) - 1]
                            + (AF3_CCD_NAME_TO_ONE_LETTER.get(mod_type, "X"))
                            + entity_seq[int(mod_pos) :]
                        )
                    entity_dict["sequence"] = entity_seq
                    entity_dict["modifications"] = modifications
                return {"protein": entity_dict}

            elif sequence.entity_type in {DNA, RNA}:
                if sequence.modifications:
                    modifications = []
                    entity_seq = copy.deepcopy(sequence.sequence)
                    for mod_pos, mod_type in sequence.modifications:
                        modifications.append(
                            {
                                "modificationType": mod_type,
                                "basePosition": mod_pos,
                            }
                        )

                        entity_seq = (
                            entity_seq[: int(mod_pos) - 1]
                            + (AF3_CCD_NAME_TO_ONE_LETTER.get(mod_type, "N"))
                            + entity_seq[int(mod_pos) :]
                        )
                    entity_dict["sequence"] = entity_seq
                    entity_dict["modifications"] = modifications

                return (
                    {"rna": entity_dict}
                    if sequence.entity_type == RNA
                    else {"dna": entity_dict}
                )

            else:
                raise ValueError(f"Unknown entity type: {sequence.entity_type}")

        else:
            # Ligand
            if sequence.ccd_codes:
                return {
                    "ligand": {
                        "ccdCodes": list(sequence.ccd_codes),
                        "id": entity_chain_ids,
                    }
                }
            elif sequence.smiles:
                return {"ligand": {"smiles": sequence.smiles, "id": entity_chain_ids}}
            else:
                raise ValueError("LigandChainSequence must have ccd_codes or smiles")

    @classmethod
    def from_sequences(cls, sequences: Sequences, seeds: list[int]):
        """
        Create an AlphaFold3 input from a Sequences container.

        Args:
            sequences (Sequences): Container with chain sequences and bonds.
            seeds (list[int]): List of model seeds.


        Returns:
            AlpahFold3Input: Constructed AlphaFold3 input object.
        """
        return cls(
            name=sequences.name,
            seeds=seeds,
            sequences=sequences.sequences,
            bonds=sequences.bonds,
        )

    @classmethod
    def from_json(cls, json_dict: dict) -> "AlpahFold3Input":
        """
        Reconstruct an AlphaFold3 input from a JSON dict.

        Args:
            json_dict (dict): Parsed AlphaFold3-style JSON dict.

        Returns:
            AlpahFold3Input: Reconstructed input object.

        Raises:
            ValueError: If an unknown entity type is encountered or a ligand
                entry lacks both CCD codes and SMILES.
        """
        name = json_dict["name"]
        seeds = json_dict["modelSeeds"]
        entity_type_mapping = {
            "protein": PROTEIN,
            "rna": RNA,
            "dna": DNA,
            "ligand": LIGAND,
        }

        seq_obj_lst = []
        chain_id_to_idx = {}
        for af3_seq in json_dict["sequences"]:
            for af3_entity_type, info_dict in af3_seq.items():
                chain_ids = info_dict["id"]
                if isinstance(chain_ids, str):
                    chain_ids = [chain_ids]

                if af3_entity_type == "ligand":
                    for chain_id in chain_ids:
                        if "ccdCodes" in info_dict:
                            seq_obj_lst.append(
                                LigandChainSequence(
                                    ccd_codes=tuple(info_dict["ccdCodes"]),
                                    ori_chain_id=chain_id,
                                )
                            )
                        elif "smiles" in info_dict:
                            seq_obj_lst.append(
                                LigandChainSequence(
                                    smiles=info_dict["smiles"],
                                    ori_chain_id=chain_id,
                                )
                            )
                        else:
                            raise ValueError(
                                "Ligand chain must have ccdCodes or smiles"
                            )
                        chain_id_to_idx[chain_id] = len(seq_obj_lst) - 1

                else:
                    modifications = []
                    if af3_entity_type == "protein":
                        if "modifications" in info_dict:
                            for mod in info_dict["modifications"]:
                                modifications.append(
                                    (int(mod["ptmPosition"]), mod["ptmType"])
                                )
                    elif af3_entity_type in {"rna", "dna"}:
                        if "modifications" in info_dict:
                            for mod in info_dict["modifications"]:
                                modifications.append(
                                    (int(mod["basePosition"]), mod["modificationType"])
                                )
                    else:
                        raise ValueError(f"Unknown entity type: {af3_entity_type}")

                    for chain_id in chain_ids:
                        seq_obj_lst.append(
                            PolymerChainSequence(
                                sequence=info_dict["sequence"],
                                entity_type=entity_type_mapping[af3_entity_type],
                                modifications=tuple(modifications),
                                ori_chain_id=chain_id,
                            )
                        )
                        chain_id_to_idx[chain_id] = len(seq_obj_lst) - 1

        bond_obj_lst = []
        if "bondedAtomPairs" in json_dict:
            for bond in json_dict["bondedAtomPairs"]:
                chain_index_1 = chain_id_to_idx[bond[0][0]]
                chain_index_2 = chain_id_to_idx[bond[1][0]]
                res_id_1 = int(bond[0][1])
                atom_name_1 = bond[0][2]
                res_id_2 = int(bond[1][1])
                atom_name_2 = bond[1][2]
                bond_obj_lst.append(
                    Bond(
                        chain_index_1=chain_index_1,
                        res_id_1=res_id_1,
                        atom_name_1=atom_name_1,
                        chain_index_2=chain_index_2,
                        res_id_2=res_id_2,
                        atom_name_2=atom_name_2,
                    )
                )

        return cls(
            name=name,
            seeds=seeds,
            sequences=tuple(seq_obj_lst),
            bonds=tuple(bond_obj_lst),
        )

    @classmethod
    def from_json_file(cls, json_f: Union[Path, str]) -> "AlpahFold3Input":
        """
        Load an AlphaFold3 input from a JSON file.

        Args:
            json_f (Path | str): Path to an AlphaFold3-style JSON file.

        Returns:
            AlpahFold3Input: Reconstructed input object.
        """
        with open(json_f, "r", encoding="utf-8") as f:
            json_dict = json.load(f)
        return cls.from_json(json_dict)

    def to_json(self) -> dict:
        """
        Serialize the AlphaFold3 input into a JSON dict.

        Sequences are grouped by identical sequence objects and emitted with
        appropriate chain IDs. Bonds are converted into the AlphaFold3
        ``bondedAtomPairs`` format.

        Returns:
            dict: JSON-serializable AlphaFold3 job specification.
        """
        # meta info
        af3_info_dict = {
            "name": self.name,
            "modelSeeds": self.seeds,
            "dialect": "alphafold3",
            "version": 3,
        }

        idx_to_chain_id = {}
        seqs_to_chain_ids = defaultdict(list)

        # Determine unique chain ID for each sequence
        # Prioritize ori_chain_id, generate unique ID if missing.
        existing_chain_ids = {
            seq.ori_chain_id for seq in self.sequences if seq.ori_chain_id
        }
        next_chain_id_int = 1

        for idx, seq in enumerate(self.sequences):
            if seq.ori_chain_id:
                chain_id = seq.ori_chain_id
            else:
                while True:
                    candidate = int_to_letters(next_chain_id_int)
                    next_chain_id_int += 1
                    if candidate not in existing_chain_ids:
                        chain_id = candidate
                        existing_chain_ids.add(chain_id)
                        break
            
            idx_to_chain_id[idx] = chain_id
            seqs_to_chain_ids[seq].append(chain_id)

        af3_seqs_lst = []
        for seq, chain_ids in seqs_to_chain_ids.items():
            af3_seqs_lst.append(
                AlpahFold3Input._convert_sequence_to_dict(seq, chain_ids)
            )
        af3_info_dict["sequences"] = af3_seqs_lst

        af3_bonds_lst = []
        for bond in self.bonds:
            af3_bonds_lst.append(
                [
                    [
                        idx_to_chain_id[bond.chain_index_1],
                        bond.res_id_1,
                        bond.atom_name_1,
                    ],
                    [
                        idx_to_chain_id[bond.chain_index_2],
                        bond.res_id_2,
                        bond.atom_name_2,
                    ],
                ]
            )
        if self.bonds:
            af3_info_dict["bondedAtomPairs"] = af3_bonds_lst
        return af3_info_dict

    def to_json_file(self, json_file: Path):
        """
        Write the AlphaFold3 input to a JSON file.

        Args:
            json_file (Path): Output JSON file path.
        """
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=4)

    def to_sequences(self) -> Sequences:
        """
        Convert the AlphaFold3 input back to a Sequences container.

        Returns:
            Sequences: Container with chain sequences and bonds.
        """
        return Sequences(name=self.name, sequences=self.sequences, bonds=self.bonds)

    def get_num_tokens(self) -> int:
        """
        Return the total token count for all sequences.

        Tokens are computed as the sum of tokens across `sequences`.

        Returns:
            int: Total number of modeling tokens for all sequences.
        """
        return self.to_sequences().get_num_tokens()
