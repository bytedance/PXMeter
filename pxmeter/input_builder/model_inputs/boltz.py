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

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import gemmi
import yaml

from pxmeter.constants import DNA, LIGAND, PROTEIN, PROTEIN_D, RNA
from pxmeter.input_builder.seq import (
    Bond,
    LigandChainSequence,
    PolymerChainSequence,
    Sequences,
)
from pxmeter.utils import int_to_letters

logger = logging.getLogger(__name__)


def gemmi_replace_seq_3to1(seq: PolymerChainSequence) -> PolymerChainSequence:
    """
    Replace pxmeter 3 to 1 letter mapping to gemmi 3 to 1 letter mapping.

    Args:
        seq: PolymerChainSequence object.

    Returns:
        PolymerChainSequence object with replaced sequence.
    """
    new_sequece = [i for i in seq.sequence]
    for mod_pos, mod_res_name in seq.modifications:
        old = new_sequece[mod_pos - 1]
        new = gemmi.one_letter_code([mod_res_name])[0]
        if old != new:
            logger.debug(
                f"Change {mod_res_name}->{old} to {mod_res_name}->{new} for consistent with boltz (use gemmi.one_letter_code)."
            )
            new_sequece[mod_pos - 1] = new
    return PolymerChainSequence(
        entity_type=seq.entity_type,
        sequence="".join(new_sequece),
        modifications=seq.modifications,
        ori_entity_id=seq.ori_entity_id,
        ori_chain_id=seq.ori_chain_id,
    )


@dataclass
class BoltzInput:
    """
    Boltz input wrapper with sequences and covalent bonds.
    """

    name: str
    sequences: tuple[Union[PolymerChainSequence, LigandChainSequence]]
    bonds: tuple[Bond] = tuple()

    @staticmethod
    def _convert_sequence_to_dict(sequence, chain_ids) -> dict:
        """
        Convert a sequence object to Boltz YAML dictionary.

        Args:
            sequence: A PolymerChainSequence or LigandChainSequence.
            chain_ids: List of chain IDs or a single chain ID string.

        Returns:
            dict: Boltz-style sequence entry.

        Raises:
            ValueError: If entity type or ligand specification is unsupported.
        """
        entity_type_mapping = {
            PROTEIN: "protein",
            PROTEIN_D: "protein",
            RNA: "rna",
            DNA: "dna",
            LIGAND: "ligand",
        }
        if sequence.entity_type not in entity_type_mapping:
            raise ValueError(f"Unknown entity type: {sequence.entity_type}")

        entity_chain_ids = chain_ids if len(chain_ids) > 1 else chain_ids[0]
        if sequence.is_polymer():
            entity_dict = {
                "sequence": sequence.sequence,
                "id": entity_chain_ids,
            }

            if sequence.modifications:
                modifications = []
                for mod_pos, mod_type in sequence.modifications:
                    modifications.append(
                        {
                            "ccd": mod_type,
                            "position": mod_pos,
                        }
                    )

                entity_dict["modifications"] = modifications
            return {entity_type_mapping[sequence.entity_type]: entity_dict}

        else:
            # Ligand
            if sequence.ccd_codes:
                if len(sequence.ccd_codes) > 1:
                    raise ValueError("Boltz not support multiple CCD codes for ligand.")
                else:
                    return {
                        "ligand": {
                            "ccd": sequence.ccd_codes[0],
                            "id": entity_chain_ids,
                        }
                    }
            elif sequence.smiles:
                return {"ligand": {"smiles": sequence.smiles, "id": entity_chain_ids}}
            else:
                raise ValueError("LigandChainSequence must have ccd_codes or smiles")

    @classmethod
    def from_sequences(cls, sequences: Sequences):
        """
        Build a BoltzInput from a Sequences container.

        Args:
            sequences (Sequences): Input sequences and bonds.

        Returns:
            BoltzInput: Constructed BoltzInput object.
        """
        new_seqs = []
        for seq in sequences.sequences:
            if seq.is_polymer():
                new_seqs.append(gemmi_replace_seq_3to1(seq))
            else:
                new_seqs.append(seq)

        return cls(
            name=sequences.name,
            sequences=new_seqs,
            bonds=sequences.bonds,
        )

    @classmethod
    def from_yaml(cls, yaml_dict: dict, name: str = "None") -> "BoltzInput":
        """
         Create a BoltzInput from a parsed Boltz YAML dict.

        Args:
            yaml_dict (dict): Parsed YAML content of a Boltz input.
            name (str, optional): Logical name for this input. Defaults to "None".

        Returns:
            BoltzInput: Constructed BoltzInput object.
        """
        entity_type_mapping = {
            "protein": PROTEIN,
            "rna": RNA,
            "dna": DNA,
            "ligand": LIGAND,
        }

        seq_obj_lst = []
        chain_id_to_idx = {}
        for boltz_seq in yaml_dict["sequences"]:
            for entity_type, info_dict in boltz_seq.items():
                chain_ids = info_dict["id"]
                if isinstance(chain_ids, str):
                    chain_ids = [chain_ids]

                if entity_type == "ligand":
                    for chain_id in chain_ids:
                        if "ccd" in info_dict:
                            seq_obj_lst.append(
                                LigandChainSequence(
                                    ccd_codes=tuple([info_dict["ccd"]]),
                                )
                            )
                        elif "smiles" in info_dict:
                            seq_obj_lst.append(
                                LigandChainSequence(
                                    smiles=info_dict["smiles"],
                                )
                            )
                        else:
                            raise ValueError(
                                "Ligand chain must have ccdCodes or smiles"
                            )
                        chain_id_to_idx[chain_id] = len(seq_obj_lst) - 1

                else:
                    modifications = []
                    if entity_type in {"protein", "rna", "dna"}:
                        if "modifications" in info_dict:
                            for mod in info_dict["modifications"]:
                                modifications.append((int(mod["position"]), mod["ccd"]))
                    else:
                        raise ValueError(f"Unknown entity type: {entity_type}")

                    for chain_id in chain_ids:
                        seq_obj_lst.append(
                            PolymerChainSequence(
                                sequence=info_dict["sequence"],
                                entity_type=entity_type_mapping[entity_type],
                                modifications=tuple(modifications),
                            )
                        )
                        chain_id_to_idx[chain_id] = len(seq_obj_lst) - 1

        bond_obj_lst = []
        for constraint in yaml_dict.get("constraints", []):
            for constraint_type, info_dict in constraint.items():
                if constraint_type == "bond":
                    chain_index_1 = chain_id_to_idx[info_dict["atom1"][0]]
                    chain_index_2 = chain_id_to_idx[info_dict["atom2"][0]]
                    res_id_1 = int(info_dict["atom1"][1])
                    atom_name_1 = info_dict["atom1"][2]
                    res_id_2 = int(info_dict["atom2"][1])
                    atom_name_2 = info_dict["atom2"][2]

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
                else:
                    # Not support other constraint types
                    continue

        return cls(
            name=name,
            sequences=tuple(seq_obj_lst),
            bonds=tuple(bond_obj_lst),
        )

    @classmethod
    def from_yaml_file(cls, yaml_f: Union[Path, str]) -> "BoltzInput":
        """
        Load a BoltzInput from a YAML file.

        Args:
            yaml_f (Path | str): Path to the Boltz YAML file.

        Returns:
            BoltzInput: Constructed BoltzInput object.
        """
        yaml_f = Path(yaml_f)
        with open(yaml_f, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_yaml(data, name=yaml_f.stem)

    def to_yaml(self) -> dict:
        """
        Serialize the BoltzInput to a Boltz-style YAML dict.

        Returns:
            dict: YAML-serializable dictionary with `sequences` and `constraints`.
        """
        boltz_info_dict = {}

        idx_to_chain_id = {}
        seqs_to_chain_ids = defaultdict(list)
        for idx, seq in enumerate(self.sequences):
            if not seq.is_polymer():
                if seq.ccd_codes and len(seq.ccd_codes) > 1:
                    logger.debug(
                        "Boltz not support multiple CCD codes for ligand. "
                        "Skip ligand: %s",
                        seq.ccd_codes,
                    )
                    continue
            chain_id = int_to_letters(idx + 1)
            idx_to_chain_id[idx] = chain_id
            seqs_to_chain_ids[seq].append(chain_id)

        seqs_lst = []
        for seq, chain_ids in seqs_to_chain_ids.items():
            boltz_seq_dict = BoltzInput._convert_sequence_to_dict(seq, chain_ids)
            if not boltz_seq_dict:
                continue
            seqs_lst.append(boltz_seq_dict)
        boltz_info_dict["sequences"] = seqs_lst

        bonds_lst = []
        for bond in self.bonds:
            chain_id_1 = idx_to_chain_id.get(bond.chain_index_1)
            chain_id_2 = idx_to_chain_id.get(bond.chain_index_2)
            if chain_id_1 is None or chain_id_2 is None:
                continue

            bonds_lst.append(
                {
                    "bond": {
                        "atom1": [
                            chain_id_1,
                            bond.res_id_1,
                            bond.atom_name_1,
                        ],
                        "atom2": [
                            chain_id_2,
                            bond.res_id_2,
                            bond.atom_name_2,
                        ],
                    }
                }
            )
        if bonds_lst:
            boltz_info_dict["constraints"] = bonds_lst
        return boltz_info_dict

    def to_yaml_file(self, yaml_file: Path):
        """
        Write the Boltz input to a YAML file.

        Args:
            yaml_file (Path): Output YAML file path.
        """
        with open(yaml_file, "w", encoding="utf-8") as f:
            yaml.dump(self.to_yaml(), f)

    def to_sequences(self) -> Sequences:
        """
        Convert the Boltz input back to a Sequences container.

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
