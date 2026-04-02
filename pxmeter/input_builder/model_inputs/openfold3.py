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

import json
import logging
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
    SequencesFilter,
)
from pxmeter.utils import int_to_letters

logger = logging.getLogger(__name__)


@dataclass
class OpenFold3Input:
    """
    OpenFold3-style model input wrapper.

    This class holds the minimal information needed to describe an OpenFold3
    job: a name, chain-level sequences (polymer or ligand), optional covalent
    bonds (though OpenFold3 itself doesn't directly support them in JSON at this time),
    and an optional list of random seeds.

    Attributes:
        name (str): Name of the job or complex.
        sequences (tuple[Union[PolymerChainSequence, LigandChainSequence]]):
            Chain-level sequence descriptors.
        bonds (tuple[Bond]): Optional covalent bonds between chains.
    """

    name: str
    sequences: tuple[Union[PolymerChainSequence, LigandChainSequence]]
    bonds: tuple[Bond] = tuple()

    @staticmethod
    def _convert_sequence_to_dict(sequence, chain_ids) -> dict:
        """
        Convert a single sequence object into OpenFold3 JSON format.

        Args:
            sequence (Union[PolymerChainSequence, LigandChainSequence]): Sequence
                object to convert.
            chain_ids (Union[list[str], tuple[str], str]): One or more chain IDs
                associated with this sequence.

        Returns:
            dict: JSON-serializable chain entry in OpenFold3 schema.

        Raises:
            ValueError: If the ligand input is missing both CCD codes and SMILES,
                or if the entity type is unknown.
        """
        if sequence.is_polymer():
            entity_type_mapping = {
                PROTEIN: "protein",
                PROTEIN_D: "protein",
                RNA: "rna",
                DNA: "dna",
            }
            if sequence.entity_type not in entity_type_mapping:
                raise ValueError(f"Unknown entity type: {sequence.entity_type}")

            entity_dict = {
                "molecule_type": entity_type_mapping[sequence.entity_type],
                "chain_ids": chain_ids if len(chain_ids) > 1 else chain_ids[0],
                "sequence": sequence.sequence,
            }
            if sequence.modifications:
                non_canonical_residues = {}
                for mod_pos, mod_type in sequence.modifications:
                    non_canonical_residues[str(mod_pos)] = mod_type
                entity_dict["non_canonical_residues"] = non_canonical_residues

            return entity_dict

        else:
            # Ligand
            if sequence.ccd_codes and len(sequence.ccd_codes) > 1:
                logger.debug(
                    "OpenFold3 currently does not support multiple CCDs in a single chain. "
                    "Skipping entity with CCD codes: %s",
                    sequence.ccd_codes
                )
                return None

            entity_dict = {
                "molecule_type": "ligand",
                "chain_ids": chain_ids if len(chain_ids) > 1 else chain_ids[0],
            }
            if sequence.ccd_codes:
                entity_dict["ccd_codes"] = list(sequence.ccd_codes)
            elif sequence.smiles:
                entity_dict["smiles"] = sequence.smiles
            else:
                raise ValueError("LigandChainSequence must have ccd_codes or smiles")
            return entity_dict

    @classmethod
    def from_sequences(cls, sequences: Sequences):
        """
        Create an OpenFold3 input from a Sequences container.

        Since OpenFold3 does not currently support explicit covalent bonds via JSON,
        a warning is raised and covalent ligands are filtered out to prevent misleading
        the model or user.

        Args:
            sequences (Sequences): Container with chain sequences and bonds.

        Returns:
            OpenFold3Input: Constructed OpenFold3 input object.
        """
        if sequences.bonds:
            logger.debug(
                "OpenFold3 currently does not support explicit covalent bonds via JSON. "
                "The provided bonds will be ignored, and any covalent ligands will be removed."
            )
            # Remove covalent ligands because we can't express the bond,
            # and leaving an unbonded ligand might mislead the user or model.
            sequences = SequencesFilter(sequences).remove_covalent_ligand()

        return cls(
            name=sequences.name,
            sequences=sequences.sequences,
            bonds=sequences.bonds,
        )

    @classmethod
    def from_json(cls, json_dict: dict, name: str = None) -> "OpenFold3Input":
        """
        Reconstruct an OpenFold3 input from a JSON dict.

        Args:
            json_dict (dict): Parsed OpenFold3-style JSON dict.
            name (str, optional): The query name to extract. If None, uses the first query.

        Returns:
            OpenFold3Input: Reconstructed input object.

        Raises:
            ValueError: If the required query name is not found in the input,
                or if a ligand entry lacks both CCD codes and SMILES.
        """
        queries = json_dict.get("queries", {})
        if name is None:
            name = list(queries.keys())[0] if queries else "None"

        if name not in queries:
            raise ValueError(f"Query {name} not found in OpenFold3 input.")

        chains = queries[name].get("chains", [])

        entity_type_mapping = {
            "protein": PROTEIN,
            "rna": RNA,
            "dna": DNA,
            "ligand": LIGAND,
        }

        seq_obj_lst = []
        for chain_dict in chains:
            mol_type = chain_dict["molecule_type"]
            chain_ids = chain_dict["chain_ids"]
            if isinstance(chain_ids, str):
                chain_ids = [chain_ids]

            if mol_type == "ligand":
                for chain_id in chain_ids:
                    if "ccd_codes" in chain_dict:
                        ccd_codes = chain_dict["ccd_codes"]
                        if isinstance(ccd_codes, str):
                            ccd_codes = [ccd_codes]
                        seq_obj_lst.append(
                            LigandChainSequence(
                                ccd_codes=tuple(ccd_codes),
                                ori_chain_id=chain_id,
                            )
                        )
                    elif "smiles" in chain_dict:
                        seq_obj_lst.append(
                            LigandChainSequence(
                                smiles=chain_dict["smiles"],
                                ori_chain_id=chain_id,
                            )
                        )
                    else:
                        raise ValueError("Ligand chain must have ccd_codes or smiles")
            else:
                modifications = []
                if "non_canonical_residues" in chain_dict:
                    for mod_pos, mod_type in chain_dict[
                        "non_canonical_residues"
                    ].items():
                        modifications.append((int(mod_pos), mod_type))

                for chain_id in chain_ids:
                    seq_obj_lst.append(
                        PolymerChainSequence(
                            sequence=chain_dict["sequence"],
                            entity_type=entity_type_mapping[mol_type],
                            modifications=tuple(modifications),
                            ori_chain_id=chain_id,
                        )
                    )

        return cls(
            name=name,
            sequences=tuple(seq_obj_lst),
        )

    @classmethod
    def from_json_file(cls, json_f: Union[Path, str]) -> "OpenFold3Input":
        """
        Load an OpenFold3 input from a JSON file. Returns the first query.

        Args:
            json_f (Union[Path, str]): Path to an OpenFold3-style JSON file.

        Returns:
            OpenFold3Input: Reconstructed input object for the first query.
        """
        with open(json_f, "r", encoding="utf-8") as f:
            json_dict = json.load(f)
        return cls.from_json(json_dict)

    @classmethod
    def from_json_file_multi(cls, json_f: Union[Path, str]) -> list["OpenFold3Input"]:
        """
        Load all OpenFold3 queries from a JSON file.

        Args:
            json_f (Union[Path, str]): Path to an OpenFold3-style JSON file.

        Returns:
            list[OpenFold3Input]: List of reconstructed input objects for each query.
        """
        with open(json_f, "r", encoding="utf-8") as f:
            json_dict = json.load(f)

        queries = json_dict.get("queries", {})
        inputs = []
        for name in queries.keys():
            inputs.append(cls.from_json(json_dict, name=name))
        return inputs

    def to_json(self) -> dict:
        """
        Serialize the OpenFold3 input into a JSON dict.

        Sequences are grouped by identical sequence objects and emitted with
        appropriate chain IDs.

        Returns:
            dict: JSON-serializable OpenFold3 job specification.
        """
        idx_to_chain_id = {}
        seqs_to_chain_ids = defaultdict(list)

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

        chains_lst = []
        for seq, chain_ids in seqs_to_chain_ids.items():
            chain_dict = OpenFold3Input._convert_sequence_to_dict(seq, chain_ids)
            if chain_dict is not None:
                chains_lst.append(chain_dict)

        out_dict = {"queries": {self.name: {"chains": chains_lst}}}
        return out_dict

    def to_json_file(self, json_file: Path):
        """
        Write the OpenFold3 input to a JSON file.

        Args:
            json_file (Path): Output JSON file path.
        """
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=4)

    def to_sequences(self) -> Sequences:
        """
        Convert the OpenFold3 input back to a Sequences container.

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
