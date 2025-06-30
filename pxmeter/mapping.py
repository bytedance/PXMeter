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
import logging
from typing import Any, Sequence

import numpy as np
from biotite.sequence.align import Alignment
from biotite.structure import AtomArray, get_chain_starts, get_residues
from biotite.structure.io import pdb
from ml_collections.config_dict import ConfigDict
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import TanimotoSimilarity

from pxmeter.configs.run_config import RUN_CONFIG
from pxmeter.data.ccd import get_ccd_mol_from_chain_atom_array
from pxmeter.data.struct import Structure
from pxmeter.data.utils import (
    get_chain_ccd_seq,
    get_mol_graph_matches,
    get_seq_alignment_score,
    rdkit_mol_to_nx_graph,
)
from pxmeter.permutation.atom import AtomPermutation
from pxmeter.permutation.chain import ChainPermutation


class MappingCIF:
    """
    Mapping entities between reference and model structures
    and retrieving the corresponding atom indices.

    Args:
        ref_cif (str): Path to the reference CIF file.
        model_cif (str): Path to the model CIF file.
        ref_assembly_id (str, optional): Assembly ID for the reference structure.
            Defaults to None.
        ref_altloc (str, optional): Alternate location indicator for the reference structure.
            Defaults to "first".
        ref_model (int, optional): Model number for the reference structure.
            Defaults to 1.
        model_chain_id_to_lig_mol (dict[str, Chem.Mol], optional): Mapping or
            model chain IDs to ligand molecules. Defaults to None.
        mapping_config (dict, optional): Configuration dictionary for mapping.
                       If None, the default mapping configuration from
                       `RUN_CONFIG.mapping` is used.
    """

    def __init__(
        self,
        ref_cif: str,
        model_cif: str,
        ref_assembly_id: str | None = None,
        ref_altloc: str = "first",
        ref_model: int = 1,
        model_chain_id_to_lig_mol: dict[str, Chem.Mol] | None = None,
        mapping_config: ConfigDict = RUN_CONFIG.mapping,
    ):
        self.ref_cif = ref_cif
        self.model_cif = model_cif

        self.mapping_config = mapping_config

        self.model_chain_id_to_lig_mol = model_chain_id_to_lig_mol

        # the assembly id, altloc and model of model structure are not considered
        self.ref_struct = self._ref_parser(
            ref_model=ref_model, ref_altloc=ref_altloc, ref_assembly_id=ref_assembly_id
        )
        self.model_struct = self._model_parser()

    def _rearrange_entities_for_model(self, model_atom_array: AtomArray) -> np.ndarray:
        """
        Rearrange entities in the model structure based on CCD sequences.

        This function takes an AtomArray representing the model
        structure and rearranges the entities such that chains
        with the same CCD sequence are grouped into the same entity.
        If ligand molecules are provided, their CCD sequences are
        used to update the chain to sequence mapping.

        Args:
            model_atom_array (AtomArray): The input AtomArray representing the model structure.

        Returns:
            np.ndarray: An array of new entity IDs for each atom in the model structure.
        """
        chain_id_to_ccd_seq = get_chain_ccd_seq(model_atom_array)

        # Update chain to sequence mapping if ligands are provided
        if self.model_chain_id_to_lig_mol:
            for chain_id, lig_mol in self.model_chain_id_to_lig_mol.items():
                ccd_seq = Chem.MolToSmiles(lig_mol)
                if chain_id in chain_id_to_ccd_seq:
                    chain_id_to_ccd_seq[chain_id] = ccd_seq
                else:
                    logging.warning(
                        "Chain id %s not found in CIF: %s",
                        chain_id,
                        list(chain_id_to_ccd_seq.keys()),
                    )

        # Gather unique entity IDs
        entity_ids = np.unique(model_atom_array.label_entity_id)
        largest_entity_id = max(entity_ids.astype(int)) + 1

        entity_id_to_ccd_seq = {}
        ccd_seq_to_entity = {}
        chain_id_to_entity_id = {}

        for chain_id, ccd_seq in chain_id_to_ccd_seq.items():
            entity_id = model_atom_array.label_entity_id[
                model_atom_array.chain_id == chain_id
            ][0]

            if ccd_seq not in ccd_seq_to_entity:
                # Assign new entity ID if needed
                if entity_id in entity_id_to_ccd_seq:
                    entity_id = str(largest_entity_id)
                    largest_entity_id += 1

                entity_id_to_ccd_seq[entity_id] = ccd_seq
                ccd_seq_to_entity[ccd_seq] = entity_id

            # Map chain IDs to entity IDs
            chain_id_to_entity_id[chain_id] = ccd_seq_to_entity.get(ccd_seq, entity_id)

        # Vectorize entity ID assignment
        new_label_entity_id = np.vectorize(chain_id_to_entity_id.get)(
            model_atom_array.chain_id
        )
        return new_label_entity_id

    def _ref_parser(
        self,
        ref_model: int | None = 1,
        ref_altloc: str | None = "first",
        ref_assembly_id: int | None = None,
    ) -> Structure:
        """
        Parses the reference structure from an mmCIF file and cleans it.

        Args:
            ref_model (int, optional): The model number to use from the mmCIF file.
                                       Defaults to 1.
            ref_altloc (str, optional): The alternate location identifier to use.
                                        Defaults to "first".
            ref_assembly_id (int, optional): The assembly ID to use.
                                             Defaults to None.

        Returns:
            Structure: The cleaned reference structure.
        """
        ref_struct = Structure.from_mmcif(
            mmcif=self.ref_cif,
            model=ref_model,
            altloc=ref_altloc,
            assembly_id=ref_assembly_id,
            include_bonds=True,
        )
        ref_struct.reset_entity_poly_by_atom_array()

        cleaned_ref_struct = ref_struct.clean_structure(
            mse_to_met=True,
            modified_arg_atom_naming=True,
            remove_water=True,
            remove_hydrogens=True,
            remove_element_x=True,
            remove_crystallization_aids=True,
        )
        return cleaned_ref_struct

    def _model_parser(self) -> Structure:
        """
        Parses the model structure from the provided mmCIF file
        and performs various cleaning and rearranging operations.

        This method performs the following steps:
        1. Loads the model structure from the mmCIF file.
        2. Cleans the structure by removing unwanted elements such as water,
           hydrogens, and certain other elements.
        3. Rearranges entities within the model to ensure identical chains
           are included within the same entity.
        4. Updates the entity IDs and ensures elements are in uppercase.

        Returns:
            Structure: The cleaned and rearranged model structure.
        """
        model_struct = Structure.from_mmcif(
            mmcif=self.model_cif,
            model=1,
            altloc="first",
            assembly_id=None,
            include_bonds=True,
        )
        model_struct.reset_entity_poly_by_atom_array()

        # Organize atoms with the same chain_id consecutively in the AtomArray
        unique_values, indices = np.unique(
            model_struct.atom_array.chain_id, return_index=True
        )
        sorted_unique_values = unique_values[np.argsort(indices)]
        consecutive_chain_id_index = np.concatenate(
            [
                np.where(model_struct.atom_array.chain_id == value)[0]
                for value in sorted_unique_values
            ]
        )
        model_struct = model_struct.select_substructure(
            consecutive_chain_id_index, reset_uni_id=True
        )

        model_struct = model_struct.clean_structure(
            mse_to_met=True,
            modified_arg_atom_naming=True,
            remove_water=True,
            remove_hydrogens=True,
            remove_element_x=True,
            remove_crystallization_aids=False,
        )

        # Identical chains should be included within the same entity.
        new_label_entity_id = self._rearrange_entities_for_model(
            model_struct.atom_array
        )

        unique_array, indices = np.unique(new_label_entity_id, return_index=True)
        model_entity_id_old_and_new = [
            (model_struct.atom_array.label_entity_id[index], key)
            for key, index in zip(unique_array, indices)
        ]

        model_struct.reset_atom_array_annot("label_entity_id", new_label_entity_id)
        model_struct.update_entity_poly(model_entity_id_old_and_new)

        # Elements should be in uppercase; some model outputs use lowercase in CIF files
        upper_element = np.char.upper(model_struct.atom_array.element)
        model_struct.reset_atom_array_annot("element", upper_element)
        return model_struct

    @staticmethod
    def get_polymer_entity_mapping(
        ref_struct: Structure, model_struct: Structure
    ) -> tuple[dict[str, str], dict[tuple[str, str], Any]]:
        """
        Maps polymer entities between a reference Structure and
        a model Structure based on sequence alignment scores.

        Args:
            ref_struct (Structure): The reference Structure containing polymer entities.
            model_struct (Structure): The model Structure containing polymer entities.

        Returns:
            tuple[dict[str, str], dict]: A tuple containing two dictionaries:
                - A dictionary mapping model entity IDs to reference entity IDs.
                - A dictionary containing sequence alignments for each
                  entity pair (ref entity ID, model entity ID).
        """
        entity_score_dict = {}
        entity_alignments_dict = {}

        for ref_id, ref_seq in ref_struct.entity_poly_seq.items():
            ref_type = ref_struct.entity_poly_type[ref_id]
            for model_id, model_seq in model_struct.entity_poly_seq.items():
                model_type = model_struct.entity_poly_type[model_id]

                if model_type != ref_type:
                    continue

                # Determine sequence type
                seq_type = "protein" if "polypeptide" in model_type else "nuc"
                alignments, seq_identity = get_seq_alignment_score(
                    ref_seq, model_seq, seq_type
                )

                entity_score_dict[(ref_id, model_id)] = seq_identity
                entity_alignments_dict[(ref_id, model_id)] = alignments

        model_to_ref_poly_entity_id = {}
        # Sort and map entities based on sequence identity
        for (ref_id, model_id), seq_identity in sorted(
            entity_score_dict.items(), key=lambda x: x[1], reverse=True
        ):
            # Ensure each entity is mapped only once
            if (
                model_id in model_to_ref_poly_entity_id
                or ref_id in model_to_ref_poly_entity_id.values()
            ):
                continue
            model_to_ref_poly_entity_id[model_id] = ref_id

        return model_to_ref_poly_entity_id, entity_alignments_dict

    @staticmethod
    def _get_entity_ccd_seq(struct: Structure, entity_ids: list | tuple):
        """
        Generate a mapping of entity IDs to their corresponding CCD sequences.

        Args:
            struct: A structure object containing atom array and chain information.
            entity_ids (list): A list of entity IDs to process.

        Returns:
            dict: A dictionary where keys are entity IDs and values are CCD sequences
                  represented as strings with residue names joined by underscores.
        """
        entity_id_to_ccd_seq = {}
        for entity_id in entity_ids:
            entity_mask = struct.atom_array.label_entity_id == entity_id
            first_asym_chain_id = np.unique(struct.uni_chain_id[entity_mask])[0]
            first_asym_chain = struct.atom_array[
                struct.uni_chain_id == first_asym_chain_id
            ]
            _res_ids, res_names = get_residues(first_asym_chain)
            ccd_seq = "_".join(res_names)
            entity_id_to_ccd_seq[entity_id] = ccd_seq
        return entity_id_to_ccd_seq

    @staticmethod
    def _get_ligand_entity_ccd_seq_mapping(
        ref_struct: Structure,
        model_struct: Structure,
        ref_lig_entity_ids: Sequence[str],
        model_lig_entity_ids: Sequence[str],
    ) -> dict[str, str]:
        """
        Maps ligand entity IDs from a model structure to a reference
        structure based on CCD sequences.

        This function takes two structures (reference and model)
        and their corresponding ligand entity IDs, and returns a
        dictionary mapping ligand entity IDs from the model structure
        to the reference structure based on their CCD sequences.

        Args:
            ref_struct (Structure): The reference structure containing ligand entities.
            model_struct (Structure): The model structure containing ligand entities.
            ref_lig_entity_ids (Sequence[str]): A sequence of ligand entity IDs
                                                in the reference structure.
            model_lig_entity_ids (Sequence[str]): A sequence of ligand entity IDs
                                                  in the model structure.

        Returns:
            dict[str, str]: A dictionary where keys are ligand entity IDs
                            from the model structure and values are the
                            corresponding ligand entity IDs from the reference structure.
        """
        ref_lig_entity_id_to_ccd_seq = MappingCIF._get_entity_ccd_seq(
            ref_struct, ref_lig_entity_ids
        )

        model_lig_entity_id_to_ccd_seq = MappingCIF._get_entity_ccd_seq(
            model_struct, model_lig_entity_ids
        )

        model_to_ref_lig_entity_id = {}
        for model_entity_id, model_entity_seq in model_lig_entity_id_to_ccd_seq.items():
            for (
                ref_entity_id,
                ref_entity_seq,
            ) in ref_lig_entity_id_to_ccd_seq.items():
                if ref_entity_id in model_to_ref_lig_entity_id.values():
                    # 1 to 1 mapping
                    continue

                if model_entity_seq == ref_entity_seq:
                    model_to_ref_lig_entity_id[model_entity_id] = ref_entity_id
                    break
        return model_to_ref_lig_entity_id

    @staticmethod
    def _get_rdkit_mol_from_pdb_of_atom_array(atom_array: AtomArray) -> Chem.Mol:
        """
        Convert an AtomArray to an RDKit Mol object using PDB format.

        Args:
            atom_array (AtomArray): The AtomArray object to be converted.

        Returns:
            Chem.Mol: The RDKit Mol object generated from the AtomArray.
        """
        # PDB need an 1 character chain_id
        atom_array.chain_id = atom_array.chain_id.astype("<U1")

        # PDB need an <3 character res_name
        atom_array.res_name = atom_array.res_name.astype("<U3")

        # PDB need an <4 character res_name
        atom_array.atom_name = atom_array.atom_name.astype("<U4")

        file = pdb.PDBFile()
        file.set_structure(atom_array)
        mol = Chem.MolFromPDBBlock(str(file), removeHs=False, proximityBonding=True)
        return mol

    @staticmethod
    def _set_atom_prop_for_lig_mol(atom_array: AtomArray, mol: Chem.Mol) -> Chem.Mol:
        """
        Set properties for each atom in a ligand molecule.

        This function assigns atom properties such as atom name, residue name,
        and residue ID from an AtomArray to the corresponding atoms in a
        RDKit molecule (Chem.Mol).

        Args:
            atom_array (AtomArray): An array containing atom properties.
            mol (Chem.Mol): An RDKit molecule object whose atoms will be updated.

        Returns:
            Chem.Mol: The updated RDKit molecule with new atom properties.
        """
        for idx, atom in enumerate(mol.GetAtoms()):
            atom.SetProp("atom_name", atom_array.atom_name[idx])
            atom.SetProp("res_name", atom_array.res_name[idx])
            atom.SetProp("res_id", str(atom_array.res_id[idx]))
        return mol

    def _get_entity_rdkit_mol(
        self, struct, entity_ids: Sequence[int], source_ref: bool = True
    ) -> tuple[dict[str, Chem.Mol], dict]:
        """
        Generate RDKit molecule objects and fingerprints for specified entity IDs.

        Args:
            struct: The structure object containing atom arrays and chain information.
            entity_ids (Sequence[int]): A sequence of entity IDs for which to
                                        generate RDKit molecules.
            source_ref (bool, optional): Flag indicating whether to use source
                                         reference CCD molecules.
                                         Defaults to True.

        Returns:
            tuple[dict[str, Chem.Mol], dict]: A tuple containing two dictionaries:
                - entity_id_to_mol: Maps entity IDs to their corresponding RDKit molecule objects.
                - entity_id_to_fp: Maps entity IDs to their corresponding molecular fingerprints.
        """
        entity_id_to_mol = {}
        entity_id_to_fp = {}

        mfpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=2048, includeChirality=True
        )

        for entity_id in entity_ids:
            entity_mask = struct.atom_array.label_entity_id == entity_id
            first_asym_chain_id = np.unique(struct.uni_chain_id[entity_mask])[0]
            first_asym_chain = struct.atom_array[
                struct.uni_chain_id == first_asym_chain_id
            ]
            try:
                first_asym_chain_copy = copy.deepcopy(first_asym_chain)

                if source_ref:
                    mol = get_ccd_mol_from_chain_atom_array(first_asym_chain_copy)
                else:
                    # for model structure
                    if self.model_chain_id_to_lig_mol is not None:
                        mol = self.model_chain_id_to_lig_mol[
                            first_asym_chain.label_asym_id[0]
                        ]
                    else:
                        mol = self._get_rdkit_mol_from_pdb_of_atom_array(
                            first_asym_chain_copy
                        )
                    mol = self._set_atom_prop_for_lig_mol(first_asym_chain_copy, mol)
                entity_id_to_mol[entity_id] = mol
                fp_use_chirality = mfpgen.GetFingerprint(mol)
                entity_id_to_fp[entity_id] = fp_use_chirality
            except Exception as e:
                logging.warning(
                    "Generate rdkit mol failed for entity_id=%s due to %s. Model CIF: %s",
                    entity_id,
                    e,
                    self.model_cif,
                )
        return entity_id_to_mol, entity_id_to_fp

    def _get_ligand_entity_similarity_mapping(
        self,
        ref_struct: Structure,
        model_struct: Structure,
        ref_lig_smi_entity_ids: Sequence[str],
        model_lig_smi_entity_ids: Sequence[str],
    ) -> tuple[dict[tuple[str, str], float], dict[str, Chem.Mol], dict[str, Chem.Mol]]:
        """
        Calculate the similarity mapping between ligand entities in reference and model structures.

        Args:
            ref_struct (Structure): The reference structure containing ligand entities.
            model_struct (Structure): The model structure containing ligand entities.
            ref_lig_smi_entity_ids (Sequence[str]): A sequence of entity IDs for ligands in the reference structure.
            model_lig_smi_entity_ids (Sequence[str]): A sequence of entity IDs for ligands in the model structure.

        Returns:
            tuple: A tuple containing:
                - dict[tuple[str, str], float]: A dictionary mapping pairs of
                                                (model_entity_id, ref_entity_id) to their Tanimoto similarity score.
                - dict[str, Chem.Mol]: A dictionary mapping reference entity IDs to their RDKit molecule objects.
                - dict[str, Chem.Mol]: A dictionary mapping model entity IDs to their RDKit molecule objects.
        """
        ref_entity_to_mol, ref_entity_to_fp = self._get_entity_rdkit_mol(
            ref_struct, ref_lig_smi_entity_ids, source_ref=True
        )

        model_entity_to_mol, model_entity_to_fp = self._get_entity_rdkit_mol(
            model_struct, model_lig_smi_entity_ids, source_ref=False
        )

        entity_simi_dict = {}
        for model_entity_id, model_entity_fp in model_entity_to_fp.items():
            for ref_entity_id, ref_entity_fp in ref_entity_to_fp.items():
                try:
                    entity_simi_dict[
                        (model_entity_id, ref_entity_id)
                    ] = TanimotoSimilarity(model_entity_fp, ref_entity_fp)
                except Exception as e:
                    logging.warning(
                        "Calculate similarity failed for model_entity_id=%s and ref_entity_id=%s due to %s",
                        model_entity_id,
                        ref_entity_id,
                        e,
                    )
        return entity_simi_dict, ref_entity_to_mol, model_entity_to_mol

    @staticmethod
    def _get_ligand_entity_atom_mapping(
        entity_simi_dict: dict[tuple[str, str], float],
        ref_entity_to_mol: dict[str, Chem.Mol],
        model_entity_to_mol: dict[str, Chem.Mol],
    ) -> tuple[dict[str, str], dict[tuple, tuple]]:
        """
        Maps ligand entities and their atoms between a reference
        and a model based on structural similarity.

        Args:
            entity_simi_dict (dict[tuple[str, str], float]): A dictionary where keys
                            are tuples of model and ref entity IDs, and values are
                            similarity scores between the entities.
            ref_entity_to_mol (dict[str, Chem.Mol]): A dictionary mapping reference
                              entity IDs to RDKit molecule objects.
            model_entity_to_mol (dict[str, Chem.Mol]): A dictionary mapping model
                                entity IDs to RDKit molecule objects.
        Returns:
            tuple[dict[str, str], dict[tuple, tuple]]:
                - A dictionary mapping model entity IDs to reference entity IDs
                  based on similarity.
                - A dictionary mapping tuples of (model entity ID, residue ID,
                  residue name, and atom name) to tuples of (reference
                  entity ID, residue ID, residue name, and atom name).
        """
        # entity match
        model_to_ref_lig_simi_entity_id = {}

        # atom match
        model_to_ref_atom_mapping = (
            {}
        )  # (entity_id, res_id, res_name, atom_name) mapping
        for model_and_ref_entity_id, _similarity in sorted(
            entity_simi_dict.items(), key=lambda x: x[1], reverse=True
        ):
            model_entity_id, ref_entity_id = model_and_ref_entity_id

            if model_entity_id in model_to_ref_lig_simi_entity_id.keys():
                continue
            if ref_entity_id in model_to_ref_lig_simi_entity_id.values():
                continue

            ref_mol = ref_entity_to_mol[ref_entity_id]
            model_mol = model_entity_to_mol[model_entity_id]
            ref_graph = rdkit_mol_to_nx_graph(ref_mol)
            model_graph = rdkit_mol_to_nx_graph(model_mol)
            try:
                matches = get_mol_graph_matches(model_graph, ref_graph)
                assert len(matches) > 0
                match = matches[0]

            except Exception as _e:
                logging.warning(
                    "Atom match failed for model_entity_id=%s and ref_entity_id=%s, delete this mapping",
                    model_entity_id,
                    ref_entity_id,
                )
                continue

            for model_atom_idx, ref_atom_idx in match.items():
                model_atom = model_mol.GetAtomWithIdx(model_atom_idx)
                ref_atom = ref_mol.GetAtomWithIdx(ref_atom_idx)

                model_res_name = model_atom.GetProp("res_name")
                model_res_id = int(model_atom.GetProp("res_id"))
                model_atom_name = model_atom.GetProp("atom_name")

                ref_res_name = ref_atom.GetProp("res_name")
                ref_res_id = int(ref_atom.GetProp("res_id"))
                ref_atom_name = ref_atom.GetProp("atom_name")
                model_to_ref_atom_mapping[
                    (model_entity_id, model_res_id, model_res_name, model_atom_name)
                ] = (ref_entity_id, ref_res_id, ref_res_name, ref_atom_name)

            # match success
            model_to_ref_lig_simi_entity_id[model_entity_id] = ref_entity_id
        return model_to_ref_lig_simi_entity_id, model_to_ref_atom_mapping

    def get_ligand_entity_mapping(
        self, ref_struct: Structure, model_struct: Structure
    ) -> tuple[dict[str, str], dict[str, str], dict[tuple, tuple]]:
        """
        Generate mappings between ligand entities in reference and model structures.

        Args:
            ref_struct (Structure): The reference structure containing ligand entities.
            model_struct (Structure): The model structure containing ligand entities.

        Returns:
            tuple: A tuple containing three dictionaries:
                - model_to_ref_lig_ccd_entity_id (dict[str, str]): Mapping of ligand entity
                    IDs based on CCD sequences from model to reference.
                - model_to_ref_lig_simi_entity_id (dict[str, str]): Mapping of ligand entity
                    IDs based on similarity.
                - model_to_ref_atom_mapping (dict[tuple, tuple]): Mapping of atom indices
                    between model and reference ligands.
        """
        ref_entity_ids = np.unique(ref_struct.atom_array.label_entity_id)
        model_entity_ids = np.unique(model_struct.atom_array.label_entity_id)

        ref_lig_entity_ids = ref_entity_ids[
            ~np.isin(ref_entity_ids, list(ref_struct.entity_poly_seq.keys()))
        ]
        model_lig_entity_ids = model_entity_ids[
            ~np.isin(model_entity_ids, list(model_struct.entity_poly_seq.keys()))
        ]

        model_to_ref_lig_ccd_entity_id = MappingCIF._get_ligand_entity_ccd_seq_mapping(
            ref_struct, model_struct, ref_lig_entity_ids, model_lig_entity_ids
        )

        # get remained lig entity mapping by similarity
        ref_lig_smi_entity_ids = ref_lig_entity_ids[
            ~np.isin(ref_lig_entity_ids, list(model_to_ref_lig_ccd_entity_id.values()))
        ]
        model_lig_smi_entity_ids = model_lig_entity_ids[
            ~np.isin(model_lig_entity_ids, list(model_to_ref_lig_ccd_entity_id.keys()))
        ]

        if model_lig_smi_entity_ids.size > 0:
            (
                entity_simi_dict,
                ref_entity_to_mol,
                model_entity_to_mol,
            ) = self._get_ligand_entity_similarity_mapping(
                ref_struct,
                model_struct,
                ref_lig_smi_entity_ids,
                model_lig_smi_entity_ids,
            )
            (
                model_to_ref_lig_simi_entity_id,
                model_to_ref_atom_mapping,
            ) = MappingCIF._get_ligand_entity_atom_mapping(
                entity_simi_dict,
                ref_entity_to_mol,
                model_entity_to_mol,
            )

        else:
            model_to_ref_lig_simi_entity_id, model_to_ref_atom_mapping = {}, {}

        return (
            model_to_ref_lig_ccd_entity_id,
            model_to_ref_lig_simi_entity_id,
            model_to_ref_atom_mapping,
        )

    @staticmethod
    def _align_model_lig_atom_to_ref(
        model_struct: Structure, model_to_ref_atom_mapping: dict[tuple, tuple]
    ):
        unmapped_lig_mask = np.zeros(len(model_struct.atom_array), dtype=bool)
        for model_atom_info, _ref_atom_info in model_to_ref_atom_mapping.items():
            model_entity_id = model_atom_info[0]
            entity_mask = model_struct.atom_array.label_entity_id == model_entity_id
            # Initialize unmapped_lig_mask for the ligand entities
            unmapped_lig_mask[entity_mask] = True

        model_new_res_id = copy.deepcopy(model_struct.atom_array.res_id)
        model_new_res_name = copy.deepcopy(model_struct.atom_array.res_name)
        model_new_atom_name = copy.deepcopy(model_struct.atom_array.atom_name)

        for model_atom_info, ref_atom_info in model_to_ref_atom_mapping.items():
            (
                model_entity_id,
                model_res_id,
                _model_res_name,
                model_atom_name,
            ) = model_atom_info
            _ref_entity_id, ref_res_id, ref_res_name, ref_atom_name = ref_atom_info

            entity_mask = model_struct.atom_array.label_entity_id == model_entity_id
            res_id_mask = model_struct.atom_array.res_id == model_res_id
            atom_name_mask = model_struct.atom_array.atom_name == model_atom_name
            model_mask = entity_mask & res_id_mask & atom_name_mask

            model_new_res_name[model_mask] = ref_res_name
            model_new_atom_name[model_mask] = ref_atom_name
            model_new_res_id[
                entity_mask & res_id_mask
            ] = ref_res_id  # prevent chain breakage
            # Remove the mapped ligand entities from unmapped_lig_mask
            unmapped_lig_mask[model_mask] = False

        # Set unmapped ligand entities to -1 and "." for res_id, res_name, and atom_name
        model_new_res_name[unmapped_lig_mask] = "."
        model_new_atom_name[unmapped_lig_mask] = "."

        # update model_struct.atom_array
        model_struct.reset_atom_array_annot("res_id", model_new_res_id)
        model_struct.reset_atom_array_annot("res_name", model_new_res_name)
        model_struct.reset_atom_array_annot("atom_name", model_new_atom_name)

    def _check_unmapped_entity(
        self,
        model_to_ref_entity_id: dict[str, str],
    ):
        """
        Check for unmapped entities between reference and model structures.

        This function compares the entity IDs in the reference structure and the model
        structure to identify any entities that are not mapped between the two. It logs
        warnings for any unmapped entities found in either the reference or model structures.

        Args:
            ref_struct (Structure): The reference structure containing entity IDs.
            model_struct (Structure): The model structure containing entity IDs.
            model_to_ref_entity_id (dict[str, str]): A dictionary mapping model entity IDs to reference entity IDs.

        Logs:
            Warnings for any unmapped entities in the reference or model structures.
        """
        ref_entity_ids = np.unique(self.ref_struct.atom_array.label_entity_id)
        model_entity_ids = np.unique(self.model_struct.atom_array.label_entity_id)

        unmapped_ref_entity_mask = ~np.isin(
            ref_entity_ids, list(model_to_ref_entity_id.values())
        )

        unmapped_model_entity_mask = ~np.isin(
            model_entity_ids, list(model_to_ref_entity_id.keys())
        )

        if unmapped_model_entity_mask.any():
            logging.warning(
                "Unmapped model entities for %s:\n %s",
                self.model_cif,
                model_entity_ids[unmapped_model_entity_mask],
            )
        if unmapped_ref_entity_mask.any():
            logging.warning(
                "Unmapped ref entities for %s:\n %s",
                self.ref_cif,
                ref_entity_ids[unmapped_ref_entity_mask],
            )

    @staticmethod
    def _get_inter_atoms_in_a_entity(
        atom_array: AtomArray, entity_id: str, unique_atom_id: np.ndarray
    ) -> set[str]:
        """
        Get the set of unique atom IDs that are present in all chains of a given entity.

        This function filters the input atom array to only include atoms belonging to the specified entity.
        It then identifies the unique atom IDs that are common across all chains within that entity.

        Args:
            atom_array (AtomArray): An array of atoms with associated metadata.
            entity_id (str): The identifier of the entity to filter atoms by.
            unique_atom_id (np.ndarray): An array of unique atom IDs corresponding to the atoms in `atom_array`.

        Returns:
            set[str]: A set of unique atom IDs that are present in all chains of the specified entity.
        """
        entity_atom_array = atom_array[atom_array.label_entity_id == entity_id]
        unique_atom_id_in_a_entity = unique_atom_id[
            atom_array.label_entity_id == entity_id
        ]
        chain_starts = get_chain_starts(entity_atom_array, add_exclusive_stop=True)

        chain_unique_atom_ids_list = []
        for start, stop in zip(chain_starts[:-1], chain_starts[1:]):
            chain_unique_atom_ids = set(unique_atom_id_in_a_entity[start:stop])
            chain_unique_atom_ids_list.append(chain_unique_atom_ids)

        entity_unique_atom_id_set = set.intersection(*chain_unique_atom_ids_list)
        return entity_unique_atom_id_set

    @staticmethod
    def _get_chain_atoms_by_unique_atom_id(
        struct: Structure,
        entity_id: str,
        unique_atom_id: np.ndarray,
        inter_entity_unique_atom_id: set | list,
    ) -> list[np.ndarray]:
        """
        Retrieve the indices of atoms in a structure that belong to
        a specific entity and are identified by unique atom IDs.

        Args:
            struct (Structure): The structure containing the atom data.
            entity_id (str): The identifier of the entity to filter atoms by.
            unique_atom_id (np.ndarray): An array of unique atom IDs.
            inter_entity_unique_atom_id (set or list): A set or list of
                                        unique atom IDs that are of interest.

        Returns:
            list[np.ndarray]: A list of arrays, where each array contains the
                              indices of atoms in the structure that belong to
                              the specified entity and have a unique atom ID
                              in `inter_entity_unique_atom_id`. The indices are
                              sorted by the corresponding unique atom ID.
        """
        entity_chains = np.unique(
            struct.uni_chain_id[struct.atom_array.label_entity_id == entity_id]
        )
        uid_mask = np.isin(unique_atom_id, list(inter_entity_unique_atom_id))

        indices_list = []
        for chain_id in entity_chains:
            chain_mask = struct.uni_chain_id == chain_id
            mask = uid_mask & chain_mask

            uid_filtered = unique_atom_id[mask]
            indices = np.where(mask)[0]
            sorted_idx = np.argsort(uid_filtered)  # sort by uid
            sorted_indices = indices[sorted_idx]
            indices_list.append(sorted_indices)
        return indices_list

    def _reset_model_res_id_by_seq_alignment(
        self, entity_alignments_dict: dict[tuple[str, str], Alignment]
    ):
        """
        Reset the residue IDs in the model structure based on sequence alignments.

        This method aligns the residue IDs of entities in the model structure with those
        in the reference structure based on sequence alignments. It updates the residue
        IDs in the model structure to match the reference structure.

        Args:
            entity_alignments_dict (dict[tuple[str, str]]): A dictionary containing
                                   the sequence alignments for entity pairs.
                                   The keys are tuples of reference and model entity IDs,
                                   and the values are Alignment objects representing
                                   the alignments between the corresponding entities.
        """
        for (_ref_entity_id, model_entity_id), ali in entity_alignments_dict.items():
            model_entity_mask = (
                self.model_struct.atom_array.label_entity_id == model_entity_id
            )

            filtered_ali = ali.trace[np.all(ali.trace != -1, axis=1)]
            model_to_ref_res_id = dict((filtered_ali[:, [1, 0]] + 1).tolist())

            model_res_ids = self.model_struct.atom_array.res_id[model_entity_mask]
            mapped_res_ids = [model_to_ref_res_id.get(x, -1) for x in model_res_ids]
            self.model_struct.atom_array.res_id[model_entity_mask] = mapped_res_ids

    def get_mapping_result(self) -> dict[str, str]:
        """
        Generates entity mappings between reference and model structures for polymers and ligands.

        Returns:
            dict[str, str]: A dictionary mapping model entity IDs to reference entity IDs.
        """
        # Mapping entities
        if self.mapping_config["mapping_polymer"]:
            (
                model_to_ref_poly_entity_id,
                poly_entity_alignments_dict,
            ) = self.get_polymer_entity_mapping(self.ref_struct, self.model_struct)

            if not self.mapping_config["res_id_alignments"]:
                self._reset_model_res_id_by_seq_alignment(poly_entity_alignments_dict)
        else:
            model_to_ref_poly_entity_id = {}

        if self.mapping_config["mapping_ligand"]:
            (
                model_to_ref_lig_ccd_entity_id,
                model_to_ref_lig_simi_entity_id,
                model_to_ref_atom_mapping,
            ) = self.get_ligand_entity_mapping(self.ref_struct, self.model_struct)

        else:
            (
                model_to_ref_lig_ccd_entity_id,
                model_to_ref_lig_simi_entity_id,
                model_to_ref_atom_mapping,
            ) = (
                {},
                {},
                {},
            )

        # Merge polymer and ligand mappings
        model_to_ref_entity_id = copy.deepcopy(model_to_ref_poly_entity_id)
        model_to_ref_entity_id.update(model_to_ref_lig_ccd_entity_id)
        model_to_ref_entity_id.update(model_to_ref_lig_simi_entity_id)

        self._check_unmapped_entity(model_to_ref_entity_id)

        # This step will be change the res_id, res_name, atom_name in self.model_struct.atom_array
        self._align_model_lig_atom_to_ref(self.model_struct, model_to_ref_atom_mapping)

        # Remove unmapped ligand of model and reset unique atom id
        self.model_struct = self.model_struct.select_substructure(
            ~(
                (
                    (self.model_struct.atom_array.res_name == ".")
                    & (self.model_struct.atom_array.atom_name == ".")
                )
                | (self.model_struct.atom_array.res_id < 0)  # -1 for unmapped residues
            ),
            reset_uni_id=True,
        )

        # Update entity mapping after removing unmapped ligand
        model_entity_ids = np.unique(self.model_struct.atom_array.label_entity_id)
        model_to_ref_entity_id = {
            model_entity_id: ref_entity_id
            for model_entity_id, ref_entity_id in model_to_ref_entity_id.items()
            if model_entity_id in model_entity_ids
        }

        return model_to_ref_entity_id


@dataclasses.dataclass(frozen=True)
class MappingResult:
    """
    A class to represent the result of mapping between reference and model structures.
    """

    ref_struct: Structure
    model_struct: Structure
    ref_indices: np.ndarray
    model_indices: np.ndarray
    chain_mapping: dict[str, str]
    chain_mapping_anchors: dict[str, str]
    model_to_ref_entity_id: dict[str, str]

    def get_mapped_structures(self) -> tuple[Structure, Structure]:
        """
        Selects and returns substructures from reference and model structures based on specified indices.

        Returns:
            tuple: A tuple containing two substructures:
                - sele_ref_struct: The selected substructure from the reference structure.
                - sele_model_struct: The selected substructure from the model structure.
        """

        sele_ref_struct = self.ref_struct.select_substructure(self.ref_indices)
        sele_model_struct = self.model_struct.select_substructure(self.model_indices)
        return sele_ref_struct, sele_model_struct

    @classmethod
    def from_cifs(
        cls,
        ref_cif: str,
        model_cif: str,
        ref_assembly_id: str | None = None,
        ref_altloc: str = "first",
        ref_model: int = 1,
        model_chain_id_to_lig_mol: dict[str, Chem.Mol] | None = None,
        mapping_config: ConfigDict = RUN_CONFIG.mapping,
    ) -> "MappingResult":
        """
        Creates a MappingResult object from the provided CIF files.

        Args:
            ref_cif (str): Path to the reference CIF file.
            model_cif (str): Path to the model CIF file.
            ref_assembly_id (str, optional): Assembly ID for the reference structure. Defaults to None.
            ref_altloc (str): Alternate location indicator for the reference structure. Defaults to "first".
            ref_model (int): Model number for the reference structure. Defaults to 1.
            model_chain_id_to_lig_mol (dict[str, Chem.Mol], optional): Mapping of model chain IDs
                to ligand molecules. Defaults to None.
            mapping_config (ConfigDict, optional): Configuration for the mapping process.
                            Defaults to RUN_CONFIG.mapping.

        Returns:
            MappingResult: An instance of MappingResult containing the mapping information.
        """

        map_cif = MappingCIF(
            ref_cif,
            model_cif,
            ref_assembly_id,
            ref_altloc,
            ref_model,
            model_chain_id_to_lig_mol,
            mapping_config,
        )
        model_to_ref_entity_id = map_cif.get_mapping_result()

        # Permutation
        chain_perm = ChainPermutation(
            map_cif.ref_struct,
            map_cif.model_struct,
            model_to_ref_entity_id,
            enumerate_all_anchors=mapping_config.enumerate_all_anchors,
        )
        chain_mapping, chain_mapping_anchors = chain_perm.get_heurisitic_chain_mapping()
        (
            chain_perm_ref_indices,
            chain_perm_model_indices,
        ) = chain_perm.get_permuted_indices(chain_mapping)

        atom_perm = AtomPermutation(
            map_cif.ref_struct.select_substructure(chain_perm_ref_indices),
            map_cif.model_struct.select_substructure(chain_perm_model_indices),
        )
        atom_permuted_indices = atom_perm.run()

        return cls(
            ref_struct=map_cif.ref_struct,
            model_struct=map_cif.model_struct,
            ref_indices=chain_perm_ref_indices,
            model_indices=chain_perm_model_indices[atom_permuted_indices],
            chain_mapping=chain_mapping,
            chain_mapping_anchors=chain_mapping_anchors,
            model_to_ref_entity_id=model_to_ref_entity_id,
        )
