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
import gzip
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import DockQ.parsers as dockq_parsers
import numpy as np
import pandas as pd
from biotite.structure.io import pdb
from DockQ.DockQ import run_on_all_native_interfaces
from ml_collections.config_dict import ConfigDict
from posebusters import PoseBusters
from rdkit import Chem

from pxmeter.configs.run_config import RUN_CONFIG
from pxmeter.constants import IONS, LIGAND
from pxmeter.data.ccd import get_ccd_mol_from_chain_atom_array
from pxmeter.data.struct import Structure
from pxmeter.metrics.lddt_metrics import LDDT
from pxmeter.metrics.rmsd_metrics import RMSDMetrics

logging.getLogger("posebusters").setLevel(logging.ERROR)


def load_PDB(path, chains=None, small_molecule=False, n_model=0):
    """
    Modified from DockQ.DockQ.load_PDB to avoid ResourceWarning warnings.
    ResourceWarning: Enable tracemalloc to get the object allocation traceback
    DockQ/DockQ.py:660: ResourceWarning: unclosed file
    """
    if chains is None:
        chains = []
    try:
        pdb_parser = dockq_parsers.PDBParser(QUIET=True)
        with (
            gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")
        ) as file_obj:
            model = pdb_parser.get_structure(
                "-",
                file_obj,
                chains=chains,
                parse_hetatms=small_molecule,
                model_number=n_model,
            )
    except Exception:
        pdb_parser = dockq_parsers.MMCIFParser(QUIET=True)
        with (
            gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")
        ) as file_obj:
            model = pdb_parser.get_structure(
                "-",
                file_obj,
                chains=chains,
                parse_hetatms=small_molecule,
                auth_chains=not small_molecule,
                model_number=n_model,
            )
    model.id = path
    return model


def compute_dockq(
    ref_struct: Structure,
    model_struct: Structure,
    ref_to_model_chain_map: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """
    Computes the DockQ score between a reference structure and a model structure.

    Args:
        ref_struct (Structure): The reference structure.
        model_struct (Structure): The model structure to be evaluated.
        ref_to_model_chain_map (dict[str, str]): A dictionary mapping reference chain IDs to model chain IDs.

    Returns:
        dict[str, dict[str, Any]]: A dictionary containing the DockQ score and other related metrics.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        tmp_ref_cif = tmp_dir / "tmp_ref.cif"
        tmp_model_cif = tmp_dir / "tmp_model.cif"

        # Calculate DockQ using exclusively valid atoms
        # Use uni_chain_id as label_asym_id
        ref_struct.to_cif(tmp_ref_cif, use_uni_chain_id=True)
        model_struct.to_cif(tmp_model_cif, use_uni_chain_id=True)

        # small_molecule=False means only polymer is considered
        model = load_PDB(str(tmp_model_cif), small_molecule=False)
        native = load_PDB(str(tmp_ref_cif), small_molecule=False)

        native_chains = [c.id for c in native]
        model_chains = [c.id for c in model]

        valid_ref_to_model_chain_map = {}
        for k, v in ref_to_model_chain_map.items():
            if (
                k in ref_struct.uni_chain_id
                and k in native_chains
                and v in model_chains
            ):
                # some all UNK structure will not be load by load_PDB(), e.g. chain Q in 7q6i
                valid_ref_to_model_chain_map[k] = v
                assert v in model_struct.uni_chain_id

    dockq_result_dict, _total_dockq = run_on_all_native_interfaces(
        model, native, chain_map=valid_ref_to_model_chain_map
    )
    return dockq_result_dict


def compute_pb_valid(
    ref_struct: Structure,
    model_struct: Structure,
    ref_lig_label_asym_id: str | list[str],
) -> pd.DataFrame | None:
    """
    Compute pose-busting validation metrics for a given reference structure, model structure, and reference features.

    Args:
        ref_struct (Structure): The reference structure containing atom arrays and valid atom masks.
        model_struct (Structure): The model structure containing atom arrays.
        ref_lig_label_asym_id (str | list[str]): The label asym ID of the ligand of
                              interest in the reference structure.

    Returns:
        pd.DataFrame or None: A DataFrame containing the pose-busting validation metrics for each ligand mask.
    """

    if isinstance(ref_lig_label_asym_id, str):
        ref_lig_label_asym_ids = [ref_lig_label_asym_id]
    else:
        ref_lig_label_asym_ids = list(ref_lig_label_asym_id)

    df_list = []
    for lig_label_asym_id in ref_lig_label_asym_ids:
        lig_mask = ref_struct.atom_array.label_asym_id == lig_label_asym_id

        ref_lig_chain_id = ref_struct.uni_chain_id[lig_mask][0]
        model_lig_chain_id = model_struct.uni_chain_id[lig_mask][0]

        ref_lig_atom_array = ref_struct.atom_array[lig_mask]
        model_lig_atom_array = copy.deepcopy(model_struct.atom_array[lig_mask])
        # reset res_name for model ligand atoms by ref Structure
        model_lig_atom_array.res_name = ref_lig_atom_array.res_name
        model_cond_atom_array = model_struct.atom_array[~lig_mask]

        ref_lig_mol = get_ccd_mol_from_chain_atom_array(ref_lig_atom_array)
        model_lig_mol = get_ccd_mol_from_chain_atom_array(model_lig_atom_array)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            ref_lig_sdf = tmp_dir / "tmp_ref_lig.sdf"
            model_lig_sdf = tmp_dir / "tmp_model_lig.sdf"
            model_cond_pdb = tmp_dir / "tmp_model_cond.pdb"

            sdf_writer = Chem.SDWriter(str(ref_lig_sdf))
            sdf_writer.write(ref_lig_mol)
            sdf_writer.close()

            sdf_writer = Chem.SDWriter(str(model_lig_sdf))
            sdf_writer.write(model_lig_mol)
            sdf_writer.close()

            pdb_file = pdb.PDBFile()
            model_cond_atom_array = copy.deepcopy(model_cond_atom_array)
            # PDB file only support one letter chain_id
            model_cond_atom_array.chain_id = [
                i[0] for i in model_cond_atom_array.chain_id
            ]
            pdb_file.set_structure(model_cond_atom_array)
            pdb_file.write(model_cond_pdb)

            buster = PoseBusters(config="redock")
            df = buster.bust(
                mol_pred=model_lig_sdf,
                mol_true=ref_lig_sdf,
                mol_cond=model_cond_pdb,
                full_report=True,
            )

            # record ligand chain id
            df["ref_lig_chain_id"] = ref_lig_chain_id
            df["model_lig_chain_id"] = model_lig_chain_id
            df_list.append(df)
    df_cat = pd.concat(df_list)
    return df_cat


class CalcLDDTMetric:
    """
    A class to calculate the Local Distance Difference Test (LDDT) metric for protein structures.

    Args:
        ref_struct (Structure): The reference structure.
        ref_features (Features): The reference features.
        model_features (Features): The model features.
        lddt_config (ConfigDict, optional): The configuration for the LDDT metric.
                    Defaults to RUN_CONFIG.metric.lddt.
    """

    def __init__(
        self,
        ref_struct: Structure,
        model_struct: Structure,
        lddt_config: ConfigDict = RUN_CONFIG.metric.lddt,
    ):
        self.ref_struct = ref_struct
        self.model_struct = model_struct
        self.lddt_calculator = LDDT(
            ref_struct=self.ref_struct,
            model_struct=self.model_struct,
            is_nucleotide_threshold=lddt_config.nucleotide_threshold,
            is_not_nucleotide_threshold=lddt_config.non_nucleotide_threshold,
            eps=lddt_config.eps,
        )

    def get_chains_mask(
        self, chains: list[str], interfaces: list[tuple[str, str]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate masks for chains and interfaces.

        Args:
            chains (list[str]): A list of chain identifiers.
            interfaces (list[tuple[str, str]]): A list of tuples,
                each containing two chain identifiers representing an interface.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - merged_chain_1_mask: A mask for the first
                                       chain in each chain/interface.
                - merged_chain_2_mask: A mask for the second
                                       chain in each chain/interface.
        """
        chains_and_interfaces = chains + interfaces

        merged_chain_1_masks = []  # [N_eval, N_atoms]
        merged_chain_2_masks = []  # [N_eval, N_atoms]
        for chain_or_interface in chains_and_interfaces:
            is_chain = isinstance(chain_or_interface, str)
            if is_chain:
                # chain_1_mask == chain_2_mask for chain
                chain_1 = chain_or_interface
                chain_2 = chain_1
            else:
                # interface
                chain_1, chain_2 = chain_or_interface

            chain_1_mask = self.ref_struct.uni_chain_id == chain_1
            chain_2_mask = self.ref_struct.uni_chain_id == chain_2

            assert np.sum(chain_1_mask) > 0, f"chain_1 ({chain_1}) not found"
            assert np.sum(chain_2_mask) > 0, f"chain_2 ({chain_2}) not found"

            merged_chain_1_masks.append(chain_1_mask)
            merged_chain_2_masks.append(chain_2_mask)
        merged_chain_1_masks = np.array(merged_chain_1_masks)
        merged_chain_2_masks = np.array(merged_chain_2_masks)
        return merged_chain_1_masks, merged_chain_2_masks

    def get_complex_lddt(self) -> float:
        """
        Calculate the LDDT score for a complex.

        This method uses the LDDT calculator to compute the LDDT score based on the predicted
        and true coordinates of the complex. The LDDT score is a measure of the
        structural similarity between the predicted and true structures.

        Returns:
            float: The LDDT score for the complex.
        """
        # complex_lddt = [1]
        complex_lddt = self.lddt_calculator.run(
            chain_1_masks=None,
            chain_2_masks=None,
        )
        return complex_lddt

    def get_chain_interface_lddt(
        self, chains: list[str], interfaces: list[tuple[str, str]]
    ) -> list[float]:
        """
        Calculate the LDDT scores for chains and interfaces.

        Args:
            chains (list[str]): A list of chain identifiers.
            interfaces (list[tuple[str, str]]): A list of tuples, each containing
                                                two chain identifiers representing an interface.

        Returns:
            list[float]: A list of LDDT scores for chains and interfaces.
        """
        merged_chain_1_masks, merged_chain_2_masks = self.get_chains_mask(
            chains, interfaces
        )

        lddt_list = self.lddt_calculator.run(
            chain_1_masks=merged_chain_1_masks,
            chain_2_masks=merged_chain_2_masks,
        )
        return lddt_list


@dataclasses.dataclass(frozen=True)
class MetricResult:
    """
    A class to represent the results of various metrics calculated
    for a given structure and its features.
    """

    ref_struct: Structure
    model_struct: Structure

    meta_info: dict[str, Any]

    # {metric: value}
    complex: dict[str, float]

    # {chain_id: {metric: value}}
    chain: dict[tuple[str], dict[str, Any]]

    # {(chain_id_1, chain_id_2): {metric: value}}
    interface: dict[tuple[str, str], dict[str, Any]]

    # [ref_chain_id: {metric: value}]
    pb_valid: dict[str, dict[str, Any]] | None = None

    ori_model_chain_ids: list[str] | None = None

    @staticmethod
    def _get_chain_info(ref_struct: Structure) -> dict[str, dict[str, str]]:
        """
        Extracts chain information from a given structure.

        Args:
            ref_struct (Structure): The reference structure containing chain and atom information.

        Returns:
            dict[str, dict[str, str]]: A dictionary where each key is a chain ID and the value is another dictionary
                                       containing 'label_entity_id' and 'entity_type' for that chain.
        """
        chain_info_dict = {}
        for chain_id in np.unique(ref_struct.uni_chain_id):
            chain_mask = ref_struct.uni_chain_id == chain_id
            label_entity_id = ref_struct.atom_array.label_entity_id[chain_mask][0]
            entity_type = ref_struct.entity_poly_type.get(label_entity_id, LIGAND)

            chain_info_dict[chain_id] = {
                "label_entity_id": label_entity_id,
                "entity_type": entity_type,
            }
        return chain_info_dict

    @staticmethod
    def _remove_ion_from_chain_and_interface(
        ref_struct: Structure, chains: list[str], interfaces: list[tuple[str, str]]
    ) -> tuple[list[str], list[tuple[str, str]]]:
        """
        Remove ions from the list of chains and interfaces.
        This function filters out chains and interfaces that may contain ions from the provided lists.

        Args:
            ref_struct (Structure): The reference structure containing chain information.
            interfaces (list[tuple[str, str]]): A list of tuples, where each tuple contains
                                                a pair of chain identifiers that have interfaces
                                                within the specified radius.

        Returns:
            tuple[list[str], list[tuple[str, str]]]: A tuple containing two lists:
                - chains_wo_ions: A list of chain identifiers without ions.
                - interfaces_wo_ions: A list of tuples representing interfaces without ions.
        """
        ions_ccd_list = list(IONS)
        chain_ids = np.unique(ref_struct.uni_chain_id)

        ion_chains = []
        chain_id_to_atom_num = {}  # Calc LDDT need at least 2 atoms
        for chain_id in chain_ids:
            chain_mask = ref_struct.uni_chain_id == chain_id
            res_names = ref_struct.atom_array.res_name[chain_mask]
            if np.all(np.isin(res_names, ions_ccd_list)):
                ion_chains.append(chain_id)
            chain_id_to_atom_num[chain_id] = chain_mask.sum()

        chains_wo_ions = [
            chain_id
            for chain_id in chains
            if chain_id not in ion_chains
            if chain_id_to_atom_num[chain_id] > 1
        ]
        interfaces_wo_ions = [
            (chain_1, chain_2)
            for chain_1, chain_2 in interfaces
            if chain_1 not in ion_chains and chain_2 not in ion_chains
        ]
        return chains_wo_ions, interfaces_wo_ions

    @staticmethod
    def _post_process_chain_interface_lddt(
        chains: list[str],
        interfaces: list[tuple[str, str]],
        chain_interface_lddt: list[float],
    ) -> tuple[dict[str, dict[str, float]], dict[tuple[str, str], dict[str, float]]]:
        chain_lddt_dict = {}
        interface_lddt_dict = {}
        num_chains = len(chains)
        for idx, chain_id in enumerate(chains):
            chain_lddt_dict[chain_id] = {"lddt": chain_interface_lddt[idx]}

        for idx, interface in enumerate(interfaces):
            sorted_interface = tuple(
                sorted(interface)
            )  # Sort chains to ensure consistent order
            interface_lddt_dict[sorted_interface] = {
                "lddt": chain_interface_lddt[idx + num_chains]
            }
        return chain_lddt_dict, interface_lddt_dict

    @staticmethod
    def _post_process_dockq(
        dockq_result_dict: dict[str, Any],
    ) -> dict[str, float | dict[str, float]]:
        polymer_dockq_metrics = {"F1", "iRMSD", "LRMSD", "fnat", "nat_correct",
                                 "nat_total", "fnonnat", "nonnat_count", "model_total",
                                 "clashes", "len1", "len2", "class1", "class2", "is_het",
                                 }  # fmt:skip
        ligand_dockq_metrics = {"LRMSD", "is_het"}

        interface_dockq_dict = {}
        for _interface, result in dockq_result_dict.items():
            ref_to_model_chain_map = result["chain_map"]
            model_to_ref_chain_map = {v: k for k, v in ref_to_model_chain_map.items()}
            ref_chain1 = model_to_ref_chain_map[result["chain1"]]
            ref_chain2 = model_to_ref_chain_map[result["chain2"]]
            sorted_interface = tuple(
                sorted([ref_chain1, ref_chain2])
            )  # Sort chains to ensure consistent order

            is_ligand = "F1" in result
            if is_ligand:
                interface_dockq_dict[sorted_interface] = {
                    "dockq": result["DockQ"],
                    "dockq_info": {
                        k: v for k, v in result.items() if k in polymer_dockq_metrics
                    },
                }
            else:
                interface_dockq_dict[sorted_interface] = {
                    "dockq": result["DockQ"],
                    "dockq_info": {
                        k: v for k, v in result.items() if k in ligand_dockq_metrics
                    },
                }
        return interface_dockq_dict

    @staticmethod
    def _post_process_pb_valid(
        pb_valid_result_df: pd.DataFrame | None,
    ) -> dict[str, dict[str, Any]] | None:
        if pb_valid_result_df is None:
            return

        # Replace "NaN" to "None"
        pb_valid_result_df = pb_valid_result_df.replace({np.nan: None})

        chain_pb_valid_dict = {}
        for _row_idx, row in pb_valid_result_df.iterrows():
            ref_lig_chain_id = row["ref_lig_chain_id"]
            row_dict = row.to_dict()
            # Remove ref_lig_chain_id from row_dict
            del row_dict["ref_lig_chain_id"]

            assert (
                ref_lig_chain_id not in chain_pb_valid_dict
            ), "Duplicate chain for ligand"

            chain_pb_valid_dict[ref_lig_chain_id] = row_dict
        return chain_pb_valid_dict

    @staticmethod
    def _update_src_to_tar_dict(src_dict: dict[Any, dict], tar_dict: dict[Any, dict]):
        for key, value in src_dict.items():
            if key in tar_dict:
                tar_dict[key].update(value)
            else:
                tar_dict[key] = value

    @classmethod
    def from_struct(
        cls,
        ref_struct: Structure,
        model_struct: Structure,
        ori_model_chain_ids: list[str] | None = None,
        interested_lig_label_asym_id: str | list[str] | None = None,
        metric_config: ConfigDict = RUN_CONFIG.metric,
    ) -> "MetricResult":
        """
        Create a MetricResult instance from given structures and features.

        Args:
            ref_struct (Structure): The reference structure.
            model_struct (Structure): The model structure.
            ori_model_chain_ids (list[str]): A list of original model chain IDs.
            interested_lig_label_asym_id (str | list[str]): A string or list of strings
                specifying the ligand label asym IDs of interest.
            metric_config (dict[str, Any]): A dictionary containing configuration for
                          metrics. Defaults to RUN_CONFIG.metric.

        Returns:
            MetricResult: An instance of MetricResult containing the calculated metrics.

        The function performs the following steps:
        1. Maps chains from the reference structure to the model structure.
        2. Calculates RMSD (Root Mean Square Deviation) and updates the interface result dictionary.
        3. Calculates LDDT (Local Distance Difference Test) for the complex, chains, and interfaces.
        4. Calculates DockQ score and updates the interface result dictionary.
        5. Calculates PoseBusters validation score and set to the pb_valid attribute.
        """
        meta_info_dict = {}
        complex_result_dict = {}
        chain_result_dict = {}
        interface_result_dict = {}

        # Get chain mapping
        unique_ref_chain_id, indices = np.unique(
            ref_struct.uni_chain_id, return_index=True
        )
        chain_map = {
            ref_chain: model_struct.uni_chain_id[index]
            for ref_chain, index in zip(unique_ref_chain_id, indices)
        }

        # Update meta_info
        meta_info_dict["entry_id"] = ref_struct.entry_id
        meta_info_dict["ref_to_model_chain_mapping"] = chain_map
        meta_info_dict["ref_chain_info"] = cls._get_chain_info(ref_struct)

        # Calculate RMSD (if ligand and pocket specified in ref_features)
        if metric_config.calc_rmsd and interested_lig_label_asym_id:
            rmsd_metrics = RMSDMetrics(
                ref_struct,
                model_struct,
                ref_lig_label_asym_id=interested_lig_label_asym_id,
            )
            chain_rmsd_dict = rmsd_metrics.calc_pocket_aligned_rmsd()
            cls._update_src_to_tar_dict(
                src_dict=chain_rmsd_dict, tar_dict=chain_result_dict
            )

        # Calculate LDDT
        if metric_config.calc_lddt:
            chains, interfaces = ref_struct.get_chains_and_interfaces(
                interface_radius=5
            )
            chains, interfaces = MetricResult._remove_ion_from_chain_and_interface(
                ref_struct, chains, interfaces
            )
            calc_lddt = CalcLDDTMetric(
                ref_struct=ref_struct,
                model_struct=model_struct,
                lddt_config=metric_config.lddt,
            )
            complex_lddt = calc_lddt.get_complex_lddt()
            complex_result_dict["lddt"] = complex_lddt

            chain_interface_lddt = calc_lddt.get_chain_interface_lddt(
                chains, interfaces
            )
            (
                chain_lddt_dict,
                interface_lddt_dict,
            ) = cls._post_process_chain_interface_lddt(
                chains, interfaces, chain_interface_lddt
            )
            cls._update_src_to_tar_dict(chain_lddt_dict, chain_result_dict)
            cls._update_src_to_tar_dict(interface_lddt_dict, interface_result_dict)

        # Calculate DockQ
        if metric_config.calc_dockq:
            dockq_result_dict = compute_dockq(
                ref_struct=ref_struct,
                model_struct=model_struct,
                ref_to_model_chain_map=chain_map,
            )
            interface_dockq_dict = cls._post_process_dockq(dockq_result_dict)
            cls._update_src_to_tar_dict(interface_dockq_dict, interface_result_dict)

        # Calculate PoseBusters valid check
        if metric_config.calc_pb_valid and interested_lig_label_asym_id:
            pb_valid_result_df = compute_pb_valid(
                ref_struct=ref_struct,
                model_struct=model_struct,
                ref_lig_label_asym_id=interested_lig_label_asym_id,
            )
            chain_pb_valid_dict = cls._post_process_pb_valid(pb_valid_result_df)
        else:
            chain_pb_valid_dict = None

        return cls(
            ref_struct=ref_struct,
            model_struct=model_struct,
            meta_info=meta_info_dict,
            complex=complex_result_dict,
            chain=chain_result_dict,
            interface=interface_result_dict,
            pb_valid=chain_pb_valid_dict,
            ori_model_chain_ids=ori_model_chain_ids,
        )

    def to_json_dict(self) -> dict[str, Any]:
        """
        Convert the MetricResult instance to a dictionary.

        Returns:
            dict[str, Any]: A dictionary representation of the MetricResult instance.
        """

        json_dict = {}
        json_dict.update(self.meta_info)
        json_dict.update({"complex": self.complex})
        json_dict.update({"chain": self.chain})

        interface_json_dict = {}
        for k, v in self.interface.items():
            # chain_1_id, chain_2_id as the key for interface
            interface_json_dict[",".join(k)] = v
        json_dict.update({"interface": interface_json_dict})

        if self.pb_valid is not None:
            json_dict.update({"pb_valid": self.pb_valid})

        if self.ori_model_chain_ids is not None:
            json_dict["ori_model_chain_ids"] = self.ori_model_chain_ids
        return json_dict

    def to_json(self, json_file: Path, update_data: dict | None = None):
        """
        Convert the MetricResult instance to a JSON string.

        Args:
            json_file (str): The path to the JSON file where the result will be saved.
            update_data (dict, optional): Additional data to update the JSON dictionary.

        """
        json_dict = self.to_json_dict()

        if update_data:
            json_dict.update(update_data)

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(json_dict, f, indent=4, ensure_ascii=False)
