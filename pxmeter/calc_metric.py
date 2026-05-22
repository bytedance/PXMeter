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

import dataclasses
import json
import warnings
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from ml_collections.config_dict import ConfigDict

from pxmeter.configs.run_config import RUN_CONFIG
from pxmeter.constants import IONS, LIGAND
from pxmeter.data.ccd import get_ccd_mol_from_chain_atom_array
from pxmeter.data.struct import Structure
from pxmeter.metrics.dockq import compute_dockq
from pxmeter.metrics.lddt_metrics import LDDT
from pxmeter.metrics.pb_valid import run_pb_valid
from pxmeter.metrics.rmsd_metrics import RMSDMetrics

warnings.filterwarnings("ignore", message="The coordinates are missing for some atoms")


def compute_pb_valid(
    ref_struct: Structure,
    model_struct: Structure,
    ref_lig_label_asym_id: Union[str, list[str]],
) -> Optional[pd.DataFrame]:
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
        model_lig_atom_array = model_struct.atom_array[lig_mask].copy()
        # reset res_name for model ligand atoms by ref Structure
        model_lig_atom_array.res_name = ref_lig_atom_array.res_name
        model_cond_atom_array = model_struct.atom_array[~lig_mask].copy()

        if model_struct.valid_mask is not None:
            model_cond_valid_mask = model_struct.valid_mask[~lig_mask].copy()
            model_lig_valid_mask = model_struct.valid_mask[lig_mask].copy()
        else:
            model_cond_valid_mask = None
            model_lig_valid_mask = None

        ref_lig_mol = get_ccd_mol_from_chain_atom_array(ref_lig_atom_array)
        model_lig_mol = get_ccd_mol_from_chain_atom_array(model_lig_atom_array)

        df = run_pb_valid(
            mol_pred=model_lig_mol,
            mol_true=ref_lig_mol,
            mol_cond=model_cond_atom_array,
            mol_cond_valid_mask=model_cond_valid_mask,
            mol_pred_valid_mask=model_lig_valid_mask,
        )

        # record ligand chain id
        df["ref_lig_chain_id"] = ref_lig_chain_id
        df["model_lig_chain_id"] = model_lig_chain_id
        if not df.empty:
            df_list.append(df)
    if not df_list:
        return pd.DataFrame()

    # Avoid FutureWarning in pandas 2.1+ by excluding all-NA columns from entries
    # and ensuring we only concat non-empty DataFrames.
    df_list = [d.dropna(axis=1, how="all") for d in df_list if not d.empty]
    if not df_list:
        return pd.DataFrame()
    df_cat = pd.concat(df_list, ignore_index=True)
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
            stereochecks=lddt_config.stereochecks,
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

    def get_complex_lddt(self, atom_mask: Optional[np.ndarray] = None) -> float:
        """
        Calculate the LDDT score for a complex.

        This method uses the LDDT calculator to compute the LDDT score based on the predicted
        and true coordinates of the complex. The LDDT score is a measure of the
        structural similarity between the predicted and true structures.

        Args:
            atom_mask (np.ndarray): A mask for the atoms to include in the calculation.

        Returns:
            float: The LDDT score for the complex.
        """
        # complex_lddt = [1]
        complex_lddt = self.lddt_calculator.run(
            chain_1_masks=None,
            chain_2_masks=None,
            atom_mask=atom_mask,
        )
        return complex_lddt

    def get_chain_interface_lddt(
        self,
        chains: list[str],
        interfaces: list[tuple[str, str]],
        atom_mask: Optional[np.ndarray] = None,
    ) -> list[float]:
        """
        Calculate the LDDT scores for chains and interfaces.

        Args:
            chains (list[str]): A list of chain identifiers.
            interfaces (list[tuple[str, str]]): A list of tuples, each containing
                two chain identifiers representing an interface.
            atom_mask (np.ndarray, optional): A mask for the atoms to include in the calculation.
                Defaults to None.

        Returns:
            list[float]: A list of LDDT scores for chains and interfaces.
        """
        merged_chain_1_masks, merged_chain_2_masks = self.get_chains_mask(
            chains, interfaces
        )

        lddt_list = self.lddt_calculator.run(
            chain_1_masks=merged_chain_1_masks,
            chain_2_masks=merged_chain_2_masks,
            atom_mask=atom_mask,
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
    pb_valid: Optional[dict[str, dict[str, Any]]] = None

    ori_model_chain_ids: Optional[list[str]] = None

    update_data: Optional[dict[str, Any]] = None

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
        metric_name: str = "lddt",
    ) -> tuple[dict[str, dict[str, float]], dict[tuple[str, str], dict[str, float]]]:
        chain_lddt_dict = {}
        interface_lddt_dict = {}
        num_chains = len(chains)
        for idx, chain_id in enumerate(chains):
            lddt_value = chain_interface_lddt[idx]
            if np.isnan(lddt_value):
                continue
            chain_lddt_dict[chain_id] = {metric_name: lddt_value}

        for idx, interface in enumerate(interfaces):
            sorted_interface = tuple(
                sorted(interface)
            )  # Sort chains to ensure consistent order
            lddt_value = chain_interface_lddt[idx + num_chains]
            if np.isnan(lddt_value):
                continue
            interface_lddt_dict[sorted_interface] = {metric_name: lddt_value}
        return chain_lddt_dict, interface_lddt_dict

    @staticmethod
    def _post_process_dockq(
        dockq_result_dict: dict[str, Any],
    ) -> dict[str, Union[float, dict[str, float]]]:
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
        pb_valid_result_df: Optional[pd.DataFrame],
    ) -> Optional[dict[str, dict[str, Any]]]:
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

    @staticmethod
    def _calc_stereochecks_summary(
        atom_mask: np.ndarray,
        clash_df: pd.DataFrame,
        bad_bond_df: pd.DataFrame,
        bad_angle_df: pd.DataFrame,
    ) -> dict[str, int]:
        """
        ggregate stereochemistry violations within an atom subset.

        - `clash_atoms`: number of unique atoms involved in clashes (within subset)
        - `bad_bonds`: number of bad bonds (within subset)
        - `bad_angles`: number of bad angles (within subset)

        The `idx*` columns in DataFrames are indices into the mapped atom arrays.
        """

        atom_mask = np.asarray(atom_mask, dtype=bool)

        clash_atoms = 0
        if clash_df is not None and (not clash_df.empty):
            idx1 = clash_df["idx1"].to_numpy(dtype=np.int64, copy=False)
            idx2 = clash_df["idx2"].to_numpy(dtype=np.int64, copy=False)
            row_mask = atom_mask[idx1] & atom_mask[idx2]
            if np.any(row_mask):
                clash_atoms = int(
                    np.unique(np.concatenate([idx1[row_mask], idx2[row_mask]])).size
                )

        bond_cnt = 0
        if bad_bond_df is not None and (not bad_bond_df.empty):
            idx1 = bad_bond_df["idx1"].to_numpy(dtype=np.int64, copy=False)
            idx2 = bad_bond_df["idx2"].to_numpy(dtype=np.int64, copy=False)
            bond_cnt = int(np.sum(atom_mask[idx1] & atom_mask[idx2]))

        angle_cnt = 0
        if bad_angle_df is not None and (not bad_angle_df.empty):
            idx_a = bad_angle_df["idx_a"].to_numpy(dtype=np.int64, copy=False)
            idx_b = bad_angle_df["idx_b"].to_numpy(dtype=np.int64, copy=False)
            idx_c = bad_angle_df["idx_c"].to_numpy(dtype=np.int64, copy=False)
            angle_cnt = int(
                np.sum(atom_mask[idx_a] & atom_mask[idx_b] & atom_mask[idx_c])
            )

        return {
            "clash_atoms": clash_atoms,
            "bad_bonds": bond_cnt,
            "bad_angles": angle_cnt,
        }

    @classmethod
    def _maybe_add_lddt_stereochecks_summaries(
        cls,
        *,
        lddt_config: ConfigDict,
        lddt_calculator: LDDT,
        ref_struct: Structure,
        chains: list[str],
        interfaces: list[tuple[str, str]],
        complex_result_dict: dict[str, Any],
        chain_result_dict: dict[str, dict[str, Any]],
        interface_result_dict: dict[tuple[str, str], dict[str, Any]],
    ) -> None:
        """Attach stereochemistry violation summaries to output dicts.

        Only active when `metric.lddt.stereochecks=True` and the underlying
        stereochemistry checker produced violation tables.
        """

        if not lddt_config.stereochecks:
            return

        stereo_violation_dfs = getattr(lddt_calculator, "stereo_violation_dfs", None)
        if stereo_violation_dfs is None:
            return

        clash_df, bad_bond_df, bad_angle_df = stereo_violation_dfs
        n_atoms = len(ref_struct.atom_array)

        # Complex-level summary
        complex_result_dict["stereochecks"] = cls._calc_stereochecks_summary(
            atom_mask=np.ones(n_atoms, dtype=bool),
            clash_df=clash_df,
            bad_bond_df=bad_bond_df,
            bad_angle_df=bad_angle_df,
        )

        # Chain-level summary (keyed by reference chain IDs)
        for chain_id in chains:
            chain_atom_mask = ref_struct.uni_chain_id == chain_id
            chain_result_dict.setdefault(chain_id, {})[
                "stereochecks"
            ] = cls._calc_stereochecks_summary(
                atom_mask=chain_atom_mask,
                clash_df=clash_df,
                bad_bond_df=bad_bond_df,
                bad_angle_df=bad_angle_df,
            )

        # Interface-level summary (keyed by sorted(reference chain IDs))
        for chain_1, chain_2 in interfaces:
            interface_key = tuple(sorted((chain_1, chain_2)))
            interface_atom_mask = (ref_struct.uni_chain_id == chain_1) | (
                ref_struct.uni_chain_id == chain_2
            )
            interface_result_dict.setdefault(interface_key, {})[
                "stereochecks"
            ] = cls._calc_stereochecks_summary(
                atom_mask=interface_atom_mask,
                clash_df=clash_df,
                bad_bond_df=bad_bond_df,
                bad_angle_df=bad_angle_df,
            )

    @classmethod
    def from_struct(
        cls,
        ref_struct: Structure,
        model_struct: Structure,
        ori_model_chain_ids: Optional[list[str]] = None,
        interested_lig_label_asym_id: Optional[Union[str, list[str]]] = None,
        metric_config: ConfigDict = RUN_CONFIG.metric,
        update_data: Optional[dict[str, Any]] = None,
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
            update_data (dict[str, Any] | None): A dictionary containing additional data to update.
                Defaults to None.

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

            cls._maybe_add_lddt_stereochecks_summaries(
                lddt_config=metric_config.lddt,
                lddt_calculator=calc_lddt.lddt_calculator,
                ref_struct=ref_struct,
                chains=chains,
                interfaces=interfaces,
                complex_result_dict=complex_result_dict,
                chain_result_dict=chain_result_dict,
                interface_result_dict=interface_result_dict,
            )

            complex_lddt = calc_lddt.get_complex_lddt()
            if not np.isnan(complex_lddt):
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

            if metric_config.lddt.calc_backbone_lddt:
                backbone_mask = ref_struct.get_backbone_atom_masks(only_rep_atom=True)
                complex_bb_lddt = calc_lddt.get_complex_lddt(atom_mask=backbone_mask)
                if not np.isnan(complex_bb_lddt):
                    complex_result_dict["bb_lddt"] = complex_bb_lddt

                # It reuses the chains and interfaces from the previous step
                chain_interface_lddt = calc_lddt.get_chain_interface_lddt(
                    chains, interfaces, atom_mask=backbone_mask
                )
                (
                    chain_bb_lddt_dict,
                    interface_bb_lddt_dict,
                ) = cls._post_process_chain_interface_lddt(
                    chains, interfaces, chain_interface_lddt, metric_name="bb_lddt"
                )
                cls._update_src_to_tar_dict(chain_bb_lddt_dict, chain_result_dict)
                cls._update_src_to_tar_dict(
                    interface_bb_lddt_dict, interface_result_dict
                )

            # Calculate LDDT-PLI
            if interested_lig_label_asym_id and metric_config.lddt.calc_lddt_pli:
                lig_chain_to_calc = interested_lig_label_asym_id
                if isinstance(interested_lig_label_asym_id, str):
                    lig_chain_to_calc = [interested_lig_label_asym_id]

                for lig_chain_id in lig_chain_to_calc:
                    lddt_pli = calc_lddt.lddt_calculator.calc_lddt_pli(
                        ref_lig_label_asym_id=lig_chain_id,
                        inclusion_radius=6.0,
                    )
                    if not np.isnan(lddt_pli):
                        if lig_chain_id not in chain_result_dict:
                            chain_result_dict[lig_chain_id] = {}
                        chain_result_dict[lig_chain_id]["lddt_pli"] = lddt_pli

        # Calculate DockQ
        if metric_config.calc_dockq:
            dockq_result_dict = compute_dockq(
                ref_struct=ref_struct,
                model_struct=model_struct,
                ref_to_model_chain_map=chain_map,
                exclude_hetatms=metric_config.dockq.exclude_hetatms,
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

        # Calculate CDR-H3 RMSD
        if metric_config.calc_cdr_h3_bb_rmsd:
            from pxmeter.metrics.antibody.cdr_h3_rmsd import calc_cdr_h3_bb_rmsd

            cdr_h3_rmsd_result = calc_cdr_h3_bb_rmsd(
                ref_struct=ref_struct,
                model_struct=model_struct,
            )
            # update chain_result_dict
            for chain_id, rmsd_val in cdr_h3_rmsd_result.items():
                if chain_id not in chain_result_dict:
                    chain_result_dict[chain_id] = {}
                chain_result_dict[chain_id]["cdr_h3_bb_rmsd"] = rmsd_val

        return cls(
            ref_struct=ref_struct,
            model_struct=model_struct,
            meta_info=meta_info_dict,
            complex=complex_result_dict,
            chain=chain_result_dict,
            interface=interface_result_dict,
            pb_valid=chain_pb_valid_dict,
            ori_model_chain_ids=ori_model_chain_ids,
            update_data=update_data,
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

    def to_json(self, json_file: Path, update_data: Optional[dict] = None):
        """
        Convert the MetricResult instance to a JSON string.

        Args:
            json_file (str): The path to the JSON file where the result will be saved.
            update_data (dict, optional): Additional data to update the JSON dictionary.

        """
        json_dict = self.to_json_dict()

        if update_data:
            json_dict.update(update_data)

        if self.update_data is not None:
            json_dict.update(self.update_data)

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(json_dict, f, indent=4, ensure_ascii=False)
