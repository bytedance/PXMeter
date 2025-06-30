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
from typing import Sequence

import numpy as np
import pandas as pd

from benchmark.configs.eval_type_config import (
    EVAL_TYPE_TO_ENTITIY_TYPES,
    PB_VALID_CHECK_COL,
)
from benchmark.configs.rankers_config import MODEL_TO_RANKER_KEYS


class ChainInterfaceDisplayer:
    """
    Displays the results of chains or interfaces evaluation.

    Args:
        metrics_df (pd.DataFrame): The DataFrame containing the metrics data.
        model (str, optional): The model name.
        seeds (list[str | int], optional): The list of seeds.
    """

    def __init__(
        self,
        metrics_df: pd.DataFrame,
        model: str | None = None,
        seeds: list[str | int] | None = None,
    ):
        self.metrics_df = metrics_df
        self.seeds = [str(i) for i in seeds] if seeds else None
        self.ranker_keys = MODEL_TO_RANKER_KEYS.get(model, MODEL_TO_RANKER_KEYS["nan"])

    @staticmethod
    def select_df_by_eval_types(
        metrics_df: pd.DataFrame, eval_types: list[str]
    ) -> pd.DataFrame:
        """
        Selects a subset of the metrics DataFrame based on the specified evaluation types.

        Args:
            metrics_df (pd.DataFrame): The DataFrame containing the metrics data.
            eval_types (list[str]): A list of evaluation types to consider.

        Returns:
            pd.DataFrame: A DataFrame containing the subset of metrics
                          data that matches the specified evaluation types.
        """
        mask = np.zeros(len(metrics_df), dtype=bool)
        for eval_type in eval_types:
            entity_type = EVAL_TYPE_TO_ENTITIY_TYPES[eval_type]
            if eval_type.startswith("Intra-"):
                # chain
                entity_type_mask = metrics_df.apply(
                    lambda row, e_type=entity_type: str(row["entity_type_1"])
                    == e_type[0]
                    and row["type"] == "chain",
                    axis=1,
                )
            else:
                # interface
                entity_type = sorted(entity_type)
                entity_type_mask = metrics_df.apply(
                    lambda row, e_type=entity_type: sorted(
                        [
                            str(row["entity_type_1"]),
                            str(row["entity_type_2"]),
                        ]
                    )
                    == e_type
                    and row["type"] == "interface",
                    axis=1,
                )
            mask |= entity_type_mask

        subset_metrics_df = metrics_df[mask].copy()
        return subset_metrics_df

    def _get_group_agg_funcs(
        self,
        metric_key: str,
        level: list[str] = None,
    ) -> dict[str, callable]:
        if level is None:
            level = [
                "complex",
                "chain",
                "interface",
            ]

        # Initialize with basic aggregation functions
        agg_funcs = {
            "best": lambda grp, m_key=metric_key: grp.loc[grp[m_key].idxmax()],
            "worst": lambda grp, m_key=metric_key: grp.loc[grp[m_key].idxmin()],
            "rand": lambda grp: grp.sample(n=1).iloc[0],
            "median": lambda grp, m_key=metric_key: grp.loc[
                (grp[m_key] - grp[m_key].median()).abs().idxmin()
            ],
        }

        ranker_keys = []
        for lv in level:
            ranker_keys += self.ranker_keys[lv]

        for ranker_key, ascending in ranker_keys:
            if ranker_key not in self.metrics_df.columns:
                continue

            def rank_func(
                grp,
                ranker_key=ranker_key,
                ascending=ascending,
            ):
                return grp.sort_values(by=ranker_key, ascending=ascending).iloc[0]

            agg_funcs[f"best.{ranker_key}"] = rank_func
        return agg_funcs

    def get_dockq_sr_by_cluster(
        self,
        eval_types: list[str],
        mask_on_metrics_df: Sequence[bool] | None = None,
        success_threshold=0.23,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate the average DockQ score and success rate for each cluster.

        Args:
            mask_on_metrics_df (Sequence[bool], optional): A boolean mask to apply to the metrics DataFrame.
            eval_types (list[str]): A list of evaluation types to consider.
            success_threshold (float): The threshold for considering a DockQ score as a success.
                              Default is 0.23.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
                - dockq_results_df: A DataFrame containing the average DockQ score and success rate for each cluster.
                - dockq_details_df: A DataFrame containing the details of each sample, including the seed, sample,
                                    ranker, eval type, entry ID, chain ID, cluster ID, and DockQ score.
        """
        if not eval_types:
            raise ValueError("At least one eval type must be provided.")

        if mask_on_metrics_df is not None:
            metrics_df = self.metrics_df[mask_on_metrics_df].copy()
        else:
            metrics_df = self.metrics_df.copy()

        if len(metrics_df) == 0:
            logging.warning("No data found for the given metrics_df.")
            return pd.DataFrame(), pd.DataFrame()

        if "cluster_id" not in metrics_df.columns:
            # Set a default cluster_id
            metrics_df["cluster_id"] = "nan_cluster_id"

        if self.seeds:
            # Selected by seeds
            metrics_df = metrics_df[
                metrics_df["seed"].astype(str).isin(self.seeds)
            ].copy()

        # Drop NaN rows in dockq column
        # For example: a peptide-peptide interface has no dockq score (7x6x C,D)
        metrics_df.dropna(subset=["dockq"], inplace=True, how="all", axis=0)

        # Func name: func apply to a group
        dockq_agg_func = self._get_group_agg_funcs(
            "dockq", level=["complex", "interface"]
        )

        dockq_results = []
        dockq_details = []
        for eval_type in eval_types:
            eval_df = self.select_df_by_eval_types(metrics_df, [eval_type])
            if len(eval_df) == 0:
                # No data for this eval type
                continue

            entry_id_num = len(eval_df["entry_id"].unique())

            for agg_func_name, agg_func in dockq_agg_func.items():
                cluster_id_to_dockq_scores = defaultdict(list)
                # DockQ only has interface metric
                group_by_key = ["entry_id", "chain_id_1", "chain_id_2"]
                for group_id, group_df in eval_df.groupby(by=group_by_key):
                    cluster_id = group_df["cluster_id"].iloc[0]
                    sample_dockq_row = agg_func(group_df)
                    sample_dockq_value = sample_dockq_row["dockq"]
                    cluster_id_to_dockq_scores[cluster_id].append(sample_dockq_value)

                    dockq_details.append(
                        {
                            "seed": sample_dockq_row["seed"],
                            "sample": sample_dockq_row["sample"],
                            "ranker": agg_func_name,
                            "eval_type": eval_type,
                            "entry_id": group_id[0],
                            "entity_id_1": sample_dockq_row["entity_id_1"],
                            "entity_id_2": sample_dockq_row["entity_id_2"],
                            "chain_id_1": group_id[1],
                            "chain_id_2": group_id[2],
                            "cluster_id": cluster_id,
                            "dockq": sample_dockq_value,
                        }
                    )

                # avg_dockq_sr_avg_sr: mean DockQ SR in a cluster, and mean for all SR for all clusters
                # avg_dockq_avg_sr: mean DockQ in a cluster, and mean SR for all clusters
                all_avg_dockq = []
                all_avg_dockq_sr = []
                for cluster_id, dockq_scores in cluster_id_to_dockq_scores.items():
                    avg_dockq = np.mean(dockq_scores)
                    avg_dockq_sr = np.mean(np.array(dockq_scores) > success_threshold)

                    all_avg_dockq.append(avg_dockq)
                    all_avg_dockq_sr.append(avg_dockq_sr)

                avg_dockq_avg_sr = np.mean(np.array(all_avg_dockq) > success_threshold)
                avg_dockq_sr_avg_sr = np.mean(all_avg_dockq_sr)
                dockq_result = {
                    "eval_type": eval_type,
                    "entry_id_num": entry_id_num,
                    "cluster_num": len(cluster_id_to_dockq_scores),
                    "ranker": agg_func_name,
                    "avg_dockq_avg_sr": avg_dockq_avg_sr,
                    "avg_dockq_sr_avg_sr": avg_dockq_sr_avg_sr,
                }
                dockq_results.append(dockq_result)
        dockq_results_df = pd.DataFrame(dockq_results)
        dockq_details_df = pd.DataFrame(dockq_details)
        return dockq_results_df, dockq_details_df

    def get_lddt_by_cluster(
        self,
        eval_types: list[str] | None = None,
        mask_on_metrics_df: Sequence[bool] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate the average LDDT score for each cluster.

        Args:
            eval_types (list[str], optional): A list of evaluation types to consider.
            mask_on_metrics_df (Sequence[bool], optional): A boolean mask to
                               apply to the metrics DataFrame.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
                - lddt_results_df: A DataFrame containing the average LDDT score for each cluster.
                - lddt_details_df: A DataFrame containing the details of each sample, including the seed, sample,
                                   ranker, eval type, entry ID, chain ID, cluster ID, and LDDT score.
        """
        if mask_on_metrics_df is not None:
            metrics_df = self.metrics_df[mask_on_metrics_df].copy()
        else:
            metrics_df = self.metrics_df.copy()

        if len(metrics_df) == 0:
            logging.warning("No data found for the given metrics_df.")
            return pd.DataFrame(), pd.DataFrame()

        if "cluster_id" not in metrics_df.columns:
            # Set a default cluster_id
            metrics_df["cluster_id"] = "nan_cluster_id"

        if self.seeds:
            # Selected by seeds
            metrics_df = metrics_df[
                metrics_df["seed"].astype(str).isin(self.seeds)
            ].copy()

        if eval_types is not None:
            all_eval_types = {
                k: v for k, v in EVAL_TYPE_TO_ENTITIY_TYPES.items() if k in eval_types
            }
        else:
            all_eval_types = EVAL_TYPE_TO_ENTITIY_TYPES

        lddt_results = []
        lddt_details = []
        lddt_chain_agg_funcs = self._get_group_agg_funcs(
            "lddt", level=["complex", "chain"]
        )
        lddt_interface_agg_funcs = self._get_group_agg_funcs(
            "lddt", level=["complex", "interface"]
        )
        for eval_type in all_eval_types.keys():
            eval_df = self.select_df_by_eval_types(metrics_df, [eval_type])

            # Drop NaN rows in lddt column
            eval_df.dropna(subset=["lddt"], inplace=True, how="all", axis=0)

            if len(eval_df) == 0:
                # No data for this eval_type
                continue

            if eval_type.startswith("Intra"):
                eval_type_level = "chain"
                lddt_agg_funcs = lddt_chain_agg_funcs
            else:
                eval_type_level = "interface"
                lddt_agg_funcs = lddt_interface_agg_funcs

            entry_id_num = len(eval_df["entry_id"].unique())

            for agg_func_name, agg_func in lddt_agg_funcs.items():
                cluster_id_to_lddt_scores = defaultdict(list)
                if eval_type_level == "chain":
                    group_by_key = ["entry_id", "chain_id_1"]
                else:
                    group_by_key = ["entry_id", "chain_id_1", "chain_id_2"]

                for group_id, group_df in eval_df.groupby(by=group_by_key):
                    cluster_id = group_df["cluster_id"].iloc[0]
                    sample_lddt_row = agg_func(group_df)
                    sample_lddt_value = sample_lddt_row["lddt"]
                    cluster_id_to_lddt_scores[cluster_id].append(sample_lddt_value)

                    if eval_type_level == "chain":
                        chain_id_2 = ""
                    else:
                        chain_id_2 = group_id[2]

                    lddt_details.append(
                        {
                            "seed": sample_lddt_row["seed"],
                            "sample": sample_lddt_row["sample"],
                            "ranker": agg_func_name,
                            "eval_type": eval_type,
                            "entry_id": group_id[0],
                            "entity_id_1": sample_lddt_row["entity_id_1"],
                            "entity_id_2": sample_lddt_row["entity_id_2"],
                            "chain_id_1": group_id[1],
                            "chain_id_2": chain_id_2,
                            "cluster_id": cluster_id,
                            "lddt": sample_lddt_value,
                        }
                    )

                all_avg_lddt = []
                for cluster_id, lddt_scores in cluster_id_to_lddt_scores.items():
                    avg_lddt = np.mean(lddt_scores)
                    all_avg_lddt.append(avg_lddt)
                avg_avg_lddt = np.mean(all_avg_lddt)

                lddt_result = {
                    "eval_type": eval_type,
                    "entry_id_num": entry_id_num,
                    "cluster_num": len(cluster_id_to_lddt_scores),
                    "ranker": agg_func_name,
                    "lddt": avg_avg_lddt,
                }
                lddt_results.append(lddt_result)

        lddt_results_df = pd.DataFrame(lddt_results)
        lddt_details_df = pd.DataFrame(lddt_details)
        return lddt_results_df, lddt_details_df


class RMSDDisplayer:
    """
    Displayer for RMSD metrics.

    Args:
        metrics_df (pd.DataFrame): The DataFrame containing the metrics data.
        pb_valid_df (pd.DataFrame, optional): The DataFrame containing the PB valid data.
        model (str, optional): The model name.
        seeds (list[str or int], optional): The list of seeds.
    """

    def __init__(
        self,
        metrics_df: pd.DataFrame,
        pb_valid_df: pd.DataFrame | None = None,
        model: str | None = None,
        seeds: list[str | int] | None = None,
    ):
        self.ranker_keys = MODEL_TO_RANKER_KEYS.get(model, MODEL_TO_RANKER_KEYS["nan"])
        self.metrics_df = (
            metrics_df
            if pb_valid_df is None
            else RMSDDisplayer._add_pb_valid_to_metrics_df(metrics_df, pb_valid_df)
        )

        self.seeds = [str(i) for i in seeds] if seeds else None

    @staticmethod
    def _add_pb_valid_to_metrics_df(
        metrics_df: pd.DataFrame, pb_valid_df: pd.DataFrame
    ) -> pd.DataFrame:
        match_keys = ["entry_id", "seed", "sample", "chain_id_1", "type"]

        merged_metrics_df = pd.merge(
            metrics_df,
            pb_valid_df[match_keys + PB_VALID_CHECK_COL],
            how="left",
            on=match_keys,
        )

        # Add penalty column as denominator: 0 or 100
        merged_metrics_df["penalty"] = merged_metrics_df.apply(
            lambda row: (
                not row["minimum_distance_to_protein"]
                or not row["tetrahedral_chirality"]
            )
            * 100,
            axis=1,
        )

        return merged_metrics_df

    def _get_rmsd_agg_funcs(
        self,
    ) -> dict[str, callable]:
        # Initialize with basic aggregation functions
        agg_funcs = {
            "best": lambda grp: grp.sort_values(
                by="lig_rmsd_wo_refl", ascending=True
            ).iloc[0],
            "worst": lambda grp: grp.sort_values(
                by="lig_rmsd_wo_refl", ascending=False
            ).iloc[0],
            "rand": lambda grp: grp.sample(n=1).iloc[0],
            "median": lambda grp: grp.loc[
                (grp["lig_rmsd_wo_refl"] - grp["lig_rmsd_wo_refl"].median())
                .abs()
                .idxmin()
            ],
        }

        for _level, ranker_list in self.ranker_keys.items():
            for ranker_key, ascending in ranker_list:
                if ranker_key not in self.metrics_df.columns:
                    continue

                def rank_func(grp, ranker_key=ranker_key, ascending=ascending):
                    return grp.sort_values(by=ranker_key, ascending=ascending).iloc[0]

                agg_funcs[f"best.{ranker_key}"] = rank_func

                if "penalty" in self.metrics_df.columns:

                    def penalized_rank_func(
                        grp, ranker_key=ranker_key, ascending=ascending
                    ):
                        penalized_ranker_key = f"{ranker_key}.penalized"
                        if ascending:
                            grp[penalized_ranker_key] = grp[ranker_key] + grp["penalty"]
                        else:
                            grp[penalized_ranker_key] = grp[ranker_key] - grp["penalty"]
                        return grp.sort_values(
                            by=penalized_ranker_key, ascending=ascending
                        ).iloc[0]

                    agg_funcs[f"best.{ranker_key}.penalized"] = penalized_rank_func

        return agg_funcs

    def get_rmsd(self, success_threshold=2.0) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate RMSD metrics for ligand and pocket chains.

        Args:
            success_threshold (float): The threshold for considering an RMSD value as a success.
                              Defaults to 2.0.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
                - rmsd_results_df: A DataFrame containing the calculated RMSD metrics.
                - rmsd_details_df: A DataFrame containing the details of each sample, including the seed, sample,
                                   ranker, eval type, entry ID, chain ID, cluster ID, and RMSD values.
        """
        # rmsd label on chain level (ligand chain)
        rmsd_df = self.metrics_df[self.metrics_df["type"] == "chain"].copy()

        if self.seeds:
            # Selected by seeds
            rmsd_df = rmsd_df[rmsd_df["seed"].astype(str).isin(self.seeds)].copy()

        # Drop NaN rows in lig_rmsd_wo_refl column
        rmsd_df.dropna(subset=["lig_rmsd_wo_refl"], inplace=True, how="all", axis=0)

        entry_id_num = len(rmsd_df["entry_id"].unique())

        existed_pb_check_rows = [i for i in PB_VALID_CHECK_COL if i in rmsd_df.columns]

        rmsd_results = []
        rmsd_details = []
        rmsd_agg_funcs = self._get_rmsd_agg_funcs()
        for agg_func_name, agg_func in rmsd_agg_funcs.items():
            all_lig_rmsd = []
            all_pocket_rmsd = []
            all_pb_valid = defaultdict(list)
            for group_id, group_df in rmsd_df.groupby(by=["entry_id", "chain_id_1"]):
                sample_row = agg_func(group_df)
                sample_lig_rmsd = sample_row["lig_rmsd_wo_refl"]
                sample_pocket_rmsd = sample_row["pocket_rmsd_wo_refl"]

                rmsd_detail = {
                    "seed": sample_row["seed"],
                    "sample": sample_row["sample"],
                    "eval_type": "RMSD",
                    "ranker": agg_func_name,
                    "entry_id": group_id[0],
                    "entity_id_1": sample_row["entity_id_1"],
                    "entity_id_2": "",
                    "chain_id_1": group_id[1],
                    "chain_id_2": "",
                    "cluster_id": "",
                    "lig_rmsd_wo_refl": sample_lig_rmsd,
                    "pocket_rmsd_wo_refl": sample_pocket_rmsd,
                }

                for check_col in existed_pb_check_rows:
                    check_result = sample_row[check_col]
                    rmsd_detail[check_col] = check_result
                    all_pb_valid[check_col].append(int(sample_row[check_col]))

                rmsd_details.append(rmsd_detail)
                all_lig_rmsd.append(sample_lig_rmsd)
                all_pocket_rmsd.append(sample_pocket_rmsd)

            all_lig_rmsd = np.array(all_lig_rmsd)
            lig_sr = np.mean(all_lig_rmsd < success_threshold)
            lig_avg_rmsd = np.mean(all_lig_rmsd)

            all_pocket_rmsd = np.array(all_pocket_rmsd)
            pocket_sr = np.mean(all_pocket_rmsd < success_threshold)
            pocket_avg_rmsd = np.mean(all_pocket_rmsd)

            rmsd_result = {
                "entry_id_num": entry_id_num,
                "ranker": agg_func_name,
                "lig_avg_rmsd": lig_avg_rmsd,
                "lig_rmsd_sr": lig_sr,
                "pocket_avg_rmsd": pocket_avg_rmsd,
                "pocket_rmsd_sr": pocket_sr,
            }

            for check_col in existed_pb_check_rows:
                rmsd_result[check_col] = np.mean(all_pb_valid[check_col])

            rmsd_results.append(rmsd_result)
        rmsd_results_df = pd.DataFrame(rmsd_results)
        rmsd_details_df = pd.DataFrame(rmsd_details)
        return rmsd_results_df, rmsd_details_df
