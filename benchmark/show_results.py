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

from collections import defaultdict
from typing import Sequence

import numpy as np
import pandas as pd

from benchmark.configs.eval_type_config import (
    EVAL_TYPE_TO_ENTITIY_TYPES,
    PB_VALID_CHECK_COL,
)
from benchmark.evaluators import MODEL_TO_RANKER_KEYS
from benchmark.utils import get_binomial_ci, get_bootstrap_ci, select_df_by_eval_types


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
        eval_types: list[str] | None = None,
        mask_on_metrics_df: Sequence[bool] | None = None,
        success_threshold: float = 0.23,
        subset_name: str = "All",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate the average DockQ score and success rate for each cluster.

        Args:
            eval_types (list[str] | None, optional): A list of evaluation types to consider.
                        Defaults to None.
            mask_on_metrics_df (Sequence[bool], optional): A boolean mask to apply to the metrics DataFrame.
                        Defaults to None.
            success_threshold (float): The threshold for considering a DockQ score as a success.
                              Default is 0.23.
            subset_name (str, optional): The name of the subset.
                        It will be added to the "subset" column of result DataFrame.
                        Defaults to 'All'.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
                - dockq_results_df: A DataFrame containing the average DockQ score and success rate for each cluster.
                - dockq_details_df: A DataFrame containing the details of each sample, including the seed, sample,
                                    ranker, eval type, entry ID, chain ID, cluster ID, and DockQ score.
        """
        if "dockq" not in self.metrics_df.columns:
            # single chain only
            return pd.DataFrame(), pd.DataFrame()

        if eval_types is not None:
            all_eval_types = {
                k: v for k, v in EVAL_TYPE_TO_ENTITIY_TYPES.items() if k in eval_types
            }
        else:
            all_eval_types = EVAL_TYPE_TO_ENTITIY_TYPES

        if mask_on_metrics_df is not None:
            metrics_df = self.metrics_df[mask_on_metrics_df].copy()
        else:
            metrics_df = self.metrics_df.copy()

        if len(metrics_df) == 0:
            return pd.DataFrame(), pd.DataFrame()

        if "cluster_id" not in metrics_df.columns:
            # Set a default cluster_id
            metrics_df["cluster_id"] = [str(i) for i in range(len(metrics_df))]

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
        for eval_type in all_eval_types:
            eval_df = select_df_by_eval_types(metrics_df, [eval_type])
            if len(eval_df) == 0:
                # No data for this eval type
                continue

            entry_id_num = len(eval_df["entry_id"].unique())

            for agg_func_name, agg_func in dockq_agg_func.items():
                cluster_id_to_dockq_scores = defaultdict(list)
                # DockQ only has interface metric
                group_by_key = ["entry_id", "chain_id_1", "chain_id_2"]
                for group_id, group_df in eval_df.groupby(
                    by=group_by_key, observed=True
                ):
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

                avg_dockq_sr = np.array(all_avg_dockq) > success_threshold
                avg_dockq_avg_sr = np.mean(avg_dockq_sr)
                avg_dockq_sr_avg_sr = np.mean(all_avg_dockq_sr)
                dockq_result = {
                    "eval_type": eval_type,
                    "entry_id_num": entry_id_num,
                    "cluster_num": len(cluster_id_to_dockq_scores),
                    "ranker": agg_func_name,
                    "avg_dockq_avg_sr": avg_dockq_avg_sr,
                    "avg_dockq_sr_avg_sr": avg_dockq_sr_avg_sr,
                    "ci_avg_dockq_avg_sr": get_binomial_ci(
                        total_num=len(avg_dockq_sr), success_num=avg_dockq_sr.sum()
                    ),
                    "ci_avg_dockq_sr_avg_sr": get_bootstrap_ci(all_avg_dockq_sr),
                }
                dockq_results.append(dockq_result)
        dockq_results_df = pd.DataFrame(dockq_results)
        dockq_details_df = pd.DataFrame(dockq_details)

        dockq_results_df["subset"] = subset_name
        dockq_details_df["subset"] = subset_name
        return dockq_results_df, dockq_details_df

    def get_lddt_by_cluster(
        self,
        eval_types: list[str] | None = None,
        mask_on_metrics_df: Sequence[bool] | None = None,
        subset_name: str = "All",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate the average LDDT score for each cluster.

        Args:
            eval_types (list[str], optional): A list of evaluation types to consider.
            mask_on_metrics_df (Sequence[bool], optional): A boolean mask to
                               apply to the metrics DataFrame.
            subset_name (str, optional): The name of the subset.
                        It will be added to the "subset" column of result DataFrame.
                        Defaults to 'All'.

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
            return pd.DataFrame(), pd.DataFrame()

        if "cluster_id" not in metrics_df.columns:
            # Set a default cluster_id
            metrics_df["cluster_id"] = [str(i) for i in range(len(metrics_df))]

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
            eval_df = select_df_by_eval_types(metrics_df, [eval_type])

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

                for group_id, group_df in eval_df.groupby(
                    by=group_by_key, observed=True
                ):
                    cluster_id = group_df["cluster_id"].iloc[0]
                    sample_lddt_row = agg_func(group_df)
                    sample_lddt_value = sample_lddt_row["lddt"]
                    cluster_id_to_lddt_scores[cluster_id].append(sample_lddt_value)

                    if eval_type_level == "chain":
                        chain_id_2 = ""
                        entity_id_2 = ""
                    else:
                        chain_id_2 = group_id[2]
                        entity_id_2 = sample_lddt_row["entity_id_2"]

                    lddt_details.append(
                        {
                            "seed": sample_lddt_row["seed"],
                            "sample": sample_lddt_row["sample"],
                            "ranker": agg_func_name,
                            "eval_type": eval_type,
                            "entry_id": group_id[0],
                            "entity_id_1": sample_lddt_row["entity_id_1"],
                            "entity_id_2": entity_id_2,
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

                lddt_ci = get_bootstrap_ci(all_avg_lddt)

                lddt_result = {
                    "eval_type": eval_type,
                    "entry_id_num": entry_id_num,
                    "cluster_num": len(cluster_id_to_lddt_scores),
                    "ranker": agg_func_name,
                    "lddt": avg_avg_lddt,
                    "ci_lddt": lddt_ci,
                }
                lddt_results.append(lddt_result)

        lddt_results_df = pd.DataFrame(lddt_results)
        lddt_details_df = pd.DataFrame(lddt_details)
        lddt_results_df["subset"] = subset_name
        lddt_details_df["subset"] = subset_name
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
            metrics_df.reset_index(),
            pb_valid_df[match_keys + PB_VALID_CHECK_COL],
            how="left",
            on=match_keys,
        ).set_index("index")
        merged_metrics_df.index.name = metrics_df.index.name

        # Add penalty column as denominator: 0 or 100
        merged_metrics_df["penalty"] = merged_metrics_df.apply(
            lambda row: (
                (
                    not pd.isna(row["minimum_distance_to_protein"])
                    and not row["minimum_distance_to_protein"]
                )
                or (
                    not pd.isna(row["tetrahedral_chirality"])
                    and not row["tetrahedral_chirality"]
                )
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
            "best": lambda grp: grp.sort_values(by="lig_rmsd", ascending=True).iloc[0],
            "worst": lambda grp: grp.sort_values(by="lig_rmsd", ascending=False).iloc[
                0
            ],
            "rand": lambda grp: grp.sample(n=1).iloc[0],
            "median": lambda grp: grp.loc[
                (grp["lig_rmsd"] - grp["lig_rmsd"].median()).abs().idxmin()
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

    def get_rmsd(
        self,
        success_threshold=2.0,
        mask_on_metrics_df: Sequence[bool] | None = None,
        subset_name: str = "All",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate RMSD metrics for ligand and pocket chains.

        Args:
            success_threshold (float): The threshold for considering an RMSD value as a success.
                            Defaults to 2.0.
            mask_on_metrics_df (Sequence[bool] | None): A mask to filter the metrics DataFrame. Defaults to None.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
                - rmsd_results_df: A DataFrame containing the calculated RMSD metrics.
                - rmsd_details_df: A DataFrame containing the details of each sample, including the seed, sample,
                                ranker, eval type, entry ID, chain ID, cluster ID, and RMSD values.
        """
        if mask_on_metrics_df is not None:
            metrics_df = self.metrics_df[mask_on_metrics_df].copy()
        else:
            metrics_df = self.metrics_df.copy()

        rmsd_df = metrics_df[metrics_df["type"] == "chain"].copy()

        if self.seeds:
            rmsd_df = rmsd_df[rmsd_df["seed"].astype(str).isin(self.seeds)].copy()

        # Rename columns for legacy compatibility
        if (
            "lig_rmsd_wo_refl" in rmsd_df.columns
            and "pocket_rmsd_wo_refl" in rmsd_df.columns
        ) and (
            "lig_rmsd" not in rmsd_df.columns and "pocket_rmsd" not in rmsd_df.columns
        ):
            rmsd_df = rmsd_df.rename(
                columns={
                    "lig_rmsd_wo_refl": "lig_rmsd",
                    "pocket_rmsd_wo_refl": "pocket_rmsd",
                }
            )

        # Drop NaN rows in lig_rmsd column
        rmsd_df.dropna(subset=["lig_rmsd"], inplace=True, how="all", axis=0)

        if "cluster_id" in rmsd_df.columns:
            # Drop NaN rows in cluster_id column (ligand not in low homology subset)
            rmsd_df.dropna(subset=["cluster_id"], inplace=True, how="all", axis=0)
        else:
            rmsd_df["cluster_id"] = (
                rmsd_df["entry_id"].astype(str)
                + "_"
                + rmsd_df["chain_id_1"].astype(str)
            )

        entry_id_num = len(rmsd_df["entry_id"].unique())

        existed_pb_check_rows = [i for i in PB_VALID_CHECK_COL if i in rmsd_df.columns]

        rmsd_results = []
        rmsd_details = []
        rmsd_agg_funcs = self._get_rmsd_agg_funcs()
        for agg_func_name, agg_func in rmsd_agg_funcs.items():
            all_lig_rmsd = []

            # cluster_pb_all_valid_flags[cluster_id] -> list[int]
            cluster_pb_all_valid_flags = defaultdict(list)
            cluster_pb_all_valid_and_good_rmsd_flags = defaultdict(list)

            # all_pb_valid[check_col][cluster_id] -> list[int]
            all_pb_valid = defaultdict(lambda: defaultdict(list))

            cluster_id_to_rmsd_scores = defaultdict(list)

            for group_id, group_df in rmsd_df.groupby(
                by=["entry_id", "chain_id_1"], observed=True
            ):
                sample_row = agg_func(group_df)
                sample_lig_rmsd = sample_row["lig_rmsd"]
                sample_pocket_rmsd = sample_row["pocket_rmsd"]
                cluster_id = sample_row["cluster_id"]

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
                    "cluster_id": cluster_id,
                    "lig_rmsd": sample_lig_rmsd,
                    "pocket_rmsd": sample_pocket_rmsd,
                }

                sample_pb_flags = []
                for check_col in existed_pb_check_rows:
                    check_result = sample_row[check_col]
                    # if NaN, it will be True -> 1
                    check_flag = 1 if pd.isna(check_result) or bool(check_result) else 0
                    rmsd_detail[check_col] = check_flag
                    all_pb_valid[check_col][cluster_id].append(check_flag)
                    sample_pb_flags.append(check_flag)

                if existed_pb_check_rows:
                    all_valid_flag = int(all(sample_pb_flags))
                    cluster_pb_all_valid_flags[cluster_id].append(all_valid_flag)

                    good_rmsd_flag = int(
                        (sample_lig_rmsd is not None)
                        and (sample_lig_rmsd < success_threshold)
                        and bool(all_valid_flag)
                    )
                    cluster_pb_all_valid_and_good_rmsd_flags[cluster_id].append(
                        good_rmsd_flag
                    )

                rmsd_details.append(rmsd_detail)
                all_lig_rmsd.append(sample_lig_rmsd)
                cluster_id_to_rmsd_scores[cluster_id].append(sample_lig_rmsd)

            if len(all_lig_rmsd) == 0:
                continue

            all_avg_lig_rmsd = []
            all_avg_lig_rmsd_sr = []
            for cluster_id, rmsd_scores in cluster_id_to_rmsd_scores.items():
                rmsd_arr = np.asarray(rmsd_scores, dtype=float)
                avg_lig_rmsd = float(np.mean(rmsd_arr))
                avg_lig_rmsd_sr = float(np.mean(rmsd_arr < success_threshold))
                all_avg_lig_rmsd.append(avg_lig_rmsd)
                all_avg_lig_rmsd_sr.append(avg_lig_rmsd_sr)

            avg_lig_rmsd = float(np.mean(all_avg_lig_rmsd))
            avg_lig_rmsd_sr_avg_sr = float(np.mean(all_avg_lig_rmsd_sr))

            all_lig_rmsd_arr = np.asarray(all_lig_rmsd, dtype=float)
            if len(all_lig_rmsd) == len(cluster_id_to_rmsd_scores):
                # N_cluster == N_sample
                lig_sr_ci = get_binomial_ci(
                    total_num=len(all_lig_rmsd),
                    success_num=int((all_lig_rmsd_arr < success_threshold).sum()),
                )
            else:
                lig_sr_ci = get_bootstrap_ci(all_avg_lig_rmsd_sr)

            rmsd_result = {
                "entry_id_num": entry_id_num,
                "cluster_num": len(cluster_id_to_rmsd_scores),
                "ranker": agg_func_name,
                "lig_avg_rmsd": avg_lig_rmsd,
                "lig_rmsd_sr": avg_lig_rmsd_sr_avg_sr,
                "ci_lig_avg_rmsd": get_bootstrap_ci(all_avg_lig_rmsd),
                "ci_lig_rmsd_sr": lig_sr_ci,
            }

            for check_col in existed_pb_check_rows:
                cluster_means = []
                for cluster_results in all_pb_valid[check_col].values():
                    cluster_means.append(float(np.mean(cluster_results)))
                rmsd_result[check_col] = (
                    float(np.mean(cluster_means)) if cluster_means else np.nan
                )

            if existed_pb_check_rows:
                cluster_all_valid_means = []
                for flags in cluster_pb_all_valid_flags.values():
                    cluster_all_valid_means.append(float(np.mean(flags)))
                rmsd_result["pb_all_valid_sr"] = (
                    float(np.mean(cluster_all_valid_means))
                    if cluster_all_valid_means
                    else np.nan
                )

                cluster_all_valid_and_good_rmsd_means = []
                for flags in cluster_pb_all_valid_and_good_rmsd_flags.values():
                    cluster_all_valid_and_good_rmsd_means.append(float(np.mean(flags)))
                rmsd_result["pb_all_valid_and_good_rmsd_sr"] = (
                    float(np.mean(cluster_all_valid_and_good_rmsd_means))
                    if cluster_all_valid_and_good_rmsd_means
                    else np.nan
                )

            rmsd_results.append(rmsd_result)

        rmsd_results_df = pd.DataFrame(rmsd_results)
        rmsd_details_df = pd.DataFrame(rmsd_details)
        rmsd_results_df["subset"] = subset_name
        rmsd_details_df["subset"] = subset_name
        return rmsd_results_df, rmsd_details_df
