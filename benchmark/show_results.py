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

    def _get_ranker_names(
        self,
        level: list[str] = None,
    ) -> list[str]:
        if level is None:
            level = [
                "complex",
                "chain",
                "interface",
            ]

        ranker_names = ["best", "worst", "rand", "median"]

        ranker_keys = []
        for lv in level:
            ranker_keys += self.ranker_keys[lv]

        for ranker_key, _ in ranker_keys:
            if ranker_key not in self.metrics_df.columns:
                continue
            ranker_names.append(f"best.{ranker_key}")
        return ranker_names

    def _select_representative_samples(
        self,
        df: pd.DataFrame,
        agg_func_name: str,
        group_by_key: list[str],
        metric_key: str,
        ranker_lookup: dict[str, bool],
    ) -> pd.DataFrame | None:
        """
        Selects representative samples for each group based on the aggregation strategy.
        Returns a sorted and deduplicated DataFrame if the strategy is vectorizable.
        Otherwise returns None.
        """
        # Define tie-breakers for deterministic sorting
        tie_breakers = []
        tie_asc = []
        if "seed" in df.columns:
            tie_breakers.append("seed")
            tie_asc.append(True)
        if "sample" in df.columns:
            tie_breakers.append("sample")
            tie_asc.append(True)

        if agg_func_name == "best":
            return df.sort_values(
                by=[metric_key] + tie_breakers, ascending=[False] + tie_asc
            ).drop_duplicates(subset=group_by_key)
        elif agg_func_name == "worst":
            return df.sort_values(
                by=[metric_key] + tie_breakers, ascending=[True] + tie_asc
            ).drop_duplicates(subset=group_by_key)
        elif agg_func_name == "rand":
            return df.sample(frac=1).drop_duplicates(subset=group_by_key)
        elif agg_func_name == "median":
            # Vectorized median selection
            # Calculate median for each group
            median_series = df.groupby(group_by_key, observed=True)[
                metric_key
            ].transform("median")
            # Calculate absolute difference
            # Use a temporary column name unlikely to clash
            diff_col = f"_median_diff_{metric_key}"
            df = df.copy()
            df[diff_col] = (df[metric_key] - median_series).abs()
            # Sort by difference
            return (
                df.sort_values(
                    by=[diff_col] + tie_breakers,
                    ascending=[True] + tie_asc,
                )
                .drop_duplicates(subset=group_by_key)
                .drop(columns=[diff_col])
            )
        if agg_func_name in ranker_lookup:
            asc = ranker_lookup[agg_func_name]
            # agg_func_name is like "best.{ranker_key}"
            rk = agg_func_name.split(".", 1)[1]
            if rk in df.columns:
                # Ensure the column used for sorting is numeric
                # Create a temporary numeric column for sorting to handle string issues
                # such as "10.0" < "9.0" being True when sorting as strings
                temp_col = f"_tmp_sort_{rk}"
                df = df.copy()
                df[temp_col] = pd.to_numeric(df[rk], errors="coerce")

                sorted_df = df.sort_values(
                    by=[temp_col] + tie_breakers, ascending=[asc] + tie_asc
                )

                return sorted_df.drop_duplicates(subset=group_by_key).drop(
                    columns=[temp_col]
                )

        raise ValueError(f"Unknown aggregation strategy: {agg_func_name}")

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
        else:
            metrics_df = metrics_df.dropna(subset=["cluster_id"]).copy()

        if self.seeds:
            # Selected by seeds
            metrics_df = metrics_df[
                metrics_df["seed"].astype(str).isin(self.seeds)
            ].copy()

        # Drop NaN rows in dockq column
        # For example: a peptide-peptide interface has no dockq score (7x6x C,D)
        metrics_df.dropna(subset=["dockq"], inplace=True, how="all", axis=0)

        # Func name: func apply to a group
        dockq_ranker_names = self._get_ranker_names(level=["complex", "interface"])

        ranker_lookup = {}
        for lv in ["complex", "interface"]:
            if lv in self.ranker_keys:
                for rk, asc in self.ranker_keys[lv]:
                    ranker_lookup[f"best.{rk}"] = asc

        dockq_results = []
        dockq_details = []
        for eval_type in all_eval_types:
            eval_df = select_df_by_eval_types(metrics_df, [eval_type])
            if len(eval_df) == 0:
                # No data for this eval type
                continue

            entry_id_num = len(eval_df["entry_id"].unique())

            for agg_func_name in dockq_ranker_names:
                cluster_id_to_dockq_scores = defaultdict(list)
                # DockQ only has interface metric
                group_by_key = ["entry_id", "chain_id_1", "chain_id_2"]

                selected_df = self._select_representative_samples(
                    eval_df, agg_func_name, group_by_key, "dockq", ranker_lookup
                )

                # Vectorized path
                grouped_scores = selected_df.groupby("cluster_id", observed=True)[
                    "dockq"
                ].apply(list)
                for cid, scores in grouped_scores.items():
                    cluster_id_to_dockq_scores[cid].extend(scores)

                details_subset = selected_df.copy()
                details_subset["ranker"] = agg_func_name
                details_subset["eval_type"] = eval_type

                ranker_metric = None
                if agg_func_name.startswith("best."):
                    ranker_metric = agg_func_name.split(".", 1)[1]
                elif agg_func_name in ["best", "worst", "median", "rand"]:
                    ranker_metric = "dockq"

                if ranker_metric and ranker_metric in details_subset.columns:
                    details_subset["ranker_val"] = details_subset[ranker_metric]
                else:
                    details_subset["ranker_val"] = np.nan

                detail_records = details_subset[
                    [
                        "seed",
                        "sample",
                        "ranker",
                        "ranker_val",
                        "eval_type",
                        "entry_id",
                        "entity_id_1",
                        "entity_id_2",
                        "chain_id_1",
                        "chain_id_2",
                        "cluster_id",
                        "dockq",
                    ]
                ].to_dict("records")
                dockq_details.extend(detail_records)

                # avg_dockq_sr_avg_sr: mean DockQ SR in a cluster, and mean for all SR for all clusters
                # avg_dockq_avg_sr: mean DockQ in a cluster, and mean SR for all clusters
                all_avg_dockq = []
                all_avg_dockq_sr = []
                for _cluster_id, dockq_scores in cluster_id_to_dockq_scores.items():
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
        else:
            metrics_df = metrics_df.dropna(subset=["cluster_id"]).copy()

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
        lddt_chain_ranker_names = self._get_ranker_names(level=["complex", "chain"])
        lddt_interface_ranker_names = self._get_ranker_names(
            level=["complex", "interface"]
        )

        chain_ranker_lookup = {}
        for lv in ["complex", "chain"]:
            if lv in self.ranker_keys:
                for rk, asc in self.ranker_keys[lv]:
                    chain_ranker_lookup[f"best.{rk}"] = asc

        interface_ranker_lookup = {}
        for lv in ["complex", "interface"]:
            if lv in self.ranker_keys:
                for rk, asc in self.ranker_keys[lv]:
                    interface_ranker_lookup[f"best.{rk}"] = asc

        for eval_type in all_eval_types.keys():
            eval_df = select_df_by_eval_types(metrics_df, [eval_type])

            # Drop NaN rows in lddt column
            eval_df.dropna(subset=["lddt"], inplace=True, how="all", axis=0)

            if len(eval_df) == 0:
                # No data for this eval_type
                continue

            if eval_type.startswith("Intra"):
                eval_type_level = "chain"
                lddt_ranker_names = lddt_chain_ranker_names
                ranker_lookup = chain_ranker_lookup
            else:
                eval_type_level = "interface"
                lddt_ranker_names = lddt_interface_ranker_names
                ranker_lookup = interface_ranker_lookup

            entry_id_num = len(eval_df["entry_id"].unique())

            for agg_func_name in lddt_ranker_names:
                cluster_id_to_lddt_scores = defaultdict(list)
                if eval_type_level == "chain":
                    group_by_key = ["entry_id", "chain_id_1"]
                else:
                    group_by_key = ["entry_id", "chain_id_1", "chain_id_2"]

                selected_df = self._select_representative_samples(
                    eval_df, agg_func_name, group_by_key, "lddt", ranker_lookup
                )

                grouped_scores = selected_df.groupby("cluster_id", observed=True)[
                    "lddt"
                ].apply(list)
                for cid, scores in grouped_scores.items():
                    cluster_id_to_lddt_scores[cid].extend(scores)

                details_subset = selected_df.copy()
                details_subset["ranker"] = agg_func_name
                details_subset["eval_type"] = eval_type

                ranker_metric = None
                if agg_func_name.startswith("best."):
                    ranker_metric = agg_func_name.split(".", 1)[1]
                elif agg_func_name in ["best", "worst", "median", "rand"]:
                    ranker_metric = "lddt"

                if ranker_metric and ranker_metric in details_subset.columns:
                    details_subset["ranker_val"] = details_subset[ranker_metric]
                else:
                    details_subset["ranker_val"] = np.nan

                if eval_type_level == "chain":
                    details_subset["chain_id_2"] = ""
                    details_subset["entity_id_2"] = ""

                fields = [
                    "seed",
                    "sample",
                    "ranker",
                    "ranker_val",
                    "eval_type",
                    "entry_id",
                    "entity_id_1",
                    "chain_id_1",
                    "cluster_id",
                    "lddt",
                    "entity_id_2",
                    "chain_id_2",
                ]

                lddt_details_records = details_subset[fields].to_dict("records")
                lddt_details.extend(lddt_details_records)

                all_avg_lddt = []
                for _cluster_id, lddt_scores in cluster_id_to_lddt_scores.items():
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

    def get_lddt_pli(
        self,
        mask_on_metrics_df: Sequence[bool] | None = None,
        subset_name: str = "All",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate LDDT-PLI metrics for ligand chains.

        Args:
            mask_on_metrics_df (Sequence[bool] | None): A mask to filter the metrics DataFrame. Defaults to None.
            subset_name (str): The name of the subset.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
                - lddt_results_df: A DataFrame containing the calculated LDDT-PLI metrics.
                - lddt_details_df: A DataFrame containing the details of each sample.
        """
        if mask_on_metrics_df is not None:
            metrics_df = self.metrics_df[mask_on_metrics_df].copy()
        else:
            metrics_df = self.metrics_df.copy()

        lddt_df = metrics_df[metrics_df["type"] == "chain"].copy()

        if self.seeds:
            lddt_df = lddt_df[lddt_df["seed"].astype(str).isin(self.seeds)].copy()

        if "lddt_pli" not in lddt_df.columns:
            return pd.DataFrame(), pd.DataFrame()

        # Drop NaN rows in lddt_pli column
        lddt_df.dropna(subset=["lddt_pli"], inplace=True, how="all", axis=0)
        if lddt_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        # LDDT-PLI is agregated without cluster_id
        lddt_df["cluster_id"] = (
            lddt_df["entry_id"].astype(str) + "_" + lddt_df["chain_id_1"].astype(str)
        )

        entry_id_num = len(lddt_df["entry_id"].unique())

        lddt_results = []
        lddt_details = []

        ranker_names = self._get_ranker_names(level=["complex", "chain"])

        ranker_lookup = {}
        for lv in ["complex", "chain"]:
            if lv in self.ranker_keys:
                for rk, asc in self.ranker_keys[lv]:
                    ranker_lookup[f"best.{rk}"] = asc

        for agg_func_name in ranker_names:
            cluster_id_to_lddt_scores = defaultdict(list)

            group_by_key = ["entry_id", "chain_id_1"]
            selected_df = self._select_representative_samples(
                lddt_df, agg_func_name, group_by_key, "lddt_pli", ranker_lookup
            )

            # Vectorized
            grouped_scores = selected_df.groupby("cluster_id", observed=True)[
                "lddt_pli"
            ].apply(list)
            for cid, scores in grouped_scores.items():
                cluster_id_to_lddt_scores[cid].extend(scores)

            details_subset = selected_df.copy()
            details_subset["ranker"] = agg_func_name
            details_subset["eval_type"] = "LDDT-PLI"

            ranker_metric = None
            if agg_func_name.startswith("best."):
                ranker_metric = agg_func_name.split(".", 1)[1]
            elif agg_func_name in ["best", "worst", "median", "rand"]:
                ranker_metric = "lddt_pli"

            if ranker_metric and ranker_metric in details_subset.columns:
                details_subset["ranker_val"] = details_subset[ranker_metric]
            else:
                details_subset["ranker_val"] = np.nan

            details_subset["entity_id_2"] = ""
            details_subset["chain_id_2"] = ""
            # lddt column name is "lddt" in result, but metric is "lddt_pli"
            details_subset["lddt"] = details_subset["lddt_pli"]

            fields = [
                "seed",
                "sample",
                "ranker",
                "ranker_val",
                "eval_type",
                "entry_id",
                "entity_id_1",
                "entity_id_2",
                "chain_id_1",
                "chain_id_2",
                "cluster_id",
                "lddt",
            ]
            lddt_details.extend(details_subset[fields].to_dict("records"))

            all_avg_lddt = []
            for _cluster_id, lddt_scores in cluster_id_to_lddt_scores.items():
                avg_lddt = np.mean(lddt_scores)
                all_avg_lddt.append(avg_lddt)

            avg_avg_lddt = np.mean(all_avg_lddt)
            lddt_ci = get_bootstrap_ci(all_avg_lddt)

            lddt_result = {
                "eval_type": "LDDT-PLI",
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
        exist_pb_keys = [i for i in PB_VALID_CHECK_COL if i in pb_valid_df.columns]

        merged_metrics_df = pd.merge(
            metrics_df.reset_index(),
            pb_valid_df[match_keys + exist_pb_keys],
            how="left",
            on=match_keys,
        ).set_index("index")
        merged_metrics_df.index.name = metrics_df.index.name

        if (
            "minimum_distance_to_protein" in exist_pb_keys
            and "tetrahedral_chirality" in exist_pb_keys
        ):
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

    def _get_ranker_names(
        self,
    ) -> list[str]:
        # Initialize with basic aggregation functions
        ranker_names = ["best", "worst", "rand", "median"]

        for _level, ranker_list in self.ranker_keys.items():
            for ranker_key, _ in ranker_list:
                if ranker_key not in self.metrics_df.columns:
                    continue

                ranker_names.append(f"best.{ranker_key}")

                if "penalty" in self.metrics_df.columns:
                    ranker_names.append(f"best.{ranker_key}.penalized")

        return ranker_names

    def _select_representative_samples(
        self,
        df: pd.DataFrame,
        agg_func_name: str,
        group_by_key: list[str],
        metric_key: str,
        ranker_lookup: dict[str, bool],
    ) -> pd.DataFrame | None:
        """
        Selects representative samples for each group based on the aggregation strategy.
        Returns a sorted and deduplicated DataFrame if the strategy is vectorizable.
        Otherwise returns None.
        """
        # Define tie-breakers for deterministic sorting
        tie_breakers = []
        tie_asc = []
        if "seed" in df.columns:
            tie_breakers.append("seed")
            tie_asc.append(True)
        if "sample" in df.columns:
            tie_breakers.append("sample")
            tie_asc.append(True)

        if agg_func_name == "best":
            # For RMSD, lower is better
            return df.sort_values(
                by=[metric_key] + tie_breakers, ascending=[True] + tie_asc
            ).drop_duplicates(subset=group_by_key)
        elif agg_func_name == "worst":
            return df.sort_values(
                by=[metric_key] + tie_breakers, ascending=[False] + tie_asc
            ).drop_duplicates(subset=group_by_key)
        elif agg_func_name == "rand":
            return df.sample(frac=1).drop_duplicates(subset=group_by_key)
        elif agg_func_name == "median":
            # Vectorized median selection
            # Calculate median for each group
            median_series = df.groupby(group_by_key, observed=True)[
                metric_key
            ].transform("median")
            # Calculate absolute difference
            diff_col = f"_median_diff_{metric_key}"
            df = df.copy()
            df[diff_col] = (df[metric_key] - median_series).abs()
            # Sort by difference
            return (
                df.sort_values(by=[diff_col] + tie_breakers, ascending=[True] + tie_asc)
                .drop_duplicates(subset=group_by_key)
                .drop(columns=[diff_col])
            )

        # Handle best.{ranker_key} and best.{ranker_key}.penalized
        is_penalized = agg_func_name.endswith(".penalized")
        lookup_name = agg_func_name
        if is_penalized:
            lookup_name = agg_func_name.replace(".penalized", "")

        if lookup_name in ranker_lookup:
            asc = ranker_lookup[lookup_name]
            rk = lookup_name.split(".", 1)[1]

            if rk not in df.columns:
                return None

            # Ensure the column used for sorting is numeric
            # Use a copy to avoid SettingWithCopyWarning
            df = df.copy()
            df[rk] = pd.to_numeric(df[rk], errors="coerce")

            sort_col = rk
            if is_penalized and "penalty" in df.columns:
                penalized_col = f"{rk}_penalized_temp"
                if asc:
                    df[penalized_col] = df[rk] + df["penalty"]
                else:
                    df[penalized_col] = df[rk] - df["penalty"]
                sort_col = penalized_col

            return df.sort_values(
                by=[sort_col] + tie_breakers, ascending=[asc] + tie_asc
            ).drop_duplicates(subset=group_by_key)
        return None

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
        ranker_names = self._get_ranker_names()

        ranker_lookup = {}
        for _lv, ranker_list in self.ranker_keys.items():
            for rk, asc in ranker_list:
                ranker_lookup[f"best.{rk}"] = asc

        for agg_func_name in ranker_names:
            all_lig_rmsd = []

            # cluster_pb_all_valid_flags[cluster_id] -> list[int]
            cluster_pb_all_valid_flags = defaultdict(list)
            cluster_pb_all_valid_and_good_rmsd_flags = defaultdict(list)

            # all_pb_valid[check_col][cluster_id] -> list[int]
            all_pb_valid = defaultdict(lambda: defaultdict(list))

            cluster_id_to_rmsd_scores = defaultdict(list)

            group_by_key = ["entry_id", "chain_id_1"]

            selected_df = self._select_representative_samples(
                rmsd_df, agg_func_name, group_by_key, "lig_rmsd", ranker_lookup
            )

            # 1. Collect lig_rmsd by cluster_id
            grouped_scores = selected_df.groupby("cluster_id", observed=True)[
                "lig_rmsd"
            ].apply(list)
            for cid, scores in grouped_scores.items():
                cluster_id_to_rmsd_scores[cid].extend(scores)

            # 2. Collect all lig_rmsd (flat)
            all_lig_rmsd.extend(selected_df["lig_rmsd"].tolist())

            # 3. Handle PB checks
            temp_details = selected_df.copy()

            # Calculate flags for each row
            sample_pb_flags_cols = []
            for check_col in existed_pb_check_rows:
                flag_col_name = f"{check_col}_flag"
                # Use .where to avoid pandas FutureWarning about downcasting behavior in .fillna
                filled_series = temp_details[check_col].where(
                    temp_details[check_col].notna(), True
                )
                temp_details[flag_col_name] = filled_series.astype(bool).astype(int)
                sample_pb_flags_cols.append(flag_col_name)

                grouped_flags = temp_details.groupby("cluster_id", observed=True)[
                    flag_col_name
                ].apply(list)
                for cid, flags in grouped_flags.items():
                    all_pb_valid[check_col][cid].extend(flags)

            if existed_pb_check_rows:
                temp_details["all_valid_flag"] = (
                    temp_details[sample_pb_flags_cols].all(axis=1).astype(int)
                )

                grouped_all_valid = temp_details.groupby("cluster_id", observed=True)[
                    "all_valid_flag"
                ].apply(list)
                for cid, flags in grouped_all_valid.items():
                    cluster_pb_all_valid_flags[cid].extend(flags)

                temp_details["good_rmsd_flag"] = (
                    (temp_details["lig_rmsd"] < success_threshold)
                    & (temp_details["all_valid_flag"] == 1)
                ).astype(int)

                grouped_good_rmsd = temp_details.groupby("cluster_id", observed=True)[
                    "good_rmsd_flag"
                ].apply(list)
                for cid, flags in grouped_good_rmsd.items():
                    cluster_pb_all_valid_and_good_rmsd_flags[cid].extend(flags)

            temp_details["eval_type"] = "RMSD"
            temp_details["ranker"] = agg_func_name

            ranker_metric = None
            if agg_func_name.startswith("best."):
                ranker_metric = agg_func_name.split(".", 1)[1]
                if ranker_metric.endswith(".penalized"):
                    base_metric = ranker_metric.replace(".penalized", "")
                    ranker_metric = f"{base_metric}_penalized_temp"
            elif agg_func_name in ["best", "worst", "median"]:
                ranker_metric = "lig_rmsd"

            if ranker_metric and ranker_metric in temp_details.columns:
                temp_details["ranker_val"] = temp_details[ranker_metric]
            else:
                temp_details["ranker_val"] = np.nan

            temp_details["entity_id_2"] = ""
            temp_details["chain_id_2"] = ""

            for check_col, flag_col in zip(existed_pb_check_rows, sample_pb_flags_cols):
                temp_details[check_col] = temp_details[flag_col]

            fields = [
                "seed",
                "sample",
                "eval_type",
                "ranker",
                "ranker_val",
                "entry_id",
                "entity_id_1",
                "entity_id_2",
                "chain_id_1",
                "chain_id_2",
                "cluster_id",
                "lig_rmsd",
                "pocket_rmsd",
            ] + existed_pb_check_rows

            rmsd_details.extend(temp_details[fields].to_dict("records"))

            if len(all_lig_rmsd) == 0:
                continue

            all_avg_lig_rmsd = []
            all_avg_lig_rmsd_sr = []
            for _cluster_id, rmsd_scores in cluster_id_to_rmsd_scores.items():
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

    def get_ligand_others_metrics(
        self,
        mask_on_metrics_df: Sequence[bool] | None = None,
        subset_name: str = "All",
    ) -> pd.DataFrame:
        """
        Calculate others metrics for ligand predictions directly from metrics dataframe.

        The others metric is the success rate of (lig_rmsd < 2.0 AND lddt_pli > 0.8).
        It is calculated by averaging the success flags across all samples (not aggregated by cluster).

        Args:
            mask_on_metrics_df (Sequence[bool] | None): A mask to filter the metrics DataFrame. Defaults to None.
            subset_name (str): The name of the subset.

        Returns:
            pd.DataFrame: A DataFrame containing 'ranker', 'subset', and 'lig_rmsd_lddt_pli_sr'.
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

        if "lddt_pli" not in rmsd_df.columns:
            # Cannot calculate if lddt_pli is missing
            return pd.DataFrame()

        # Drop NaN rows in lig_rmsd and lddt_pli columns
        rmsd_df.dropna(subset=["lig_rmsd", "lddt_pli"], inplace=True, how="all", axis=0)

        results = []
        ranker_names = self._get_ranker_names()

        ranker_lookup = {}
        for _lv, ranker_list in self.ranker_keys.items():
            for rk, asc in ranker_list:
                ranker_lookup[f"best.{rk}"] = asc

        for agg_func_name in ranker_names:
            others_success_flags = []

            group_by_key = ["entry_id", "chain_id_1"]
            selected_df = self._select_representative_samples(
                rmsd_df, agg_func_name, group_by_key, "lig_rmsd", ranker_lookup
            )

            if "lddt_pli" in selected_df.columns:
                flags = (
                    (selected_df["lig_rmsd"] < 2.0) & (selected_df["lddt_pli"] > 0.8)
                ).astype(int)
                others_success_flags = flags.tolist()
            else:
                others_success_flags = [0] * len(selected_df)

            if not others_success_flags:
                continue

            sr = np.mean(others_success_flags)
            results.append(
                {
                    "ranker": agg_func_name,
                    "lig_rmsd_lddt_pli_sr": sr,
                    "subset": subset_name,
                    "entry_id_num": len(rmsd_df["entry_id"].unique()),
                    "cluster_num": len(
                        others_success_flags
                    ),  # Calculated per case (entry+chain)
                }
            )

        return pd.DataFrame(results)


class CDRH3Displayer:
    """
    Displayer for CDR H3 RMSD metrics.

    Args:
        metrics_df (pd.DataFrame): The DataFrame containing the metrics data.
        model (str, optional): The model name.
        seeds (list[str or int], optional): The list of seeds.
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

    def _get_ranker_names(self) -> list[str]:
        ranker_names = ["best", "worst", "rand", "median"]

        seen_rankers = set()
        for lv in ["complex", "chain", "interface"]:
            if lv in self.ranker_keys:
                for r_key, _ in self.ranker_keys[lv]:
                    if r_key in self.metrics_df.columns and r_key not in seen_rankers:
                        seen_rankers.add(r_key)
                        ranker_names.append(f"best.{r_key}")
        return ranker_names

    def _select_representative_samples(
        self,
        df: pd.DataFrame,
        agg_func_name: str,
        group_by_key: list[str],
        metric_key: str,
        ranker_lookup: dict[str, bool],
    ) -> pd.DataFrame | None:
        """
        Selects representative samples for each group based on the aggregation strategy.
        Returns a sorted and deduplicated DataFrame if the strategy is vectorizable.
        Otherwise returns None.
        """
        # Define tie-breakers for deterministic sorting
        tie_breakers = []
        tie_asc = []
        if "seed" in df.columns:
            tie_breakers.append("seed")
            tie_asc.append(True)
        if "sample" in df.columns:
            tie_breakers.append("sample")
            tie_asc.append(True)

        if agg_func_name == "best":
            return df.sort_values(
                by=[metric_key] + tie_breakers, ascending=[True] + tie_asc
            ).drop_duplicates(subset=group_by_key)
        elif agg_func_name == "worst":
            return df.sort_values(
                by=[metric_key] + tie_breakers, ascending=[False] + tie_asc
            ).drop_duplicates(subset=group_by_key)
        elif agg_func_name == "rand":
            return df.sample(frac=1).drop_duplicates(subset=group_by_key)
        elif agg_func_name == "median":
            # Vectorized median selection
            # Calculate median for each group
            median_series = df.groupby(group_by_key, observed=True)[
                metric_key
            ].transform("median")
            # Calculate absolute difference
            diff_col = f"_median_diff_{metric_key}"
            df = df.copy()
            df[diff_col] = (df[metric_key] - median_series).abs()
            # Sort by difference
            return (
                df.sort_values(by=[diff_col] + tie_breakers, ascending=[True] + tie_asc)
                .drop_duplicates(subset=group_by_key)
                .drop(columns=[diff_col])
            )
        if agg_func_name in ranker_lookup:
            asc = ranker_lookup[agg_func_name]
            # agg_func_name is like "best.{ranker_key}"
            rk = agg_func_name.split(".", 1)[1]
            if rk in df.columns:
                # Ensure the column used for sorting is numeric
                # Create a temporary numeric column for sorting to handle string issues
                # such as "10.0" < "9.0" being True when sorting as strings
                temp_col = f"_tmp_sort_{rk}"
                df = df.copy()
                df[temp_col] = pd.to_numeric(df[rk], errors="coerce")

                sorted_df = df.sort_values(
                    by=[temp_col] + tie_breakers, ascending=[asc] + tie_asc
                )

                return sorted_df.drop_duplicates(subset=group_by_key).drop(
                    columns=[temp_col]
                )

        raise ValueError(f"Unknown aggregation strategy: {agg_func_name}")

    def get_cdr_h3_rmsd(
        self,
        success_threshold: float = 1.0,
        mask_on_metrics_df: Sequence[bool] | None = None,
        subset_name: str = "All",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate CDR H3 RMSD metrics.

        Args:
            success_threshold (float): The threshold for considering an RMSD value as a success.
                            Defaults to 1.0.
            mask_on_metrics_df (Sequence[bool] | None): A mask to filter the metrics DataFrame. Defaults to None.
            subset_name (str): The name of the subset.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
                - results_df: A DataFrame containing the calculated metrics.
                - details_df: A DataFrame containing the details of each sample.
        """
        if "cdr_h3_bb_rmsd" not in self.metrics_df.columns:
            return pd.DataFrame(), pd.DataFrame()

        if mask_on_metrics_df is not None:
            df = self.metrics_df[mask_on_metrics_df].copy()
        else:
            df = self.metrics_df.copy()

        # Filter NaN
        df = df.dropna(subset=["cdr_h3_bb_rmsd"]).copy()
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()

        if self.seeds:
            df = df[df["seed"].astype(str).isin(self.seeds)].copy()

        ranker_names = self._get_ranker_names()

        ranker_lookup = {}
        for _lv, ranker_list in self.ranker_keys.items():
            for rk, asc in ranker_list:
                ranker_lookup[f"best.{rk}"] = asc

        results = []
        details = []

        for ranker_name in ranker_names:
            group_by_key = ["entry_id", "chain_id_1"]
            selected_df = self._select_representative_samples(
                df, ranker_name, group_by_key, "cdr_h3_bb_rmsd", ranker_lookup
            )

            selected_df["cluster_id"] = (
                selected_df["entry_id"].astype(str)
                + "_"
                + selected_df["entity_id_1"].astype(str)
            )
            # Aggregate by cluster_id
            cluster_means = selected_df.groupby("cluster_id", observed=True)[
                "cdr_h3_bb_rmsd"
            ].mean()
            selected_df["_success"] = (
                selected_df["cdr_h3_bb_rmsd"] <= success_threshold
            ).astype(int)
            cluster_sr = selected_df.groupby("cluster_id", observed=True)[
                "_success"
            ].mean()

            vals = cluster_means.values.astype(float)
            sr_vals = cluster_sr.values.astype(float)

            mean_val = np.mean(vals)
            sr_val = np.mean(sr_vals)

            ranker_metric = None
            if ranker_name.startswith("best."):
                ranker_metric = ranker_name.split(".", 1)[1]
            elif ranker_name in ["best", "worst", "median", "rand"]:
                ranker_metric = "cdr_h3_bb_rmsd"

            if ranker_metric and ranker_metric in selected_df.columns:
                selected_df["ranker_val"] = selected_df[ranker_metric]
            else:
                selected_df["ranker_val"] = np.nan

            det_df = selected_df[
                [
                    "seed",
                    "sample",
                    "entry_id",
                    "chain_id_1",
                    "entity_id_1",
                    "cdr_h3_bb_rmsd",
                    "ranker_val",
                    "cluster_id",
                ]
            ].copy()
            det_df["ranker"] = ranker_name
            det_df["eval_type"] = "CDR_H3_BB_RMSD"
            # cluster_id is already in det_df
            details.append(det_df)

            if len(vals) == len(selected_df):
                # N_cluster == N_sample
                # sr_vals contains only 0.0 or 1.0
                sr_ci = get_binomial_ci(
                    total_num=len(vals), success_num=int(sr_vals.sum())
                )
            else:
                sr_ci = get_bootstrap_ci(list(sr_vals))

            res = {
                "ranker": ranker_name,
                "cdr_h3_bb_avg_rmsd": mean_val,
                "cdr_h3_bb_rmsd_sr": sr_val,
                "ci_cdr_h3_bb_avg_rmsd": get_bootstrap_ci(list(vals)),
                "ci_cdr_h3_bb_rmsd_sr": sr_ci,
                "entry_id_num": selected_df["entry_id"].nunique(),
                "cluster_num": len(vals),
            }
            results.append(res)

        results_df = pd.DataFrame(results)
        details_df = pd.concat(details) if details else pd.DataFrame()
        results_df["subset"] = subset_name
        details_df["subset"] = subset_name
        return results_df, details_df
