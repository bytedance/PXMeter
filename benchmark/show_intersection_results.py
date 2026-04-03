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

import argparse
import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from joblib import delayed, Parallel
from pxmeter.constants import PROTEIN
from tqdm import tqdm

from benchmark.aggregator import run_aggregator
from benchmark.configs.data_config import SUPPORTED_DATA
from benchmark.configs.dataset_metrics_config import DATASET_METRICS_CONFIG
from benchmark.show_results import (
    CDRH3Displayer,
    ChainInterfaceDisplayer,
    RMSDDisplayer,
)
from benchmark.simplified_results import run_reduce
from benchmark.utils import (
    add_cluster_id_to_df,
    add_comp_chain_iface_id,
    query_subset_labels,
    select_df_by_eval_types,
)


@dataclass
class MetricsDisplayers:
    displayer: ChainInterfaceDisplayer
    valid_chain_displayer: ChainInterfaceDisplayer
    rmsd_displayer: RMSDDisplayer | None = None
    cdr_displayer: CDRH3Displayer | None = None


def get_af3_ab_sub_df(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a DataFrame containing AF3 Antibody-Antigen Subset metrics.
    Args:
        metrics_df (pd.DataFrame): The metrics DataFrame.
    Returns:
        pd.DataFrame: A DataFrame containing AF3 Antibody-Antigen Subset metrics.
    """
    meta_df = pd.read_csv(SUPPORTED_DATA.af3_ab_metadata)
    interface_to_cluster = {}
    for _idx, row in meta_df.iterrows():
        # The chain ids already sorted
        interface_to_cluster[
            (row["pdb_id"], row["chain_id_1"], row["chain_id_2"])
        ] = row["interface_cluster_key"]

    in_af3_ab_mask = metrics_df.apply(
        lambda row, cluster_dict=interface_to_cluster: (
            row["type"] == "interface"
            and row["entity_type_1"] == PROTEIN
            and row["entity_type_2"] == PROTEIN
            and (row["entry_id"], row["chain_id_1"], row["chain_id_2"]) in cluster_dict
        ),
        axis=1,
    )
    af3_ab_sub_df = metrics_df[in_af3_ab_mask].copy()

    # reset cluster_id
    af3_ab_sub_df["cluster_id"] = af3_ab_sub_df.apply(
        lambda row, cluster_dict=interface_to_cluster: cluster_dict[
            (row["entry_id"], row["chain_id_1"], row["chain_id_2"])
        ],
        axis=1,
    )
    return af3_ab_sub_df


def get_low_homology_subset(
    metrics_df: pd.DataFrame, lowh_df: pd.DataFrame
) -> pd.Series:
    """
    Filter a metrics DataFrame of RecentPDB to include only low homology entries.

    Args:
        metrics_df (pd.DataFrame): The metrics DataFrame.
        lowh_df (pd.DataFrame): The low homology DataFrame.

    Returns:
        pd.Series: A boolean series indicating whether each row
                   in the metrics DataFrame is a low homology entry.
    """

    # lowh_df cols: type, entry_id, entity_id_1, entity_id_2
    def _make_lowh_keys(row):
        if row["type"] == "chain":
            return row["entry_id"] + "_" + row["entity_id_1"]
        elif row["type"] == "interface":
            return (
                row["entry_id"]
                + "_"
                + "_".join(sorted([row["entity_id_1"], row["entity_id_2"]]))
            )
        elif row["type"] == "complex":
            return row["entry_id"]
        else:
            raise NotImplementedError(f"Unknown type: {row['type']}")

    lowh_df_keys = lowh_df.apply(_make_lowh_keys, axis=1)
    metrics_keys = metrics_df.apply(_make_lowh_keys, axis=1)
    return metrics_keys.isin(lowh_df_keys)


def _filter_by_valid_chain_id(
    metrics_df: pd.DataFrame, aux_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Filter metrics DataFrame based on chain ID validity in aux_df.

    Args:
        metrics_df (pd.DataFrame): The metrics DataFrame to filter.
        aux_df (pd.DataFrame): The auxiliary DataFrame containing valid chain information.

    Returns:
        pd.DataFrame: A filtered DataFrame containing only rows with valid chain IDs.
    """
    if metrics_df.empty or aux_df.empty:
        return metrics_df.iloc[:0]

    # chain_rows: keep entry_id, chain_id_1
    chain_mask = aux_df["type"] == "chain"
    chain_df = aux_df.loc[chain_mask, ["entry_id", "chain_id_1"]].copy()

    # interface_rows: keep entry_id, chain_id_1 AND entry_id, chain_id_2 (renamed to chain_id_1)
    interface_mask = aux_df["type"] == "interface"

    if1 = aux_df.loc[interface_mask, ["entry_id", "chain_id_1"]].copy()

    if2 = aux_df.loc[interface_mask, ["entry_id", "chain_id_2"]].copy()
    if2 = if2.rename(columns={"chain_id_2": "chain_id_1"})

    valid_df = pd.concat([chain_df, if1, if2], ignore_index=True).drop_duplicates()

    # Set index for faster lookup
    # Using MultiIndex intersection
    valid_index = pd.MultiIndex.from_frame(valid_df)

    # We use a temporary index based on entry_id and chain_id_1 for matching
    metrics_index = pd.MultiIndex.from_frame(metrics_df[["entry_id", "chain_id_1"]])

    mask = metrics_index.isin(valid_index)
    return metrics_df[mask].copy()


def get_valid_chain_mask(metrics_df: pd.DataFrame, aux_df: pd.DataFrame) -> pd.Series:
    """
    Get a boolean mask for metrics_df based on valid chains in aux_df.
    Similar to _filter_by_valid_chain_id but returns a mask.
    """
    if metrics_df.empty or aux_df.empty:
        return pd.Series(False, index=metrics_df.index)

    # chain_rows: keep entry_id, chain_id_1
    chain_mask = aux_df["type"] == "chain"
    chain_df = aux_df.loc[chain_mask, ["entry_id", "chain_id_1"]].copy()

    # interface_rows: keep entry_id, chain_id_1 AND entry_id, chain_id_2 (renamed to chain_id_1)
    interface_mask = aux_df["type"] == "interface"

    if1 = aux_df.loc[interface_mask, ["entry_id", "chain_id_1"]].copy()
    if2 = aux_df.loc[interface_mask, ["entry_id", "chain_id_2"]].copy()
    if2 = if2.rename(columns={"chain_id_2": "chain_id_1"})

    valid_df = pd.concat([chain_df, if1, if2], ignore_index=True).drop_duplicates()

    # Set index for faster lookup
    valid_index = pd.MultiIndex.from_frame(valid_df)

    # We use a temporary index based on entry_id and chain_id_1 for matching
    metrics_index = pd.MultiIndex.from_frame(metrics_df[["entry_id", "chain_id_1"]])

    return metrics_index.isin(valid_index)


def _load_pb_valid_df(
    trial_name_to_metrics_file: dict[str, tuple[Path, ...]], dataset_name: str
) -> pd.DataFrame | None:
    metrics_files = trial_name_to_metrics_file[dataset_name]
    pb_dfs = []

    for metrics_file in metrics_files:
        pb_metrics_file = Path(str(metrics_file).replace("_metrics.", "_pb_valid."))

        if not pb_metrics_file.exists():
            continue

        if str(pb_metrics_file).endswith(".parquet"):
            pb_valid_df = pd.read_parquet(pb_metrics_file, engine="pyarrow")
            for col in ["entry_id", "seed", "sample"]:
                pb_valid_df[col] = pb_valid_df[col].astype("string")
        else:
            pb_valid_df = pd.read_csv(
                pb_metrics_file,
                dtype={"entry_id": str, "seed": str, "sample": str},
                low_memory=False,
            )
        pb_dfs.append(pb_valid_df)

    if not pb_dfs:
        return None

    return pd.concat(pb_dfs, ignore_index=True)


def _prepare_metrics_df(
    sub_metrics_df: pd.DataFrame, eval_dataset: str
) -> tuple[pd.DataFrame, pd.DataFrame | None, str]:
    working_metrics_df = sub_metrics_df
    aux_df = None
    config_key = eval_dataset
    if eval_dataset == "dsDNA-Protein":
        config_key = "DNA-Protein"

    if config_key not in DATASET_METRICS_CONFIG:
        raise NotImplementedError(f"Unknown dataset {eval_dataset}")

    if eval_dataset == "RecentPDB":
        aux_df = pd.read_csv(
            SUPPORTED_DATA.recentpdb_low_homology,
            dtype={
                "entry_id": str,
                "entity_id_1": str,
                "entity_id_2": str,
                "chain_id_1": str,
                "chain_id_2": str,
            },
        )
        working_metrics_df = sub_metrics_df[
            get_low_homology_subset(sub_metrics_df, aux_df)
        ]
    elif eval_dataset == "AF3-AB":
        working_metrics_df = get_af3_ab_sub_df(sub_metrics_df)
    elif eval_dataset == "dsDNA-Protein":
        dsdna_metrics_df = sub_metrics_df.copy()
        dsdna_metrics_df = select_df_by_eval_types(dsdna_metrics_df, ["DNA-Protein"])
        dsdna_metrics_df["lddt_mean"] = dsdna_metrics_df.groupby(
            ["entry_id", "sample", "seed"]
        )["lddt"].transform("mean")
        working_metrics_df = (
            dsdna_metrics_df.drop_duplicates(subset=["entry_id", "sample", "seed"])
            .drop(columns=["lddt"])
            .rename(columns={"lddt_mean": "lddt"})
        )

    return working_metrics_df, aux_df, config_key


def _initialize_displayers(
    working_metrics_df: pd.DataFrame,
    sub_metrics_df: pd.DataFrame,
    aux_df: pd.DataFrame | None,
    pb_valid_df: pd.DataFrame | None,
    model: str,
    seeds: list[str | int] | None,
    eval_dataset: str,
) -> tuple[MetricsDisplayers, pd.DataFrame]:
    if eval_dataset == "RecentPDB":
        valid_chain_metrics_df = _filter_by_valid_chain_id(sub_metrics_df, aux_df)
    else:
        valid_chain_metrics_df = working_metrics_df

    displayer = ChainInterfaceDisplayer(working_metrics_df, model=model, seeds=seeds)

    rmsd_displayer = None
    if (
        "lig_rmsd" in valid_chain_metrics_df.columns
        or "lig_rmsd_wo_refl" in valid_chain_metrics_df.columns
    ):
        rmsd_displayer = RMSDDisplayer(
            valid_chain_metrics_df,
            pb_valid_df=pb_valid_df,
            model=model,
            seeds=seeds,
        )

    valid_chain_displayer = ChainInterfaceDisplayer(
        valid_chain_metrics_df, model=model, seeds=seeds
    )

    cdr_displayer = None
    if "cdr_h3_bb_rmsd" in valid_chain_metrics_df.columns:
        cdr_displayer = CDRH3Displayer(valid_chain_metrics_df, model, seeds)

    return (
        MetricsDisplayers(
            displayer=displayer,
            valid_chain_displayer=valid_chain_displayer,
            rmsd_displayer=rmsd_displayer,
            cdr_displayer=cdr_displayer,
        ),
        valid_chain_metrics_df,
    )


def _get_subset_masks(
    config: dict,
    eval_dataset: str,
    aux_df: pd.DataFrame | None,
    working_metrics_df: pd.DataFrame,
    valid_chain_metrics_df: pd.DataFrame,
) -> tuple[pd.Series | None, pd.Series | None]:
    mask = None
    mask_valid = None

    if config["subset_label"] and eval_dataset == "RecentPDB" and aux_df is not None:
        subset_labels = [
            lbl.strip()
            for lbl in str(config["subset_label"]).split(";")
            if str(lbl).strip()
        ]

        for subset_label in subset_labels:
            target_lowh_entries = query_subset_labels(aux_df["subset"], subset_label)

            label_mask = get_low_homology_subset(
                working_metrics_df, aux_df[target_lowh_entries]
            )

            if mask is None:
                mask = label_mask
            else:
                mask &= label_mask

            # Use get_valid_chain_mask to handle interface-to-chain decomposition
            label_mask_valid = get_valid_chain_mask(
                valid_chain_metrics_df, aux_df[target_lowh_entries]
            )
            if mask_valid is None:
                mask_valid = label_mask_valid
            else:
                mask_valid &= label_mask_valid

    if config["inverse_subset"]:
        if mask is not None:
            mask = ~mask
        if mask_valid is not None:
            mask_valid = ~mask_valid

    return mask, mask_valid


def _compute_metric_results(
    config: dict,
    displayers: MetricsDisplayers,
    masks: dict,
    eval_dataset: str,
    results_lists: dict,
    details_lists: dict,
):
    subset_name = config["subset"]
    eval_types = config["eval_type"]
    metrics = config["metric"]

    displayer = displayers.displayer
    valid_chain_displayer = displayers.valid_chain_displayer
    rmsd_displayer = displayers.rmsd_displayer
    cdr_displayer = displayers.cdr_displayer

    mask = masks["mask"]
    mask_valid = masks["mask_valid"]

    for metric in metrics:
        if metric == "dockq":
            res, det = displayer.get_dockq_sr_by_cluster(
                mask_on_metrics_df=mask,
                eval_types=eval_types,
                subset_name=subset_name,
            )
            results_lists["dockq"].append(res)
            details_lists["dockq"].append(det)

        elif metric == "lddt":
            res, det = displayer.get_lddt_by_cluster(
                mask_on_metrics_df=mask,
                eval_types=eval_types,
                subset_name=subset_name,
            )
            if eval_dataset == "dsDNA-Protein":
                res["eval_type"] = "dsDNA-Protein"
                det["eval_type"] = "dsDNA-Protein"
                det["chain_id_1"] = pd.NA
                det["chain_id_2"] = pd.NA

            results_lists["lddt"].append(res)
            details_lists["lddt"].append(det)

        elif metric == "lddt_pli":
            # Check if lddt_pli column exists and compute it using ChainInterfaceDisplayer
            if "lddt_pli" in valid_chain_displayer.metrics_df.columns:
                res_pli, det_pli = valid_chain_displayer.get_lddt_pli(
                    mask_on_metrics_df=mask_valid,
                    subset_name=subset_name,
                )
                if not res_pli.empty:
                    results_lists["lddt"].append(res_pli)
                    details_lists["lddt"].append(det_pli)

        elif metric == "rmsd":
            if rmsd_displayer:
                res, det = rmsd_displayer.get_rmsd(
                    mask_on_metrics_df=mask_valid,
                    subset_name=subset_name,
                )

                res_others = rmsd_displayer.get_ligand_others_metrics(
                    mask_on_metrics_df=mask_valid,
                    subset_name=subset_name,
                )
                if not res_others.empty:
                    results_lists["others"].append(res_others)

                results_lists["rmsd"].append(res)
                details_lists["rmsd"].append(det)

        elif metric == "cdr_h3_bb_rmsd":
            if cdr_displayer:
                res, det = cdr_displayer.get_cdr_h3_rmsd(
                    success_threshold=1.0,
                    mask_on_metrics_df=mask_valid,
                    subset_name=subset_name,
                )
                results_lists["rmsd"].append(res)
                details_lists["rmsd"].append(det)

        else:
            raise NotImplementedError(f"Unknown metric {metric}")


def _find_result_csv(
    eval_info_dict: dict, trials: list[str]
) -> dict[str, dict[str, tuple[Path, ...]]]:
    """
    Find the result CSV files for the specified trials.

    This function iterates over the evaluation trials and finds the result CSV files
    for each trial. It checks if the trial is in the list of specified trials and
    if the evaluation result directory exists. If the result CSV file exists, it adds
    it to the dictionary of trial to result files.

    Args:
        eval_info_dict (dict): A dictionary containing evaluation information.
        trials (list[str]): A list of trial names for which to find result CSV files.

    Returns:
        dict[str, dict[str, tuple[Path, ...]]]: A dictionary mapping dataset names to a dictionary of
        trial names to result CSV file paths.
    """
    # {"RecentPDB" or "PoseBusters": {trial_name: tuple(csv_path)}}
    trial_name_to_result_files = defaultdict(dict)
    for trial_name, trial_dict in eval_info_dict.items():
        if trial_name not in trials:
            continue

        for eval_dataset, dataset_path in trial_dict["dataset_path"].items():
            dataset_paths = [p.strip() for p in str(dataset_path).split(",")]
            result_csvs = []
            for dp in dataset_paths:
                eval_result_dir = Path(dp)
                if not eval_result_dir.exists():
                    logging.warning("%s does not exist", eval_result_dir)
                    continue

                if eval_result_dir.name.endswith(
                    ".csv"
                ) or eval_result_dir.name.endswith(".parquet"):
                    result_csv = eval_result_dir
                else:
                    result_csv = Path(
                        eval_result_dir.parent
                        / f"{eval_result_dir.name}_metrics.parquet"
                    )
                    if result_csv.with_suffix(".csv").exists():
                        result_csv = result_csv.with_suffix(".csv")

                result_csvs.append(result_csv)
            if result_csvs:
                trial_name_to_result_files[eval_dataset][trial_name] = tuple(
                    result_csvs
                )
    return trial_name_to_result_files


def gen_aggregated_results(
    eval_info_dict: dict,
    trials: list[str],
    num_cpu: int = 1,
    overwrite: bool = False,
):
    """
    Generate aggregated results for specified datasets.

    Args:
        eval_info_dict (dict): A dictionary containing evaluation information.
        trials (list[str]): A list of trial names for which to generate aggregated results.
                      generate aggregated results.
        num_cpu (int, optional): The number of CPU cores to use for
                parallel processing. Defaults to 1.
        overwrite (bool, optional): If True, overwrite existing
                  result CSV files. Defaults to False.
    """
    trial_name_to_result_files = _find_result_csv(eval_info_dict, trials)
    for _eval_dataset, trial_name_to_csv_path in trial_name_to_result_files.items():
        if not trial_name_to_csv_path:
            continue
        for _trial_name, metrics_csvs in trial_name_to_csv_path.items():
            for metrics_csv in metrics_csvs:
                eval_result_dir = metrics_csv.parent / str(metrics_csv.name).replace(
                    "_metrics.csv", ""
                ).replace("_metrics.parquet", "")

                if (not metrics_csv.exists()) or overwrite:
                    logging.info("Aggregating for: %s", eval_result_dir)
                    run_aggregator(
                        eval_result_dir,
                        num_cpu=num_cpu,
                    )


def _get_a_dataset_result(
    trial_name_to_metrics_file: dict[str, tuple[Path, ...]],
    sub_metrics_df: pd.DataFrame,
    eval_dataset: str,
    model: str,
    dataset_name: str,
    seeds: list[str | int] | None = None,
) -> tuple[str, dict[str, tuple[pd.DataFrame, pd.DataFrame]]]:
    """
    Compute metrics for a single dataset and return them organized by metric name.

    Returns:
        tuple[str, dict[str, tuple[pd.DataFrame, pd.DataFrame]]]:
            eval_dataset, and a mapping:
            {
                "dockq": (result_df, details_df),
                "lddt": (result_df, details_df),
                "rmsd": (result_df, details_df),
                ...
            }
            Only metrics relevant to `eval_dataset` are present.
    """
    pb_valid_df = _load_pb_valid_df(trial_name_to_metrics_file, dataset_name)

    working_metrics_df, aux_df, config_key = _prepare_metrics_df(
        sub_metrics_df, eval_dataset
    )

    (displayers, valid_chain_metrics_df,) = _initialize_displayers(
        working_metrics_df,
        sub_metrics_df,
        aux_df,
        pb_valid_df,
        model,
        seeds,
        eval_dataset,
    )

    results_lists = defaultdict(list)
    details_lists = defaultdict(list)

    for config in DATASET_METRICS_CONFIG[config_key]:
        mask, mask_valid = _get_subset_masks(
            config, eval_dataset, aux_df, working_metrics_df, valid_chain_metrics_df
        )

        masks = {"mask": mask, "mask_valid": mask_valid}

        _compute_metric_results(
            config, displayers, masks, eval_dataset, results_lists, details_lists
        )

    metric_results = {}
    for metric_name in ["dockq", "lddt", "rmsd", "others"]:
        if results_lists[metric_name]:
            result_df = pd.concat(results_lists[metric_name])
            if details_lists[metric_name]:
                details_df = pd.concat(details_lists[metric_name])
            else:
                details_df = pd.DataFrame()

            if not result_df.empty:
                result_df.insert(0, "name", dataset_name)
                result_df.insert(1, "eval_dataset", eval_dataset)
            if not details_df.empty:
                details_df.insert(0, "name", dataset_name)
                details_df.insert(1, "eval_dataset", eval_dataset)

            metric_results[metric_name] = (result_df, details_df)

    return eval_dataset, metric_results


def _save_to_output_csv(
    output_dir: Path,
    dockq_results: list[pd.DataFrame],
    lddt_results: list[pd.DataFrame],
    rmsd_results: list[pd.DataFrame],
    others_results: list[pd.DataFrame],
    dockq_details: list[pd.DataFrame],
    lddt_details: list[pd.DataFrame],
    rmsd_details: list[pd.DataFrame],
):
    dockq_csv = None
    lddt_csv = None
    rmsd_csv = None
    others_csv = None

    if len(dockq_results) > 0:
        all_dockq_df = pd.concat(dockq_results)
        if len(all_dockq_df) > 0:
            dockq_csv = output_dir / "DockQ_results.csv"
            all_dockq_df["entry_id_num"] = all_dockq_df["entry_id_num"].astype(int)
            all_dockq_df["cluster_num"] = all_dockq_df["cluster_num"].astype(int)
            all_dockq_df = all_dockq_df.round(4)
            all_dockq_df.to_csv(
                dockq_csv,
                index=False,
                quoting=csv.QUOTE_NONNUMERIC,
            )
            logging.info("DockQ results saved to %s", dockq_csv)

    if len(dockq_details) > 0:
        all_dockq_details_df = pd.concat(dockq_details)
        if len(all_dockq_details_df) > 0:
            dockq_details_csv = output_dir / "DockQ_details.csv"
            all_dockq_details_df = add_comp_chain_iface_id(all_dockq_details_df)
            all_dockq_details_df.to_csv(
                dockq_details_csv,
                index=False,
                quoting=csv.QUOTE_NONNUMERIC,
            )
            logging.info("DockQ details saved to %s", dockq_details_csv)

    if len(lddt_results) > 0:
        all_lddt_df = pd.concat(lddt_results)
        if len(all_lddt_df) > 0:
            all_lddt_df["entry_id_num"] = all_lddt_df["entry_id_num"].astype(int)
            all_lddt_df["cluster_num"] = all_lddt_df["cluster_num"].astype(int)
            lddt_csv = output_dir / "LDDT_results.csv"
            all_lddt_df["lddt"] = all_lddt_df["lddt"].astype(float)
            all_lddt_df = all_lddt_df.round(4)
            all_lddt_df.to_csv(
                lddt_csv,
                index=False,
                quoting=csv.QUOTE_NONNUMERIC,
            )
            logging.info("LDDT results saved to %s", lddt_csv)

    if len(lddt_details) > 0:
        all_lddt_details_df = pd.concat(lddt_details)
        if len(all_lddt_details_df) > 0:
            lddt_details_csv = output_dir / "LDDT_details.csv"
            all_lddt_details_df = add_comp_chain_iface_id(all_lddt_details_df)
            all_lddt_details_df.to_csv(
                lddt_details_csv,
                index=False,
                quoting=csv.QUOTE_NONNUMERIC,
            )
            logging.info("LDDT details saved to %s", lddt_details_csv)

    if len(rmsd_results) > 0:
        all_rmsd_df = pd.concat(rmsd_results)
        if len(all_rmsd_df) > 0:
            rmsd_csv = output_dir / "RMSD_results.csv"
            cols_to_float = ["lig_avg_rmsd", "cdr_h3_bb_avg_rmsd"]
            for col in cols_to_float:
                if col in all_rmsd_df.columns:
                    all_rmsd_df[col] = all_rmsd_df[col].astype(float)

            all_rmsd_df = all_rmsd_df.round(4)
            all_rmsd_df.to_csv(
                rmsd_csv,
                index=False,
                quoting=csv.QUOTE_NONNUMERIC,
            )
            logging.info("RMSD results saved to %s", rmsd_csv)

    if len(rmsd_details) > 0:
        all_rmsd_details_df = pd.concat(rmsd_details)
        if len(all_rmsd_details_df) > 0:
            rmsd_details_csv = output_dir / "RMSD_details.csv"
            all_rmsd_details_df = add_comp_chain_iface_id(all_rmsd_details_df)
            all_rmsd_details_df.to_csv(
                rmsd_details_csv,
                index=False,
                quoting=csv.QUOTE_NONNUMERIC,
            )
            logging.info("RMSD details saved to %s", rmsd_details_csv)

    if len(others_results) > 0:
        all_others_df = pd.concat(others_results)
        if len(all_others_df) > 0:
            others_csv = output_dir / "Others_results.csv"
            cols_to_float = ["lig_rmsd_lddt_pli_sr"]
            for col in cols_to_float:
                if col in all_others_df.columns:
                    all_others_df[col] = all_others_df[col].astype(float)

            all_others_df = all_others_df.round(4)

            # Ensure consistent column ordering like other results files
            # Preferred order: name, eval_dataset, eval_type (if exists), entry_id_num, cluster_num, ranker, metrics, subset
            base_cols = ["name", "eval_dataset"]
            if "eval_type" in all_others_df.columns:
                base_cols.append("eval_type")
            base_cols.extend(["entry_id_num", "cluster_num", "ranker"])

            end_cols = ["subset"]
            metrics_cols = [
                c
                for c in all_others_df.columns
                if c not in base_cols and c not in end_cols
            ]

            ordered_cols = base_cols + metrics_cols + end_cols
            # Reorder with available columns to be safe
            ordered_cols = [c for c in ordered_cols if c in all_others_df.columns]
            all_others_df = all_others_df[ordered_cols]

            all_others_df.to_csv(
                others_csv,
                index=False,
                quoting=csv.QUOTE_NONNUMERIC,
            )
            logging.info("Others results saved to %s", others_csv)

    return dockq_csv, lddt_csv, rmsd_csv, others_csv


def _integrity_check(
    metrics_df: pd.DataFrame, seeds_to_check: list[str] | None = None
) -> set[str]:
    if seeds_to_check is None:
        seeds_to_check = list(metrics_df["seed"].unique())

    df_to_check = metrics_df[metrics_df["seed"].isin(seeds_to_check)].copy()

    # Here we assume that a sample will not be missing across all seeds
    expected_samples = metrics_df["sample"].nunique()
    expected_count = len(seeds_to_check) * expected_samples

    df_to_check["_seed_sample"] = (
        df_to_check["seed"].astype(str) + "_" + df_to_check["sample"].astype(str)
    )
    counts = df_to_check.groupby("entry_id", observed=True)["_seed_sample"].nunique()
    incomplete_list = counts[counts < expected_count].index.tolist()

    all_set = set(metrics_df["entry_id"])
    present_set = set(counts.index)
    completely_missing = all_set - present_set
    return set(incomplete_list) | completely_missing


def _prepare_tasks(
    eval_info_dict: dict,
    dataset_to_result_files: dict[str, dict[str, tuple[Path, ...]]],
    pdb_id_list: list[str] = None,
    subset_csv: Path = None,
) -> list:
    if subset_csv is not None:
        subset_df = pd.read_csv(subset_csv, dtype=str, low_memory=False).astype(str)

        def _make_key(entry_id: str, c1: str, c2: str, row_type: str) -> tuple:
            if row_type == "interface":
                a, b = sorted([c1, c2])
                return (entry_id, a, b)
            return (entry_id, c1, c2)

        subset_df["match_key"] = [
            _make_key(r.entry_id, r.chain_id_1, r.chain_id_2, getattr(r, "type", ""))
            for r in subset_df.itertuples(index=False)
        ]
        subset_match_key = set(subset_df["match_key"])
    else:
        subset_match_key = set()

    tasks = []
    for eval_dataset, dataset_name_to_csv_paths in dataset_to_result_files.items():
        filepath_to_df = {}
        filepath_to_seeds = defaultdict(set)
        filepath_to_dataset_name = defaultdict(list)
        for dataset_name, metrics_files in dataset_name_to_csv_paths.items():
            filepath_to_dataset_name[metrics_files].append(dataset_name)

            if metrics_files not in filepath_to_df:
                dfs_to_concat = []
                for metrics_file in metrics_files:
                    if metrics_file.suffix == ".csv":
                        metrics_df = pd.read_csv(
                            metrics_file,
                            dtype={
                                "entry_id": str,
                                "entity_id_1": str,
                                "entity_id_2": str,
                                "seed": str,
                                "sample": str,
                                "chain_id_1": str,
                                "chain_id_2": str,
                            },
                            low_memory=False,
                        )
                    else:
                        metrics_df = pd.read_parquet(
                            metrics_file,
                            engine="pyarrow",
                        )
                        for col in [
                            "entry_id",
                            "entity_id_1",
                            "entity_id_2",
                            "seed",
                            "sample",
                        ]:
                            if col in metrics_df.columns:
                                metrics_df[col] = metrics_df[col].astype("string")

                    # Ensure entity_id columns are int strings
                    for col in ("entity_id_1", "entity_id_2"):
                        if col in metrics_df.columns:
                            metrics_df[col] = (
                                pd.to_numeric(metrics_df[col], errors="coerce")
                                .dropna()
                                .astype(int)
                                .astype(str)
                                .reindex(metrics_df.index)
                            )

                    metrics_df["match_key"] = list(
                        zip(
                            metrics_df["entry_id"].astype(str),
                            metrics_df["chain_id_1"].astype(str),
                            metrics_df["chain_id_2"].astype(str),
                        )
                    )
                    dfs_to_concat.append(metrics_df)

                others_df = pd.concat(dfs_to_concat, ignore_index=True)
                filepath_to_df[metrics_files] = others_df
                logging.info(
                    "%s entries loaded from '%s'\n",
                    others_df["entry_id"].nunique(),
                    ", ".join([str(p) for p in metrics_files]),
                )

            seeds = eval_info_dict[dataset_name].get("seeds", [])
            if seeds:
                filepath_to_seeds[metrics_files].update(str(s) for s in seeds)

        intersection = set(subset_match_key).copy()
        for metrics_files, metrics_df in list(filepath_to_df.items()):
            df = metrics_df

            if pdb_id_list is not None:
                df = df[df["entry_id"].isin(pdb_id_list)].copy()
                logging.info(
                    '%d entries remain in "%s" after pdb_id_list filter\n',
                    df["entry_id"].nunique(),
                    ",".join(filepath_to_dataset_name[metrics_files]),
                )
                if df.empty:
                    raise AssertionError(
                        f"No PDB IDs found in the pdb_id_list for {metrics_files}"
                    )

            seeds_to_check = None
            all_seeds = filepath_to_seeds.get(metrics_files, set())
            if all_seeds:
                df = df[df["seed"].isin(all_seeds)].copy()
                if df.empty:
                    raise ValueError(
                        f"No matching seeds found in {metrics_files}; "
                        f"expected any of {all_seeds}."
                    )
                seeds_to_check = list(all_seeds)

            incomplete_entries = _integrity_check(df, seeds_to_check=seeds_to_check)
            if incomplete_entries:
                logging.warning(
                    '%d entries are incomplete (N_seed * N_sample) in "%s"; \
                        dropping them before dataset intersection\n',
                    len(incomplete_entries),
                    ",".join(filepath_to_dataset_name[metrics_files]),
                )
                df = df[~df["entry_id"].isin(incomplete_entries)].copy()

            unique_match_key = set(df["match_key"])
            if not intersection:
                intersection = unique_match_key
            else:
                intersection &= unique_match_key
            assert (
                intersection
            ), f'Intersection became empty after processing "{metrics_files}"'

            filepath_to_df[metrics_files] = df

        # Filter DataFrames by intersection
        for metrics_files, metrics_df in list(filepath_to_df.items()):
            sub_metrics_df = metrics_df[
                metrics_df["match_key"].isin(intersection)
            ].copy()

            if eval_dataset == "RecentPDB":
                # Add cluster_id on-the-fly
                cluster_df = pd.read_csv(
                    SUPPORTED_DATA.recentpdb_low_homology_cluster, dtype=str
                )
                # ensure entity_id is a string of int
                cluster_df["label_entity_id"] = (
                    pd.to_numeric(cluster_df["label_entity_id"], errors="coerce")
                    .dropna()
                    .astype(int)
                    .astype(str)
                    .reindex(cluster_df.index)
                )

                sub_metrics_df = add_cluster_id_to_df(
                    cluster_df,
                    sub_metrics_df,
                    interface_only_use_polymer_cluster=True,
                )

            elif eval_dataset == "Custom" and subset_csv is not None:
                if "cluster_id" in subset_df.columns:
                    # Add cluster_id from subset_df
                    sub_metrics_df = sub_metrics_df.merge(
                        subset_df[["match_key", "cluster_id"]],
                        on=["match_key"],
                        how="left",
                    )

            filepath_to_df[metrics_files] = sub_metrics_df
            logging.info(
                '%s entries in the intersection from "%s"\n',
                sub_metrics_df["entry_id"].nunique(),
                ",".join(filepath_to_dataset_name[metrics_files]),
            )

        for dataset_name, metrics_files in dataset_name_to_csv_paths.items():
            model = eval_info_dict[dataset_name]["model"]
            sub_metrics_df = filepath_to_df[metrics_files]
            dataset_seeds = [
                str(i)
                for i in sorted(set(eval_info_dict[dataset_name].get("seeds", [])))
            ]

            if dataset_seeds:
                sub_metrics_df = sub_metrics_df[
                    sub_metrics_df["seed"].isin(set(dataset_seeds))
                ].copy()

            remain_seeds = sorted(
                [str(s) for s in sub_metrics_df["seed"].unique()],
                key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else s),
            )
            logging.info(
                '%d entries after filtering from "%s" by %d seeds: [%s]\n',
                sub_metrics_df["entry_id"].nunique(),
                dataset_name,
                len(remain_seeds),
                ",".join(remain_seeds),
            )

            tasks.append(
                [
                    dataset_name_to_csv_paths,
                    sub_metrics_df,
                    eval_dataset,
                    model,
                    dataset_name,
                    dataset_seeds,
                ]
            )
    return tasks


def save_all_results(
    eval_info_dict: dict,
    trials: list[str],
    output_dir: Path | str = Path("."),
    pdb_id_list: list[str] | None = None,
    subset_csv: Path | None = None,
    num_cpu: int = 1,
) -> tuple[Path, ...]:
    """
    Save all results for the specified trials.
    Args:
        eval_info_dict (dict): A dictionary containing evaluation information.
        trials (list[str]): A list of trial names for which to save results.
        output_dir (Path or str, optional): The output directory where
                   results will be saved. Defaults to Path(".").
        pdb_id_list (list[str] or None, optional): A list of PDB IDs to evaluate.
                   If None, all PDB IDs in the trials will be evaluated.
                   Defaults to None.
        subset_csv (Path, optional): A CSV file containing subset information.
                   It should have columns ["type", "entry_id", "chain_id_1", "chain_id_2"].
                   "type" can be "chain" or "interface".Defaults to None.
        num_cpu (int, optional): The number of CPU cores to use for parallel
                processing. Defaults to 1.

    Returns:
        tuple[Path]: A tuple containing the paths to the saved DockQ, LDDT, and RMSD results CSV files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # {"RecentPDB" or "PoseBusters": {trial_name: csv_path}}
    trial_name_to_result_files = _find_result_csv(eval_info_dict, trials)

    tasks = _prepare_tasks(
        eval_info_dict,
        trial_name_to_result_files,
        pdb_id_list,
        subset_csv,
    )

    results = [
        r
        for r in tqdm(
            Parallel(n_jobs=num_cpu, return_as="generator_unordered")(
                delayed(_get_a_dataset_result)(*task) for task in tasks
            ),
            total=len(tasks),
            desc="Show intersection results",
        )
    ]

    dockq_results = []
    lddt_results = []
    rmsd_results = []
    others_results = []

    dockq_details = []
    lddt_details = []
    rmsd_details = []
    for _eval_dataset, metric_dict in results:
        for metric_name, (result_df, details_df) in metric_dict.items():
            if metric_name == "dockq":
                dockq_results.append(result_df)
                dockq_details.append(details_df)
            elif metric_name == "lddt":
                lddt_results.append(result_df)
                lddt_details.append(details_df)
            elif metric_name == "rmsd":
                rmsd_results.append(result_df)
                rmsd_details.append(details_df)
            elif metric_name == "others":
                others_results.append(result_df)
            else:
                raise NotImplementedError(f"Unknown metric type {metric_name}")

    # Save results to CSV files
    dockq_csv, lddt_csv, rmsd_csv, others_csv = _save_to_output_csv(
        output_dir,
        dockq_results,
        lddt_results,
        rmsd_results,
        others_results,
        dockq_details,
        lddt_details,
        rmsd_details,
    )
    return dockq_csv, lddt_csv, rmsd_csv, others_csv


def get_intersection_results(
    eval_info_dict: dict,
    trials: list[str],
    output_dir: Path | str = Path("."),
    pdb_id_list: list[str] | None = None,
    subset_csv: Path | None = None,
    num_cpu: int = 1,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """
    Generate and save intersection results for specified trials.

    This function generates aggregated results for the specified trials and saves them to CSV files.
    It then returns the paths to the saved DockQ, LDDT, and RMSD results CSV files.

    Args:
        eval_info_dict (dict): A dictionary containing evaluation information.
        trials (list[str]): A list of trial names for which to generate
                      intersection results.
        output_dir (Path or str, optional): The output directory where results
                   will be saved. Defaults to Path(".").
        pdb_id_list (list[str] or None, optional): A list of PDB IDs to evaluate.
                   If None, all PDB IDs in the trials will be evaluated.
                   Defaults to None.
        subset_csv (Path, optional): A CSV file containing subset information.
                   It should have columns ["type", "entry_id", "chain_id_1", "chain_id_2"].
                   "type" can be "chain" or "interface".Defaults to None.
        num_cpu (int, optional): The number of CPU cores to use for parallel processing. Defaults to 1.
        overwrite (bool, optional): If True, overwrite aggregating result CSV files. Defaults to False.

    Returns:
        tuple[Path]: A tuple containing the paths to the saved DockQ, LDDT, and RMSD results CSV files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    gen_aggregated_results(eval_info_dict, trials, num_cpu=num_cpu, overwrite=overwrite)
    dockq_csv, lddt_csv, rmsd_csv, others_csv = save_all_results(
        eval_info_dict,
        trials,
        output_dir=output_dir,
        pdb_id_list=pdb_id_list,
        subset_csv=subset_csv,
        num_cpu=num_cpu,
    )
    return dockq_csv, lddt_csv, rmsd_csv, others_csv


def run(
    eval_info_json: Path,
    output_path: Path | str,
    pdb_id_list_file: Path | None = None,
    subset_csv: Path | None = None,
    num_cpu: int = 1,
    overwrite_agg: bool = False,
    out_file_name: str = "Summary_table",
):
    """
    Run the process to generate and save intersection results for specified datasets.

    Args:
        eval_info_json (Path): A JSON file containing evaluation information.
        output_path (Path or str): The output directory where results will be saved.
        pdb_id_list_file (Path, optional): A file containing a list of PDB IDs to evaluation. Defaults to None.
        subset_csv (Path, optional): A CSV file containing subset information.
                   It should have columns ["type", "entry_id", "chain_id_1", "chain_id_2"].
                   "type" can be "chain" or "interface".Defaults to None.
        num_cpu (int, optional): The number of CPU cores to use for parallel processing. Defaults to 1.
        overwrite_agg (bool, optional): If True, overwrite aggregating result CSV files. Defaults to False.
        out_file_name (str, optional): The base name for the output CSV files. Defaults to "Summary_table".
    """
    with open(eval_info_json) as f:
        eval_info_dict = json.load(f)

    trials = list(eval_info_dict.keys())

    logging.info("Processing trials: %s\n", trials)

    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    if pdb_id_list_file is not None:
        with open(pdb_id_list_file) as f:
            pdb_id_list = [line.strip() for line in f.readlines()]
    else:
        pdb_id_list = None

    dockq_csv, lddt_csv, rmsd_csv, others_csv = get_intersection_results(
        eval_info_dict,
        trials,
        output_dir=output_path,
        pdb_id_list=pdb_id_list,
        subset_csv=subset_csv,
        num_cpu=num_cpu,
        overwrite=overwrite_agg,
    )

    output_summary_csv_path = output_path / f"{out_file_name}.csv"
    output_ranked_csv_path = output_path / f"{out_file_name}_ranked.csv"
    run_reduce(
        output_summary_csv_path,
        output_ranked_csv_path,
        dockq_csv,
        lddt_csv,
        rmsd_csv,
        others_csv,
        order=trials,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_json",
        type=Path,
        help="Path to the input JSON file.",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=Path,
        default=Path("./pxm_results"),
        help="Path to the output directory.",
    )
    parser.add_argument(
        "-n",
        "--num_cpu",
        type=int,
        default=-1,
        help="Number of CPU cores to use for parallel processing.",
    )
    parser.add_argument(
        "-p",
        "--pdb_id_list_file",
        type=Path,
        default=None,
        help="A txt file containing a list of PDB IDs to process. (Each line is a PDB ID)",
    )
    parser.add_argument(
        "-c",
        "--subset_csv",
        type=Path,
        default=None,
        help='A csv file containing ["type", "entry_id", "chain_id_1", "chain_id_2"] columns. \
        It use to subset the results. "type" can be "chain" or "interface"',
    )
    parser.add_argument(
        "--overwrite_agg",
        action="store_true",
        help="Overwrite aggregating result CSV files.",
    )
    parser.add_argument(
        "--out_file_name",
        type=str,
        default="Summary_table",
        help="Name of the output file.",
    )
    args = parser.parse_args()

    run(
        eval_info_json=args.input_json,
        output_path=args.output_path,
        pdb_id_list_file=args.pdb_id_list_file,
        subset_csv=args.subset_csv,
        num_cpu=args.num_cpu,
        overwrite_agg=args.overwrite_agg,
        out_file_name=args.out_file_name,
    )
