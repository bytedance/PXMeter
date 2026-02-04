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
from pathlib import Path

import pandas as pd
from joblib import delayed, Parallel
from pxmeter.constants import PROTEIN
from tqdm import tqdm

from benchmark.aggregator import run_aggregator
from benchmark.configs.data_config import SUPPORTED_DATA
from benchmark.configs.dataset_metrics_config import DATASET_METRICS_CONFIG
from benchmark.show_results import ChainInterfaceDisplayer, RMSDDisplayer
from benchmark.simplified_results import run_reduce
from benchmark.utils import (
    add_cluster_id_to_df,
    add_comp_chain_iface_id,
    query_subset_labels,
    select_df_by_eval_types,
)


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


def _find_result_csv(
    eval_info_dict: dict, trials: list[str]
) -> dict[str, dict[str, Path]]:
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
        dict[str, dict[str, Path]]: A dictionary mapping trial names to a dictionary of
        trial names to result CSV file paths.
    """
    # {"RecentPDB" or "PoseBusters": {trial_name: csv_path}}
    trial_name_to_result_files = defaultdict(dict)
    for trial_name, trial_dict in eval_info_dict.items():
        if trial_name not in trials:
            continue

        for eval_dataset, dataset_path in trial_dict["dataset_path"].items():
            eval_result_dir = Path(dataset_path)
            if not eval_result_dir.exists():
                logging.warning("%s does not exist", eval_result_dir)
                continue

            if eval_result_dir.name.endswith(".csv") or eval_result_dir.name.endswith(
                ".parquet"
            ):
                result_csv = eval_result_dir
            else:
                result_csv = Path(
                    eval_result_dir.parent / f"{eval_result_dir.name}_metrics.parquet"
                )
                if result_csv.with_suffix(".csv").exists():
                    result_csv = result_csv.with_suffix(".csv")

            trial_name_to_result_files[eval_dataset][trial_name] = result_csv
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
        for _trial_name, metrics_csv in trial_name_to_csv_path.items():
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
    trial_name_to_metrics_file: dict[str, str],
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
    pb_metrics_file = Path(
        str(trial_name_to_metrics_file[dataset_name]).replace("_metrics.", "_pb_valid.")
    )

    if pb_metrics_file.exists():
        if str(pb_metrics_file).endswith(".parquet"):
            pb_valid_df = pd.read_parquet(
                pb_metrics_file,
                engine="pyarrow",
            )
            for col in ["entry_id", "seed", "sample"]:
                pb_valid_df[col] = pb_valid_df[col].astype("string")
        else:
            pb_valid_df = pd.read_csv(
                pb_metrics_file,
                dtype={"entry_id": str, "seed": str, "sample": str},
                low_memory=False,
            )
    else:
        pb_valid_df = None

    # Dataset specific preparation
    working_metrics_df = sub_metrics_df
    aux_df = None  # e.g. lowh_df for RecentPDB

    config_key = eval_dataset
    if eval_dataset == "dsDNA-Protein":
        config_key = "DNA-Protein"

    if config_key not in DATASET_METRICS_CONFIG:
        raise NotImplementedError(f"Unknown dataset {eval_dataset}")

    if eval_dataset == "RecentPDB":
        # Get low homology subset
        aux_df = pd.read_csv(
            SUPPORTED_DATA.recentpdb_low_homology,
            dtype={"entry_id": str, "entity_id_1": str, "entity_id_2": str},
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

        dsdna_metrics_df = (
            dsdna_metrics_df.drop_duplicates(subset=["entry_id", "sample", "seed"])
            .drop(columns=["lddt"])
            .rename(columns={"lddt_mean": "lddt"})
        )
        working_metrics_df = dsdna_metrics_df

    displayer = ChainInterfaceDisplayer(working_metrics_df, model=model, seeds=seeds)
    rmsd_displayer = None

    if eval_dataset == "RecentPDB":
        lowh_entry_ids = set(aux_df["entry_id"])
        is_lowh_entry = sub_metrics_df["entry_id"].isin(lowh_entry_ids)
        rmsd_metrics_df = sub_metrics_df[is_lowh_entry]
    else:
        rmsd_metrics_df = working_metrics_df

    if ("lig_rmsd" in rmsd_metrics_df.columns) or (
        "lig_rmsd_wo_refl" in rmsd_metrics_df.columns
    ):
        rmsd_displayer = RMSDDisplayer(
            rmsd_metrics_df,
            pb_valid_df=pb_valid_df,
            model=model,
            seeds=seeds,
        )

    results_lists = defaultdict(list)
    details_lists = defaultdict(list)

    for config in DATASET_METRICS_CONFIG[config_key]:
        mask = None
        if (
            config["subset_label"]
            and eval_dataset == "RecentPDB"
            and aux_df is not None
        ):
            # Support multiple subset_label values separated by semicolons
            # in the config, e.g. "[antibody-protein];[peptide-interface]".
            # For each label, build an individual mask and then intersect
            # them on the metrics DataFrame.
            subset_labels = [
                lbl.strip()
                for lbl in str(config["subset_label"]).split(";")
                if str(lbl).strip()
            ]

            for subset_label in subset_labels:
                # We need to map subset_label to mask on working_metrics_df
                # aux_df is lowh_df
                target_lowh_entries = query_subset_labels(
                    aux_df["subset"], subset_label
                )

                # The working_metrics_df is already filtered by lowh_df keys.
                # We assume get_low_homology_subset matches accurately.
                # But get_low_homology_subset takes metrics_df and lowh_df.
                label_mask = get_low_homology_subset(
                    working_metrics_df, aux_df[target_lowh_entries]
                )

                if mask is None:
                    mask = label_mask
                else:
                    # Take the intersection across multiple subset_label masks
                    mask &= label_mask

        # First compute the intersection of all labels, then apply inverse_subset
        if config["inverse_subset"] and mask is not None:
            mask = ~mask

        subset_name = config["subset"]
        eval_types = config["eval_type"]
        metrics = config["metric"]

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

            elif metric == "rmsd":
                if rmsd_displayer:
                    res, det = rmsd_displayer.get_rmsd()
                    results_lists["rmsd"].append(res)
                    details_lists["rmsd"].append(det)
            else:
                raise NotImplementedError(f"Unknown metric {metric}")

    metric_results = {}
    for metric_name in ["dockq", "lddt", "rmsd"]:
        if results_lists[metric_name]:
            result_df = pd.concat(results_lists[metric_name])
            details_df = pd.concat(details_lists[metric_name])

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
    dockq_details: list[pd.DataFrame],
    lddt_details: list[pd.DataFrame],
    rmsd_details: list[pd.DataFrame],
):
    dockq_csv = None
    lddt_csv = None
    rmsd_csv = None

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
            all_rmsd_df["lig_avg_rmsd"] = all_rmsd_df["lig_avg_rmsd"].astype(float)
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
    return dockq_csv, lddt_csv, rmsd_csv


def _integrity_check(
    metrics_df: pd.DataFrame, seeds_to_check: list[str] | None = None
) -> set[str]:
    if seeds_to_check is None:
        seeds_to_check = list(metrics_df["seed"].unique())

    incomplete_entries = set()
    all_set = set(metrics_df["entry_id"])
    for seed in seeds_to_check:
        # Here we assume that a sample will not be missing across all seeds
        for sample in metrics_df["sample"].unique():
            failed = all_set - set(
                metrics_df["entry_id"][
                    (metrics_df["seed"] == seed) & (metrics_df["sample"] == sample)
                ]
            )
            incomplete_entries |= failed
    return incomplete_entries


def _prepare_tasks(
    eval_info_dict: dict,
    dataset_to_result_files: dict[str, dict[str, str]],
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
    for eval_dataset, dataset_name_to_csv_path in dataset_to_result_files.items():
        filepath_to_df = {}
        filepath_to_seeds = defaultdict(set)
        filepath_to_dataset_name = defaultdict(list)
        for dataset_name, metrics_file in dataset_name_to_csv_path.items():
            filepath_to_dataset_name[metrics_file].append(dataset_name)

            if metrics_file not in filepath_to_df:
                if metrics_file.suffix == ".csv":
                    metrics_df = pd.read_csv(
                        metrics_file,
                        dtype={
                            "entry_id": str,
                            "entity_id_1": str,
                            "entity_id_2": str,
                            "seed": str,
                            "sample": str,
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

                filepath_to_df[metrics_file] = metrics_df
                logging.info(
                    "%s entries loaded from '%s'\n",
                    metrics_df["entry_id"].nunique(),
                    metrics_file,
                )

            seeds = eval_info_dict[dataset_name].get("seeds", [])
            if seeds:
                filepath_to_seeds[metrics_file].update(str(s) for s in seeds)

        intersection = set(subset_match_key).copy()
        for metrics_file, metrics_df in list(filepath_to_df.items()):
            df = metrics_df

            if pdb_id_list is not None:
                df = df[df["entry_id"].isin(pdb_id_list)].copy()
                logging.info(
                    '%d entries remain in "%s" after pdb_id_list filter\n',
                    df["entry_id"].nunique(),
                    ",".join(filepath_to_dataset_name[metrics_file]),
                )
                if df.empty:
                    raise AssertionError(
                        f"No PDB IDs found in the pdb_id_list for {metrics_file}"
                    )

            seeds_to_check = None
            all_seeds = filepath_to_seeds.get(metrics_file, set())
            if all_seeds:
                df = df[df["seed"].isin(all_seeds)].copy()
                if df.empty:
                    raise ValueError(
                        f"No matching seeds found in {metrics_file}; "
                        f"expected any of {all_seeds}."
                    )
                seeds_to_check = list(all_seeds)

            incomplete_entries = _integrity_check(df, seeds_to_check=seeds_to_check)
            if incomplete_entries:
                logging.warning(
                    '%d entries are incomplete (N_seed * N_sample) in "%s"; \
                        dropping them before dataset intersection\n',
                    len(incomplete_entries),
                    ",".join(filepath_to_dataset_name[metrics_file]),
                )
                df = df[~df["entry_id"].isin(incomplete_entries)].copy()

            unique_match_key = set(df["match_key"])
            if not intersection:
                intersection = unique_match_key
            else:
                intersection &= unique_match_key
            assert (
                intersection
            ), f'Intersection became empty after processing "{metrics_file}"'

            filepath_to_df[metrics_file] = df

        # Filter DataFrames by intersection
        for metrics_file, metrics_df in list(filepath_to_df.items()):
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

            filepath_to_df[metrics_file] = sub_metrics_df
            logging.info(
                '%s entries in the intersection from "%s"\n',
                sub_metrics_df["entry_id"].nunique(),
                ",".join(filepath_to_dataset_name[metrics_file]),
            )

        for dataset_name, metrics_file in dataset_name_to_csv_path.items():
            model = eval_info_dict[dataset_name]["model"]
            sub_metrics_df = filepath_to_df[metrics_file]
            dataset_seeds = [
                str(i)
                for i in sorted(set(eval_info_dict[dataset_name].get("seeds", [])))
            ]

            if dataset_seeds:
                sub_metrics_df = sub_metrics_df[
                    sub_metrics_df["seed"].isin(set(dataset_seeds))
                ].copy()

            remain_seeds = list(sub_metrics_df["seed"].unique())
            logging.info(
                '%d entries after filtering from "%s" by %d seeds: [%s]\n',
                sub_metrics_df["entry_id"].nunique(),
                dataset_name,
                len(remain_seeds),
                ",".join(remain_seeds),
            )

            tasks.append(
                [
                    dataset_name_to_csv_path,
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
            else:
                raise NotImplementedError(f"Unknown metric type {metric_name}")

    # Save results to CSV files
    dockq_csv, lddt_csv, rmsd_csv = _save_to_output_csv(
        output_dir,
        dockq_results,
        lddt_results,
        rmsd_results,
        dockq_details,
        lddt_details,
        rmsd_details,
    )
    return dockq_csv, lddt_csv, rmsd_csv


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
    dockq_csv, lddt_csv, rmsd_csv = save_all_results(
        eval_info_dict,
        trials,
        output_dir=output_dir,
        pdb_id_list=pdb_id_list,
        subset_csv=subset_csv,
        num_cpu=num_cpu,
    )
    return dockq_csv, lddt_csv, rmsd_csv


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

    dockq_csv, lddt_csv, rmsd_csv = get_intersection_results(
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
