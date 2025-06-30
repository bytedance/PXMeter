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
from joblib import Parallel, delayed
from tqdm import tqdm

from benchmark.aggregator import run_aggregator
from benchmark.configs.data_config import EVAL_RESULTS, SUPPORTED_DATA
from benchmark.show_results import ChainInterfaceDisplayer, RMSDDisplayer
from benchmark.simplified_results import reduce_csv_content
from pxmeter.constants import PROTEIN


def get_antibody_entities():
    """
    Retrieve a set of antibody entities from a PDB cluster file.

    This function reads a PDB cluster file, extracts the first two lines (Top2 largest clusters),
    and creates a set of unique antibody entities by converting them to lowercase.

    Returns:
        set: A set of unique antibody entities.
    """
    with open(SUPPORTED_DATA.pdb_cluster_file, "r") as f:
        lines = f.readlines()[:2]
    cluster_list = [line.strip().split() for line in lines]
    antibody_entities = set(
        [i.lower() for i in cluster_list[0]] + [i.lower() for i in cluster_list[1]]
    )
    return antibody_entities


def get_antibody_col(metrics_df: pd.DataFrame) -> pd.Series:
    """
    Generate a mask for antibody entities in a metrics DataFrame.

    Args:
        metrics_df (pd.DataFrame): The metrics DataFrame.

    Returns:
        pd.Series: A pd.Series indicating whether each entity
                   is an antibody or an interface of antibody-antigen.
    """
    antibody_entities = get_antibody_entities()

    def is_antibody_or_antibody_antigen(row):
        pdb_entity_id_1 = str(row["entry_id"]).lower() + "_" + str(row["entity_id_1"])

        if row["type"] == "chain":
            if pdb_entity_id_1 in antibody_entities and row["entity_type_1"] == PROTEIN:
                return "antibody"
        elif row["type"] == "interface":
            pdb_entity_id_2 = (
                str(row["entry_id"]).lower() + "_" + str(row["entity_id_2"])
            )
            if (
                (row["entity_type_1"], row["entity_type_2"]) == (PROTEIN, PROTEIN)
            ) and (
                (
                    (pdb_entity_id_1 in antibody_entities)
                    and (pdb_entity_id_2 not in antibody_entities)
                )
                or (
                    (pdb_entity_id_1 not in antibody_entities)
                    and (pdb_entity_id_2 in antibody_entities)
                )
            ):
                return "antibody_antigen"
        return None

    # "antibody", "antibody_antigen" or None
    ab_type = metrics_df.apply(is_antibody_or_antibody_antigen, axis=1)
    return ab_type


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
        interface_to_cluster[(row["pdb_id"], row["chain_id_1"], row["chain_id_2"])] = (
            row["interface_cluster_key"]
        )

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


def get_protein_monomer_pdb_ids() -> set[str]:
    """
    Get a set of PDB IDs for protein monomers from the RecentPDB dataset.
    """
    df = pd.read_csv(
        SUPPORTED_DATA.recentpdb_low_homology_entity_type_count,
        dtype={"entry_id": str},
    )
    return set(df[df["is_protein_monomer"]]["entry_id"])


def get_low_homology_subset(
    metrics_df: pd.DataFrame, lowh_csv: Path | str
) -> pd.DataFrame:
    """
    Filter a metrics DataFrame of RecentPDB to include only low homology entries.

    Args:
        metrics_df (pd.DataFrame): The metrics DataFrame.
        lowh_csv (Path or str): The path to the low homology CSV file.

    Returns:
        pd.DataFrame: The filtered metrics DataFrame containing only low homology entries.
    """
    # cols: type, entry_id, entity_id_1, entity_id_2
    lowh_df = pd.read_csv(lowh_csv, dtype=str)

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

    lowh_metrics_df = metrics_df[metrics_keys.isin(lowh_df_keys)]
    lowh_metrics_df_wo_nan_cluster = lowh_metrics_df.dropna(
        subset=["cluster_id"], how="all", axis=0
    )
    return lowh_metrics_df_wo_nan_cluster


def _find_result_csv(dataset_names: list[str]) -> dict[str, dict[str, Path]]:
    """
    Find the result CSV files for the specified datasets.

    This function iterates over the evaluation datasets and finds the result CSV files
    for each dataset. It checks if the dataset is in the list of specified datasets and
    if the evaluation result directory exists. If the result CSV file exists, it adds
    it to the dictionary of dataset to result files.

    Args:
        dataset_names (list[str]): A list of dataset names for which to find result CSV files.

    Returns:
        dict[str, dict[str, Path]]: A dictionary mapping dataset names to a dictionary of
        dataset names to result CSV file paths.
    """
    # {"RecentPDB" or "PoseBusters": {dataset_name: csv_path}}
    dataset_to_result_files = defaultdict(dict)
    for dataset_name, dataset_dict in EVAL_RESULTS.items():
        if dataset_name not in dataset_names:
            continue

        for eval_dataset, dataset_path in dataset_dict["dataset_path"].items():
            eval_result_dir = Path(dataset_path)
            if not eval_result_dir.exists():
                logging.warning("%s does not exist", eval_result_dir)
                continue

            result_csv = Path(
                eval_result_dir.parent / f"{eval_result_dir.name}_metrics.csv"
            )

            dataset_to_result_files[eval_dataset][dataset_name] = result_csv
    return dataset_to_result_files


def gen_aggregated_results(
    dataset_names: list[str], num_cpu: int = 1, overwrite: bool = False
):
    """
    Generate aggregated results for specified datasets.

    Args:
        dataset_names (list[str]): A list of dataset names for which to
                      generate aggregated results.
        num_cpu (int, optional): The number of CPU cores to use for
                parallel processing. Defaults to 1.
        overwrite (bool, optional): If True, overwrite existing
                  result CSV files. Defaults to False.
    """
    dataset_to_result_files = _find_result_csv(dataset_names)
    for eval_dataset, dataset_name_to_csv_path in dataset_to_result_files.items():
        if not dataset_name_to_csv_path:
            continue
        for _dataset_name, metrics_csv in dataset_name_to_csv_path.items():
            eval_result_dir = metrics_csv.parent / str(metrics_csv.name).replace(
                "_metrics.csv", ""
            )

            interface_only_use_polymer_cluster = False
            if eval_dataset == "RecentPDB":
                cluster_file = SUPPORTED_DATA.recentpdb_low_homology_cluster
                interface_only_use_polymer_cluster = True

            elif eval_dataset == "RecentPDB-NA":
                cluster_file = SUPPORTED_DATA.recentpdb_na_cluster
            else:
                cluster_file = None

            if (not metrics_csv.exists()) or overwrite:
                logging.info("Aggregating for: %s", eval_result_dir)
                run_aggregator(
                    eval_result_dir,
                    cluster_csv=cluster_file,
                    num_cpu=num_cpu,
                    interface_only_use_polymer_cluster=interface_only_use_polymer_cluster,
                )


def get_recent_pdb_results_df(
    sub_metrics_df: pd.DataFrame,
    model: str = "nan",
    seeds: list[str | int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate and return DockQ and LDDT results for RecentPDB dataset.

    Args:
        sub_metrics_df (pd.DataFrame): Subset of metrics DataFrame for RecentPDB.
        model (str): Name of the model used for evaluation.
        seeds (list[str | int] | None, optional): List of seeds for evaluation. Defaults to None.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
            - The first DataFrame contains all DockQ results.
            - The second DataFrame contains all LDDT results.
    """
    # Get low homology subset
    lowh_csv = SUPPORTED_DATA.recentpdb_low_homology
    lowh_sub_metrics_df = get_low_homology_subset(sub_metrics_df, lowh_csv)
    displayer = ChainInterfaceDisplayer(lowh_sub_metrics_df, model=model, seeds=seeds)

    # DockQ
    all_pp_dockq_df, all_pp_dockq_details_df = displayer.get_dockq_sr_by_cluster(
        eval_types=["Protein-Protein"]
    )

    # DockQ for antibody
    ab_col = get_antibody_col(lowh_sub_metrics_df)
    ab_pp_dockq_df, ab_pp_dockq_details_df = displayer.get_dockq_sr_by_cluster(
        eval_types=["Protein-Protein"],
        mask_on_metrics_df=ab_col == "antibody_antigen",
    )
    ab_pp_dockq_df["eval_type"] = "Protein-Protein (Antibody=True)"
    ab_pp_dockq_details_df["eval_type"] = "Protein-Protein (Antibody=True)"

    not_ab_pp_dockq_df, not_ab_pp_dockq_details_df = displayer.get_dockq_sr_by_cluster(
        eval_types=["Protein-Protein"],
        mask_on_metrics_df=ab_col != "antibody_antigen",
    )
    not_ab_pp_dockq_df["eval_type"] = "Protein-Protein (Antibody=False)"
    not_ab_pp_dockq_details_df["eval_type"] = "Protein-Protein (Antibody=False)"

    # LDDT
    all_lddt_results_df, all_lddt_details_df = displayer.get_lddt_by_cluster()

    # LDDT for antibody
    ab_pp_lddt_df, ab_pp_lddt_details_df = displayer.get_lddt_by_cluster(
        eval_types=["Protein-Protein"],
        mask_on_metrics_df=ab_col == "antibody_antigen",
    )
    ab_pp_lddt_df["eval_type"] = "Protein-Protein (Antibody=True)"
    ab_pp_lddt_details_df["eval_type"] = "Protein-Protein (Antibody=True)"

    not_ab_pp_lddt_df, not_ab_pp_lddt_details_df = displayer.get_lddt_by_cluster(
        eval_types=["Protein-Protein"],
        mask_on_metrics_df=ab_col != "antibody_antigen",
    )
    not_ab_pp_lddt_df["eval_type"] = "Protein-Protein (Antibody=False)"
    not_ab_pp_lddt_details_df["eval_type"] = "Protein-Protein (Antibody=False)"

    # LDDT for protein monomer
    monomer_ids = get_protein_monomer_pdb_ids()
    monomer_lddt_df, monomer_lddt_details_df = displayer.get_lddt_by_cluster(
        eval_types=["Intra-Protein"],
        mask_on_metrics_df=lowh_sub_metrics_df["entry_id"].isin(monomer_ids),
    )
    monomer_lddt_df["eval_type"] = "Intra-Protein (Monomer)"
    monomer_lddt_details_df["eval_type"] = "Intra-Protein (Monomer)"

    # LDDT for ligand-protein (not low homology set)
    displayer_not_lowh = ChainInterfaceDisplayer(
        sub_metrics_df, model=model, seeds=seeds
    )
    ligand_protein_lddt_df, ligand_protein_lddt_details_df = (
        displayer_not_lowh.get_lddt_by_cluster(
            eval_types=["Protein-Ligand"],
        )
    )

    # Merge results
    dockq_results_df = pd.concat([all_pp_dockq_df, ab_pp_dockq_df, not_ab_pp_dockq_df])
    lddt_results_df = pd.concat(
        [
            all_lddt_results_df,
            ab_pp_lddt_df,
            not_ab_pp_lddt_df,
            monomer_lddt_df,
            ligand_protein_lddt_df,
        ]
    )

    dockq_details_df = pd.concat(
        [all_pp_dockq_details_df, ab_pp_dockq_details_df, not_ab_pp_dockq_details_df]
    )
    lddt_details_df = pd.concat(
        [
            all_lddt_details_df,
            ab_pp_lddt_details_df,
            not_ab_pp_lddt_details_df,
            monomer_lddt_details_df,
            ligand_protein_lddt_details_df,
        ]
    )

    return dockq_results_df, lddt_results_df, dockq_details_df, lddt_details_df


def get_recent_pdb_na_results_df(
    sub_metrics_df: pd.DataFrame,
    model: str = "nan",
    seeds: list[str | int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate and return DockQ success rate and LDDT results for the RecentPDB-NA dataset.

    Args:
        sub_metrics_df (pd.DataFrame): Subset of the metrics DataFrame for the RecentPDB-NA dataset.
        model (str): Name of the model used for evaluation.
        seeds (list[str | int] | None, optional): List of seeds for evaluation. Defaults to None.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
            - The first DataFrame contains DockQ success rate results for DNA/RNA - Protein and Intra - DNA/RNA.
            - The second DataFrame contains LDDT results for DNA/RNA - Protein and Intra - DNA/RNA.
    """
    lowh_csv = SUPPORTED_DATA.recentpdb_na_low_homology
    lowh_sub_metrics_df = get_low_homology_subset(sub_metrics_df, lowh_csv)
    displayer = ChainInterfaceDisplayer(lowh_sub_metrics_df, model=model, seeds=seeds)

    # DockQ
    dockq_results_df, dockq_details_df = displayer.get_dockq_sr_by_cluster(
        eval_types=["DNA-Protein", "RNA-Protein"]
    )

    # LDDT
    lddt_results_df, lddt_details_df = displayer.get_lddt_by_cluster(
        eval_types=["DNA-Protein", "RNA-Protein", "Intra-RNA", "Intra-DNA"]
    )
    return dockq_results_df, lddt_results_df, dockq_details_df, lddt_details_df


def get_af3_ab_results_df(
    sub_metrics_df: pd.DataFrame,
    model: str = "nan",
    seeds: list[str | int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate and return DockQ success rate and LDDT results for AF3 Antibody-Antigen Subset.

    Args:
        sub_metrics_df (pd.DataFrame): Subset of the metrics DataFrame.
        model (str): Name of the model used for evaluation.
        seeds (list[str | int] | None, optional): List of seeds for evaluation. Defaults to None.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
            - The first DataFrame contains DockQ success rate results for antibody-protein interfaces.
            - The second DataFrame contains LDDT results for antibody-protein interfaces.
    """
    af3_ab_sub_df = get_af3_ab_sub_df(sub_metrics_df)

    displayer = ChainInterfaceDisplayer(af3_ab_sub_df, model=model, seeds=seeds)
    ab_pp_dockq_df, ab_pp_dockq_details_df = displayer.get_dockq_sr_by_cluster(
        eval_types=["Protein-Protein"],
    )
    ab_pp_dockq_df["eval_type"] = "Protein-Protein (Antibody=True)"
    ab_pp_dockq_details_df["eval_type"] = "Protein-Protein (Antibody=True)"

    # LDDT for antibody
    ab_pp_lddt_df, ab_pp_lddt_details_df = displayer.get_lddt_by_cluster(
        eval_types=["Protein-Protein"]
    )
    ab_pp_lddt_df["eval_type"] = "Protein-Protein (Antibody=True)"
    ab_pp_lddt_details_df["eval_type"] = "Protein-Protein (Antibody=True)"

    return ab_pp_dockq_df, ab_pp_lddt_df, ab_pp_dockq_details_df, ab_pp_lddt_details_df


def _get_a_dataset_result(
    dataset_name_to_csv_path: dict[str, str],
    sub_metrics_df: pd.DataFrame,
    eval_dataset: str,
    model: str,
    dataset_name: str,
    seeds: list[str | int] | None = None,
) -> tuple[str, list[pd.DataFrame], list[pd.DataFrame]]:
    result_df_list = []
    details_df_list = []
    if eval_dataset == "RecentPDB":
        dockq_results_df, lddt_results_df, dockq_details_df, lddt_details_df = (
            get_recent_pdb_results_df(sub_metrics_df, model, seeds)
        )
        result_df_list += [dockq_results_df, lddt_results_df]
        details_df_list += [dockq_details_df, lddt_details_df]

    elif eval_dataset == "RecentPDB-NA":
        dockq_results_df, lddt_results_df, dockq_details_df, lddt_details_df = (
            get_recent_pdb_na_results_df(sub_metrics_df, model, seeds)
        )
        result_df_list += [dockq_results_df, lddt_results_df]
        details_df_list += [dockq_details_df, lddt_details_df]

    elif eval_dataset == "AF3-AB":
        dockq_results_df, lddt_results_df, dockq_details_df, lddt_details_df = (
            get_af3_ab_results_df(sub_metrics_df, model, seeds)
        )
        result_df_list += [dockq_results_df, lddt_results_df]
        details_df_list += [dockq_details_df, lddt_details_df]

    elif eval_dataset == "PoseBusters":
        # Replace "_metrics.csv" with "_pb_valid.csv"
        pb_valid_csv = str(dataset_name_to_csv_path[dataset_name]).replace(
            "_metrics.csv", "_pb_valid.csv"
        )

        pb_valid_df = pd.read_csv(
            pb_valid_csv,
            dtype={"entry_id": str, "seed": str, "sample": str},
            low_memory=False,
        )
        displayer = RMSDDisplayer(
            sub_metrics_df,
            pb_valid_df=pb_valid_df,
            model=model,
            seeds=seeds,
        )
        rmsd_df, rmsd_details_df = displayer.get_rmsd()
        result_df_list.append(rmsd_df)
        details_df_list.append(rmsd_details_df)

    elif eval_dataset == "dsDNA-Protein":
        dsdna_metrics_df = sub_metrics_df.copy()

        # Filter out the rows with eval_type == "DNA-Protein"
        dsdna_metrics_df = ChainInterfaceDisplayer.select_df_by_eval_types(
            dsdna_metrics_df, ["DNA-Protein"]
        )

        # Average the LDDT for each CIF (DNA1-Protein, DNA2-protein)
        # Each CIF keeps one row only with the average LDDT of dsDNA-Protein
        dsdna_metrics_df["lddt_mean"] = dsdna_metrics_df.groupby(
            ["entry_id", "sample", "seed"]
        )["lddt"].transform("mean")

        dsdna_metrics_df = (
            dsdna_metrics_df.drop_duplicates(subset=["entry_id", "sample", "seed"])
            .drop(columns=["lddt"])
            .rename(columns={"lddt_mean": "lddt"})
        )

        displayer = ChainInterfaceDisplayer(dsdna_metrics_df, model=model, seeds=seeds)
        dna_prot_df, dna_prot_details_df = displayer.get_lddt_by_cluster(
            eval_types=["DNA-Protein"]
        )
        dna_prot_df["eval_type"] = "dsDNA-Protein"
        dna_prot_details_df["eval_type"] = "dsDNA-Protein"
        result_df_list.append(dna_prot_df)
        details_df_list.append(dna_prot_details_df)

    elif eval_dataset == "RNA-Protein":
        displayer = ChainInterfaceDisplayer(sub_metrics_df, model=model, seeds=seeds)
        rna_prot_df, rna_prot_details_df = displayer.get_lddt_by_cluster(
            eval_types=["RNA-Protein"]
        )
        result_df_list.append(rna_prot_df)
        details_df_list.append(rna_prot_details_df)

    else:
        raise NotImplementedError(f"Unknown dataset {eval_dataset}")

    for result_df in result_df_list:
        result_df.insert(0, "name", dataset_name)
        result_df.insert(1, "eval_dataset", eval_dataset)
    for details_df in details_df_list:
        details_df.insert(0, "name", dataset_name)
        details_df.insert(1, "eval_dataset", eval_dataset)

    return eval_dataset, result_df_list, details_df_list


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
            all_rmsd_details_df.to_csv(
                rmsd_details_csv,
                index=False,
                quoting=csv.QUOTE_NONNUMERIC,
            )
            logging.info("RMSD details saved to %s", rmsd_details_csv)
    return dockq_csv, lddt_csv, rmsd_csv


def save_all_results(
    dataset_names: list[str],
    output_dir: Path | str = Path("."),
    pdb_id_list: list[str] | None = None,
    subset_csv: Path | None = None,
    num_cpu: int = 1,
) -> tuple[Path, ...]:
    """
    Save all results for the specified datasets.

    Args:
        dataset_names (list[str]): A list of dataset names for which to save results.
        output_dir (Path or str, optional): The output directory where
                   results will be saved. Defaults to Path(".").
        pdb_id_list (list[str] or None, optional): A list of PDB IDs to evaluate.
                   If None, all PDB IDs in the datasets will be evaluated.
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

    # {"RecentPDB" or "PoseBusters": {dataset_name: csv_path}}
    dataset_to_result_files = _find_result_csv(dataset_names)

    if subset_csv is not None:
        subset_df = pd.read_csv(subset_csv, dtype=str)
        subset_match_key = set(
            subset_df.apply(
                lambda row: (
                    row["entry_id"]
                    + "_"
                    + "_".join(sorted([row["chain_id_1"], row["chain_id_2"]]))
                    if row["type"] == "interface"
                    else "_".join(
                        [
                            row["entry_id"],
                            str(row["chain_id_1"]),
                            str(row["chain_id_2"]),
                        ]
                    )
                ),
                axis=1,
            )
        )

    else:
        subset_match_key = set()

    tasks = []
    for eval_dataset, dataset_name_to_csv_path in dataset_to_result_files.items():
        # Find intersections
        intersection = subset_match_key.copy()  # Empty set if subset_csv is None
        dataset_name_to_df = {}
        for dataset_name, metrics_csv in dataset_name_to_csv_path.items():

            # Load the CSV and convert "entry_id", "entity_id", "seed", "sample" to string
            metrics_df = pd.read_csv(
                metrics_csv,
                dtype={
                    "entry_id": str,
                    "entity_id_1": str,
                    "entity_id_2": str,
                    "seed": str,
                    "sample": str,
                },
                low_memory=False,
            )

            # Add "match_key" column to the DataFrame
            # In the JSON file output by PXMeter,
            # the chain_id pair for the interfaces has already been sorted
            metrics_df["match_key"] = metrics_df.apply(
                lambda row: "_".join(
                    [row["entry_id"], str(row["chain_id_1"]), str(row["chain_id_2"])]
                ),
                axis=1,
            )
            unique_match_key = set(metrics_df["match_key"])
            if intersection:
                intersection &= unique_match_key
            else:
                intersection = unique_match_key

            dataset_name_to_df[dataset_name] = metrics_df

        # Get intersection results
        for dataset_name, metrics_csv in dataset_name_to_csv_path.items():
            model = EVAL_RESULTS[dataset_name]["model"]
            seeds = EVAL_RESULTS[dataset_name].get("seeds")
            metrics_df = dataset_name_to_df[dataset_name]

            if pdb_id_list is not None:
                # Filter the DataFrame based on the list of PDB IDs
                metrics_df = metrics_df[metrics_df["entry_id"].isin(pdb_id_list)]
                assert (
                    len(metrics_df) > 0
                ), f"No PDB IDs found in the pdb_id_list for {metrics_csv}"

            sub_metrics_df = metrics_df[metrics_df["match_key"].isin(intersection)]

            tasks.append(
                [
                    dataset_name_to_csv_path,
                    sub_metrics_df,
                    eval_dataset,
                    model,
                    dataset_name,
                    seeds,
                ]
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
    for eval_dataset, result, details in results:
        if eval_dataset in {"RecentPDB", "RecentPDB-NA", "AF3-AB"}:
            dockq_results.append(result[0])
            lddt_results.append(result[1])
            dockq_details.append(details[0])
            lddt_details.append(details[1])
        elif eval_dataset == "PoseBusters":
            rmsd_results.append(result[0])
            rmsd_details.append(details[0])
        elif eval_dataset in {"dsDNA-Protein", "RNA-Protein"}:
            lddt_results.append(result[0])
            lddt_details.append(details[0])
        else:
            raise NotImplementedError(f"Unknown dataset {eval_dataset}")

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
    dataset_names: list[str],
    output_dir: Path | str = Path("."),
    pdb_id_list: list[str] | None = None,
    subset_csv: Path | None = None,
    num_cpu: int = 1,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """
    Generate and save intersection results for specified datasets.

    This function generates aggregated results for the specified datasets and saves them to CSV files.
    It then returns the paths to the saved DockQ, LDDT, and RMSD results CSV files.

    Args:
        dataset_names (list[str]): A list of dataset names for which to generate
                      intersection results.
        output_dir (Path or str, optional): The output directory where results
                   will be saved. Defaults to Path(".").
        pdb_id_list (list[str] or None, optional): A list of PDB IDs to evaluate.
                   If None, all PDB IDs in the datasets will be evaluated.
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
    gen_aggregated_results(dataset_names, num_cpu=num_cpu, overwrite=overwrite)
    dockq_csv, lddt_csv, rmsd_csv = save_all_results(
        dataset_names,
        output_dir=output_dir,
        pdb_id_list=pdb_id_list,
        subset_csv=subset_csv,
        num_cpu=num_cpu,
    )
    return dockq_csv, lddt_csv, rmsd_csv


def run(
    output_path: Path | str,
    dataset_names: list[str] = None,
    pdb_id_list_file: Path | None = None,
    subset_csv: Path | None = None,
    num_cpu: int = 1,
    overwrite_agg: bool = False,
    out_file_name: str = "Summary_table",
):
    """
    Run the process to generate and save intersection results for specified datasets.

    Args:
        dataset_names (list[str]): A list of dataset names for which to generate intersection results.
        output_path (Path or str): The output directory where results will be saved.
        pdb_id_list_file (Path, optional): A file containing a list of PDB IDs to evaluation. Defaults to None.
        subset_csv (Path, optional): A CSV file containing subset information.
                   It should have columns ["type", "entry_id", "chain_id_1", "chain_id_2"].
                   "type" can be "chain" or "interface".Defaults to None.
        num_cpu (int, optional): The number of CPU cores to use for parallel processing. Defaults to 1.
        overwrite_agg (bool, optional): If True, overwrite aggregating result CSV files. Defaults to False.
    """
    if dataset_names is None:
        logging.info("No dataset names provided, using all datasets")
        dataset_names = list(EVAL_RESULTS.keys())

    logging.info("Processing datasets: %s", dataset_names)

    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    if pdb_id_list_file is not None:
        with open(pdb_id_list_file) as f:
            pdb_id_list = [line.strip() for line in f.readlines()]
    else:
        pdb_id_list = None

    dockq_csv, lddt_csv, rmsd_csv = get_intersection_results(
        dataset_names,
        output_dir=output_path,
        pdb_id_list=pdb_id_list,
        subset_csv=subset_csv,
        num_cpu=num_cpu,
        overwrite=overwrite_agg,
    )

    table_df, table_str = reduce_csv_content(dockq_csv, lddt_csv, rmsd_csv)
    table_df.to_csv(
        output_path / f"{out_file_name}.csv", index=False, quoting=csv.QUOTE_NONNUMERIC
    )

    # Summary to a string of table
    with open(output_path / f"{out_file_name}.txt", "w") as f:
        f.write(table_str)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_json",
        type=Path,
        default=None,
        help="Path to the input JSON file.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=Path,
        default=Path("./pxm_results"),
        help="Path to the output directory.",
    )
    parser.add_argument(
        "-d",
        "--dataset_names",
        type=str,
        default=None,
        help="Comma-separated names of the datasets to process.",
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

    if args.input_json is not None:
        with open(args.input_json) as f:
            # replace the EVAL_RESULTS with the json file
            EVAL_RESULTS = json.load(f)

    if args.dataset_names is None:
        input_dataset_names = None
    else:
        input_dataset_names = args.dataset_names.split(",")

    run(
        output_path=args.output_path,
        dataset_names=input_dataset_names,
        pdb_id_list_file=args.pdb_id_list_file,
        subset_csv=args.subset_csv,
        num_cpu=args.num_cpu,
        overwrite_agg=args.overwrite_agg,
        out_file_name=args.out_file_name,
    )
