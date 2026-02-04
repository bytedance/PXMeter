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
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pxmeter.constants import LIGAND
from pxmeter.data.utils import is_valid_date_format

from benchmark.configs.data_config import SRC_DATA, SUPPORT_DATA_DIR, SUPPORTED_DATA
from benchmark.configs.eval_type_config import EVAL_TYPE_TO_ENTITIY_TYPES
from benchmark.utils import (
    add_cluster_id_to_df,
    query_subset_labels,
    select_df_by_eval_types,
)

NMR_METHODS = {"SOLID-STATE NMR", "SOLUTION NMR"}


def stat_filtered_num(
    pdb_meta_info_csv: Path,
    after_date: str | None = None,
    before_date: str | None = None,
    non_nmr_filter: bool = True,
    resolution_threshold: float | None = 4.5,
    num_token_threshold: int | None = 2560,
    std_polymer_only: bool = True,
    max_polymer_copies_threshold: int | None = 20,
) -> dict[str, int]:
    """
    Stat the number of PDBs after filtering.

    Args:
        pdb_meta_info_csv: The path to the PDB meta info CSV file.
        after_date: The date after which the PDBs should be filtered.
        before_date: The date before which the PDBs should be filtered.
        non_nmr_filter: Whether to filter out NMR PDBs.
        resolution_threshold: The resolution threshold.
        num_token_threshold: The number of tokens threshold.
        std_polymer_only: Whether to filter out PDBs with non-standard polymers.
        max_polymer_copies_threshold: The maximum number of polymer copies threshold.

    Returns:
        A dictionary with the number of PDBs after each filtering step.
    """
    assert after_date is None or is_valid_date_format(
        after_date
    ), f"Invalid date format: {after_date}, it should be yyyy-mm-dd format"
    assert before_date is None or is_valid_date_format(
        before_date
    ), f"Invalid date format: {before_date}, it should be yyyy-mm-dd format"

    pdb_num_stat = {}
    pdb_meta_info_df = pd.read_csv(
        pdb_meta_info_csv, dtype={"entry_id": str, "release_date": str}
    )
    pdb_num_stat["Total"] = len(pdb_meta_info_df)
    if after_date is not None:
        pdb_meta_info_df = pdb_meta_info_df[
            pdb_meta_info_df["release_date"] >= after_date
        ]
    if before_date is not None:
        pdb_meta_info_df = pdb_meta_info_df[
            pdb_meta_info_df["release_date"] <= before_date
        ]
    if after_date is not None or before_date is not None:
        pdb_num_stat["FilteredByDate"] = len(pdb_meta_info_df)

    if non_nmr_filter:
        non_nmr = ~pdb_meta_info_df["exptl_methods"].apply(
            lambda x: bool(set(x.split(";")) & NMR_METHODS)
        )
        pdb_meta_info_df = pdb_meta_info_df[non_nmr]
        pdb_num_stat["ExcludeNMR"] = len(pdb_meta_info_df)

    if resolution_threshold is not None:
        pdb_meta_info_df = pdb_meta_info_df[
            pdb_meta_info_df["resolution"] <= resolution_threshold
        ]
        pdb_num_stat["FilteredByResolution"] = len(pdb_meta_info_df)

    if num_token_threshold is not None:
        pdb_meta_info_df = pdb_meta_info_df[
            pdb_meta_info_df["num_tokens"] <= num_token_threshold
        ]
        pdb_num_stat["FilteredByTokenCount"] = len(pdb_meta_info_df)

    if std_polymer_only:
        pdb_meta_info_df = pdb_meta_info_df[~pdb_meta_info_df["no_standard_polymer"]]
        pdb_num_stat["RequireStandardPolymer"] = len(pdb_meta_info_df)

    if max_polymer_copies_threshold is not None:
        pdb_meta_info_df = pdb_meta_info_df[
            pdb_meta_info_df["max_polymer_chain_copies"] <= max_polymer_copies_threshold
        ]
        pdb_num_stat["LimitPolymerChainCopies"] = len(pdb_meta_info_df)

    pdb_meta_info_df = pdb_meta_info_df[~pdb_meta_info_df["all_chains_unk"]]
    pdb_num_stat["ExcludeAllChainsUnknown"] = len(pdb_meta_info_df)

    pdb_meta_info_df = pdb_meta_info_df[~pdb_meta_info_df["all_chains_break"]]
    pdb_num_stat["ExcludeAllChainsBreak"] = len(pdb_meta_info_df)

    pdb_meta_info_df = pdb_meta_info_df[~pdb_meta_info_df["lacking_resolved"]]
    pdb_num_stat["RequireResolvedStructure"] = len(pdb_meta_info_df)
    return pdb_num_stat


def _log_filtered_num_stat(pdb_num_stat: dict) -> str:
    steps = [
        "Total",
        "FilteredByDate",
        "ExcludeNMR",
        "FilteredByResolution",
        "FilteredByTokenCount",
        "RequireStandardPolymer",
        "LimitPolymerChainCopies",
        "ExcludeAllChainsUnknown",
        "RequireResolvedStructure",
    ]
    stat_info_str = ""
    last_num = pdb_num_stat["Total"]
    for k in steps:
        num = pdb_num_stat.get(k)
        if num is None:
            continue
        if k != "Total":
            filtered_num = f"({num - last_num})"
            last_num = num
        else:
            filtered_num = ""
        stat_info_str += f"# {k}: {num} {filtered_num}\n"
    return stat_info_str


def stat_lowh_num(
    pdb_meta_info_df: pd.DataFrame, lowh_df: pd.DataFrame, cluster_csv: Path
) -> dict:
    """
    Statistics the number of low-homology chains/interfaces.
    """
    cluster_df = pd.read_csv(cluster_csv, dtype=str)
    # ensure entity_id is a string of int
    cluster_df["label_entity_id"] = (
        pd.to_numeric(cluster_df["label_entity_id"], errors="coerce")
        .dropna()
        .astype(int)
        .astype(str)
        .reindex(cluster_df.index)
    )

    lowh_df = add_cluster_id_to_df(
        cluster_df,
        df=lowh_df,
        interface_only_use_polymer_cluster=True,
    )

    all_pdb_ids = set(lowh_df["entry_id"])
    polymer_only_eval_types = [
        k for k, v in EVAL_TYPE_TO_ENTITIY_TYPES.items() if LIGAND not in v
    ]
    lowh_polymer_pdb_ids = set(
        select_df_by_eval_types(lowh_df, polymer_only_eval_types)["entry_id"]
    )

    complex_num = len(all_pdb_ids)
    chain_num = (lowh_df["type"] == "chain").sum()
    interface_num = (lowh_df["type"] == "interface").sum()

    lowh_entry_mask = pdb_meta_info_df["entry_id"].isin(set(lowh_df["entry_id"]))
    token_num = pdb_meta_info_df["num_tokens"][lowh_entry_mask].to_list()

    eval_type_to_pdb_ids = {
        "all": all_pdb_ids,
        "lowh_polymer_only": lowh_polymer_pdb_ids,
    }
    lowh_entity_type_num = {}
    lowh_chain_interface_token_num = {}
    lowh_entity_type_cluster_num = {}
    lowh_entity_type_pdb_id_num = {}
    for k in EVAL_TYPE_TO_ENTITIY_TYPES:
        lowh_eval_type_df = select_df_by_eval_types(lowh_df, [k])
        lowh_eval_type_num = len(lowh_eval_type_df)
        lowh_eval_cluster_num = lowh_eval_type_df["cluster_id"].nunique()
        pdb_ids = set(lowh_eval_type_df["entry_id"])
        lowh_eval_pdb_id_num = len(pdb_ids)
        if len(lowh_eval_type_df) == 0:
            continue
        lowh_entity_type_num[k] = lowh_eval_type_num
        lowh_entity_type_cluster_num[k] = lowh_eval_cluster_num
        lowh_entity_type_pdb_id_num[k] = lowh_eval_pdb_id_num
        eval_type_to_pdb_ids[k] = pdb_ids

        if k.startswith("Intra"):
            lowh_chain_interface_token_num[k] = lowh_eval_type_df[
                "seq_length_1"
            ].to_list()
        else:
            lowh_chain_interface_token_num[k] = (
                lowh_eval_type_df["seq_length_1"].to_list()
                + lowh_eval_type_df["seq_length_2"].to_list()
            )

    df_exploded = lowh_df["subset"].str.split(";").explode()
    subset_counts = df_exploded.value_counts().to_dict()

    subset_pdb_id_num = {}
    subset_cluster_num = {}
    for k in subset_counts.keys():
        if not k:
            continue
        subset_lowh_df = lowh_df[query_subset_labels(lowh_df["subset"], k)]
        pdb_ids = set(subset_lowh_df["entry_id"])
        subset_pdb_id_num[k] = len(pdb_ids)
        subset_cluster_num[k] = subset_lowh_df["cluster_id"].nunique()
        eval_type_to_pdb_ids[k] = pdb_ids

    info_dict = {
        "complex_num": complex_num,
        "chain_num": chain_num,
        "interface_num": interface_num,
        "entity_type_num": lowh_entity_type_num,
        "entity_type_cluster_num": lowh_entity_type_cluster_num,
        "entity_type_pdb_id_num": lowh_entity_type_pdb_id_num,
        "token_num_list": token_num,
        "chain_interface_to_token_num_list": lowh_chain_interface_token_num,
        "subset_counts": subset_counts,
        "subset_pdb_id_num": subset_pdb_id_num,
        "subset_cluster_num": subset_cluster_num,
        "eval_type_to_pdb_ids": eval_type_to_pdb_ids,
    }

    return info_dict


def _log_lowh_num(info_dict: dict) -> str:
    stat_info_str = ""
    stat_info_str += f"# Complex: {info_dict['complex_num']}\n"
    stat_info_str += f"# Chain: {info_dict['chain_num']}\n"
    stat_info_str += f"# Interface: {info_dict['interface_num']}\n"
    stat_info_str += "------------------\n"

    for k, v in info_dict["entity_type_num"].items():
        pdb_id_num = info_dict["entity_type_pdb_id_num"][k]
        cluster_num = info_dict["entity_type_cluster_num"][k]
        stat_info_str += f"# {k}: {v} (# PDB: {pdb_id_num}, # Cluster: {cluster_num})\n"

    stat_info_str += "------------------\n"
    for k, v in info_dict["subset_counts"].items():
        if not k:
            continue
        pdb_id_num = info_dict["subset_pdb_id_num"][k]
        cluster_num = info_dict["subset_cluster_num"][k]
        stat_info_str += f"# {k}: {v} (# PDB: {pdb_id_num}, # Cluster: {cluster_num})\n"
    return stat_info_str


def _plot_number_distribution(numbers, title, xlabel):
    plt.clf()
    plt.hist(
        numbers,
        bins=range(min(numbers), max(numbers) + 2),
        edgecolor="black",
        align="left",
    )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")

    fig = plt.gcf()
    return fig


def _draw_token_num_plot(info_dict: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    token_fig = _plot_number_distribution(
        info_dict["token_num_list"], "Token Number Distribution", "Token Number"
    )
    token_fig.savefig(output_dir / "token_num_distribution.png")

    for k, v in info_dict["chain_interface_to_token_num_list"].items():
        fig = _plot_number_distribution(
            v, f"{k} Token Number Distribution", "Token Number"
        )
        fig.savefig(output_dir / f"{k}_token_num_distribution.png")


def run_data_analysis(
    pdb_meta_info_csv: Path,
    lowh_csv: Path,
    cluster_csv: Path,
    output_dir: Path,
    after_date: str | None = None,
    before_date: str | None = None,
):
    """
    Run data analysis for the dataset.

    Args:
        pdb_meta_info_csv (Path): Path to the PDB meta info CSV file.
        lowh_csv (Path): Path to the low-homology CSV file.
        cluster_csv (Path): Path to the cluster CSV file.
        output_dir (Path): Path to the output directory.
        after_date (str | None): Date after which the PDB meta info is valid.
    """
    pdb_meta_df = pd.read_csv(
        pdb_meta_info_csv,
        dtype={"entry_id": str},
    )
    lowh_df = pd.read_csv(
        lowh_csv,
        dtype={"entry_id": str, "entity_id_1": str, "entity_id_2": str},
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    pdb_num_stat = stat_filtered_num(
        pdb_meta_info_csv=pdb_meta_info_csv,
        after_date=after_date,
        before_date=before_date,
    )
    stat_filtered_info_str = _log_filtered_num_stat(pdb_num_stat)

    info_dict = stat_lowh_num(pdb_meta_df, lowh_df, cluster_csv=cluster_csv)
    lowh_stat_info_str = _log_lowh_num(info_dict)

    stat_info_str = (
        f"<DATA FILTERING PIPELINE STATISTICS>\n{stat_filtered_info_str}\n"
        f"\n<LOW-HOMOLOGY SUBSET STATISTICS>\n{lowh_stat_info_str}"
    )
    output_stat_file = output_dir / "stat.txt"
    with open(output_stat_file, "w") as f:
        f.write(stat_info_str)
    logging.info(stat_info_str)
    logging.info("Save Stat Info to %s", output_stat_file)

    fig_output_dir = output_dir / "figs"
    _draw_token_num_plot(info_dict, fig_output_dir)
    logging.info("Save Token Number Distribution figs to %s", fig_output_dir)

    # Save PDB IDs
    pdb_ids_dir = output_dir / "pdb_ids"
    subset_pdb_ids_dir = pdb_ids_dir / "subset"
    pdb_ids_dir.mkdir(parents=True, exist_ok=True)
    subset_pdb_ids_dir.mkdir(parents=True, exist_ok=True)
    for k, v in info_dict["eval_type_to_pdb_ids"].items():
        if k.startswith("["):
            # Subset
            k = k.strip("[").strip("]")
            output_file = subset_pdb_ids_dir / f"{k}.txt"
        else:
            output_file = pdb_ids_dir / f"{k}.txt"
        with open(output_file, "w") as f:
            for pdb_id in v:
                f.write(f"{pdb_id}\n")
    logging.info("Save PDB IDs to %s", pdb_ids_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=Path, default=(SUPPORT_DATA_DIR / "stat_data")
    )
    parser.add_argument(
        "--pdb_meta_info_csv", type=Path, default=SRC_DATA.pdb_meta_info
    )
    parser.add_argument(
        "--lowh_csv", type=Path, default=SUPPORTED_DATA.recentpdb_low_homology
    )
    parser.add_argument(
        "--cluster_csv",
        type=Path,
        default=SUPPORTED_DATA.recentpdb_low_homology_cluster,
    )
    parser.add_argument("-a", "--after_date", type=str, default=None)
    parser.add_argument("-b", "--before_date", type=str, default=None)
    args = parser.parse_args()
    run_data_analysis(
        pdb_meta_info_csv=args.pdb_meta_info_csv,
        lowh_csv=args.lowh_csv,
        cluster_csv=args.cluster_csv,
        output_dir=args.output_dir,
        after_date=args.after_date,
        before_date=args.before_date,
    )
