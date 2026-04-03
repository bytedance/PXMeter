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
from pathlib import Path

import pandas as pd
from tabulate import tabulate

KEPT_RANKER = [
    "best",
    "median",
    # protenix
    "best.ranking_score",
    "best.chain_pair_iptm",
    "best.chain_ptm",
    # af2
    "best.plddt",
    # af2m
    "best.iptm+ptm",
    # chai
    "best.aggregate_score",
    "best.per_chain_ptm",
    "best.per_chain_pair_iptm",
    # boltz
    "best.chains_ptm",
    "best.confidence_score",
    "best.pair_chains_iptm",
    # openfold3
    "best.bespoke_iptm",
]


EVAL_TYPE_MAP = {
    "Intra-Protein": "intra_prot",
    "Intra-RNA": "intra_rna",
    "Intra-DNA": "intra_dna",
    "Intra-Ligand": "intra_lig",
    "Protein-Protein": "prot_prot",
    "DNA-DNA": "dna_dna",
    "RNA-RNA": "rna_rna",
    "Ligand-Ligand": "lig_lig",
    "Protein-Ligand": "prot_lig",
    "RNA-Protein": "rna_prot",
    "DNA-Protein": "dna_prot",
    "DNA-RNA": "dna_rna",
    "DNA-Ligand": "dna_lig",
    "RNA-Ligand": "rna_lig",
    "LDDT-PLI": "lddt_pli",
}


def reduce_dockq_csv(dockq_csv: Path | str) -> pd.DataFrame:
    """
    Reduce the DockQ CSV file to a DataFrame with selected columns and renamed columns.

    Args:
        dockq_csv (Path or str): The path to the DockQ CSV file.

    Returns:
        pd.DataFrame: The reduced DataFrame with selected columns and renamed columns.
    """
    df = pd.read_csv(dockq_csv)

    df_list = []
    num_cols = {"name": "entry_id_num/cluster_num"}
    for eval_info, eval_type_df in df.groupby(["eval_dataset", "eval_type", "subset"]):
        eval_dataset, eval_type, subset = eval_info

        eval_type_df = eval_type_df[eval_type_df["ranker"].isin(KEPT_RANKER)]
        sub_eval_type_df = eval_type_df[["name", "ranker", "avg_dockq_sr_avg_sr"]]

        if eval_dataset == "RecentPDB":
            new_col_name = f"{EVAL_TYPE_MAP.get(eval_type, eval_type)} DockQ SR"
        else:
            new_col_name = (
                f"[{eval_dataset}]{EVAL_TYPE_MAP.get(eval_type, eval_type)} DockQ SR"
            )

        if subset != "All":
            new_col_name += f" ({subset})"

        new_sub_eval_type_df = sub_eval_type_df.rename(
            columns={"avg_dockq_sr_avg_sr": new_col_name}
        )
        num_cols[
            new_col_name
        ] = f'{eval_type_df["entry_id_num"].iloc[0]}/{eval_type_df["cluster_num"].iloc[0]}'
        df_list.append(new_sub_eval_type_df)

    new_df = df_list[0]
    for other_df in df_list[1:]:
        new_df = new_df.merge(other_df, on=["name", "ranker"], how="outer")

    new_df = new_df.round(4)

    new_df = pd.concat([pd.DataFrame(num_cols, index=[0]), new_df])
    return new_df


def reduce_lddt_csv(lddt_csv: Path | str) -> pd.DataFrame:
    """
    Reduce the LDDT CSV file to a DataFrame with selected columns and renamed columns.

    Args:
        lddt_csv (Path or str): The path to the LDDT CSV file.

    Returns:
        pd.DataFrame: The reduced DataFrame with selected columns and renamed columns.
    """
    df = pd.read_csv(lddt_csv)

    df_list = []
    num_cols = {"name": "entry_id_num/cluster_num"}
    for eval_info, eval_type_df in df.groupby(["eval_dataset", "eval_type", "subset"]):
        eval_dataset, eval_type, subset = eval_info

        eval_type_df = eval_type_df[eval_type_df["ranker"].isin(KEPT_RANKER)]
        sub_eval_type_df = eval_type_df[["name", "ranker", "lddt"]]

        if eval_dataset == "RecentPDB":
            new_col_name = f"{EVAL_TYPE_MAP.get(eval_type, eval_type)}"
        else:
            new_col_name = f"[{eval_dataset}]{EVAL_TYPE_MAP.get(eval_type, eval_type)}"

        if subset != "All":
            new_col_name += f" ({subset})"

        new_sub_eval_type_df = sub_eval_type_df.rename(columns={"lddt": new_col_name})
        num_cols[
            new_col_name
        ] = f'{eval_type_df["entry_id_num"].iloc[0]}/{eval_type_df["cluster_num"].iloc[0]}'
        df_list.append(new_sub_eval_type_df)

    new_df = df_list[0]
    for other_df in df_list[1:]:
        new_df = new_df.merge(other_df, on=["name", "ranker"], how="outer")

    new_df = new_df.round(4)
    new_df = pd.concat([pd.DataFrame(num_cols, index=[0]), new_df])
    return new_df


def reduce_rmsd_csv(lddt_csv: Path | str) -> pd.DataFrame:
    """
    Reduce the RMSD CSV file to a DataFrame with selected columns and renamed columns.

    Args:
        lddt_csv (Path or str): The path to the RMSD CSV file.

    Returns:
        pd.DataFrame: The reduced DataFrame with selected columns and renamed columns.
    """
    df = pd.read_csv(lddt_csv)

    sub_df = df[df["ranker"].isin(KEPT_RANKER)].copy()

    metric_rename_map = {
        "lig_rmsd_sr": "Ligand RMSD SR",
        "lig_rmsd_lddt_pli_sr": "Ligand SR",
        "pb_all_valid_sr": "PB Valid SR",
        "pb_all_valid_and_good_rmsd_sr": "PB Valid and Good RMSD SR",
        "cdr_h3_bb_rmsd_sr": "CDR H3 RMSD SR",
    }

    available_metric_cols = [
        col for col in metric_rename_map.keys() if col in sub_df.columns
    ]
    available_metric_rename = {
        col: metric_rename_map[col] for col in available_metric_cols
    }

    base_cols = ["name", "ranker"]
    sub_df = sub_df[base_cols + available_metric_cols].copy()
    new_sub_df = sub_df.rename(columns=available_metric_rename)

    num_cols = new_sub_df.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        new_sub_df[num_cols] = new_sub_df[num_cols].round(4)

    entry_cluster = f"{df['entry_id_num'].iloc[0]}/{df['cluster_num'].iloc[0]}"

    summary_row = {"name": "entry_id_num/cluster_num"}
    for pretty_name in available_metric_rename.values():
        summary_row[pretty_name] = entry_cluster

    header_df = pd.DataFrame([summary_row], columns=new_sub_df.columns)
    new_sub_df = pd.concat([header_df, new_sub_df], ignore_index=True)
    return new_sub_df


def reduce_others_csv(others_csv: Path | str) -> pd.DataFrame:
    """
    Reduce the Others CSV file to a DataFrame with selected columns and renamed columns.

    Args:
        others_csv (Path or str): The path to the Others CSV file.

    Returns:
        pd.DataFrame: The reduced DataFrame with selected columns and renamed columns.
    """
    df = pd.read_csv(others_csv)

    sub_df = df[df["ranker"].isin(KEPT_RANKER)].copy()

    metric_rename_map = {
        "lig_rmsd_lddt_pli_sr": "Ligand SR",
    }

    available_metric_cols = [
        col for col in metric_rename_map.keys() if col in sub_df.columns
    ]
    available_metric_rename = {
        col: metric_rename_map[col] for col in available_metric_cols
    }

    base_cols = ["name", "ranker"]
    sub_df = sub_df[base_cols + available_metric_cols].copy()
    new_sub_df = sub_df.rename(columns=available_metric_rename)

    num_cols = new_sub_df.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        new_sub_df[num_cols] = new_sub_df[num_cols].round(4)

    entry_cluster = f"{df['entry_id_num'].iloc[0]}/{df['cluster_num'].iloc[0]}"

    summary_row = {"name": "entry_id_num/cluster_num"}
    for pretty_name in available_metric_rename.values():
        summary_row[pretty_name] = entry_cluster

    header_df = pd.DataFrame([summary_row], columns=new_sub_df.columns)
    new_sub_df = pd.concat([header_df, new_sub_df], ignore_index=True)
    return new_sub_df


def reduce_csv_content(
    dockq_csv: Path | str | None = None,
    lddt_csv: Path | str | None = None,
    rmsd_csv: Path | str | None = None,
    others_csv: Path | str | None = None,
    cdr_h3_csv: Path | str | None = None,
    order: list[str] | None = None,
) -> tuple[pd.DataFrame, str]:
    """
    Reduce the content of DockQ, LDDT, RMSD and CDR H3 CSV files to a DataFrame and a formatted string.

    Args:
        dockq_csv (Path or str): The path to the DockQ CSV file.
        lddt_csv (Path or str): The path to the LDDT CSV file.
        rmsd_csv (Path or str): The path to the RMSD CSV file.
        cdr_h3_csv (Path or str): The path to the CDR H3 CSV file.
        order (list of str, optional): The order of rankers to be displayed in the DataFrame.
            Defaults to None.

    Returns:
        tuple[pd.DataFrame, str]: A tuple containing the reduced DataFrame and a formatted string.
    """
    assert not (
        dockq_csv is None
        and lddt_csv is None
        and rmsd_csv is None
        and others_csv is None
        and cdr_h3_csv is None
    ), "At least one of dockq_csv, lddt_csv, rmsd_csv, others_csv or cdr_h3_csv must be provided."

    df_list = []
    if dockq_csv is not None and Path(dockq_csv).exists():
        short_dockq_df = reduce_dockq_csv(dockq_csv)
        df_list.append(short_dockq_df)

    if lddt_csv is not None and Path(lddt_csv).exists():
        short_lddt_df = reduce_lddt_csv(lddt_csv)
        df_list.append(short_lddt_df)

    if rmsd_csv is not None and Path(rmsd_csv).exists():
        short_rmsd_df = reduce_rmsd_csv(rmsd_csv)
        df_list.append(short_rmsd_df)

    if others_csv is not None and Path(others_csv).exists():
        short_others_df = reduce_others_csv(others_csv)
        df_list.append(short_others_df)

    if cdr_h3_csv is not None and Path(cdr_h3_csv).exists():
        short_cdr_df = reduce_rmsd_csv(cdr_h3_csv)
        df_list.append(short_cdr_df)

    total_df = df_list[0]
    for other_df in df_list[1:]:
        total_df = total_df.merge(other_df, on=["name", "ranker"], how="outer")

    rows_with_num = total_df[total_df["name"] == "entry_id_num/cluster_num"]
    other_rows = total_df[total_df["name"] != "entry_id_num/cluster_num"]

    if order is not None:
        rank = {v: i for i, v in enumerate(order)}
        tmp = other_rows.assign(
            _key=other_rows["name"].map(rank).fillna(len(order)).astype(int)
        )
        other_rows = tmp.sort_values(["_key", "name"], kind="mergesort").drop(
            columns="_key"
        )

    df_reordered = pd.concat([rows_with_num, other_rows], ignore_index=True)

    preferred_columns = [
        "name",
        "num_token",
        "ranker",
        "prot_prot DockQ SR",
        "prot_prot DockQ SR (Antibody=False)",
        "prot_prot DockQ SR (Antibody=True)",
        "CDR H3 RMSD SR",
        "Ligand SR",
        "dna_dna",
        "dna_lig",
        "dna_prot",
        "dna_rna",
        "intra_dna",
        "intra_lig",
        "intra_prot",
        "intra_prot (Monomer)",
        "intra_rna",
        "prot_lig",
        "lddt_pli",
        "prot_prot",
        "prot_prot (Antibody=False)",
        "prot_prot (Antibody=True)",
        "prot_prot (Peptide)",
        "rna_lig",
        "rna_prot",
        "rna_rna",
    ]

    existing_cols = list(df_reordered.columns)
    priority_cols = [c for c in preferred_columns if c in existing_cols]
    remaining_cols = [c for c in existing_cols if c not in priority_cols]
    new_order = priority_cols + remaining_cols
    df_reordered = df_reordered[new_order]

    df_reordered = df_reordered.fillna("")

    table_str = tabulate(
        df_reordered, headers="keys", tablefmt="simple_grid", showindex=False
    )
    return df_reordered, table_str


def rank_results_df(result_df: pd.DataFrame, metrics_col: str) -> pd.DataFrame:
    """
    Rank evaluation results within grouped subsets of a DataFrame.

    Groups results by evaluation metadata (dataset, subset, type, ranker), sorts
    models by a specified metric in descending order, and generates a ranked
    string list of models with their scores.

    Args:
        result_df (pd.DataFrame): Input DataFrame containing evaluation results.
                    Must include at least the following columns:
                    - "eval_dataset"
                    - "eval_type"
                    - "ranker"
                    - "name"
                    - "entry_id_num"
                    - "cluster_num"
                    - metrics_col (specified)
                    Optionally "subset". If absent, a default value "All" is added.
        metrics_col (str): Name of the column containing metric values to rank by.

    Returns:
        pd.DataFrame: A DataFrame with the same grouping columns plus a "rank"
        column, where each entry is a string of ranked models in the format
        "model_name (metric_value) > ...".
    """
    if "subset" not in result_df.columns:
        result_df["subset"] = "All"

    group_cols = ["eval_dataset"]
    for i in ["subset", "eval_type", "ranker"]:
        if i in result_df.columns:
            group_cols.append(i)

    results = []
    for group, sub_df in result_df.groupby(group_cols):
        group_list = list(group)
        sorted_sub_df = sub_df.dropna(subset=[metrics_col]).sort_values(
            by=metrics_col, ascending=False
        )
        if sorted_sub_df.empty:
            continue
        entry_id_num = sub_df["entry_id_num"].iloc[0]
        cluster_id_num = sub_df["cluster_num"].iloc[0]

        rank_list = []
        for _, row in sorted_sub_df.iterrows():
            row_str = f"{row['name']} ({round(row[metrics_col], 4)})"
            rank_list.append(row_str)
        results.append(
            group_list + [entry_id_num, cluster_id_num] + [" > ".join(rank_list)]
        )

    result_rank_df = pd.DataFrame(
        results, columns=group_cols + ["entry_id_num", "cluster_num", "rank"]
    )
    return result_rank_df


def get_ranked_results(
    dockq_csv: Path | str | None = None,
    lddt_csv: Path | str | None = None,
    rmsd_csv: Path | str | None = None,
    others_csv: Path | str | None = None,
    cdr_h3_csv: Path | str | None = None,
    output_csv: Path | str | None = None,
):
    """
    Generate ranked evaluation results from DockQ and/or LDDT CSV files.

    Reads DockQ and LDDT evaluation CSVs if provided, computes ranking tables
    using rank_results_df, and combines them into a single DataFrame. The
    resulting DataFrame is sorted and written to a CSV file.

    Args:
        dockq_csv (Path | str | None): Path to a DockQ evaluation CSV file. If
        None or file does not exist, it is skipped.
        lddt_csv (Path | str | None): Path to an LDDT evaluation CSV file. If
        None or file does not exist, it is skipped.
        rmsd_csv (Path | str | None): Path to an RMSD evaluation CSV file. If
        None or file does not exist, it is skipped.
        cdr_h3_csv (Path | str | None): Path to a CDR H3 evaluation CSV file. If
        None or file does not exist, it is skipped.
        output_csv (Path | str | None): Path to the output CSV file containing
        ranked results. Must be provided if ranking results are to be saved.
    """
    df_list = []
    if dockq_csv is not None and Path(dockq_csv).exists():
        dockq_df = pd.read_csv(dockq_csv)
        ranked_dockq_df = rank_results_df(dockq_df, metrics_col="avg_dockq_sr_avg_sr")
        ranked_dockq_df.insert(0, "metric", "DockQ_SR")
        df_list.append(ranked_dockq_df)

    if lddt_csv is not None and Path(lddt_csv).exists():
        lddt_df = pd.read_csv(lddt_csv)
        ranked_lddt_df = rank_results_df(lddt_df, metrics_col="lddt")
        ranked_lddt_df.insert(0, "metric", "LDDT")
        df_list.append(ranked_lddt_df)

    if rmsd_csv is not None and Path(rmsd_csv).exists():
        rmsd_df = pd.read_csv(rmsd_csv)

        for metrics_col, metric_name in [
            ("lig_rmsd_sr", "Ligand_RMSD_SR"),
            ("pb_all_valid_sr", "PB_Valid_SR"),
            ("pb_all_valid_and_good_rmsd_sr", "PB_Valid_and_Good_RMSD_SR"),
            ("cdr_h3_bb_rmsd_sr", "CDR_H3_RMSD_SR"),
        ]:
            if metrics_col not in rmsd_df.columns:
                continue
            ranked_rmsd_df = rank_results_df(rmsd_df, metrics_col=metrics_col)
            ranked_rmsd_df.insert(0, "metric", metric_name)
            df_list.append(ranked_rmsd_df)

    if others_csv is not None and Path(others_csv).exists():
        others_df = pd.read_csv(others_csv)

        for metrics_col, metric_name in [
            ("lig_rmsd_lddt_pli_sr", "Ligand_SR"),
        ]:
            if metrics_col not in others_df.columns:
                continue
            ranked_others_df = rank_results_df(others_df, metrics_col=metrics_col)
            ranked_others_df.insert(0, "metric", metric_name)
            df_list.append(ranked_others_df)

    if cdr_h3_csv is not None and Path(cdr_h3_csv).exists():
        cdr_df = pd.read_csv(cdr_h3_csv)
        if "cdr_h3_bb_rmsd_sr" in cdr_df.columns:
            ranked_cdr_df = rank_results_df(cdr_df, metrics_col="cdr_h3_bb_rmsd_sr")
            ranked_cdr_df.insert(0, "metric", "CDR_H3_RMSD_SR")
            df_list.append(ranked_cdr_df)

    output_df = pd.concat(df_list)
    sorted_keys = [
        i
        for i in ["metric", "ranker", "eval_dataset", "subset", "eval_type"]
        if i in output_df.columns
    ]
    output_df.sort_values(by=sorted_keys, inplace=True)
    output_df.to_csv(output_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)


def run_reduce(
    output_summary_csv: Path,
    output_ranked_csv: Path,
    dockq_csv: Path | None = None,
    lddt_csv: Path | None = None,
    rmsd_csv: Path | None = None,
    others_csv: Path | None = None,
    cdr_h3_csv: Path | None = None,
    order: list[str] | None = None,
):
    """
    Aggregate and summarize evaluation results, then produce ranked tables.

    Reduces multiple evaluation result CSVs (DockQ, LDDT, RMSD, CDR H3) into a summary
    table and corresponding ranked results. Writes both CSV and text summary
    files, as well as ranked results in CSV format.

    Args:
        output_summary_csv (Path): Path where the reduced summary CSV file will
        be saved.
        output_ranked_csv (Path): Path where the ranked results CSV file will
        be saved.
        dockq_csv (Path | None): Optional path to DockQ evaluation CSV file.
        lddt_csv (Path | None): Optional path to LDDT evaluation CSV file.
        rmsd_csv (Path | None): Optional path to RMSD evaluation CSV file.
        cdr_h3_csv (Path | None): Optional path to CDR H3 evaluation CSV file.
        order (list[str] | None): Optional list of dataset names to order the
            summary table. If None, default ordering is used.
    """
    table_df, table_str = reduce_csv_content(
        dockq_csv, lddt_csv, rmsd_csv, others_csv, cdr_h3_csv, order=order
    )

    output_summary_csv.parent.mkdir(exist_ok=True, parents=True)
    table_df.to_csv(
        output_summary_csv,
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
    )

    # Summary to a string of table
    with open(output_summary_csv.with_suffix(".txt"), "w") as f:
        f.write(table_str)

    output_ranked_csv.parent.mkdir(exist_ok=True, parents=True)
    get_ranked_results(
        dockq_csv,
        lddt_csv,
        rmsd_csv,
        others_csv,
        cdr_h3_csv,
        output_csv=output_ranked_csv,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dockq_csv", type=Path, default=None)
    parser.add_argument("-l", "--lddt_csv", type=Path, default=None)
    parser.add_argument("-r", "--rmsd_csv", type=Path, default=None)
    parser.add_argument("-c", "--others_csv", type=Path, default=None)
    parser.add_argument("-o", "--output_path", type=Path, default=".")
    parser.add_argument("-n", "--out_file_name", type=str, default="Summary_table")
    args = parser.parse_args()

    output_summary_csv_path = args.output_path / f"{args.out_file_name}.csv"
    output_ranked_csv_path = args.output_path / f"{args.out_file_name}_ranked.csv"
    run_reduce(
        output_summary_csv_path,
        output_ranked_csv_path,
        dockq_csv=args.dockq_csv,
        lddt_csv=args.lddt_csv,
        rmsd_csv=args.rmsd_csv,
        others_csv=args.others_csv,
    )
