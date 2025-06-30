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
]


EVAL_TYPE_MAP = {
    "Intra-Protein": "intra_prot",
    "Intra-RNA": "intra_rna",
    "Intra-DNA": "intra_dna",
    "Intra-Ligand": "intra_lig",
    "Protein-Protein (Antibody=True)": "prot_prot (ab=True)",
    "Protein-Protein (Antibody=False)": "prot_prot (ab=False)",
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
    "Intra-Protein (Monomer)": "intra_prot (monomer)",
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
    for eval_info, eval_type_df in df.groupby(["eval_dataset", "eval_type"]):
        eval_dataset, eval_type = eval_info

        eval_type_df = eval_type_df[eval_type_df["ranker"].isin(KEPT_RANKER)]
        sub_eval_type_df = eval_type_df[["name", "ranker", "avg_dockq_sr_avg_sr"]]

        if eval_dataset == "RecentPDB":
            new_col_name = f"{EVAL_TYPE_MAP.get(eval_type, eval_type)} DockQ SR"
        else:
            new_col_name = (
                f"[{eval_dataset}]{EVAL_TYPE_MAP.get(eval_type, eval_type)} DockQ SR"
            )

        new_sub_eval_type_df = sub_eval_type_df.rename(
            columns={"avg_dockq_sr_avg_sr": new_col_name}
        )
        num_cols[new_col_name] = (
            f'{eval_type_df["entry_id_num"].iloc[0]}/{eval_type_df["cluster_num"].iloc[0]}'
        )
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
    for eval_info, eval_type_df in df.groupby(["eval_dataset", "eval_type"]):
        eval_dataset, eval_type = eval_info

        eval_type_df = eval_type_df[eval_type_df["ranker"].isin(KEPT_RANKER)]
        sub_eval_type_df = eval_type_df[["name", "ranker", "lddt"]]

        if eval_dataset == "RecentPDB":
            new_col_name = f"{EVAL_TYPE_MAP.get(eval_type, eval_type)}"
        else:
            new_col_name = f"[{eval_dataset}]{EVAL_TYPE_MAP.get(eval_type, eval_type)}"

        new_sub_eval_type_df = sub_eval_type_df.rename(columns={"lddt": new_col_name})
        num_cols[new_col_name] = (
            f'{eval_type_df["entry_id_num"].iloc[0]}/{eval_type_df["cluster_num"].iloc[0]}'
        )
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

    sub_df = df[df["ranker"].isin(KEPT_RANKER)]
    sub_df = sub_df[["name", "ranker", "lig_rmsd_sr"]]
    new_sub_df = sub_df.rename(columns={"lig_rmsd_sr": "PoseBusters SR"})
    new_sub_df = new_sub_df.round(4)
    new_sub_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "name": "entry_id_num/cluster_num",
                    "PoseBusters SR": f"{df['entry_id_num'].iloc[0]}/null",
                },
                index=[0],
            ),
            new_sub_df,
        ]
    )
    return new_sub_df


def reduce_csv_content(
    dockq_csv: Path | str | None = None,
    lddt_csv: Path | str | None = None,
    rmsd_csv: Path | str | None = None,
) -> tuple[pd.DataFrame, str]:
    """
    Reduce the content of DockQ, LDDT, and RMSD CSV files to a DataFrame and a formatted string.

    Args:
        dockq_csv (Path or str): The path to the DockQ CSV file.
        lddt_csv (Path or str): The path to the LDDT CSV file.
        rmsd_csv (Path or str): The path to the RMSD CSV file.

    Returns:
        tuple[pd.DataFrame, str]: A tuple containing the reduced DataFrame and a formatted string.
    """
    assert not (
        dockq_csv is None and lddt_csv is None and rmsd_csv is None
    ), "At least one of dockq_csv, lddt_csv, or rmsd_csv must be provided."

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

    total_df = df_list[0]
    for other_df in df_list[1:]:
        total_df = total_df.merge(other_df, on=["name", "ranker"], how="outer")

    rows_with_num = total_df[total_df["name"] == "entry_id_num/cluster_num"]
    other_rows = total_df[total_df["name"] != "entry_id_num/cluster_num"]
    df_reordered = pd.concat([rows_with_num, other_rows], ignore_index=True)
    columns_to_move = ["name", "ranker"]

    new_order = columns_to_move + [
        col for col in df_reordered.columns if col not in columns_to_move
    ]
    df_reordered = df_reordered[new_order]

    df_reordered = df_reordered.fillna("")

    table_str = tabulate(
        df_reordered, headers="keys", tablefmt="simple_grid", showindex=False
    )
    return df_reordered, table_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dockq_csv", type=Path, default=None)
    parser.add_argument("-l", "--lddt_csv", type=Path, default=None)
    parser.add_argument("-r", "--rmsd_csv", type=Path, default=None)
    parser.add_argument("-o", "--output_path", type=Path, default=".")
    parser.add_argument("-n", "--out_file_name", type=str, default="Summary_table")
    args = parser.parse_args()

    table_df, table_str = reduce_csv_content(
        args.dockq_csv, args.lddt_csv, args.rmsd_csv
    )

    args.output_path.mkdir(parents=True, exist_ok=True)
    table_df.to_csv(
        args.output_path / f"{args.out_file_name}.csv",
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
    )

    # Summary to a string of table
    with open(args.output_path / f"{args.out_file_name}.txt", "w") as f:
        f.write(table_str)
