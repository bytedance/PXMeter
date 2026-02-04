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
from pxmeter.constants import DNA, LIGAND, PROTEIN, RNA

from benchmark.configs.data_config import PXM_MMCIF_DIR, SRC_DATA, SUPPORTED_DATA
from benchmark.dataset_pipeline.utils.select_ligand import (
    filter_lig_in_chain_interface_df,
)


def _check_for_lowh(row: pd.Series, test_to_train: dict[str, list[str]]) -> bool:
    if row["type"] == "chain":
        return not test_to_train.get(f'{row["entry_id"]}_{row["entity_id_1"]}', [])
    else:
        # interface
        train_pdb_entities_1 = test_to_train.get(
            f'{row["entry_id"]}_{row["entity_id_1"]}', []
        )
        train_pdb_entities_2 = test_to_train.get(
            f'{row["entry_id"]}_{row["entity_id_2"]}', []
        )

        train_pdb_ids_1 = set([i.split("_")[0] for i in train_pdb_entities_1])
        train_pdb_ids_2 = set([i.split("_")[0] for i in train_pdb_entities_2])
        return not (train_pdb_ids_1 & train_pdb_ids_2)


def _is_short_chain(seq_length: int, entity_type: str) -> bool:
    if entity_type in [PROTEIN, DNA, RNA]:
        return seq_length < 25
    else:
        raise ValueError(f"Unknown entity type: {entity_type}")


def _filter_short_polymer(
    row: pd.Series,
    test_to_train: dict[str, list[str]],
    polymer_seq_length_thres: int = 25,
) -> bool:
    if row["type"] == "chain":
        # at least N res for protein / nuc chain
        return row["seq_length_1"] >= polymer_seq_length_thres

    else:
        # interface
        chain_1_is_short = _is_short_chain(row["seq_length_1"], row["entity_type_1"])
        chain_2_is_short = _is_short_chain(row["seq_length_2"], row["entity_type_2"])

        if not chain_1_is_short and not chain_2_is_short:
            return True
        elif chain_1_is_short and chain_2_is_short:
            return False
        elif chain_2_is_short:
            return not test_to_train.get(f'{row["entry_id"]}_{row["entity_id_1"]}', [])
        else:
            # chain 1 is short
            return not test_to_train.get(f'{row["entry_id"]}_{row["entity_id_2"]}', [])


def _filter_lowh_protein(
    df: pd.DataFrame, test_to_train: dict[str, list[str]]
) -> pd.DataFrame:
    sub_df = df[
        (df["entity_type_1"] == PROTEIN)
        & ((df["type"] == "chain") | (df["entity_type_2"] == PROTEIN))
    ]

    lowh_mask = sub_df.apply(lambda row: _check_for_lowh(row, test_to_train), axis=1)
    lowh_df = sub_df[lowh_mask]

    # Remove peptide chain, peptide-peptide interface
    # Remove peptide-protein interface if the protein chain is not lowh
    non_lowh_peptide_mask = lowh_df.apply(
        lambda row: _filter_short_polymer(row, test_to_train), axis=1
    )
    lowh_protein_df = lowh_df[non_lowh_peptide_mask]
    return lowh_protein_df


def _filter_lowh_nuc(
    df: pd.DataFrame, test_to_train: dict[str, list[str]]
) -> pd.DataFrame:
    nuc_chain_mask = (df["type"] == "chain") & df["entity_type_1"].isin([DNA, RNA])
    nuc_protein_interface_mask = (df["type"] == "interface") & (
        (df["entity_type_1"].isin([DNA, RNA]) & (df["entity_type_2"] == PROTEIN))
        | (df["entity_type_2"].isin([DNA, RNA]) & (df["entity_type_1"] == PROTEIN))
    )
    nuc_nuc_mask = (df["type"] == "interface") & (
        (df["entity_type_1"].isin([DNA, RNA]) & (df["entity_type_2"].isin([DNA, RNA])))
    )
    sub_df = df[nuc_chain_mask | nuc_protein_interface_mask | nuc_nuc_mask]
    lowh_mask = sub_df.apply(lambda row: _check_for_lowh(row, test_to_train), axis=1)
    lowh_df = sub_df[lowh_mask]

    # Remove short chain, short-short interface
    # Remove short-polymer interface if the polymer chain is not lowh
    non_lowh_short_mask = lowh_df.apply(
        lambda row: _filter_short_polymer(row, test_to_train), axis=1
    )
    lowh_nuc_df = lowh_df[non_lowh_short_mask]
    return lowh_nuc_df


def _filter_lowh_lig(
    df: pd.DataFrame,
    test_to_train: dict[str, list[str]],
    polymer_seq_length_thres: int = 25,
) -> pd.DataFrame:

    lig_chain_mask = (df["type"] == "chain") & (df["entity_type_1"] == LIGAND)
    lig_polymer_mask = (df["type"] == "interface") & (
        (
            df["entity_type_1"].isin([PROTEIN, DNA, RNA])
            & (df["seq_length_1"] >= polymer_seq_length_thres)
            & (df["entity_type_2"] == LIGAND)
        )
        | (
            df["entity_type_2"].isin([PROTEIN, DNA, RNA])
            & (df["seq_length_2"] >= polymer_seq_length_thres)
            & (df["entity_type_1"] == LIGAND)
        )
    )
    sub_df = df[lig_chain_mask | lig_polymer_mask]
    lowh_mask = sub_df.apply(lambda row: _check_for_lowh(row, test_to_train), axis=1)
    lowh_lig_df = sub_df[lowh_mask]
    return lowh_lig_df


def make_lig_info_df(lowh_df: pd.DataFrame) -> pd.DataFrame:
    """
    Make a DataFrame containing PDB ID and ligand asym ID.

    Args:
        lowh_df (pd.DataFrame): DataFrame containing low homology data.

    Returns:
        pd.DataFrame: DataFrame with columns "entry_id" and "label_asym_id".
    """
    lig_info_df1 = lowh_df[lowh_df["entity_type_1"] == LIGAND][
        ["entry_id", "chain_id_1"]
    ].drop_duplicates()

    lig_info_df2 = lowh_df[lowh_df["entity_type_2"] == LIGAND][
        ["entry_id", "chain_id_2"]
    ].drop_duplicates()

    lig_info_df1 = lig_info_df1.rename(columns={"chain_id_1": "label_asym_id"})
    lig_info_df2 = lig_info_df2.rename(columns={"chain_id_2": "label_asym_id"})

    lig_info_df = pd.concat([lig_info_df1, lig_info_df2], axis=0).drop_duplicates()
    return lig_info_df


def filter_lowh(
    chain_interface_csv: Path,
    test_to_train_entity_homo_parquet: Path,
    output_lowh_csv: Path,
    mmcif_dir: Path,
    n_cpu: int = -1,
):
    """
    Filter the chain interface data to identify low homology (lowh) entries.

    Args:
        chain_interface_csv (Path): Path to the CSV file containing chain interface data.
        test_to_train_entity_homo_parquet (Path): Path to the Parquet file mapping test entities
                                               to training entities.
        output_lowh_csv (Path): Path to save the filtered low homology data as a CSV file.
        mmcif_dir (Path): Directory containing mmCIF files named as <pdbid>.cif
                          (lowercase). These files are used to inspect symmetry-mate contacts.
        n_cpu (int, optional): Number of CPUs for parallel processing when checking
                               symmetry-mate contacts. Defaults to -1 (use all available CPUs).
    """
    df = pd.read_csv(
        chain_interface_csv,
        dtype={"entry_id": str, "entity_id_1": str, "entity_id_2": str},
        keep_default_na=False,
    )

    # column: query_id, db_id, similarity, aligned_res_num
    test_to_train_df = pd.read_parquet(test_to_train_entity_homo_parquet)

    # {query_id: [db_id1, db_id2, ...]}
    test_to_train = (
        test_to_train_df.groupby("query_id", observed=True)["db_id"]
        .apply(list)
        .to_dict()
    )

    lowh_protein_df = _filter_lowh_protein(df, test_to_train)

    lowh_nuc_df = _filter_lowh_nuc(df, test_to_train)

    lowh_lig_df = _filter_lowh_lig(df, test_to_train)

    merged_df = pd.concat([lowh_protein_df, lowh_nuc_df, lowh_lig_df])

    merged_df_filter_for_lig = filter_lig_in_chain_interface_df(
        merged_df, mmcif_dir, n_cpu=n_cpu
    )

    output_lowh_csv.parent.mkdir(parents=True, exist_ok=True)
    merged_df_filter_for_lig.to_csv(
        output_lowh_csv,
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
    )

    lig_info_df = make_lig_info_df(merged_df_filter_for_lig)
    SUPPORTED_DATA.lig_info_csv.parent.mkdir(parents=True, exist_ok=True)
    lig_info_df.to_csv(
        SUPPORTED_DATA.lig_info_csv,
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-c",
        "--chain_interface_csv",
        type=Path,
        default=Path(SRC_DATA.recentpdb_chain_interface_csv),
    )
    argparser.add_argument(
        "-t",
        "--test_to_train_parquet",
        type=Path,
        default=Path(SRC_DATA.test_to_train_entity_homo_parquet),
    )
    argparser.add_argument(
        "-o",
        "--output_lowh_csv",
        type=Path,
        default=Path(SUPPORTED_DATA.recentpdb_low_homology),
    )
    argparser.add_argument(
        "-m",
        "--mmcif_dir",
        type=Path,
        required=True,
        default=PXM_MMCIF_DIR,
    )
    argparser.add_argument(
        "-n",
        "--n_cpu",
        type=int,
        default=-1,
    )

    args = argparser.parse_args()

    filter_lowh(
        chain_interface_csv=args.chain_interface_csv,
        test_to_train_entity_homo_parquet=args.test_to_train_parquet,
        output_lowh_csv=args.output_lowh_csv,
        mmcif_dir=args.mmcif_dir,
        n_cpu=args.n_cpu,
    )
