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
from pathlib import Path

import pandas as pd

from benchmark.configs.data_config import SRC_DATA, SUPPORTED_DATA
from pxmeter.constants import DNA, PROTEIN, RNA


def _check_for_lowh(row: pd.Series, test_to_train: dict[str, list[str]]) -> bool:
    if row["type"] == "chain":
        return not (test_to_train[f'{row["entry_id"]}_{row["entity_id_1"]}'])
    else:
        # interface
        train_pdb_entities_1 = test_to_train[f'{row["entry_id"]}_{row["entity_id_1"]}']
        train_pdb_entities_2 = test_to_train[f'{row["entry_id"]}_{row["entity_id_2"]}']
        train_pdb_ids_1 = set([i.split("_")[0] for i in train_pdb_entities_1])
        train_pdb_ids_2 = set([i.split("_")[0] for i in train_pdb_entities_2])
        return not (train_pdb_ids_1 & train_pdb_ids_2)


def _is_short_chain(seq_length: int, entity_type: str) -> bool:
    if entity_type == PROTEIN:
        return seq_length < 16
    elif entity_type in [DNA, RNA]:
        return seq_length < 10
    else:
        raise ValueError(f"Unknown entity type: {entity_type}")


def _filter_short(row: pd.Series, test_to_train: dict[str, list[str]]) -> bool:
    if row["type"] == "chain":
        if row["entity_type_1"] in [DNA, RNA]:
            # at least 10 res for nuc chain
            return row["seq_length_1"] >= 10
        else:
            # at least 16 res for protein chain
            return row["seq_length_1"] >= 16

    else:
        # interface
        chain_1_is_short = _is_short_chain(row["seq_length_1"], row["entity_type_1"])
        chain_2_is_short = _is_short_chain(row["seq_length_2"], row["entity_type_2"])

        if not chain_1_is_short and not chain_2_is_short:
            return True
        elif chain_1_is_short and chain_2_is_short:
            return False
        elif chain_2_is_short:
            return not (test_to_train[f'{row["entry_id"]}_{row["entity_id_1"]}'])
        else:
            # chain 1 is short
            return not (test_to_train[f'{row["entry_id"]}_{row["entity_id_2"]}'])


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
        lambda row: _filter_short(row, test_to_train), axis=1
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
        lambda row: _filter_short(row, test_to_train), axis=1
    )
    lowh_nuc_df = lowh_df[non_lowh_short_mask]
    return lowh_nuc_df


def filter_lowh(
    chain_interface_csv: Path,
    test_to_train_entity_homo_json: Path,
    output_protein_lowh_csv: Path,
    output_nuc_lowh_csv: Path,
):
    """
    Filter the chain interface data to identify low homology (lowh) protein and nucleic acid entries.

    Args:
        chain_interface_csv (Path): Path to the CSV file containing chain interface data.
        test_to_train_entity_homo_json (Path): Path to the JSON file mapping test entities to training entities.
        output_protein_lowh_csv (Path): Path to save the filtered low homology protein data as a CSV file.
        output_nuc_lowh_csv (Path): Path to save the filtered low homology nucleic acid data as a CSV file.
    """
    df = pd.read_csv(
        chain_interface_csv,
        dtype={"entry_id": str, "entity_id_1": str, "entity_id_2": str},
    )

    with open(test_to_train_entity_homo_json) as f:
        test_to_train = json.load(f)

    lowh_protein_df = _filter_lowh_protein(df, test_to_train)

    lowh_nuc_df = _filter_lowh_nuc(df, test_to_train)
    lowh_protein_df.to_csv(
        output_protein_lowh_csv,
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
    )
    lowh_nuc_df.to_csv(
        output_nuc_lowh_csv,
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
        "--test_to_train_json",
        type=Path,
        default=Path(SRC_DATA.test_to_train_entity_homo_json),
    )
    argparser.add_argument(
        "-o",
        "--output_protein_lowh_csv",
        type=Path,
        default=Path(SUPPORTED_DATA.recentpdb_low_homology),
    )
    argparser.add_argument(
        "-n",
        "--output_nuc_lowh_csv",
        type=Path,
        default=Path(SUPPORTED_DATA.recentpdb_na_low_homology),
    )
    args = argparser.parse_args()

    filter_lowh(
        chain_interface_csv=args.chain_interface_csv,
        test_to_train_entity_homo_json=args.test_to_train_json,
        output_protein_lowh_csv=args.output_protein_lowh_csv,
        output_nuc_lowh_csv=args.output_nuc_lowh_csv,
    )
