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
from pathlib import Path

import pandas as pd
from pxmeter.constants import DNA, PROTEIN, RNA

from benchmark.configs.data_config import SRC_DATA, SUPPORTED_DATA
from benchmark.dataset_pipeline.step2_make_lowh_file import calc_mmseqs_seq_identity
from benchmark.utils import shrink_dataframe


def get_lowh_sequences(
    seq_df: pd.DataFrame,
    lowh_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Get low homology sequences from seq_df.

    Args:
        seq_df: Sequence dataframe.
        lowh_df: Low homology dataframe.

    Returns:
        Low homology sequence dataframe.
    """
    e_id_1 = lowh_df["entry_id"] + "_" + lowh_df["entity_id_1"]
    is_interface = lowh_df["type"] == "interface"
    e_id_2 = (
        lowh_df[is_interface]["entry_id"] + "_" + lowh_df[is_interface]["entity_id_2"]
    )
    all_lowh_keys = set(e_id_1) | set(e_id_2)

    seq_keys = seq_df["entry_id"] + "_" + seq_df["entity_id"]
    lowh_seq_df = seq_df[seq_keys.isin(all_lowh_keys)]
    return lowh_seq_df


def get_homology_for_lowh(
    lowh_csv: Path,
    seqs_csv: Path,
    after_date: str,
    lowh_homo_parquet: Path,
):
    """
    Get homology for low homology sequences.

    Args:
        lowh_csv: Low homology csv file.
        seqs_csv: Sequence csv file.
        after_date: Date after which the sequences are released.
        lowh_homo_parquet: Output path of low homology homology Parquet file.
    """
    seq_df = pd.read_csv(seqs_csv, dtype={"entry_id": str, "entity_id": str})
    train_df = seq_df[(seq_df["release_date"] < after_date)]
    lowh_df = pd.read_csv(
        lowh_csv, dtype={"entry_id": str, "entity_id_1": str, "entity_id_2": str}
    )
    lowh_seq_df = get_lowh_sequences(seq_df, lowh_df)

    protein_test_vs_train = calc_mmseqs_seq_identity(
        db_df=train_df[train_df["entity_type"] == PROTEIN],
        query_df=lowh_seq_df[lowh_seq_df["entity_type"] == PROTEIN],
        min_seq_length=25,
        threshold=0.3,
        coverage=0.0,
        cov_mode=0,
        e_value_cutoff=0.1,
        sensitivity=7.5,
        max_seqs=500000,
        nuc=False,
    )

    rna_test_vs_train = calc_mmseqs_seq_identity(
        db_df=train_df[train_df["entity_type"] == RNA],
        query_df=lowh_seq_df[lowh_seq_df["entity_type"] == RNA],
        min_seq_length=25,
        threshold=0.3,
        coverage=0.0,
        cov_mode=0,
        e_value_cutoff=0.1,
        sensitivity=7.5,
        max_seqs=500000,
        nuc=True,
    )

    dna_test_vs_train = calc_mmseqs_seq_identity(
        db_df=train_df[train_df["entity_type"] == DNA],
        query_df=lowh_seq_df[lowh_seq_df["entity_type"] == DNA],
        min_seq_length=25,
        threshold=0.3,
        coverage=0.0,
        cov_mode=0,
        e_value_cutoff=0.1,
        sensitivity=7.5,
        max_seqs=500000,
        nuc=True,
    )

    merged_df = pd.concat(
        [
            protein_test_vs_train,
            rna_test_vs_train,
            dna_test_vs_train,
        ]
    )

    merged_df, _report = shrink_dataframe(merged_df)
    lowh_homo_parquet.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(
        lowh_homo_parquet,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-l", "--lowh_csv", type=Path, default=SUPPORTED_DATA.recentpdb_low_homology
    )
    args.add_argument("-a", "--after_date", type=str, required=True, help="YYYY-MM-DD")
    args.add_argument(
        "-o",
        "--lowh_homo_parquet",
        type=Path,
        default=SUPPORTED_DATA.recentpdb_low_homology_entity_homo_parquet,
    )
    args.add_argument("-s", "--seqs_csv", type=Path, default=SRC_DATA.pdb_seq_csv)
    args = args.parse_args()

    get_homology_for_lowh(
        lowh_csv=args.lowh_csv,
        seqs_csv=args.seqs_csv,
        after_date=args.after_date,
        lowh_homo_parquet=args.lowh_homo_parquet,
    )
