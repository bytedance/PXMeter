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
import shutil
import subprocess as sp
import tempfile
from pathlib import Path

import pandas as pd
from pxmeter.constants import DNA, LIGAND, PROTEIN, RNA

from benchmark.configs.data_config import SRC_DATA, SRC_DATA_DIR, SUPPORTED_DATA


def cluster_by_seq_identity(
    seq_df: pd.DataFrame,
    threshold: float = 0.4,
    coverage: float = 0.8,
    min_seq_length: int = 10,
) -> pd.DataFrame:
    """
    Cluster sequences based on sequence identity using MMseqs2.

    Args:
        seq_df (pd.DataFrame): DataFrame containing sequence data
                               with columns 'entry_id', 'entity_id', 'seq'.
        threshold (float, optional): Sequence identity threshold for clustering.
                                     Defaults to 0.4.
        coverage (float, optional): Coverage threshold for clustering.
                                    Defaults to 0.8.
        min_seq_length (int, optional): Minimum sequence length for clustering.
                                        Defaults to 10.

    Returns:
        pd.DataFrame: DataFrame with columns 'entry_id', 'entity_id', 'cluster_id'.
    """
    cluster_id_list = []
    fasta_txt = ""
    for _, row in seq_df.iterrows():
        if len(row["seq"]) < min_seq_length:
            cluster_id = row["seq"]
            cluster_id_list.append([row["entry_id"], row["entity_id"], cluster_id])
        else:
            fasta_txt += f">{row['entry_id']}_{row['entity_id']}\n{row['seq']}\n"

    SRC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SRC_DATA_DIR) as tmp_dir:
        tmp_dir = Path(tmp_dir)
        seq_fasta_file = tmp_dir / "seq.fasta"
        seq_fasta_file.write_text(fasta_txt)

        cmd = f"cd {tmp_dir};mmseqs easy-cluster {seq_fasta_file.name} mm"
        cmd += f" mmseqs_tmp --min-seq-id {threshold} -c {coverage} -s 8 --max-seqs 1000 --cluster-mode 1"
        sp.run(cmd, shell=True, check=True)

        mmseq_cluster_df = pd.read_csv(
            tmp_dir / "mm_cluster.tsv", sep="\t", header=None
        )

    for _, row in mmseq_cluster_df.iterrows():
        cluster_id = row[0]
        entry_id, entity_id = row[1].split("_")
        cluster_id_list.append([entry_id, entity_id, cluster_id])

    cluster_id_df = pd.DataFrame(
        cluster_id_list, columns=["entry_id", "label_entity_id", "cluster_id"]
    )
    return cluster_id_df


def get_recentpdb_lowh_cluster_csv(
    lowh_csv: Path,
    seqs_csv: Path,
    output_cluster_csv: Path,
):
    """
    Generate cluster information CSV for RecentPDB entries with low homology.

    This function processes a predefined set of RecentPDB structures with low homology,
    extracting their cluster IDs and saving results to the configured output path.

    Args:
        lowh_csv (Path): Path to the CSV file containing low homology PDB entries.
        seqs_csv (Path): Path to the CSV file containing PDB sequence information.
        output_cluster_csv (Path): Path to save the output CSV with cluster information
    """
    assert shutil.which("mmseqs"), "CMD: mmseqs not found, please install it first."

    lowh_df = pd.read_csv(lowh_csv, dtype=str)
    seqs_df = pd.read_csv(seqs_csv, dtype=str)
    seqs_keys = seqs_df.apply(lambda x: x["entry_id"] + "_" + x["entity_id"], axis=1)

    lowh_entities = []
    for _, row in lowh_df.iterrows():
        lowh_entities.append(
            [
                row["entry_id"],
                row["entity_id_1"],
                row["entity_type_1"],
                row["seq_length_1"],
            ]
        )
        if row["type"] == "interface":
            lowh_entities.append(
                [
                    row["entry_id"],
                    row["entity_id_2"],
                    row["entity_type_2"],
                    row["seq_length_2"],
                ]
            )
    lowh_entities_df = pd.DataFrame(
        lowh_entities,
        columns=["entry_id", "entity_id", "entity_type", "seq_length"],
    )
    lowh_entities_df.drop_duplicates(subset=["entry_id", "entity_id"], inplace=True)
    lowh_entities_keys = lowh_entities_df.apply(
        lambda x: x["entry_id"] + "_" + x["entity_id"], axis=1
    )

    # Protein
    protein_entities_keys = lowh_entities_keys[
        lowh_entities_df["entity_type"] == PROTEIN
    ]
    protein_seqs_df = seqs_df[seqs_keys.isin(protein_entities_keys)]
    protein_cluster_df = cluster_by_seq_identity(protein_seqs_df, threshold=0.4)
    protein_cluster_df["entity_type"] = PROTEIN

    # DNA
    dna_entities_keys = lowh_entities_keys[lowh_entities_df["entity_type"] == DNA]
    dna_seqs_df = seqs_df[seqs_keys.isin(dna_entities_keys)]
    dna_cluster_df = cluster_by_seq_identity(dna_seqs_df, threshold=0.8)
    dna_cluster_df["entity_type"] = DNA

    # RNA
    rna_entities_keys = lowh_entities_keys[lowh_entities_df["entity_type"] == RNA]
    rna_seqs_df = seqs_df[seqs_keys.isin(rna_entities_keys)]
    rna_cluster_df = cluster_by_seq_identity(rna_seqs_df, threshold=0.8)
    rna_cluster_df["entity_type"] = RNA

    # Ligand
    lig_entities_keys = lowh_entities_keys[lowh_entities_df["entity_type"] == LIGAND]
    lig_seqs_df = seqs_df[seqs_keys.isin(lig_entities_keys)]
    lig_cluster_df = lig_seqs_df[["entry_id", "entity_id", "seq"]].copy()
    lig_cluster_df["entity_type"] = "ligand"
    lig_cluster_df["seq"] = lig_cluster_df["seq"].apply(lambda x: "CCD_" + x)
    lig_cluster_df.rename(
        columns={"entity_id": "label_entity_id", "seq": "cluster_id"}, inplace=True
    )

    cluster_df = pd.concat(
        [protein_cluster_df, dna_cluster_df, rna_cluster_df, lig_cluster_df]
    )
    output_cluster_csv.parent.mkdir(parents=True, exist_ok=True)
    cluster_df.to_csv(output_cluster_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-l",
        "--lowh_csv",
        type=Path,
        default=Path(SUPPORTED_DATA.recentpdb_low_homology),
    )
    argparser.add_argument(
        "-s",
        "--seqs_csv",
        type=Path,
        default=Path(SRC_DATA.pdb_seq_csv),
    )
    argparser.add_argument(
        "-o",
        "--output_csv",
        type=Path,
        default=Path(SUPPORTED_DATA.recentpdb_low_homology_cluster),
    )

    args = argparser.parse_args()

    get_recentpdb_lowh_cluster_csv(
        lowh_csv=args.lowh_csv,
        seqs_csv=args.seqs_csv,
        output_cluster_csv=args.output_csv,
    )
