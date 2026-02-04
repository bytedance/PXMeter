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
import json
import logging
import shutil
import subprocess as sp
import tempfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
from biotite.interface.rdkit import to_mol
from biotite.structure.info import get_ccd, residue
from joblib import delayed, Parallel
from pxmeter.constants import DNA, LIGAND, PROTEIN, RNA
from pxmeter.data.utils import is_valid_date_format
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm

from benchmark.configs.data_config import SRC_DATA, SRC_DATA_DIR
from benchmark.dataset_pipeline.utils.seq_identity import smith_waterman_identity
from benchmark.utils import shrink_dataframe


def _get_ccd_mol_from_block(
    ccd_code: str,
) -> tuple[str, DataStructs.ExplicitBitVect]:
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    try:
        ccd_atom_array = residue(ccd_code, allow_missing_coord=True)
        mol = to_mol(ccd_atom_array)
        if mol is None:
            logging.warning(ccd_code, "Mol is None")
            return ccd_code, None
        Chem.SanitizeMol(mol)
        fp = mfpgen.GetFingerprint(mol)
        return ccd_code, fp
    except Exception as e:
        logging.warning(ccd_code, e)
        return ccd_code, None


def gen_ccd_fp() -> dict[str, DataStructs.ExplicitBitVect]:
    """
    Generate fingerprint for each Chemical Component Dictionary (CCD) entry.

    Returns:
        dict[str, DataStructs.ExplicitBitVect]: A dictionary mapping CCD codes
                                                to their fingerprints.
    """
    all_ccd = get_ccd()  # BinaryCIFBlock
    chem_comp = all_ccd["chem_comp"]
    codes = chem_comp["id"].as_array()

    results = [
        r
        for r in tqdm(
            Parallel(n_jobs=-1, return_as="generator_unordered")(
                delayed(_get_ccd_mol_from_block)(ccd_code) for ccd_code in codes
            ),
            total=len(codes),
            desc="Calc FP for CCD Mols",
        )
    ]

    ccd_fp = {}
    for ccd_code, fp in results:
        if fp is None:
            continue
        ccd_fp[ccd_code] = fp
    return ccd_fp


def get_ccd_similarity(output_ccd_similairty_file: Path, threshold: float = 0.6):
    """
    Calculate the pairwise Tanimoto similarity between Chemical Component Dictionary (CCD)
    entries based on their molecular fingerprints and save the results to a JSON file.

    Args:
        output_ccd_similairty_file (Path): Path to the output JSON file where the
                                           CCD similarity results will be saved.
        threshold (float, optional): Minimum Tanimoto similarity threshold for
                                     considering CCD entries as similar.
                                     Defaults to 0.6.
    """
    ccd_fp = gen_ccd_fp()
    codes = []
    fps = []
    for ccd_code, fp in ccd_fp.items():
        codes.append(ccd_code)
        fps.append(fp)

    code_to_simi_codes = defaultdict(list)
    for idx, (code_i, _fp_i) in tqdm(
        enumerate(zip(codes, fps)),
        total=len(codes),
        desc="Calc pairwise similarity for CCD Mols",
    ):
        simi_res = DataStructs.BulkTanimotoSimilarity(fps[idx], fps[idx + 1 :])
        code_to_simi_codes[code_i].append([code_i, 1.00])
        for code_j, simi in zip(codes[idx + 1 :], simi_res):
            if simi > threshold:
                code_to_simi_codes[code_i].append([code_j, round(simi, 2)])

    output_ccd_similairty_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_ccd_similairty_file, "w") as f:
        json.dump(code_to_simi_codes, f)


def get_lowh_by_ccd_similarity(
    ccd_similairty_file: Path,
    seq_df: pd.DataFrame,
    after_date: str,
    before_date: str,
    threshold: float = 0.6,
) -> pd.DataFrame:
    """
    Calculate low homology (lowh) mapping between test and train sequences based on
    the similarity of their Chemical Component Dictionary (CCD) entries.

    Args:
        ccd_similairty_file (Path): Path to the JSON file containing CCD similarity results.
        seq_df (pd.DataFrame): DataFrame containing sequence data with columns
                               'release_date', 'seq', 'entry_id', and 'entity_id'.
        after_date (str): Date string in 'YYYY-MM-DD' format to filter train sequences.
        before_date (str): Date string in 'YYYY-MM-DD' format to filter test sequences.
        threshold (float, optional): Minimum Tanimoto similarity threshold for
                                     considering CCD entries as similar.
                                     Defaults to 0.6.

    Returns:
        pd.DataFrame: DataFrame containing lowh mapping with columns
                      'query_id', 'db_id', and 'simi' (similarity score).
    """
    if not ccd_similairty_file.exists():
        get_ccd_similarity(ccd_similairty_file, threshold)

    train_df = seq_df[(seq_df["release_date"] <= after_date)]
    test_df = seq_df[
        (seq_df["release_date"] > after_date) & (seq_df["release_date"] < before_date)
    ]

    ccd_code_to_train_ids = defaultdict(list)
    for _, row in train_df.iterrows():
        ccd_code_to_train_ids[row["seq"]].append(
            f'{row["entry_id"]}_{row["entity_id"]}'
        )

    with open(ccd_similairty_file) as f:
        code_to_simi_codes = json.load(f)

    test_vs_train = []
    for _, row in test_df.iterrows():
        ccd_code = row["seq"]
        test_id = f'{row["entry_id"]}_{row["entity_id"]}'
        simi_codes = code_to_simi_codes.get(ccd_code, [])
        for simi_code, simi in simi_codes:
            for train_id in ccd_code_to_train_ids[simi_code]:
                test_vs_train.append([test_id, train_id, simi])

    test_vs_train_df = pd.DataFrame(
        test_vs_train, columns=["query_id", "db_id", "similarity"]
    )
    return test_vs_train_df


def calc_smith_waterman_seq_identity(
    db_df,
    query_df,
    min_seq_length: int = 25,
    threshold: float = 0.3,
    gap_open: int = -11,
    gap_extend: int = -1,
    nuc: bool = False,
) -> pd.DataFrame:
    """
    Calculate pairwise Smith-Waterman sequence identities between query
    sequences and a database of sequences.

    This function performs local sequence alignment (Smith-Waterman) for
    all query-database pairs above a minimum length threshold. Short
    sequences are matched by exact identity. Identities are reported as
    the fraction of exact matches over the query length.

    Args:
        db_df (pd.DataFrame): DataFrame containing database sequences.
            Must include columns "entry_id", "entity_id", and "seq".
        query_df (pd.DataFrame): DataFrame containing query sequences.
            Must include columns "entry_id", "entity_id", and "seq".
        min_seq_length (int, optional): Minimum sequence length required
            for Smith-Waterman alignment. Shorter sequences are only
            matched by exact identity. Defaults to 25.
        threshold (float, optional): Minimum sequence identity required
            to retain a hit. Defaults to 0.3.
        gap_open (int, optional): Gap opening penalty. Defaults to -11.
        gap_extend (int, optional): Gap extension penalty. Defaults to -1.
        nuc (bool, optional): If True, perform nucleotide alignment with
            'N' treated as an unknown residue. If False, perform protein
            alignment with X treated as unknown. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame containing query IDs, database IDs, sequence identities,
            and exact aligned residue counts.
    """

    db_id_and_seq = []
    db_short_seq_to_id = defaultdict(list)
    for _, row in db_df.iterrows():
        if len(row["seq"]) < min_seq_length:
            db_short_seq_to_id[row["seq"]].append(
                f"{row['entry_id']}_{row['entity_id']}"
            )
        else:
            db_id_and_seq.append((f"{row['entry_id']}_{row['entity_id']}", row["seq"]))

    query_id_and_seq = []
    query_short_id = []
    for _, row in query_df.iterrows():
        if len(row["seq"]) < min_seq_length:
            query_short_id.append((row["seq"], f"{row['entry_id']}_{row['entity_id']}"))
        else:
            query_id_and_seq.append(
                (f"{row['entry_id']}_{row['entity_id']}", row["seq"])
            )

    query_vs_db = []
    # Find identity short sequence
    for seq, query_id in query_short_id:
        if id_in_db := db_short_seq_to_id.get(seq):
            query_vs_db.extend([[query_id, i, 1.0, 0] for i in id_in_db])

    def _run_for_pair(query_id_and_seq, db_id, db_seq):
        results = []
        for query_id, query_seq in query_id_and_seq:
            identity, exact_matches, _aligned_q, _aligned_s = smith_waterman_identity(
                query_seq,
                db_seq,
                gap_open=gap_open,
                gap_extend=gap_extend,
                is_nucleic=nuc,
            )
            results.append((identity, exact_matches, query_id, db_id))
        return results

    results = [
        r
        for r in tqdm(
            Parallel(n_jobs=-1, return_as="generator_unordered")(
                delayed(_run_for_pair)(query_id_and_seq, db_id, db_seq)
                for (db_id, db_seq) in db_id_and_seq
            ),
            total=len(db_id_and_seq),
            desc=f"Pairwise SW identity for {len(query_id_and_seq)} query vs {len(db_id_and_seq)} db seqs",
        )
    ]

    for r in results:
        for identity, aligned_res_num, query_id, db_id in r:
            if identity < threshold:
                continue
            query_vs_db.append([query_id, db_id, identity, aligned_res_num])

    query_vs_db_df = pd.DataFrame(
        query_vs_db, columns=["query_id", "db_id", "similarity", "aligned_res_num"]
    )
    return query_vs_db_df


def get_lowh_by_sw_seq_identity(
    seq_df: pd.DataFrame,
    after_date: str,
    before_date: str,
    min_seq_length: int = 25,
    threshold: float = 0.3,
    gap_open: int = -11,
    gap_extend: int = -1,
    nuc: bool = False,
) -> pd.DataFrame:
    """
    Partition sequences into training and test sets based on release
    dates and compute Smith-Waterman sequence identities between them.

    The function filters sequences by date, applies local alignment
    between test and training sequences, and returns a mapping of test
    sequence IDs to their training hits above a given identity threshold.

    Args:
        seq_df (pd.DataFrame): DataFrame containing all sequences with
            at least "entry_id", "entity_id", "seq", and "release_date".
        after_date (str): Cutoff date (inclusive) for the training set.
        before_date (str): Upper cutoff date (exclusive) for the test set.
        min_seq_length (int, optional): Minimum sequence length required
            for Smith-Waterman alignment. Defaults to 25.
        threshold (float, optional): Minimum sequence identity required
            to retain a hit. Defaults to 0.3.
        gap_open (int, optional): Gap opening penalty. Defaults to -11.
        gap_extend (int, optional): Gap extension penalty. Defaults to -1.
        nuc (bool, optional): If True, perform nucleotide alignment;
            otherwise perform protein alignment. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame containing query IDs, database IDs, sequence identities,
            and exact aligned residue counts.
    """
    train_df = seq_df[(seq_df["release_date"] <= after_date)]
    test_df = seq_df[
        (seq_df["release_date"] > after_date) & (seq_df["release_date"] < before_date)
    ]

    test_vs_train_df = calc_smith_waterman_seq_identity(
        db_df=train_df,
        query_df=test_df,
        min_seq_length=min_seq_length,
        threshold=threshold,
        gap_open=gap_open,
        gap_extend=gap_extend,
        nuc=nuc,
    )

    return test_vs_train_df


def calc_mmseqs_seq_identity(
    db_df,
    query_df,
    min_seq_length: int = 25,
    threshold: float = 0.3,
    coverage: float = 0.0,
    cov_mode: int = 0,
    e_value_cutoff: float = 0.001,
    sensitivity: float = 5.7,
    max_seqs: int = 300,
    nuc: bool = False,
) -> pd.DataFrame:
    """
    Calculate low homology (lowh) mapping between test and train sequences by MMseqs.

    Args:
        db_df (pd.DataFrame): DataFrame containing sequence data with columns
                               'release_date', 'seq', 'entry_id', and 'entity_id'.
        query_df (pd.DataFrame): DataFrame containing query sequence data with columns
                               'release_date', 'seq', 'entry_id', and 'entity_id'.
        min_seq_length (int, optional): Minimum sequence length to be considered a valid sequence.
                                        Defaults to 25.
        threshold (float, optional): Minimum sequence identity threshold for MMseqs2 search.
                                     Defaults to 0.3.
        coverage (float, optional): Minimum coverage threshold for MMseqs2 search.
                                    Defaults to 0.0.
        cov_mode (int, optional): Coverage mode for MMseqs2 search.
                                  Defaults to 0.
        e_value_cutoff (float, optional): E-value cutoff for MMseqs2 search.
                                          Defaults to 0.001.
        sensitivity (float, optional): Sensitivity parameter for MMseqs2 search.
                                       Sensitivity: 1.0 faster; 4.0 fast; 7.5 sensitive
                                       Defaults to 5.7.
        max_seqs (int, optional): Maximum results per query sequence allowed to pass the prefilter for MMseqs2 search.
                                  Defaults to 300.
        nuc (bool, optional): Whether the sequence is a nucleotide sequence. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame containing query IDs, database IDs, sequence identities,
            and exact aligned residue counts.
    """
    db_fasta_txt = ""
    db_short_seq_to_id = defaultdict(list)
    for _, row in db_df.iterrows():
        if len(row["seq"]) < min_seq_length:
            db_short_seq_to_id[row["seq"]].append(
                f"{row['entry_id']}_{row['entity_id']}"
            )
        else:
            db_fasta_txt += f">{row['entry_id']}_{row['entity_id']}\n{row['seq']}\n"

    query_fasta_txt = ""
    query_short_id = []
    for _, row in query_df.iterrows():
        if len(row["seq"]) < min_seq_length:
            query_short_id.append((row["seq"], f"{row['entry_id']}_{row['entity_id']}"))
        else:
            query_fasta_txt += f">{row['entry_id']}_{row['entity_id']}\n{row['seq']}\n"

    query_vs_db = []
    # Find identity short sequence
    for seq, query_id in query_short_id:
        if id_in_db := db_short_seq_to_id.get(seq):
            # id, sequence identity
            query_vs_db.append([query_id, id_in_db[0], 1.0, 0])

    SRC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SRC_DATA_DIR) as tmp_dir:
        tmp_dir = Path(tmp_dir)
        db_fasta_path = tmp_dir / "db.fasta"
        query_fasta_path = tmp_dir / "query.fasta"
        db_fasta_path.write_text(db_fasta_txt)
        query_fasta_path.write_text(query_fasta_txt)

        cmd = (
            f"cd {tmp_dir};mmseqs easy-search query.fasta db.fasta test_vs_train.tsv mmseqs_tmp "
            f"-e {e_value_cutoff} --max-seqs {max_seqs} -s {sensitivity}"
        )
        if nuc:
            cmd += " --search-type 3"

        if threshold > 0.0:
            cmd += f" --min-seq-id {threshold}"

        if coverage > 0.0:
            cmd += f" -c {coverage}"

            if cov_mode > 0:
                cmd += f" --cov-mode {cov_mode}"

        sp.run(
            cmd,
            shell=True,
            check=True,
        )
        df = pd.read_csv(tmp_dir / "test_vs_train.tsv", sep="\t", header=None)

        for _, row in df.iterrows():
            query_id = row[0]
            db_id = row[1]
            identity = row[2]
            aligned_res_num = row[3]
            query_vs_db.append([query_id, db_id, identity, aligned_res_num])

    query_vs_db_df = pd.DataFrame(
        query_vs_db, columns=["query_id", "db_id", "similarity", "aligned_res_num"]
    )
    return query_vs_db_df


def get_lowh_by_mmseqs_seq_identity(
    seq_df: pd.DataFrame,
    after_date: str,
    before_date: str,
    min_seq_length: int = 25,
    threshold: float = 0.3,
    coverage: float = 0.0,
    cov_mode: int = 0,
    e_value_cutoff: float = 0.001,
    sensitivity: float = 5.7,
    max_seqs: int = 300,
    nuc: bool = False,
) -> dict[str, list[tuple]]:
    """
    Calculate low homology (lowh) mapping between test and train sequences by MMseqs.

    Args:
        seq_df (pd.DataFrame): DataFrame containing sequence data with columns
                               'release_date', 'seq', 'entry_id', and 'entity_id'.
        after_date (str): The start date (yyyy-mm-dd) for filtering test sequences.
        before_date (str): The end date (yyyy-mm-dd) for filtering test sequences.
        min_seq_length (int, optional): Minimum sequence length to be considered a valid sequence.
                                        Defaults to 25.
        threshold (float, optional): Minimum sequence identity threshold for MMseqs2 search.
                                     Defaults to 0.3.
        coverage (float, optional): Minimum coverage threshold for MMseqs2 search.
                                    Defaults to 0.0.
        cov_mode (int, optional): Coverage mode for MMseqs2 search.
                                  Defaults to 0.
        e_value_cutoff (float, optional): E-value cutoff for MMseqs2 search.
                                          Defaults to 0.001.
        sensitivity (float, optional): Sensitivity parameter for MMseqs2 search.
                                       Sensitivity: 1.0 faster; 4.0 fast; 7.5 sensitive
                                       Defaults to 5.7.
        max_seqs (int, optional): Maximum results per query sequence allowed to pass the prefilter for MMseqs2 search.
                                  Defaults to 300.
        nuc (bool, optional): Whether the sequence is a nucleotide sequence. Defaults to False.

    Returns:
        dict[str, list[tuple]]: A dictionary mapping test sequence IDs to a list of tuples
                        (train sequence ID, sequence identity, number of alignment residues).
    """
    assert shutil.which("mmseqs"), "CMD: mmseqs not found, please install it first."

    train_df = seq_df[(seq_df["release_date"] < after_date)]
    test_df = seq_df[
        (seq_df["release_date"] >= after_date) & (seq_df["release_date"] <= before_date)
    ]

    test_vs_train = calc_mmseqs_seq_identity(
        db_df=train_df,
        query_df=test_df,
        min_seq_length=min_seq_length,
        threshold=threshold,
        coverage=coverage,
        e_value_cutoff=e_value_cutoff,
        sensitivity=sensitivity,
        cov_mode=cov_mode,
        max_seqs=max_seqs,
        nuc=nuc,
    )

    return test_vs_train


def get_maxsim_df(
    lowh_result_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Get the maximum similarity DataFrame from the test-to-train similarity dictionary.

    Args:
        lowh_result_df (pd.DataFrame): DataFrame containing low homology (lowh) sequence mapping information.
        Columns: 'query_id', 'db_id', 'similarity', 'aligned_res_num'.

    Returns:
        pd.DataFrame: A DataFrame containing the maximum similarity information for each test sequence.
    """
    maxsim_df = []
    for query_id, group_df in lowh_result_df.groupby("query_id", observed=True):
        if not group_df.empty:
            maxsim = group_df.loc[group_df["similarity"].idxmax()]
            maxsim_df.append(
                [
                    query_id,
                    maxsim["db_id"],
                    maxsim["similarity"],
                    maxsim["aligned_res_num"],
                ]
            )

    maxsim_df = pd.DataFrame(
        maxsim_df, columns=["query_id", "db_id", "similarity", "aligned_res_num"]
    )
    return maxsim_df


def make_lowh_parquet_file(
    seqs_csv: Path,
    lowh_parquet: Path,
    ccd_similairty_file: Path,
    after_date: str,
    before_date: str,
):
    """
    Generate a JSON file containing low homology (lowh) sequence mapping information.

    Args:
        seqs_csv (Path): Path to the CSV file containing sequence data.
        lowh_parquet (Path): Path to the output Parquet file where the lowh mapping will be saved.
        ccd_similairty_file (Path): Path to the JSON file containing CCD similarity data.
        after_date (str, optional): The start date (yyyy-mm-dd) for filtering sequences.
        before_date (str, optional): The end date (yyyy-mm-dd) for filtering sequences.
    """
    assert is_valid_date_format(
        after_date
    ), f"Invalid date format: {after_date}, it should be yyyy-mm-dd format"
    assert is_valid_date_format(
        before_date
    ), f"Invalid date format: {before_date}, it should be yyyy-mm-dd format"

    df = pd.read_csv(seqs_csv, dtype=str)
    protein_test_vs_train = get_lowh_by_mmseqs_seq_identity(
        df[df["entity_type"] == PROTEIN],
        after_date,
        before_date,
        threshold=0.4,
        coverage=0.0,
        cov_mode=0,
        e_value_cutoff=0.1,
        sensitivity=7.5,
        max_seqs=500000,
    )

    rna_test_vs_train = get_lowh_by_mmseqs_seq_identity(
        df[df["entity_type"] == DNA],
        after_date,
        before_date,
        threshold=0.8,
        coverage=0.0,
        cov_mode=0,
        e_value_cutoff=0.1,
        sensitivity=7.5,
        max_seqs=500000,
        nuc=True,
    )

    dna_test_vs_train = get_lowh_by_mmseqs_seq_identity(
        df[df["entity_type"] == RNA],
        after_date,
        before_date,
        threshold=0.8,
        coverage=0.0,
        cov_mode=0,
        e_value_cutoff=0.1,
        sensitivity=7.5,
        max_seqs=500000,
        nuc=True,
    )

    lig_test_vs_train = get_lowh_by_ccd_similarity(
        ccd_similairty_file,
        df[df["entity_type"] == LIGAND],
        after_date,
        before_date,
        threshold=0.6,
    )

    merged_df = pd.concat(
        [
            protein_test_vs_train,
            rna_test_vs_train,
            dna_test_vs_train,
            lig_test_vs_train,
        ]
    )

    merged_df, _report = shrink_dataframe(merged_df)
    merged_df.to_parquet(
        lowh_parquet,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-s",
        "--seqs_csv",
        type=Path,
        default=Path(SRC_DATA.pdb_seq_csv),
    )
    arg_parser.add_argument(
        "-o",
        "--output_lowh_parquet",
        type=Path,
        default=Path(SRC_DATA.test_to_train_entity_homo_parquet),
    )
    arg_parser.add_argument(
        "-c",
        "--ccd_similairty_file",
        type=Path,
        default=Path(SRC_DATA.ccd_to_similar_ccds),
    )
    arg_parser.add_argument(
        "-a",
        "--after_date",
        type=str,
        required=True,
    )
    arg_parser.add_argument(
        "-b",
        "--before_date",
        type=str,
        required=True,
    )
    args = arg_parser.parse_args()

    make_lowh_parquet_file(
        seqs_csv=args.seqs_csv,
        lowh_parquet=args.output_lowh_parquet,
        ccd_similairty_file=args.ccd_similairty_file,
        after_date=args.after_date,
        before_date=args.before_date,
    )
