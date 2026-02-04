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
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import delayed, Parallel
from pxmeter.constants import LIGAND, PROTEIN, RNA
from pxmeter.data.struct import Structure
from tqdm import tqdm

from benchmark.configs.data_config import PXM_MMCIF_DIR, SRC_DATA, SUPPORTED_DATA


def get_entity_counts_from_cif(
    cif_path: Path | str, assembly_id: str | None = None
) -> dict[str, Any]:
    """
    Extracts entity counts from a CIF file.

    Args:
        cif_path (Path or str): The path to the CIF file.
        assembly_id (str, optional): The assembly ID to extract from the CIF file.

    Returns:
        dict[str, Any]: A dictionary containing the PDB ID and counts of different entity types.
    """
    cif_path = Path(cif_path)
    pdb_id = cif_path.stem

    structure = Structure.from_mmcif(cif_path, altloc="first", assembly_id=assembly_id)
    structure = structure.clean_structure(remove_crystallization_aids=False)

    meta_info = {"entry_id": pdb_id}

    entity_type_chain_count = Counter()
    entity_type_entity_count = Counter()
    for label_entity_id in np.unique(structure.atom_array.label_entity_id):
        entity_type = structure.entity_poly_type.get(label_entity_id, LIGAND)
        label_asym_ids = np.unique(
            structure.uni_chain_id[
                structure.atom_array.label_entity_id == label_entity_id
            ]
        )
        chains = len(label_asym_ids)
        entity_type_chain_count[entity_type] += chains
        entity_type_entity_count[f"{entity_type} entities"] += 1

    meta_info.update(entity_type_chain_count)
    meta_info.update(entity_type_entity_count)
    return meta_info


def get_entity_counts_from_cif_batch(
    cif_dir: Path | str,
    output_csv: Path | str | None = None,
    pdb_ids: list[str] | None = None,
    assembly_id: str | None = None,
    n_cpu: int = 1,
) -> pd.DataFrame:
    """
    Batch process CIF files to extract entity counts and save the results to a CSV file.

    Args:
        cif_dir (Path or str, optional): The directory containing the CIF files.
        output_csv (Path or str, optional): The path to the output CSV file.
                   If None, the results will not be saved.
        pdb_ids (list[str], optional): A list of PDB IDs to process.
                If None, all CIF files in the directory will be processed.
        assembly_id (str, optional): The assembly ID to extract from the CIF files.
                    If None, the default assembly ID will be used.
        n_cpu (int): The number of CPU cores to use for parallel processing. Default is 1.

    Returns:
        pd.DataFrame: A DataFrame containing the entity counts for each PDB ID.
    """

    cif_dir = Path(cif_dir)
    if pdb_ids is None:
        all_cif_paths = list(cif_dir.glob("*.cif"))
        random.seed(42)
        random.shuffle(all_cif_paths)
    else:
        all_cif_paths = [cif_dir / f"{i}.cif" for i in pdb_ids]

    if n_cpu == 1:
        all_meta_info = [
            get_entity_counts_from_cif(i, assembly_id)
            for i in tqdm(all_cif_paths, total=len(all_cif_paths))
        ]
    else:
        all_meta_info = [
            r
            for r in tqdm(
                Parallel(n_jobs=n_cpu, return_as="generator_unordered")(
                    delayed(get_entity_counts_from_cif)(
                        cif_path,
                        assembly_id,
                    )
                    for cif_path in all_cif_paths
                ),
                total=len(all_cif_paths),
                desc="Get Entity counts from CIFs",
            )
        ]
    counts_df = pd.DataFrame(all_meta_info)
    counts_df = counts_df.fillna(0)
    columns_to_convert = counts_df.columns.difference(["entry_id"])
    counts_df[columns_to_convert] = (
        counts_df[columns_to_convert]
        .apply(pd.to_numeric, errors="coerce")
        .astype("Int64")
    )

    if output_csv is not None:
        output_csv = Path(output_csv)
        counts_df.to_csv(output_csv, index=False)
    return counts_df


def find_monomer_and_homomer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies protein monomers in a DataFrame based on the
    presence of polypeptide chains and absence of other entities (ligand not include).

    Args:
        df (pd.DataFrame): The input DataFrame containing entity counts.

    Returns:
        pd.DataFrame: The input DataFrame with an additional 'is_protein_monomer' column
                      indicating whether each row represents a protein monomer.
    """

    def _check_is_protein_monomer(row):
        for col in df.columns:
            # Monomer include ligand
            if col in ["entry_id", PROTEIN, LIGAND] or col.endswith(" entities"):
                continue
            if row[col] > 0:
                return False

        if row[PROTEIN] == 1:
            return True
        else:
            return False

    def _check_is_protein_homomer(row):
        for col in df.columns:
            if col in ["entry_id", f"{PROTEIN} entities"] or (
                not col.endswith(" entities")
            ):
                continue
            if row[col] > 0:
                return False

        if row[PROTEIN] > 1 and row[f"{PROTEIN} entities"] == 1:
            return True
        else:
            return False

    def _check_is_rna_monomer(row):
        for col in df.columns:
            # Monomer include ligand
            if col in ["entry_id", RNA, LIGAND] or col.endswith(" entities"):
                continue
            if row[col] > 0:
                return False

        if row[RNA] == 1:
            return True
        else:
            return False

    df["is_protein_monomer"] = df.apply(_check_is_protein_monomer, axis=1)
    df["is_protein_homomer"] = df.apply(_check_is_protein_homomer, axis=1)
    df["is_rna_monomer"] = df.apply(_check_is_rna_monomer, axis=1)
    return df


def find_protein_monomer_for_recentpdb_lowh(
    mmcif_dir: Path,
    lowh_csv: Path,
    output_csv: Path,
    assembly_id: str = "1",
    n_cpu: int = -1,
):
    """
    Identifies protein monomers in the RecentPDB dataset and saves the results to a CSV file.

    Args:
        mmcif_dir (Path): The path to the directory containing the MMCIF files.
        lowh_csv (Path): The path to the CSV file containing the RecentPDB dataset.
        output_csv (Path): The path to the output CSV file.
        assembly_id (str, optional): The assembly ID to extract from the CIF files.
                    If None, the Asym Unit will be used.
        n_cpu (int, optional): The number of CPU cores to use for parallel processing. Defaults to 1.
    """
    df = pd.read_csv(
        lowh_csv,
        dtype=str,
    )
    pdb_ids = set(df["entry_id"])

    counts_df = get_entity_counts_from_cif_batch(
        cif_dir=mmcif_dir, pdb_ids=pdb_ids, assembly_id=assembly_id, n_cpu=n_cpu
    )
    counts_df = find_monomer_and_homomer(counts_df)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    counts_df.to_csv(output_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-m", "--mmcif_dir", type=Path, default=PXM_MMCIF_DIR)
    argparser.add_argument(
        "-l",
        "--lowh_csv",
        type=Path,
        default=Path(SUPPORTED_DATA.recentpdb_low_homology),
    )
    argparser.add_argument(
        "-o",
        "--output_csv",
        type=Path,
        default=Path(SRC_DATA.recentpdb_low_homology_entity_type_count),
    )
    argparser.add_argument(
        "-a",
        "--assembly_id",
        type=str,
        default="1",
    )
    argparser.add_argument(
        "-n",
        "--n_cpu",
        type=int,
        default=-1,
    )

    args = argparser.parse_args()

    find_protein_monomer_for_recentpdb_lowh(
        mmcif_dir=args.mmcif_dir,
        lowh_csv=args.lowh_csv,
        output_csv=args.output_csv,
        assembly_id=args.assembly_id,
        n_cpu=args.n_cpu,
    )
