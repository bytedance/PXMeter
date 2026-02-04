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
import subprocess as sp
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import delayed, Parallel
from pxmeter.constants import DNA, PROTEIN, RNA
from pxmeter.data.struct import Structure
from tqdm import tqdm

from benchmark.configs.data_config import PXM_MMCIF_DIR, SRC_DATA, SUPPORTED_DATA
from benchmark.utils import query_subset_labels


def get_sabdab_tsv(output_path: Path):
    """
    Get SAbDab summary tsv file.

    Args:
        output_path (Path): Path to save the tsv file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = f'wget -O {output_path} \
        "https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/summary/all/"'
    sp.run(cmd, shell=True, check=True)


def get_sabdab_ab_chain_to_type(sabdab_summary_file: Path) -> dict[str, str]:
    """
    Get a dictionary mapping chain ID to antibody type.
    The chain ID is in the format of "[pdb_id]_[auth_asym_id]".

    Args:
        sabdab_summary_file (Path): Path to the SAbDab summary tsv file.

    Returns:
        A dictionary mapping chain ID to antibody type.
    """
    sabdab_df = pd.read_csv(
        sabdab_summary_file,
        sep="\t",
    )

    # Identify antibody and antigen
    ab_chain_to_type = {}
    for _idx, row in sabdab_df.iterrows():
        pdb_id = row["pdb"]
        h_chain = row["Hchain"]
        l_chain = row["Lchain"]

        if (not pd.isna(h_chain)) and (not pd.isna(l_chain)):
            if h_chain == l_chain.upper():
                ab_chain_to_type[f"{pdb_id}_{h_chain}"] = "antibody_scFv"
                ab_chain_to_type[f"{pdb_id}_{l_chain}"] = "antibody_scFv"
            else:
                ab_chain_to_type[f"{pdb_id}_{h_chain}"] = "antibody_HL"
                ab_chain_to_type[f"{pdb_id}_{l_chain}"] = "antibody_HL"
        elif not pd.isna(h_chain):
            ab_chain_to_type[f"{pdb_id}_{h_chain}"] = "antibody_H"
        elif not pd.isna(l_chain):
            ab_chain_to_type[f"{pdb_id}_{l_chain}"] = "antibody_L"
        else:
            raise ValueError(f"PDB {pdb_id} has no antibody chain.")
    return ab_chain_to_type


def get_antibody_antigen_label(
    lowh_df: pd.DataFrame, ab_chain_to_type: dict[str, str]
) -> pd.Series:
    """
    Get antibody/antibody-protein label for each row in the DataFrame.

    Args:
        lowh_df (pd.DataFrame): Dataframe with low homology entities.
        ab_chain_to_type (dict[str, str]): Dictionary mapping chain ID to antibody type.

    Returns:
        A series of antibody/antibody-protein labels for each row in the DataFrame.

        The label is in the format of:
        - "[antibody];[antibody_type]" (chain)
        - "[antibody-protein];[antibody_type-protein]" (interface)
        - "" (not antibody/antibody-protein)
    """

    def _check_ab_per_row(row):
        entry_id = row["entry_id"]
        auth_chain_id_1 = row["auth_chain_id_1"]
        query_chain_id_1 = f"{entry_id}_{auth_chain_id_1}"

        if row["type"] == "chain":
            if ab_type := ab_chain_to_type.get(query_chain_id_1):
                return f"[antibody];[{ab_type}]"
            else:
                return pd.NA

        else:
            auth_chain_id_2 = row["auth_chain_id_2"]
            query_chain_id_2 = f"{entry_id}_{auth_chain_id_2}"

            ab_type_1 = ab_chain_to_type.get(query_chain_id_1)
            ab_type_2 = ab_chain_to_type.get(query_chain_id_2)
            is_protein_1 = row["entity_type_1"] == PROTEIN
            is_protein_2 = row["entity_type_2"] == PROTEIN

            if ab_type_1 and is_protein_1 and ab_type_2 and is_protein_2:
                return "[antibody-antibody]"
            elif (is_protein_1 and ab_type_1) and is_protein_2:
                return f"[antibody-protein];[{ab_type_1}-protein]"
            elif (is_protein_2 and ab_type_2) and is_protein_1:
                return f"[antibody-protein];[{ab_type_2}-protein]"
            else:
                return pd.NA

    ab_subset = lowh_df.apply(_check_ab_per_row, axis=1)
    return ab_subset


def identify_antibody_protein(
    lowh_df: pd.DataFrame, sabdab_summary_file: Path
) -> pd.Series:
    """
    Identify antibody/antibody-protein for each row in the DataFrame.

    Args:
        lowh_df (pd.DataFrame): Dataframe with low homology entities.
        sabdab_summary_file (Path): Path to the SAbDab summary tsv file.

    Returns:
        A series of antibody/antibody-protein labels for each row in the DataFrame.
    """
    if not sabdab_summary_file.exists():
        # Download SAbDab summary tsv file
        get_sabdab_tsv(output_path=sabdab_summary_file)

    # auth_chain_id: antibody_HL/antibody_H/antibody_L
    ab_chain_to_type = get_sabdab_ab_chain_to_type(sabdab_summary_file)
    ab_labels = get_antibody_antigen_label(lowh_df, ab_chain_to_type)
    return ab_labels


def identify_monomer_and_homomer(
    lowh_df: pd.DataFrame, entity_type_count_csv: Path
) -> pd.Series:
    """
    Identify protein/RNA monomer and protein homomer for each row in the DataFrame.

    Args:
        lowh_df (pd.DataFrame): DataFrame with low homology entities.
        entity_type_count_csv (Path): Path to the entity type count csv file.

    Returns:
        A series of protein/RNA monomer labels for each row in the DataFrame.
    """
    entity_type_count_df = pd.read_csv(
        entity_type_count_csv,
        dtype={"entry_id": str},
    )
    prot_monomer_entry_ids = set(
        entity_type_count_df[entity_type_count_df["is_protein_monomer"]]["entry_id"]
    )
    prot_homomer_entry_ids = set(
        entity_type_count_df[entity_type_count_df["is_protein_homomer"]]["entry_id"]
    )
    rna_monomer_entry_ids = set(
        entity_type_count_df[entity_type_count_df["is_rna_monomer"]]["entry_id"]
    )

    def _check_per_row(row, prot_monomer_entry_ids, rna_monomer_entry_ids):
        if (
            row["entry_id"] in prot_monomer_entry_ids
            and (row["type"] == "chain")
            and (row["entity_type_1"] == PROTEIN)
        ):
            return "[protein_monomer]"
        elif row["entry_id"] in prot_homomer_entry_ids:
            return "[protein_homomer]"
        elif (
            row["entry_id"] in rna_monomer_entry_ids
            and (row["type"] == "chain")
            and (row["entity_type_1"] == RNA)
        ):
            return "[rna_monomer]"
        else:
            return pd.NA

    monomer_labels = lowh_df.apply(
        _check_per_row, axis=1, args=(prot_monomer_entry_ids, rna_monomer_entry_ids)
    )
    return monomer_labels


def identity_peptide(lowh_df: pd.DataFrame, peptide_threshold: int = 25) -> pd.Series:
    """
    Identify peptide for each row in the DataFrame.

    Args:
        lowh_df (pd.DataFrame): DataFrame with low homology entities.
        peptide_threshold (int, optional): Threshold for peptide length.
                          Defaults to 25.

    Returns:
        A series of peptide labels for each row in the DataFrame.
    """

    def _check_per_row(row, peptide_threshold):
        labels = ""
        if row["type"] == "chain":
            return labels
        else:
            is_peptide_1 = (row["entity_type_1"] == PROTEIN) and (
                row["seq_length_1"] < peptide_threshold
            )
            is_peptide_2 = (row["entity_type_2"] == PROTEIN) and (
                row["seq_length_2"] < peptide_threshold
            )

            if is_peptide_1 and is_peptide_2:
                return "[peptide-interface];[peptide-peptide]"
            elif (is_peptide_1 and (row["entity_type_2"] == PROTEIN)) or (
                is_peptide_2 and (row["entity_type_1"] == PROTEIN)
            ):
                return "[peptide-interface];[peptide-protein]"
            elif (is_peptide_1 and (row["entity_type_2"] == DNA)) or (
                is_peptide_2 and (row["entity_type_1"] == DNA)
            ):
                return "[peptide-interface];[peptide-dna]"
            elif (is_peptide_1 and (row["entity_type_2"] == RNA)) or (
                is_peptide_2 and (row["entity_type_1"] == RNA)
            ):
                return "[peptide-interface];[peptide-rna]"
            else:
                return pd.NA

    peptide_labels = lowh_df.apply(_check_per_row, args=(peptide_threshold,), axis=1)
    return peptide_labels


def get_cyclic_peptide_entities_from_cif(
    cif_path: Path, peptide_threshold: int = 25
) -> list[str]:
    """
    Get cyclic-peptide chains from a CIF file.

    Args:
        cif_path (Path): Path to the CIF file.
        peptide_threshold (int, optional): Threshold for peptide length.
                          Defaults to 25.

    Returns:
        A list of cyclic-peptide '[PDB ID]_[Entity ID]' strings.
    """
    struct = Structure.from_mmcif(cif_path)
    entry_id = struct.entry_id
    atom_array = struct.atom_array

    peptide_entity_ids = [
        k for k, v in struct.entity_poly_type.items() if v == "polypeptide(L)"
    ]

    cyclic_peptide_entities = []
    for entity_id in peptide_entity_ids:
        chain_ids = np.unique(
            struct.uni_chain_id[atom_array.label_entity_id == entity_id]
        )
        chain_id = chain_ids[0]
        chain_atom_array = atom_array[struct.uni_chain_id == chain_id]
        res_ids = chain_atom_array.res_id
        if np.unique(res_ids).shape[0] > peptide_threshold:
            continue

        bonds = chain_atom_array.bonds.as_array()
        for i, j in bonds[:, :2]:
            if abs(res_ids[i] - res_ids[j]) > 1:
                cyclic_peptide_entities.append(f"{entry_id}_{entity_id}")
                break
    return cyclic_peptide_entities


def identity_cyclic_peptide(
    lowh_df: pd.DataFrame, peptide_mask: pd.Series, mmcif_dir: Path, n_cpu: int = -1
) -> pd.Series:
    """
    Identify cyclic-peptide for each row in the DataFrame.

    Args:
        lowh_df (pd.DataFrame): DataFrame with low homology entities.
        peptide_mask (pd.Series): Mask for peptide entities.
        mmcif_dir (Path): Path to the mmCIF directory.
        n_cpu (int, optional): Number of CPUs to use. Defaults to -1.

    Returns:
        A series of cyclic-peptide labels for each row in the DataFrame.
    """
    peptide_entry_ids = set(lowh_df[peptide_mask]["entry_id"])
    all_cif_paths = [mmcif_dir / f"{i}.cif" for i in peptide_entry_ids]

    cyclic_peptide_ids = [
        r
        for r in tqdm(
            Parallel(n_jobs=n_cpu, return_as="generator_unordered")(
                delayed(get_cyclic_peptide_entities_from_cif)(
                    cif_path,
                )
                for cif_path in all_cif_paths
            ),
            total=len(all_cif_paths),
            desc="Get cyclic-peptide chains from CIFs",
        )
    ]
    cyclic_peptide_ids = [item for sublist in cyclic_peptide_ids for item in sublist]

    def _check_per_row(row, cyclic_peptide_ids):
        if row["type"] == "chain":
            return pd.NA
        else:
            is_c_peptide_1 = (
                f'{row["entry_id"]}_{row["entity_id_1"]}' in cyclic_peptide_ids
            )
            is_c_peptide_2 = (
                f'{row["entry_id"]}_{row["entity_id_2"]}' in cyclic_peptide_ids
            )

            if is_c_peptide_1 and is_c_peptide_2:
                return "[cyclic_peptide-interface];[cyclic_peptide-cyclic_peptide]"
            elif (is_c_peptide_1 and (row["entity_type_2"] == PROTEIN)) or (
                is_c_peptide_2 and (row["entity_type_1"] == PROTEIN)
            ):
                return "[cyclic_peptide-interface];[cyclic_peptide-protein]"
            elif (is_c_peptide_1 and (row["entity_type_2"] == DNA)) or (
                is_c_peptide_2 and (row["entity_type_1"] == DNA)
            ):
                return "[cyclic_peptide-interface];[cyclic_peptide-dna]"
            elif (is_c_peptide_1 and (row["entity_type_2"] == RNA)) or (
                is_c_peptide_2 and (row["entity_type_1"] == RNA)
            ):
                return "[cyclic_peptide-interface];[cyclic_peptide-rna]"
            else:
                return pd.NA

    cyclic_peptide_labels = lowh_df.apply(
        _check_per_row, args=(cyclic_peptide_ids,), axis=1
    )
    return cyclic_peptide_labels


def identify_subset(
    lowh_csv: Path,
    sabdab_summary_file: Path,
    entity_type_count_csv: Path,
    mmcif_dir: Path,
    n_cpu: int = -1,
):
    """
    Identify subsets for each row in the DataFrame.
    The multiple subsets are separated by ';'.

    Args:
        lowh_csv (Path): Path to the low homology CSV file.
        sabdab_summary_file (Path): Path to the SABDAB summary file.
        entity_type_count_csv (Path): Path to the entity type count CSV file.
        mmcif_dir (Path): Path to the mmCIF directory.
        n_cpu (int, optional): Number of CPUs to use. Defaults to -1.
    """
    lowh_df = pd.read_csv(
        lowh_csv,
        dtype={"entry_id": str, "entity_id_1": str, "entity_id_2": str},
    )

    ab_labels = identify_antibody_protein(
        lowh_df,
        sabdab_summary_file,
    )
    monomer_labels = identify_monomer_and_homomer(
        lowh_df,
        entity_type_count_csv,
    )

    peptide_labels = identity_peptide(lowh_df)
    peptide_mask = query_subset_labels(
        peptide_labels, query_label="[peptide-interface]"
    )

    if peptide_mask.sum() > 0:
        lowh_df["subset"] = peptide_labels
        cyclic_peptide_labels = identity_cyclic_peptide(
            lowh_df, peptide_mask, mmcif_dir, n_cpu=n_cpu
        )
    else:
        cyclic_peptide_labels = pd.Series("")

    series_list = [ab_labels, monomer_labels, peptide_labels, cyclic_peptide_labels]
    combined_labels = pd.concat(series_list, axis=1).apply(
        lambda row: ";".join(row.dropna().astype(str)), axis=1
    )

    lowh_df["subset"] = combined_labels
    lowh_csv.parent.mkdir(parents=True, exist_ok=True)
    lowh_df.to_csv(lowh_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l", "--lowh_csv", type=Path, default=SUPPORTED_DATA.recentpdb_low_homology
    )
    parser.add_argument(
        "-s", "--sabdab_summary_file", type=Path, default=SRC_DATA.sabdab_summary_file
    )
    parser.add_argument(
        "-e",
        "--entity_type_count_csv",
        type=Path,
        default=SRC_DATA.recentpdb_low_homology_entity_type_count,
    )
    parser.add_argument(
        "-m",
        "--mmcif_dir",
        type=Path,
        default=PXM_MMCIF_DIR,
    )
    parser.add_argument(
        "-n",
        "--n_cpu",
        type=int,
        default=-1,
    )
    args = parser.parse_args()

    identify_subset(
        args.lowh_csv,
        args.sabdab_summary_file,
        args.entity_type_count_csv,
        args.mmcif_dir,
        args.n_cpu,
    )
