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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from benchmark.configs.data_config import SUPPORTED_DATA
from pxmeter.constants import DNA, LIGAND, PROTEIN, RNA
from pxmeter.data.struct import Structure


def parse_pdb_cluster_file_to_dict(
    cluster_file: str, remove_uniprot: bool = True
) -> dict[str, tuple]:
    """
    Parse PDB cluster file, and return a pandas dataframe
    example cluster file:
    https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-40.txt

    Args:
        cluster_file (str): cluster_file path
    Returns:
        dict(str, tuple(str, str)): {pdb_id}_{entity_id} --> [cluster_id, cluster_size]
    """
    pdb_cluster_dict = {}
    with open(cluster_file) as f:
        for line in f:
            pdb_clusters = []
            for ids in line.strip().split():
                if remove_uniprot:
                    if ids.startswith("AF_") or ids.startswith("MA_"):
                        continue
                pdb_clusters.append(ids)
            cluster_size = len(pdb_clusters)
            if cluster_size == 0:
                continue
            # use first member as cluster id.
            cluster_id = f"pdb_cluster_{pdb_clusters[0]}"
            for ids in pdb_clusters:
                pdb_cluster_dict[ids.lower()] = (cluster_id, cluster_size)
    return pdb_cluster_dict


def get_cluster_id_from_cif(
    cif_path: Path | str, pdb_cluster_dict: dict[str, tuple]
) -> dict[str, Any]:
    """
    Extract cluster ID information for each entity in a CIF file.

    - For proteins, the cluster ID is determined based on the 40% sequence similarity.
      If the sequence length is less than or equal to 9, the cluster ID is set to the sequence.
    - For DNA and RNA, the cluster ID is set to the sequence.
    - For ligands, the cluster ID is determined based on the CCD identity.
      If the ligand is a water molecule, the cluster ID is set to None.
    - For other entity types, the cluster ID is set to None.

    Args:
        cif_path (Path | str): Path to the CIF file.
        pdb_cluster_dict (dict[str, tuple]): Dictionary mapping PDB entity IDs to cluster information.

    Returns:
        list[dict[str, Any]]: List of dictionaries containing information about each entity,
                              including entry ID, entity ID, cluster ID, entity type and seq_length.
    """
    cif_path = Path(cif_path)
    pdb_id = cif_path.stem

    structure = Structure.from_mmcif(cif_path, altloc="all")

    entity_infos = []
    for label_entity_id in np.unique(structure.atom_array.label_entity_id):
        entity_info_dict = {"entry_id": pdb_id, "label_entity_id": label_entity_id}
        entity_type = structure.entity_poly_type.get(label_entity_id, LIGAND)
        first_chain_id = np.unique(
            structure.uni_chain_id[
                structure.atom_array.label_entity_id == label_entity_id
            ]
        )[0]

        if entity_type != LIGAND:
            seq_length = len(structure.entity_poly_seq[label_entity_id])
        else:
            seq_length = len(
                np.unique(
                    structure.atom_array.res_id[
                        structure.uni_chain_id == first_chain_id
                    ]
                )
            )
        if entity_type == PROTEIN:
            if seq_length <= 9:
                # 100% seq similarity
                cluster_id = structure.entity_poly_seq[label_entity_id]
            else:
                # 40% seq similarity
                cluster_id = pdb_cluster_dict.get(
                    f"{pdb_id}_{label_entity_id}", [None]
                )[0]
        elif entity_type == DNA or entity_type == RNA:
            # 100% seq similarity
            cluster_id = structure.entity_poly_seq[label_entity_id]
        elif entity_type == LIGAND:
            # 100% seq similarity (CCD identity)
            _, res_starts = np.unique(
                structure.atom_array.res_id[structure.uni_chain_id == first_chain_id],
                return_index=True,
            )

            res_names = structure.atom_array.res_name[
                structure.uni_chain_id == first_chain_id
            ][res_starts]
            if np.all((res_names == "HOH") | (res_names == "DOD")):
                # water molecule
                cluster_id = None
            else:
                cluster_id = "_".join(res_names)
        else:
            # other entity types
            cluster_id = None

        entity_info_dict["cluster_id"] = cluster_id
        entity_info_dict["entity_type"] = entity_type
        entity_info_dict["seq_length"] = seq_length

        entity_infos.append(entity_info_dict)
    return entity_infos


def get_cluster_id_from_cif_batch(
    cif_dir: Path,
    cluster_txt_file: Path,
    output_csv: Path,
    pdb_ids: list[str] | None = None,
    n_cpu: int = -1,
):
    """
    Process a batch of CIF files to extract cluster information and save results to CSV.

    Args:
        cif_dir (Path): Directory containing input CIF files
        output_csv (Path): Path to save output CSV with cluster information
        cluster_txt_file (Path): Path to the cluster text file
                        example cluster file:
                        https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-40.txt
        pdb_ids (list[str] | None): Optional list of specific PDB IDs to process.
            If None, processes all .cif files in cif_dir.
        n_cpu (int): Number of CPUs to use for parallel processing (-1 = all available)
    """
    if pdb_ids is None:
        all_cif_paths = list(cif_dir.glob("*.cif"))
        random.seed(42)
        random.shuffle(all_cif_paths)
    else:
        all_cif_paths = [cif_dir / f"{i}.cif" for i in pdb_ids]

    cluster_dict = parse_pdb_cluster_file_to_dict(cluster_txt_file)

    results = [
        r
        for r in tqdm(
            Parallel(n_jobs=n_cpu, return_as="generator_unordered")(
                delayed(get_cluster_id_from_cif)(cif_path, cluster_dict)
                for cif_path in all_cif_paths
            ),
            total=len(all_cif_paths),
        )
    ]

    all_entity_info = []
    for result in results:
        all_entity_info.extend(result)

    cluster_id_df = pd.DataFrame(all_entity_info)
    cluster_id_df.to_csv(output_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)


def get_recentpdb_lowh_cluster_csv(
    lowh_csv: Path,
    mmcif_dir: Path,
    cluster_txt_file: Path,
    output_cluster_csv: Path,
    n_cpu: int = -1,
):
    """
    Generate cluster information CSV for recent PDB entries with low homology.

    This function processes a predefined set of recent PDB structures with low sequence homology,
    extracting their cluster IDs and saving results to the configured output path.

    Args:
        lowh_csv (Path): Path to the CSV file containing low homology PDB entries
        mmcif_dir (Path): Directory containing MMCIF files for the PDB entries
        cluster_txt_file (Path): Path to the cluster text file
                        example cluster file:
                        https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-40.txt
        output_cluster_csv (Path): Path to save the output CSV with cluster information
        n_cpu (int): Number of CPUs to use for parallel processing (-1 = all available)
    """
    lowh_df = pd.read_csv(lowh_csv, dtype=str)
    pdb_list = list(lowh_df["entry_id"].unique())
    get_cluster_id_from_cif_batch(
        cif_dir=mmcif_dir,
        output_csv=output_cluster_csv,
        cluster_txt_file=cluster_txt_file,
        pdb_ids=pdb_list,
        n_cpu=n_cpu,
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-l",
        "--lowh_csv",
        type=Path,
        default=Path(SUPPORTED_DATA.recentpdb_low_homology),
    )
    argparser.add_argument(
        "-m",
        "--mmcif_dir",
        type=Path,
        default=Path(SUPPORTED_DATA.true_dir),
    )
    argparser.add_argument(
        "-c",
        "--cluster_txt_file",
        type=Path,
        default=Path(SUPPORTED_DATA.pdb_cluster_file),
    )
    argparser.add_argument(
        "-o",
        "--output_csv",
        type=Path,
        default=Path(SUPPORTED_DATA.recentpdb_low_homology_cluster),
    )
    argparser.add_argument(
        "-n",
        "--n_cpu",
        type=int,
        default=-1,
    )

    args = argparser.parse_args()

    get_recentpdb_lowh_cluster_csv(
        lowh_csv=args.lowh_csv,
        mmcif_dir=args.mmcif_dir,
        cluster_txt_file=args.cluster_txt_file,
        output_cluster_csv=args.output_csv,
        n_cpu=args.n_cpu,
    )
