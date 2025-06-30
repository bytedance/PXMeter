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
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from biotite.structure import AtomArray
from joblib import delayed, Parallel
from pxmeter.constants import DNA, LIGAND, POLYMER, PROTEIN, RNA, STD_RESIDUES
from pxmeter.data.parser import MMCIFParser
from pxmeter.data.utils import get_unique_chain_id
from scipy.spatial import KDTree
from tqdm import tqdm

from benchmark.configs.data_config import SRC_DATA, SUPPORTED_DATA

NMR_METHODS = {"SOLID-STATE NMR", "SOLUTION NMR"}


def is_valid_date_format(date_string: str) -> bool:
    """
    Check if the date string is in the format yyyy-mm-dd.

    Args:
        date_string (str): The date string to check.

    Returns:
        bool: True if the date string is in the format yyyy-mm-dd, False otherwise.
    """
    try:
        datetime.strptime(date_string, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def get_release_date(cif_block: dict) -> str:
    """
    Get first release date.

    Args:
        cif_block (dict): mmcif_io.MMCIFParser dictionary

    Returns:
        str: yyyy-mm-dd
    """

    if "pdbx_audit_revision_history" in cif_block:
        history = cif_block["pdbx_audit_revision_history"]
        # np.str_ is inherit from str, so return is str
        date = history["revision_date"].as_array()[0]
    else:
        # no release date
        date = "9999-12-31"

    valid_date = is_valid_date_format(date)
    assert valid_date, f"Invalid date format: {date}, it should be yyyy-mm-dd format"
    return date


def get_resolution(cif_block: dict) -> float:
    """
    Get resolution for X-ray and cryoEM.
    Some methods don't have resolution, set as -1.0

    Args:
        cif_block (dict): mmcif_io.MMCIFParser dictionary

    Returns:
        float: resolution (set to -1.0 if not found)
    """
    resolution_names = [
        "refine.ls_d_res_high",
        "em_3d_reconstruction.resolution",
        "reflns.d_resolution_high",
    ]
    for category_item in resolution_names:
        category, item = category_item.split(".")
        if category in cif_block and item in cif_block[category]:
            try:
                resolution = cif_block[category][item].as_array(float)[0]
                # "." will be converted to 0.0, but it is not a valid resolution.
                if resolution == 0.0:
                    continue
                return resolution
            except ValueError:
                # in some cases, resolution_str is "?"
                continue
        else:
            continue
    return -1.0


def find_all_unk_chains(
    atom_array: AtomArray, entity_poly_type: dict[str, str]
) -> list[str]:
    """
    Find chains with all unknown residues in the AtomArray.

    Args:
        atom_array (AtomArray): The AtomArray to search.
        entity_poly_type (dict[str, str]): The "label_entity_id" to the "entity_poly.type" dictionary.

    Returns:
        list[str]: A list of chain IDs with all unknown residues
    """
    all_unk_chains = []
    for chain_id in np.unique(atom_array.chain_id):
        chain_mask = atom_array.chain_id == chain_id
        label_entity_id = atom_array.label_entity_id[chain_mask][0]
        entity_type = entity_poly_type.get(label_entity_id, LIGAND)
        if entity_type == PROTEIN:
            if np.all(atom_array.res_name[chain_mask] == "UNK"):
                all_unk_chains.append(chain_id)
        elif entity_type == DNA:
            if np.all(atom_array.res_name[chain_mask] == "DN"):
                all_unk_chains.append(chain_id)
        elif entity_type == RNA:
            if np.all(atom_array.res_name[chain_mask] == "N"):
                all_unk_chains.append(chain_id)
        elif entity_type == LIGAND:
            if np.all(atom_array.res_name[chain_mask] == "UNL"):
                all_unk_chains.append(chain_id)
        else:
            continue
    return all_unk_chains


def calc_num_tokens(atom_array: AtomArray, entity_poly_seq: dict[str, str]) -> int:
    """
    Ref: AlphaFold3 SI Chapter 2.6
        • A standard amino acid residue (Table 13) is represented as a single token.
        • A standard nucleotide residue (Table 13) is represented as a single token.
        • A modified amino acid or nucleotide residue is tokenized per-atom (i.e. N tokens for an N-atom residue)
        • All ligands are tokenized per-atom

    For each token we also designate a token centre atom, used in various places below:
        • Cα for standard amino acids
        • C1′ for standard nucleotides
        • For other cases take the first and only atom as they are tokenized per-atom.

    Args:
        atom_array (AtomArray): Biotite AtomArray object
        entity_poly_seq (dict[str, str]): The "label_entity_id" to the sequence dictionary.

    Returns:
        int: The number of tokens in the AtomArray.
    """
    num_tokens = 0
    for label_entity_id in np.unique(atom_array.label_entity_id):
        if label_entity_id in entity_poly_seq:
            seq_length = len(entity_poly_seq[label_entity_id])
            chain_ids = np.unique(
                atom_array.chain_id[atom_array.label_entity_id == label_entity_id]
            )
            chain_num = len(chain_ids)
            unstd_res_in_first_chain_mask = (
                ~np.isin(atom_array.res_name, STD_RESIDUES)
            ) & (atom_array.chain_id == chain_ids[0])
            num_unstd_res_in_first_chain = len(
                np.unique(atom_array.res_id[unstd_res_in_first_chain_mask])
            )
            num_tokens_unstd_res_in_first_chain = unstd_res_in_first_chain_mask.sum()
            num_tokens += (
                seq_length
                + num_tokens_unstd_res_in_first_chain
                - num_unstd_res_in_first_chain
            ) * chain_num

        else:
            # ligand
            num_tokens += (atom_array.label_entity_id == label_entity_id).sum()
    return num_tokens


def find_interfaces(
    atom_array: AtomArray,
    radius: float = 5.0,
    keep_all_entity_chain_pair: bool = True,
) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """
    Find interface between chains of atom_array.

    Args:
        atom_array (AtomArray): Biotite AtomArray object.
        radius (float, optional): Interface radius. Defaults to 5.0.
        keep_all_entity_chain_pair (bool, optional): Whether to keep all chain pairs. Defaults to True.

    Returns:
        tuple:
            dict[tuple[str, str], list[tuple[str, str]]]: entity pair to chain pairs.
                                                        Only include chains in asym unit
                                                        and interfaces which at least have
                                                        one chain in asym unit.
    """
    chain_id_to_entity = {
        chain_id: atom_array.label_entity_id[chain_start]
        for chain_id, chain_start in zip(
            *np.unique(atom_array.chain_id, return_index=True)
        )
    }

    kdtree = KDTree(atom_array.coord)
    entity_pair_to_chain_pairs = defaultdict(list)
    for chain_i in np.unique(atom_array.chain_id):
        entity_i = chain_id_to_entity[chain_i]

        chain_mask = atom_array.chain_id == chain_i
        chain_coord = atom_array.coord[chain_mask]
        neighbors_indices = np.unique(
            np.concatenate(kdtree.query_ball_point(chain_coord, r=radius))
        )
        for chain_j in np.unique(atom_array.chain_id[neighbors_indices]):
            if chain_i == chain_j:
                continue

            entity_j = chain_id_to_entity[chain_j]

            # Sort by entity pair
            sorted_pairs = sorted(
                list(zip([entity_i, entity_j], [chain_i, chain_j])),
                key=lambda x: x[0],
            )
            entity_key, chain_pair = zip(*sorted_pairs)

            exists_chain_pair = entity_pair_to_chain_pairs.get(entity_key, [])
            if (chain_i, chain_j) in exists_chain_pair or (
                chain_j,
                chain_i,
            ) in exists_chain_pair:
                continue

            if "." in chain_i and "." in chain_j and not keep_all_entity_chain_pair:
                # skip if neither chain_i or chain_j is not in asym unit
                continue
            entity_pair_to_chain_pairs[entity_key].append(chain_pair)
    return entity_pair_to_chain_pairs


def get_chain_and_interface_from_cif(
    cif_path: Path | str,
    model: int = 1,
    altloc: str = "first",
    assembly_id: str | None = "1",
    after_date: str | None = "2022-05-01",
    before_date: str | None = "2023-01-12",
    non_nmr_filter: bool = True,
    resolution_threshold: float | None = 4.5,
    num_token_threshold: int | None = 2560,
    std_polymer_only: bool = True,
    max_copies_threshold: int | None = None,
    interface_radius: float = 5.0,
    resolved_ratio_threshold: float = 0.3,
    min_resolved_seq_length_threshold: int = 4,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Process a PDBx/mmCIF file and extract chain/interface information with filtering.

    Args:
        cif_path: Path to the mmCIF file
        model: Model number to use from the structure
        altloc: Alternate location selection strategy
        assembly_id: Biological assembly ID to load
        after_date: Only include entries released after this date (YYYY-MM-DD)
        before_date: Only include entries released before this date (YYYY-MM-DD)
        non_nmr_filter: Exclude NMR structures if True
        resolution_threshold: Max resolution to include (for X-ray/cryo-EM)
        num_token_threshold: Max token count to include (based on AlphaFold3 tokenization rules)
        std_polymer_only: Only include standard polymer
                         (polypeptide(L)/polydeoxyribonucleotide/polyribonucleotide)
                         and non-polymer entities (ligand)
        max_copies_threshold : Max number of copies for each entity
        interface_radius: Distance threshold for interface detection
        resolved_ratio_threshold: Min ratio of resolved residues to total residues for polymer
        min_resolved_seq_length_threshold: Min resolved sequence length for polymer

    Returns:
        Tuple containing metadata dict and list of chain/interface info dicts
    """

    assert after_date is None or is_valid_date_format(
        after_date
    ), f"Invalid date format: {after_date}, it should be yyyy-mm-dd format"
    assert before_date is None or is_valid_date_format(
        before_date
    ), f"Invalid date format: {before_date}, it should be yyyy-mm-dd format"

    cif_parser = MMCIFParser(cif_path)

    # template of return value
    meta_info = {
        "entry_id": cif_parser.entry_id,
        "exptl_methods": "",
        "release_date": "9999-12-31",
        "resolution": -1,
        "num_tokens": -1,
        "no_standard_polymer": False,
        "max_chain_copies": -1,
        "lacking_resolved": False,
        "all_chains_unk": False,
        "pass_filter": False,
    }
    chain_interface_info_list = []  # list[dict]

    release_date = get_release_date(cif_parser.cif.block)
    meta_info["release_date"] = release_date

    exptl_methods = tuple(cif_parser.exptl_methods)
    meta_info["exptl_methods"] = ";".join(list(cif_parser.exptl_methods))

    resolution = get_resolution(cif_parser.cif.block)
    meta_info["resolution"] = resolution

    valid_date = True
    if after_date is not None:
        valid_date = valid_date and release_date >= after_date
    if before_date is not None:
        valid_date = valid_date and release_date <= before_date

    if not valid_date:
        # Filter by release date, return empty list
        return meta_info, chain_interface_info_list

    if non_nmr_filter and NMR_METHODS.intersection(set(exptl_methods)):
        # Filter to non-NMR methods, return empty list
        return meta_info, chain_interface_info_list

    if resolution_threshold is not None:
        if not 0 < resolution < resolution_threshold:
            # Filter by resolution, return empty list
            return meta_info, chain_interface_info_list

    entity_poly_type = cif_parser.entity_poly_type
    entity_poly_seq = cif_parser.get_entity_poly_seq()

    atom_array = cif_parser.get_structure(
        model=model,
        altloc=altloc,
        assembly_id=assembly_id,
        include_bonds=True,
    )

    # Remove water and hydrogen
    atom_array = atom_array[~np.isin(atom_array.element, ["H", "D"])]
    atom_array = atom_array[~np.isin(atom_array.res_name, ["HOH", "DOD"])]

    # Reset chain id for assembly
    atom_array.set_annotation("chain_id", get_unique_chain_id(atom_array))

    num_tokens = calc_num_tokens(atom_array, entity_poly_seq)
    meta_info["num_tokens"] = num_tokens
    if num_token_threshold is not None:
        if num_tokens > num_token_threshold:
            # Filter by number of tokens, return empty list
            return meta_info, chain_interface_info_list

    if std_polymer_only:
        std_polymer_entities = [k for k, v in entity_poly_type.items() if v in POLYMER]
        non_polymer_entity_mask = ~np.isin(
            atom_array.label_entity_id, list(entity_poly_type.keys())
        )
        atom_array = atom_array[
            np.isin(atom_array.label_entity_id, std_polymer_entities)
            | non_polymer_entity_mask
        ]
        if len(atom_array) == 0:
            meta_info["no_standard_polymer"] = True
            return meta_info, chain_interface_info_list

    # Filter by max number of chains in a entity
    max_copies = 0
    for label_entity_id in np.unique(atom_array.label_entity_id):
        num_chains = len(
            np.unique(
                atom_array.chain_id[atom_array.label_entity_id == label_entity_id]
            )
        )
        max_copies = max(max_copies, num_chains)

    meta_info["max_chain_copies"] = max_copies
    if max_copies_threshold is not None:
        if max_copies > max_copies_threshold:
            return meta_info, chain_interface_info_list

    # Filter to chains with all unknown residues
    all_unk_chains = find_all_unk_chains(atom_array, entity_poly_type)
    atom_array = atom_array[~np.isin(atom_array.chain_id, all_unk_chains)]
    if len(atom_array) == 0:
        meta_info["all_chains_unk"] = True
        return meta_info, chain_interface_info_list

    chain_id_to_seq_length = {}
    chain_id_to_resolved_seq_length = {}
    lack_resolved_chain_ids = []
    for chain_id in np.unique(atom_array.chain_id):
        chain_mask = atom_array.chain_id == chain_id
        label_entity_id = atom_array.label_entity_id[atom_array.chain_id == chain_id][0]
        entity_type = entity_poly_type.get(label_entity_id, LIGAND)
        if entity_type != LIGAND:
            seq_length = len(entity_poly_seq[label_entity_id])
            resolved_seq_length = len(np.unique(atom_array.res_id[chain_mask]))
        else:
            seq_length = len(np.unique(atom_array.res_id[chain_mask]))
            resolved_seq_length = seq_length
        chain_id_to_seq_length[chain_id] = seq_length
        chain_id_to_resolved_seq_length[chain_id] = resolved_seq_length

        resolved_ratio = resolved_seq_length / seq_length
        if (
            resolved_seq_length < min_resolved_seq_length_threshold
            or resolved_ratio < resolved_ratio_threshold
        ):
            lack_resolved_chain_ids.append(chain_id)

    # Filter by resolved ratio
    atom_array = atom_array[~np.isin(atom_array.chain_id, lack_resolved_chain_ids)]
    if len(atom_array) == 0:
        meta_info["lacking_resolved"] = True
        return meta_info, chain_interface_info_list

    # Append chain info
    for chain_id in np.unique(atom_array.chain_id):
        entity_id = atom_array.label_entity_id[atom_array.chain_id == chain_id][0]
        chain_info = {
            "type": "chain",
            "entry_id": cif_parser.entry_id,
            "entity_id_1": entity_id,
            "entity_id_2": "",
            "entity_type_1": entity_poly_type.get(entity_id, LIGAND),
            "entity_type_2": "",
            "chain_id_1": chain_id,
            "chain_id_2": "",
            "seq_length_1": chain_id_to_seq_length[chain_id],
            "seq_length_2": -1,
            "resolved_seq_length_1": chain_id_to_resolved_seq_length[chain_id],
            "resolved_seq_length_2": -1,
        }
        chain_interface_info_list.append(chain_info)

    # Append interface info
    entity_pair_to_chain_pairs = find_interfaces(
        atom_array, radius=interface_radius, keep_all_entity_chain_pair=False
    )
    for entity_pair, chain_pairs in entity_pair_to_chain_pairs.items():
        entity_id_1, entity_id_2 = entity_pair
        entity_type_1 = entity_poly_type.get(entity_id_1, LIGAND)
        entity_type_2 = entity_poly_type.get(entity_id_2, LIGAND)
        for chain_id_1, chain_id_2 in chain_pairs:
            interface_info = {
                "type": "interface",
                "entry_id": cif_parser.entry_id,
                "entity_id_1": entity_id_1,
                "entity_id_2": entity_id_2,
                "entity_type_1": entity_type_1,
                "entity_type_2": entity_type_2,
                "chain_id_1": chain_id_1,
                "chain_id_2": chain_id_2,
                "seq_length_1": chain_id_to_seq_length[chain_id_1],
                "seq_length_2": chain_id_to_seq_length[chain_id_2],
                "resolved_seq_length_1": chain_id_to_resolved_seq_length[chain_id_1],
                "resolved_seq_length_2": chain_id_to_resolved_seq_length[chain_id_2],
            }
            chain_interface_info_list.append(interface_info)
    meta_info["pass_filter"] = True
    return meta_info, chain_interface_info_list


def filter_recentpdb_entry(
    mmcif_dir: Path,
    output_meta_csv: Path,
    output_chain_interface_csv: Path,
    pdb_ids: list[str] | None = None,
    n_cpu: int = -1,
):
    """
    Process a batch of mmCIF files to filter and extract chain/interface information,
    then save results to CSV files.

    Args:
        mmcif_dir: Directory containing input mmCIF files
        output_meta_csv: Output path for metadata CSV (entry-level statistics)
        output_chain_interface_csv: Output path for chain/interface CSV (detailed interactions)
        pdb_ids: Optional list of specific PDB IDs to process (if None, process all .cif files)
        n_cpu: Number of CPUs to use for parallel processing (-1 = all available)
    """
    if pdb_ids is None:
        all_cif_paths = list(mmcif_dir.glob("*.cif"))
        random.seed(42)
        random.shuffle(all_cif_paths)
    else:
        all_cif_paths = [mmcif_dir / f"{i}.cif" for i in pdb_ids]

    results = [
        r
        for r in tqdm(
            Parallel(n_jobs=n_cpu, return_as="generator_unordered")(
                delayed(get_chain_and_interface_from_cif)(cif_path)
                for cif_path in all_cif_paths
            ),
            total=len(all_cif_paths),
        )
    ]

    all_meta_info = []
    all_chain_interface_info_list = []
    for meta_info, chain_interface_info_list in results:
        all_meta_info.append(meta_info)
        all_chain_interface_info_list.extend(chain_interface_info_list)

    pd.DataFrame(all_meta_info).to_csv(
        output_meta_csv, index=False, quoting=csv.QUOTE_NONNUMERIC
    )
    pd.DataFrame(all_chain_interface_info_list).to_csv(
        output_chain_interface_csv, index=False, quoting=csv.QUOTE_NONNUMERIC
    )


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-c",
        "--cif_dir",
        type=Path,
        default=Path(SUPPORTED_DATA.true_dir),
    )
    arg_parser.add_argument(
        "-m", "--meta_csv", type=Path, default=Path(SRC_DATA.pdb_meta_info)
    )
    arg_parser.add_argument(
        "-o",
        "--chain_interface_csv",
        type=Path,
        default=Path(SRC_DATA.recentpdb_chain_interface_csv),
    )
    arg_parser.add_argument(
        "-p",
        "--pdb_ids_txt",
        type=Path,
        default=None,
        help="Path to a text file containing a list of PDB IDs to process. \
        Each ID should be on a new line. ",
    )
    arg_parser.add_argument(
        "-n",
        "--n_cpu",
        type=int,
        default=-1,
    )
    args = arg_parser.parse_args()

    if args.pdb_ids_txt is not None:
        with open(args.pdb_ids_txt, "r") as f:
            pdb_ids_list = [line.strip() for line in f.readlines()]
    else:
        pdb_ids_list = None

    filter_recentpdb_entry(
        mmcif_dir=args.cif_dir,
        output_meta_csv=args.meta_csv,
        output_chain_interface_csv=args.chain_interface_csv,
        pdb_ids=pdb_ids_list,
        n_cpu=args.n_cpu,
    )
