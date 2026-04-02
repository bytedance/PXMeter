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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from biotite.structure import AtomArray
from joblib import delayed, Parallel
from pxmeter.constants import (
    CRYSTALLIZATION_AIDS,
    CRYSTALLIZATION_METHODS,
    DNA,
    GLYCANS,
    IONS,
    LIGAND,
    POLYMER,
    PROTEIN,
    RNA,
    STD_RESIDUES,
)
from pxmeter.data.parser import MMCIFParser
from pxmeter.data.utils import get_unique_chain_id, is_valid_date_format
from scipy.spatial import KDTree
from tqdm import tqdm

from benchmark.configs.data_config import PXM_MMCIF_DIR, SRC_DATA
from benchmark.dataset_pipeline.utils.select_ligand import filter_ligand_by_ccd_info
from benchmark.utils import add_comp_chain_iface_id

NMR_METHODS = {"SOLID-STATE NMR", "SOLUTION NMR"}


def _get_nonpoly_seq(cif_parser: MMCIFParser) -> list[dict[str, str]]:
    nonpoly_scheme_df = cif_parser.get_category_table("pdbx_nonpoly_scheme")

    nonpoly_seq_info = []
    if nonpoly_scheme_df is not None:
        nonpoly_scheme_df = nonpoly_scheme_df[
            ~nonpoly_scheme_df["mon_id"].isin(["HOH", "DOD"])
        ]

        nonpoly_seq_info = []
        for entity_id, sub_df in nonpoly_scheme_df.groupby("entity_id"):
            sub_df = sub_df[sub_df["asym_id"] == sub_df["asym_id"].iloc[0]]
            seq_list = sub_df["mon_id"].tolist()

            if all(i in GLYCANS for i in seq_list):
                entity_type = "glycan"
            elif all(i in IONS for i in seq_list):
                entity_type = "ion"
            else:
                entity_type = LIGAND
            seq = "_".join(seq_list)
            seq_length = len(seq_list)
            nonpoly_seq_info.append(
                {
                    "entity_id": entity_id,
                    "entity_type": entity_type,
                    "seq": seq,
                    "seq_length": seq_length,
                }
            )

    branch_scheme_df = cif_parser.get_category_table("pdbx_branch_scheme")
    if branch_scheme_df is not None:
        for entity_id, sub_df in branch_scheme_df.groupby("entity_id"):
            sub_df = sub_df[sub_df["asym_id"] == sub_df["asym_id"].iloc[0]]
            entity_type = "glycan"
            seq_list = sub_df["mon_id"].tolist()
            seq = "_".join(seq_list)
            seq_length = len(seq_list)
            nonpoly_seq_info.append(
                {
                    "entity_id": entity_id,
                    "entity_type": entity_type,
                    "seq": seq,
                    "seq_length": seq_length,
                }
            )

    return nonpoly_seq_info


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


def get_protein_chains_with_backbone_breaks(
    atom_array: AtomArray,
    entity_poly_type: dict[str, str],
    max_distance: float = 10.0,
) -> list[str]:
    """Return protein chain IDs that contain backbone breaks.

    A backbone break is defined as a CA-CA distance between two consecutive
    residues within the same chain (i.e., same ``chain_id`` and a
    ``res_id`` difference of exactly 1) that is larger than ``max_distance``.

    Args:
        atom_array:
            Biotite ``AtomArray`` containing the structure.
        entity_poly_type:
            Mapping from ``entity_id`` to polymer type string. Only entities
            whose type equals ``PROTEIN`` are considered.
        max_distance:
            Distance threshold in Å. If the squared CA-CA distance between
            consecutive residues exceeds ``max_distance ** 2`` for any pair
            in a chain, that chain is considered to have a backbone break.

    Returns:
        list[str]: A list of chain IDs for protein chains that contain at
        least one backbone break. If no breaks are found, an empty list
        is returned.
    """
    # Collect entity IDs that correspond to protein polymers
    protein_entity_ids = [
        entity_id
        for entity_id, poly_type in entity_poly_type.items()
        if poly_type == PROTEIN
    ]

    # Select CA atoms belonging to protein entities
    protein_ca_array = atom_array[
        (atom_array.atom_name == "CA")
        & np.isin(atom_array.label_entity_id, protein_entity_ids)
    ]

    if protein_ca_array.array_length() < 2:
        return []

    # Sort once by chain_id and res_id, then scan linearly
    order = np.lexsort((protein_ca_array.res_id, protein_ca_array.chain_id))
    ca = protein_ca_array[order]

    chain_ids = ca.chain_id
    res_ids = ca.res_id
    coords = ca.coord  # shape: (N, 3)
    max_distance_sq = max_distance * max_distance

    removed_chain_ids: set[str] = set()

    # Linear scan over adjacent atoms
    for i in range(len(ca) - 1):
        chain_i = chain_ids[i]

        # Skip chains that are already marked as broken
        if chain_i in removed_chain_ids:
            continue

        # Only compare atoms within the same chain
        if chain_i != chain_ids[i + 1]:
            continue

        # Only consider consecutive residues (res_id difference == 1)
        if res_ids[i + 1] - res_ids[i] != 1:
            continue

        # Compute squared CA-CA distance
        diff = coords[i + 1] - coords[i]
        dist_sq = np.dot(diff, diff)
        if dist_sq > max_distance_sq:
            removed_chain_ids.add(chain_i)

    return list(removed_chain_ids)


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


def _lig_is_glycan_or_ion(atom_array: AtomArray, chain_id: str) -> str:
    """
    Determine whether a ligand in a specific chain is an ion or a glycan.

    Args:
        atom_array (AtomArray): Biotite AtomArray object containing atomic information.
        chain_id (str): The ID of the chain to check.

    Returns:
        str: "ion" if the ligand is an ion, "glycan" if it's a glycan, otherwise the value of LIGAND.
    """
    if np.all(
        np.isin(atom_array.res_name[atom_array.chain_id == chain_id], list(IONS))
    ):
        return "ion"
    elif np.all(
        np.isin(atom_array.res_name[atom_array.chain_id == chain_id], list(GLYCANS))
    ):
        return "glycan"
    else:
        return LIGAND


def _filter_for_ligand_chains(
    atom_array: AtomArray,
    entity_poly_type: dict[str, str],
    exptl_methods: tuple[str],
    resolution: float,
    valid_lig_codes: list[str] | None,
    remove_covalent_ligands: bool = True,
) -> list[str]:
    """
    Filter ligand chains from an AtomArray based on experimental method,
    resolution, ligand codes, and occupancy criteria.

    This function applies multiple filters to identify ligand chains that should
    be removed from the structure. It considers experimental methods, resolution
    thresholds, ligand type (glycan, ion, or standard ligand), ligand code
    validity, number of residues, and atom occupancy.

    Args:
        atom_array (AtomArray): Biotite AtomArray object containing atomic
            coordinates and annotations.
        entity_poly_type (dict[str, str]): Mapping from entity IDs to polymer
            types (e.g., "polymer", "ligand"). Used to classify entity types.
        exptl_methods (tuple[str]): Experimental methods reported for the structure
            (e.g., ("X-RAY DIFFRACTION",)).
        resolution (float): Resolution of the structure in Ångströms. Ligands are
            removed if resolution is worse than 2.0 Å.
        valid_lig_codes (list[str] | None): List of valid ligand CCD codes to keep.
            If None, no whitelist is applied.
        remove_covalent_ligands (bool, optional): Whether to remove covalent ligands.
            Defaults to True.

    Returns:
        list[str]: A list of chain IDs corresponding to ligand chains that should
        be removed based on the applied filters.
    """

    remove_all_ligands = False
    if (
        (len(exptl_methods) > 1)
        or (exptl_methods[0] != "X-RAY DIFFRACTION")
        or (resolution > 2.0)
    ):
        remove_all_ligands = True

    covalent_chain_ids = set()
    if remove_covalent_ligands and atom_array.bonds is not None:
        bonds = atom_array.bonds.as_array()
        bond_atom1_chain_ids = atom_array.chain_id[bonds[:, 0]]
        bond_atom2_chain_ids = atom_array.chain_id[bonds[:, 1]]
        inter_chain_bond_mask = bond_atom1_chain_ids != bond_atom2_chain_ids
        covalent_chain_ids = set(bond_atom1_chain_ids[inter_chain_bond_mask]) | set(
            bond_atom2_chain_ids[inter_chain_bond_mask]
        )

    removed_lig_chains = []
    for entity_id in np.unique(atom_array.label_entity_id):
        entity_type = entity_poly_type.get(entity_id, LIGAND)
        if entity_type != LIGAND:
            continue

        chains_in_entity = np.unique(
            atom_array.chain_id[atom_array.label_entity_id == entity_id]
        )
        for chain_id in chains_in_entity:
            first_ccd = atom_array.res_name[atom_array.chain_id == chain_id][0]
            sub_entity_type = _lig_is_glycan_or_ion(atom_array, chain_id)
            if sub_entity_type != LIGAND:
                continue

            if remove_all_ligands:
                removed_lig_chains.append(chain_id)

            elif remove_covalent_ligands and chain_id in covalent_chain_ids:
                removed_lig_chains.append(chain_id)

            elif len(np.unique(atom_array.res_id[atom_array.chain_id == chain_id])) > 1:
                # num of res > 1
                removed_lig_chains.append(chain_id)

            elif valid_lig_codes and (first_ccd not in valid_lig_codes):
                removed_lig_chains.append(chain_id)

            elif np.min(atom_array.occupancy[atom_array.chain_id == chain_id]) < 1.0:
                removed_lig_chains.append(chain_id)

    return removed_lig_chains


def get_chain_and_interface_from_cif(
    cif_path: Path | str,
    model: int = 1,
    altloc: str = "first",
    assembly_id: str | None = "1",
    after_date: str | None = "2022-05-01",
    before_date: str | None = "2023-01-12",
    valid_lig_codes: list[str] | None = None,
    non_nmr_filter: bool = True,
    resolution_threshold: float | None = 4.5,
    num_token_threshold: int | None = 2560,
    std_polymer_only: bool = True,
    max_polymer_copies_threshold: int | None = 20,
    interface_radius: float = 5.0,
    resolved_ratio_threshold: float = 0.3,
    min_resolved_seq_length_threshold: int = 4,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Process a PDBx/mmCIF file and extract chain/interface information with filtering.

    Args:
        cif_path: Path to the mmCIF file
        model: Model number to use from the structure
        altloc: Alternate location selection strategy
        assembly_id: Biological assembly ID to load
        after_date: Only include entries released after this date (YYYY-MM-DD)
        before_date: Only include entries released before this date (YYYY-MM-DD)
        valid_lig_codes: a list of valid ligand CCD codes
        non_nmr_filter: Exclude NMR structures if True
        resolution_threshold: Max resolution to include (for X-ray/cryo-EM)
        num_token_threshold: Max token count to include (based on AlphaFold3 tokenization rules)
        std_polymer_only: Only include standard polymer
                         (polypeptide(L)/polydeoxyribonucleotide/polyribonucleotide)
                         and non-polymer entities (ligand)
        max_polymer_copies_threshold : Max number of copies for each polymer entity
        interface_radius: Distance threshold for interface detection
        resolved_ratio_threshold: Min ratio of resolved residues to total residues for polymer
        min_resolved_seq_length_threshold: Min resolved sequence length for polymer

    Returns:
        Tuple containing metadata dict, list of chain/interface info dicts, and list of sequence info dicts
    """

    assert after_date is None or is_valid_date_format(
        after_date
    ), f"Invalid date format: {after_date}, it should be yyyy-mm-dd format"
    assert before_date is None or is_valid_date_format(
        before_date
    ), f"Invalid date format: {before_date}, it should be yyyy-mm-dd format"

    cif_parser = MMCIFParser(cif_path)
    entity_poly_type = cif_parser.entity_poly_type
    entity_poly_seq = cif_parser.get_entity_poly_seq()

    # template of return value
    meta_info = {
        "entry_id": cif_parser.entry_id,
        "exptl_methods": "",
        "classification": "",
        "release_date": "9999-12-31",
        "resolution": -1,
        "num_tokens": -1,
        "max_polymer_chain_copies": -1,
        "no_standard_polymer": False,
        "lacking_resolved": False,
        "all_chains_unk": False,
        "all_chains_break": False,
        "pass_filter": False,
    }
    chain_interface_info_list = []  # list[dict]

    release_date = MMCIFParser.get_release_date(cif_parser.cif.block)
    meta_info["release_date"] = release_date

    exptl_methods = tuple(cif_parser.exptl_methods)
    meta_info["exptl_methods"] = ";".join(list(cif_parser.exptl_methods))

    resolution = MMCIFParser.get_resolution(cif_parser.cif.block)
    meta_info["resolution"] = resolution

    meta_info["classification"] = cif_parser.cif.block["struct_keywords"][
        "pdbx_keywords"
    ].as_item()

    sequence_info = []  # list[dict]
    for entity_id, seq in entity_poly_seq.items():
        sequence_info.append(
            {
                "entry_id": cif_parser.entry_id,
                "entity_id": entity_id,
                "release_date": release_date,
                "entity_type": entity_poly_type[entity_id],
                "seq": seq,
                "seq_length": len(seq),
            }
        )

    nonpoly_seq_info = _get_nonpoly_seq(cif_parser)
    for nonpoly_seq in nonpoly_seq_info:
        nonpoly_seq_info = {
            "entry_id": cif_parser.entry_id,
            "entity_id": nonpoly_seq["entity_id"],
            "release_date": release_date,
            "entity_type": nonpoly_seq["entity_type"],
            "seq": nonpoly_seq["seq"],
            "seq_length": nonpoly_seq["seq_length"],
        }
        sequence_info.append(nonpoly_seq_info)

    valid_date = True
    if after_date is not None:
        valid_date = valid_date and release_date >= after_date
    if before_date is not None:
        valid_date = valid_date and release_date <= before_date

    if not valid_date:
        # Filter by release date, return empty list
        return meta_info, chain_interface_info_list, sequence_info

    if non_nmr_filter and NMR_METHODS.intersection(set(exptl_methods)):
        # Filter to non-NMR methods, return empty list
        return meta_info, chain_interface_info_list, sequence_info

    if resolution_threshold is not None:
        if not 0 < resolution < resolution_threshold:
            # Filter by resolution, return empty list
            return meta_info, chain_interface_info_list, sequence_info

    atom_array = cif_parser.get_structure(
        model=model,
        altloc=altloc,
        assembly_id=assembly_id,
        include_bonds=True,
    )

    # Remove water and hydrogen
    atom_array = atom_array[~np.isin(atom_array.element, ["H", "D"])]
    atom_array = atom_array[~np.isin(atom_array.res_name, ["HOH", "DOD"])]

    if set(exptl_methods) & CRYSTALLIZATION_METHODS:
        # only remove aids in non-polymer residues
        non_polymer_mask = ~np.isin(atom_array.label_entity_id, entity_poly_seq.keys())
        crys_aids_mask = np.isin(atom_array.res_name, CRYSTALLIZATION_AIDS)
        atom_array = atom_array[~(non_polymer_mask & crys_aids_mask)]

    # Reset chain id for assembly
    atom_array.set_annotation("chain_id", get_unique_chain_id(atom_array))

    num_tokens = calc_num_tokens(atom_array, entity_poly_seq)
    meta_info["num_tokens"] = num_tokens
    if num_token_threshold is not None:
        if num_tokens > num_token_threshold:
            # Filter by number of tokens, return empty list
            return meta_info, chain_interface_info_list, sequence_info

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
            return meta_info, chain_interface_info_list, sequence_info

    # Filter by max number of polymer chains in a entity
    max_copies = 0
    for label_entity_id in entity_poly_type:
        num_chains = len(
            np.unique(
                atom_array.chain_id[atom_array.label_entity_id == label_entity_id]
            )
        )
        max_copies = max(max_copies, num_chains)

    meta_info["max_polymer_chain_copies"] = max_copies
    if max_polymer_copies_threshold is not None:
        if max_copies > max_polymer_copies_threshold:
            return meta_info, chain_interface_info_list, sequence_info

    # Filter to chains with all unknown residues
    all_unk_chains = find_all_unk_chains(atom_array, entity_poly_type)
    atom_array = atom_array[~np.isin(atom_array.chain_id, all_unk_chains)]
    if len(atom_array) == 0:
        meta_info["all_chains_unk"] = True
        return meta_info, chain_interface_info_list, sequence_info

    # Filter by breaks in protein backbone
    break_chains = get_protein_chains_with_backbone_breaks(
        atom_array, entity_poly_type, max_distance=5.0
    )
    atom_array = atom_array[~np.isin(atom_array.chain_id, break_chains)]
    if len(atom_array) == 0:
        meta_info["all_chains_break"] = True
        return meta_info, chain_interface_info_list, sequence_info

    chain_id_to_seq_length = {}
    chain_id_to_resolved_seq_length = {}
    chain_id_to_auth_chain_id = {}
    lack_resolved_chain_ids = []
    for chain_id in np.unique(atom_array.chain_id):
        chain_mask = atom_array.chain_id == chain_id
        label_entity_id = atom_array.label_entity_id[chain_mask][0]
        auth_chain_id = atom_array.auth_asym_id[chain_mask][0]

        entity_type = entity_poly_type.get(label_entity_id, LIGAND)
        if entity_type != LIGAND:
            seq_length = len(entity_poly_seq[label_entity_id])
            resolved_seq_length = len(np.unique(atom_array.res_id[chain_mask]))
        else:
            seq_length = len(np.unique(atom_array.res_id[chain_mask]))
            resolved_seq_length = seq_length

        chain_id_to_seq_length[chain_id] = seq_length
        chain_id_to_resolved_seq_length[chain_id] = resolved_seq_length
        chain_id_to_auth_chain_id[chain_id] = auth_chain_id

        resolved_ratio = resolved_seq_length / seq_length
        if (
            resolved_seq_length < min_resolved_seq_length_threshold
            or resolved_ratio < resolved_ratio_threshold
        ) and entity_type != LIGAND:
            lack_resolved_chain_ids.append(chain_id)

    # Filter by resolved ratio
    atom_array = atom_array[~np.isin(atom_array.chain_id, lack_resolved_chain_ids)]
    if len(atom_array) == 0:
        meta_info["lacking_resolved"] = True
        return meta_info, chain_interface_info_list, sequence_info

    # Filter for ligand
    removed_lig_chains = _filter_for_ligand_chains(
        atom_array,
        entity_poly_type,
        exptl_methods,
        resolution,
        valid_lig_codes,
    )

    meta_info["pass_filter"] = True

    # Append chain info
    for chain_id in np.unique(atom_array.chain_id):
        entity_id = atom_array.label_entity_id[atom_array.chain_id == chain_id][0]
        seq_length = chain_id_to_seq_length[chain_id]
        entity_type = entity_poly_type.get(entity_id, LIGAND)

        if entity_type == LIGAND:
            if "." in chain_id:
                # Retain only ligands within the asymmetric unit.
                continue
            entity_type = _lig_is_glycan_or_ion(atom_array, chain_id)
            if entity_type == LIGAND and chain_id in removed_lig_chains:
                continue

        chain_info = {
            "type": "chain",
            "entry_id": cif_parser.entry_id,
            "entity_id_1": entity_id,
            "entity_id_2": "",
            "entity_type_1": entity_type,
            "entity_type_2": "",
            "chain_id_1": chain_id,
            "chain_id_2": "",
            "auth_chain_id_1": chain_id_to_auth_chain_id[chain_id],
            "auth_chain_id_2": "",
            "seq_length_1": seq_length,
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
            seq_length_1 = chain_id_to_seq_length[chain_id_1]
            seq_length_2 = chain_id_to_seq_length[chain_id_2]

            if (entity_type_1 == LIGAND or entity_type_2 == LIGAND) and (
                "." in chain_id_1 or "." in chain_id_2
            ):
                # Retain only ligand interfaces between chains within the asymmetric unit.
                continue

            if entity_type_1 == LIGAND:
                entity_type_1 = _lig_is_glycan_or_ion(atom_array, chain_id_1)
                if entity_type_1 == LIGAND and chain_id_1 in removed_lig_chains:
                    continue
            if entity_type_2 == LIGAND:
                entity_type_2 = _lig_is_glycan_or_ion(atom_array, chain_id_2)
                if entity_type_2 == LIGAND and chain_id_2 in removed_lig_chains:
                    continue

            interface_info = {
                "type": "interface",
                "entry_id": cif_parser.entry_id,
                "entity_id_1": entity_id_1,
                "entity_id_2": entity_id_2,
                "entity_type_1": entity_type_1,
                "entity_type_2": entity_type_2,
                "chain_id_1": chain_id_1,
                "chain_id_2": chain_id_2,
                "auth_chain_id_1": chain_id_to_auth_chain_id[chain_id_1],
                "auth_chain_id_2": chain_id_to_auth_chain_id[chain_id_2],
                "seq_length_1": seq_length_1,
                "seq_length_2": seq_length_2,
                "resolved_seq_length_1": chain_id_to_resolved_seq_length[chain_id_1],
                "resolved_seq_length_2": chain_id_to_resolved_seq_length[chain_id_2],
            }
            chain_interface_info_list.append(interface_info)
    return meta_info, chain_interface_info_list, sequence_info


def filter_recentpdb_entry(
    mmcif_dir: Path,
    output_meta_csv: Path,
    output_chain_interface_csv: Path,
    output_seq_csv: Path,
    after_date: str,
    before_date: str,
    assembly_id: str | None = "1",
    n_cpu: int = -1,
):
    """
    Process a batch of mmCIF files to filter and extract chain/interface information,
    then save results to CSV files.

    Args:
        mmcif_dir: Directory containing input mmCIF files
        output_meta_csv: Output path for metadata CSV (entry-level statistics)
        output_chain_interface_csv: Output path for chain/interface CSV (detailed interactions)
        output_seq_csv: Output path for sequence CSV (detailed sequence information)
        pdb_ids: Optional list of specific PDB IDs to process (if None, process all .cif files)
        assembly_id: Biological assembly ID to load (default: "1")
        n_cpu: Number of CPUs to use for parallel processing (-1 = all available)
    """
    assert mmcif_dir.exists(), f"mmCIF directory {mmcif_dir} does not exist."

    # Get all valid ligand codes
    valid_lig_codes = filter_ligand_by_ccd_info(n_cpu=n_cpu)

    all_cif_paths = list(mmcif_dir.glob("*.cif"))
    random.seed(42)
    random.shuffle(all_cif_paths)

    results = [
        r
        for r in tqdm(
            Parallel(n_jobs=n_cpu, return_as="generator_unordered", batch_size=128)(
                delayed(get_chain_and_interface_from_cif)(
                    cif_path,
                    after_date=after_date,
                    before_date=before_date,
                    assembly_id=assembly_id,
                    valid_lig_codes=valid_lig_codes,
                )
                for cif_path in all_cif_paths
            ),
            total=len(all_cif_paths),
            desc="Filter recent PDB entries",
        )
    ]

    all_meta_info = []
    all_chain_interface_info_list = []
    all_seq_info = []
    for meta_info, chain_interface_info_list, seq_info in results:
        all_meta_info.append(meta_info)
        all_chain_interface_info_list.extend(chain_interface_info_list)
        all_seq_info.extend(seq_info)

    output_meta_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_meta_info).to_csv(
        output_meta_csv, index=False, quoting=csv.QUOTE_NONNUMERIC
    )
    all_chain_interface_info_df = pd.DataFrame(all_chain_interface_info_list)
    all_chain_interface_info_df = add_comp_chain_iface_id(all_chain_interface_info_df)
    output_chain_interface_csv.parent.mkdir(parents=True, exist_ok=True)
    all_chain_interface_info_df.to_csv(
        output_chain_interface_csv, index=False, quoting=csv.QUOTE_NONNUMERIC
    )

    seq_df = pd.DataFrame(all_seq_info)
    seq_df["seq_length"] = seq_df["seq_length"].fillna(-1).astype(int)
    output_seq_csv.parent.mkdir(parents=True, exist_ok=True)
    seq_df.to_csv(output_seq_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-c",
        "--cif_dir",
        type=Path,
        default=PXM_MMCIF_DIR,
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
        "-s",
        "--seq_csv",
        type=Path,
        default=Path(SRC_DATA.pdb_seq_csv),
    )
    arg_parser.add_argument(
        "-a",
        "--after_date",
        type=str,
    )
    arg_parser.add_argument(
        "-b",
        "--before_date",
        type=str,
    )

    arg_parser.add_argument(
        "-n",
        "--n_cpu",
        type=int,
        default=-1,
    )
    args = arg_parser.parse_args()

    filter_recentpdb_entry(
        mmcif_dir=args.cif_dir,
        output_meta_csv=args.meta_csv,
        output_chain_interface_csv=args.chain_interface_csv,
        output_seq_csv=args.seq_csv,
        after_date=args.after_date,
        before_date=args.before_date,
        n_cpu=args.n_cpu,
    )
