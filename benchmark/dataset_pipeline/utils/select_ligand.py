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

import logging
import time
from pathlib import Path
from typing import Any

import biotite.structure.io.pdbx as pdbx
import numpy as np
import pandas as pd
import requests
from biotite.structure import repeat_box
from biotite.structure.info import get_ccd, get_from_ccd
from joblib import delayed, Parallel

from pxmeter.constants import LIGAND
from pxmeter.data.utils import get_unique_chain_id
from scipy.spatial import KDTree
from tqdm import tqdm

GRAPHQL_ENDPOINT = "https://data.rcsb.org/graphql"

ALLOWED_ELEM = ["H", "C", "O", "N", "P", "S", "F", "CL"]


def _check_ccd_valid_for_ligand_eval(
    ccd_code: str,
) -> str | None:
    """
    Validate a CCD (Chemical Component Dictionary) block for ligand evaluation.

    This function applies several chemical criteria to determine whether a
    chemical component is suitable for ligand evaluation. It checks molecular
    weight, allowed atom types, and the minimum number of heavy atoms.

    Args:
        ccd_code (str): The CCD code (component ID) of the chemical component.

    Returns:
        str | None: The CCD code (component ID) if the ligand passes all filters,
        otherwise None.

    Validation Criteria:
        - Molecular weight between 100 Da and 900 Da.
        - At least 3 heavy atoms (non-hydrogen).
        - All atoms must belong to the allowed set: {H, C, O, N, P, S, F, Cl}.
    """
    # Ligands weighing from 100 Da to 900 Da
    try:
        mol_weight = int(
            get_from_ccd("chem_comp", ccd_code, "formula_weight").as_item()
        )
    except ValueError:
        return
    if mol_weight < 100 or mol_weight > 900:
        return

    # Ligands with at least 3 heavy atoms
    # Ligands containing only H, C, O, N, P, S, F, Cl atoms
    symbols = get_from_ccd("chem_comp_atom", ccd_code, "type_symbol").as_array()
    valid_symbols = np.all(np.isin(symbols, ALLOWED_ELEM)) & (
        (symbols != "H").sum() >= 3
    )
    if not valid_symbols:
        return

    return ccd_code


def filter_ligand_by_ccd_info(n_cpu: int = -1) -> list[str]:
    """
    Filter CCD ligand components based on predefined chemical criteria.

    This function iterates over all CCD blocks from `CCD_BLOCKS`, validates them
    in parallel using `_check_ccd_valid_for_ligand_eval`, and collects the set of
    valid ligand codes for downstream evaluation.

    Args:
        n_cpu (int, optional): Number of CPUs to use for parallel processing.
            Defaults to -1, which uses all available CPUs.

    Returns:
        list[str]: A list of validated CCD codes (ligand identifiers) that pass
        the filtering criteria.
    """
    all_ccd = get_ccd()  # BinaryCIFBlock
    chem_comp = all_ccd["chem_comp"]
    codes = chem_comp["id"].as_array()

    results = [
        r
        for r in tqdm(
            Parallel(n_jobs=n_cpu, return_as="generator_unordered")(
                delayed(_check_ccd_valid_for_ligand_eval)(ccd_code)
                for ccd_code in codes
            ),
            total=len(codes),
            desc="Get validated CCD records for ligand evaluation",
        )
    ]

    valid_codes = []
    for ccd_code in results:
        if ccd_code is None:
            continue
        valid_codes.append(ccd_code)
    return valid_codes


def _build_batch_query(pairs: list[tuple[str, str]]) -> str:
    """
    Build a batched GraphQL query for RCSB nonpolymer entity instances.

    Given a list of (entry_id, asym_id) pairs, this function constructs a single
    GraphQL query string that requests identifiers, validation scores, and selected
    structural connectivity fields for each pair. Each pair is addressed via a
    unique alias (e.g., i0, i1, …) to enable batch retrieval in one request.

    Args:
        pairs (list[tuple[str, str]]): A list of (entry_id, asym_id) tuples.
        entry_id should be a 4-character PDB ID (case-insensitive; will be
        converted to uppercase). asym_id is the polymer/nonpolymer asymmetric
        unit identifier within the entry.

    Returns:
        str: A GraphQL query string suitable for POSTing to the RCSB GraphQL
        endpoint (https://data.rcsb.org/graphql).
    """
    lines = ["query Q {"]

    for i, (entry, asym) in enumerate(pairs):
        alias = f"i{i}"
        entry = entry.upper()
        lines.append(
            f"""  {alias}: nonpolymer_entity_instance(entry_id:"{entry}", asym_id:"{asym}") {{
    rcsb_id
    rcsb_nonpolymer_entity_instance_container_identifiers {{
      entry_id
      comp_id
      entity_id
      auth_seq_id
    }}
    rcsb_nonpolymer_instance_validation_score {{
      RSR
      RSCC
      completeness
      stereo_outliers
      is_best_instance
      intermolecular_clashes
    }}
    rcsb_nonpolymer_struct_conn {{
      connect_type
      dist_value
    }}

  }}"""
        )
    lines.append("}")
    return "\n".join(lines)


def fetch_validation_report_batch(
    entry_asym_pairs: list[tuple[str, str]], chunk_size: int = 200
) -> pd.DataFrame:
    """
    Fetch validation metadata for nonpolymer instances from the RCSB GraphQL API in batches.

    This function batches (entry_id, asym_id) pairs into GraphQL requests,
    submits them to the RCSB endpoint, and assembles a tidy pandas.DataFrame
    containing instance identifiers and validation scores for each successfully
    resolved pair.

    Args:
        entry_asym_pairs (list[tuple[str, str]]): A list of (entry_id, asym_id)
                        pairs to query. entry_id is the PDB code; asym_id is the asymmetric
                        unit identifier. Pairs are processed in chunks.
                        chunk_size (int, optional): Number of pairs per GraphQL request. Defaults
                        to 200.

    Returns:
        pd.DataFrame: A DataFrame where each row corresponds to one resolved
                    (entry_id, asym_id) pair with the following columns (when available):
                    - entry_id (str): PDB entry ID.
                    - label_asym_id (str): The queried asym_id.
                    - instance_rcsb_id (str): RCSB instance identifier.
                    - comp_id (str): Chemical component ID (CCD code).
                    - entity_id (str): RCSB entity identifier.
                    - RSR (float | None): Real-space R-factor.
                    - RSCC (float | None): Real-space correlation coefficient.
                    - completeness (float | None): Completeness metric.
                    - stereo_outliers (int | None): Count of stereochemical outliers.
                    - intermolecular_clashes (int | None): Count of intermolecular clashes.
                    - is_best_instance (bool | None): Whether RCSB marks this as the best instance.
                      The output may be stably sorted by available identifier columns.
    """
    pairs = list(entry_asym_pairs)
    all_rows: list[dict[str, Any]] = []

    for start in tqdm(range(0, len(pairs), chunk_size)):
        batch = pairs[start : start + chunk_size]
        query = _build_batch_query(batch)

        resp = requests.post(GRAPHQL_ENDPOINT, json={"query": query}, timeout=60)
        resp.raise_for_status()
        payload = resp.json()

        if "errors" in payload and payload["errors"]:
            raise RuntimeError(f"GraphQL errors: {payload['errors']}")

        data = payload.get("data", {}) or {}

        for idx, (_entry, asym) in enumerate(batch):
            alias = f"i{idx}"
            node = data.get(alias)

            if node is None:
                continue

            ids = node.get("rcsb_nonpolymer_entity_instance_container_identifiers", {})
            scores = node.get("rcsb_nonpolymer_instance_validation_score", [{}])[0]

            all_rows.append(
                {
                    "entry_id": ids.get("entry_id"),
                    "label_asym_id": asym,
                    "instance_rcsb_id": node.get("rcsb_id"),
                    "comp_id": ids.get("comp_id"),
                    "entity_id": ids.get("entity_id"),
                    "RSR": scores.get("RSR"),
                    "RSCC": scores.get("RSCC"),
                    "completeness": scores.get("completeness"),
                    "stereo_outliers": scores.get("stereo_outliers"),
                    "intermolecular_clashes": scores.get("intermolecular_clashes"),
                    "is_best_instance": scores.get("is_best_instance"),
                }
            )
        time.sleep(0.5)

    df = pd.DataFrame(all_rows)
    sort_cols = [
        c
        for c in ["entry_id", "auth_asym_id", "comp_id", "auth_seq_id"]
        if c in df.columns
    ]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return df


def _filter_report_df(report_df: pd.DataFrame) -> pd.DataFrame:
    no_clash = report_df["intermolecular_clashes"] == 0
    is_best_instance = report_df["is_best_instance"] == "Y"
    no_stereo_outliers = report_df["stereo_outliers"] == 0
    completeness = report_df["completeness"] == 1.0
    good_rsr = report_df["RSR"] <= 0.2
    good_rscc = report_df["RSCC"] >= 0.95
    return report_df[
        no_clash
        & is_best_instance
        & no_stereo_outliers
        & completeness
        & good_rsr
        & good_rscc
    ]


def check_symm_mates_contact(
    mmcif_path: Path, lig_chain_ids: list[str]
) -> list[tuple[str, str]]:
    """
    Check whether specified ligand chains in an mmCIF file have contacts with symmetry mates.

    This function loads an mmCIF file, constructs the unit cell expanded to include
    periodic symmetry mates (3x3x3 repeating box), and checks for each provided
    ligand chain whether any atom in that chain has neighboring atoms (within a
    5.0 Å radius) that belong to chains outside the asymmetric unit. Chains that
    have no contacts with symmetry mates (i.e., all neighbors are within the
    asymmetric unit) are considered valid and returned.

    Args:
        mmcif_path (Path): Path to the mmCIF file to inspect (expected to be a
                    single-entry file, e.g. <pdbid>.cif).
        lig_chain_ids (list[str]): List of chain identifiers (label_asym_id values)
                      corresponding to ligand chains to test for symmetry-mate contacts.

    Returns:
        list[tuple[str, str]]: A list of (entry_id, chain_id) tuples for ligand
                               chains that do not contact symmetry mates. entry_id is the lowercased
                               PDB ID extracted from the mmCIF file; chain_id is the ligand chain label.
    """
    with open(mmcif_path, "rt") as f:
        cif_file = pdbx.CIFFile.read(f)

    entry_id = cif_file.block["entry"]["id"].as_item().lower()

    assembly_gen_category = cif_file.block["pdbx_struct_assembly_gen"]
    assembly1_chain_ids = set(assembly_gen_category["asym_id_list"].as_array(str)[0])

    extra_fields = ["label_asym_id", "label_entity_id"]  # Chain

    atom_array = pdbx.get_unit_cell(
        pdbx_file=cif_file,
        model=1,
        altloc="first",
        extra_fields=extra_fields,
        use_author_fields=False,
        include_bonds=False,
    )

    # Remove water and hydrogen
    atom_array = atom_array[~np.isin(atom_array.element, ["H", "D"])]
    atom_array = atom_array[~np.isin(atom_array.res_name, ["HOH", "DOD"])]

    asym_unit_chain_ids = set(atom_array.chain_id)
    allowed_nbs_chain_ids = list(asym_unit_chain_ids & assembly1_chain_ids)

    # Unit Cell * 27
    repeated_atom_array, _indices = repeat_box(atom_array, amount=1)

    uni_chain_ids = get_unique_chain_id(repeated_atom_array)
    repeated_atom_array.set_annotation("chain_id", uni_chain_ids)

    no_symm_concat_lig_chain_ids = []
    kdtree = KDTree(repeated_atom_array.coord)
    for chain_i in lig_chain_ids:
        chain_mask = repeated_atom_array.chain_id == chain_i
        chain_coord = repeated_atom_array.coord[chain_mask]
        neighbors_indices = np.unique(
            np.concatenate(kdtree.query_ball_point(chain_coord, r=5.0))
        )
        neighbors_chain_ids = repeated_atom_array.chain_id[neighbors_indices]
        if np.any(~np.isin(neighbors_chain_ids, allowed_nbs_chain_ids)):
            continue
        else:
            no_symm_concat_lig_chain_ids.append(chain_i)

    results = [(entry_id, i) for i in no_symm_concat_lig_chain_ids]
    return results


def check_symm_mates_contact_batch(
    mmcif_dir: Path, pdb_id_to_lig_chain_ids: dict[str, list[str]], n_cpu: int = -1
) -> list[tuple[str, str]]:
    """
    Batch-check symmetry-mate contacts for ligand chains across multiple mmCIF files.


    Args:
        mmcif_dir (Path): Directory containing mmCIF files named <pdbid>.cif (lowercase).
        pdb_id_to_lig_chain_ids (dict[str, list[str]]): Mapping from PDB IDs (case-insensitive)
        to lists of ligand chain identifiers to be checked for each entry.
        n_cpu (int, optional): Number of worker processes for parallel execution.
        Passed to joblib.Parallel. Defaults to -1 (use all available CPUs).

    Returns:
        list[tuple[str, str]]: Aggregated list of (entry_id, chain_id) tuples for ligand
                               chains that do not contact symmetry mates
                               across all processed entries.
    """
    results = [
        r
        for r in tqdm(
            Parallel(n_jobs=n_cpu, return_as="generator_unordered")(
                delayed(check_symm_mates_contact)(
                    mmcif_dir / f"{pdb_id.lower()}.cif", lig_chain_ids
                )
                for pdb_id, lig_chain_ids in pdb_id_to_lig_chain_ids.items()
            ),
            desc="Check SymmMates contact for ligands",
            total=len(pdb_id_to_lig_chain_ids),
        )
    ]

    valid_ligands = []
    for r in results:
        valid_ligands.extend(r)
    return valid_ligands


def filter_lig_in_chain_interface_df(
    chain_interface_df: pd.DataFrame, mmcif_dir: Path, n_cpu: int = -1
) -> pd.DataFrame:
    """
    Filter a chain-interface DataFrame by validating ligand chains
    with RCSB reports and symmetry-mate contact checks.

    This function identifies ligand chains from an interface table, validates those
    ligand instances using RCSB nonpolymer validation metrics, removes ligand
    instances that fail the validation criteria, and excludes ligand chains that
    contact symmetry mates in the crystallographic unit cell. The filtered
    DataFrame retains only interface rows where any ligand side passes both the
    validation and symmetry-mate checks.

    Args:
        chain_interface_df (pd.DataFrame): Input interface table. Expected columns:
                            - "entry_id" (str): PDB entry identifier.
                            - "chain_id_1" (str) and "chain_id_2" (str): chain identifiers
                              for the two sides.
                            - "entity_type_1" and "entity_type_2" (str): entity types.
                               The function filters rows where either side is labeled
                               as a ligand and then validates those ligand chains.
        mmcif_dir (Path): Directory containing mmCIF files named as <pdbid>.cif
                          (lowercase). These files are used to inspect symmetry-mate contacts.
        n_cpu (int, optional): Number of CPUs for parallel processing when checking
                               symmetry-mate contacts. Passed to joblib.Parallel. Defaults to -1
                               (use all available CPUs).

    Returns:
        pd.DataFrame: A filtered copy of chain_interface_df where ligand-containing
                      interface rows are kept only if the ligand chain:
                      1. passes RCSB validation filters (via fetch_validation_report_batch
                      and _filter_report_df), and
                      2. does not have contacts with symmetry mates (via
                      check_symm_mates_contact_batch).
                      Non-ligand interfaces are preserved unchanged.
    """
    lig_chains_1 = list(
        chain_interface_df[chain_interface_df["entity_type_1"] == LIGAND][
            ["entry_id", "chain_id_1"]
        ].itertuples(index=False, name=None)
    )
    lig_chains_2 = list(
        chain_interface_df[chain_interface_df["entity_type_2"] == LIGAND][
            ["entry_id", "chain_id_2"]
        ].itertuples(index=False, name=None)
    )
    lig_chains = [i for i in set(lig_chains_1 + lig_chains_2) if "." not in i[1]]

    logging.info("Total %s ligand chains to check.", len(lig_chains))

    report_df = fetch_validation_report_batch(lig_chains)
    valid_report_df = _filter_report_df(report_df)

    logging.info(
        "%s ligand chains pass validation by RCSB report.", len(valid_report_df)
    )

    pdb_id_to_lig_chain_ids = (
        valid_report_df.groupby("entry_id")["label_asym_id"].apply(list).to_dict()
    )

    # allowed = [(entry_id, lig_chain_id), ...]
    allowed = check_symm_mates_contact_batch(mmcif_dir, pdb_id_to_lig_chain_ids, n_cpu)

    logging.info("%s ligand chains pass symmetry-mate contact check.", len(allowed))

    chain_interface_df["pair1"] = list(
        zip(chain_interface_df["entry_id"], chain_interface_df["chain_id_1"])
    )
    chain_interface_df["pair2"] = list(
        zip(chain_interface_df["entry_id"], chain_interface_df["chain_id_2"])
    )

    chain_interface_df["pair1_allowed"] = chain_interface_df["pair1"].isin(allowed)
    chain_interface_df["pair2_allowed"] = chain_interface_df["pair2"].isin(allowed)

    is_l1 = chain_interface_df["entity_type_1"] == LIGAND
    is_l2 = chain_interface_df["entity_type_2"] == LIGAND

    keep = (~is_l1) | chain_interface_df["pair1_allowed"]
    keep &= (~is_l2) | chain_interface_df["pair2_allowed"]

    chain_interface_df_filtered = chain_interface_df.loc[keep].drop(
        columns=["pair1", "pair2", "pair1_allowed", "pair2_allowed"]
    )
    return chain_interface_df_filtered
