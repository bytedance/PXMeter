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

from typing import Any

import numpy as np
from scipy.spatial.distance import cdist

from pxmeter.constants import POLYMER
from pxmeter.data.struct import Structure
from pxmeter.metrics.rmsd import align_src_to_tar, apply_transform, rmsd

# DockQ constants matching official implementation
FNAT_THRESHOLD = 5.0
INTERFACE_THRESHOLD = 10.0
# Official DockQ backbone atoms (including Nucleic Acids)
# Order matters for SVD!
DOCKQ_BACKBONE_ATOMS = [
    "CA",
    "C",
    "N",
    "O",
    "P",
    "OP1",
    "OP2",
    "O2'",
    "O3'",
    "O4'",
    "O5'",
    "C1'",
    "C2'",
    "C3'",
    "C4'",
    "C5'",
]


def _get_residue_atoms(struct: Structure, chain_id: str) -> list[np.ndarray]:
    """
    Get coordinates grouped by residue for a specific chain.
    """
    chain_mask = struct.uni_chain_id == chain_id
    if not np.any(chain_mask):
        return []

    # Get residue starts to group atoms
    starts = struct.get_residue_starts(add_exclusive_stop=True)
    # Filter starts to only those belonging to the specific chain
    chain_indices = np.where(chain_mask)[0]
    first_idx = chain_indices[0]
    last_idx = chain_indices[-1]

    relevant_starts = starts[:-1]
    relevant_stops = starts[1:]
    mask = (relevant_starts >= first_idx) & (relevant_stops <= last_idx + 1)

    res_coords = [
        struct.atom_array.coord[s:e]
        for s, e in zip(relevant_starts[mask], relevant_stops[mask])
    ]

    return res_coords


def _compute_residue_distances(
    coords1: list[np.ndarray], coords2: list[np.ndarray]
) -> np.ndarray:
    """
    Compute min distance squared between all pairs of residues.
    """
    if not coords1 or not coords2:
        return np.zeros((len(coords1), len(coords2)))

    # Concatenate all atoms for each chain
    all_c1 = np.concatenate(coords1)
    all_c2 = np.concatenate(coords2)

    # Compute squared distances between all atoms
    # Use sqeuclidean to avoid square root
    dists = cdist(all_c1, all_c2, "sqeuclidean")

    # Aggregate to residue-residue distances
    lens1 = [len(c) for c in coords1]
    lens2 = [len(c) for c in coords2]
    offsets1 = np.cumsum([0] + lens1)
    offsets2 = np.cumsum([0] + lens2)

    # Step 1: Reduce over columns (chain 2 residues)
    dists_reduced_cols = np.minimum.reduceat(dists, offsets2[:-1], axis=1)

    # Step 2: Reduce over rows (chain 1 residues)
    res_dist_mat = np.minimum.reduceat(dists_reduced_cols, offsets1[:-1], axis=0)

    return res_dist_mat


def _get_aligned_residues(
    struct1: Structure, chain1: str, struct2: Structure, chain2: str
):
    """
    Get residues that are common between ref and model based on residue ID.
    Official DockQ uses sequence alignment, but here we assume mapping is handled
    by residue identifiers (res_id + res_name) for simplicity.
    """

    def get_res_map(struct, chain):
        mask = struct.uni_chain_id == chain
        starts = struct.get_residue_starts(add_exclusive_stop=True)
        chain_indices = np.where(mask)[0]
        if len(chain_indices) == 0:
            return {}
        first, last = chain_indices[0], chain_indices[-1]

        relevant_starts = starts[:-1]
        relevant_stops = starts[1:]
        m = (relevant_starts >= first) & (relevant_stops <= last + 1)
        sel_starts = relevant_starts[m]
        sel_stops = relevant_stops[m]

        res_dict = {}
        # We still need to loop to build the dictionary with RIDs
        for start, stop in zip(sel_starts, sel_stops):
            rid = (
                f"{struct.atom_array.res_id[start]}_{struct.atom_array.res_name[start]}"
            )
            res_dict[rid] = (start, stop)
        return res_dict

    res_map1 = get_res_map(struct1, chain1)
    res_map2 = get_res_map(struct2, chain2)

    common_ids = set(res_map1.keys()) & set(res_map2.keys())

    # Maintain the order of residues in the reference chain.
    # Since res_map1 is built by iterating in structural order, its keys are ordered.
    ref_ordered_ids = [rid for rid in res_map1 if rid in common_ids]

    ref_aligned_res = [res_map1[rid] for rid in ref_ordered_ids]
    model_aligned_res = [res_map2[rid] for rid in ref_ordered_ids]

    return ref_aligned_res, model_aligned_res


def _get_paired_backbone_coords(
    struct1: Structure,
    struct2: Structure,
    indices1: list[tuple[int, int]],
    indices2: list[tuple[int, int]],
    align_atoms: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """
    Get paired backbone coordinates for two lists of aligned residues.
    Ensures that ONLY common backbone atoms are selected in the SAME order.
    If align_atoms is False, assumes the atoms in res_list1 and res_list2 are already aligned.
    """

    coords1 = []
    coords2 = []
    valid1 = []
    valid2 = []

    atom_names1 = struct1.atom_array.atom_name
    coords1_all = struct1.atom_array.coord
    valid1_all = (
        struct1.valid_mask
        if struct1.valid_mask is not None
        else np.ones(len(coords1_all), dtype=bool)
    )

    atom_names2 = struct2.atom_array.atom_name
    coords2_all = struct2.atom_array.coord
    valid2_all = (
        struct2.valid_mask
        if struct2.valid_mask is not None
        else np.ones(len(coords2_all), dtype=bool)
    )

    for (s1, e1), (s2, e2) in zip(indices1, indices2):
        n1 = atom_names1[s1:e1]
        n2 = atom_names2[s2:e2]
        c1 = coords1_all[s1:e1]
        c2 = coords2_all[s2:e2]
        v1 = valid1_all[s1:e1]
        v2 = valid2_all[s2:e2]

        if align_atoms:
            name_to_idx1 = {name: i for i, name in enumerate(n1)}
            name_to_idx2 = {name: i for i, name in enumerate(n2)}

            for name in DOCKQ_BACKBONE_ATOMS:
                if name in name_to_idx1 and name in name_to_idx2:
                    idx1 = name_to_idx1[name]
                    idx2 = name_to_idx2[name]
                    coords1.append(c1[idx1])
                    coords2.append(c2[idx2])
                    valid1.append(v1[idx1])
                    valid2.append(v2[idx2])
        else:
            mask1 = np.isin(n1, DOCKQ_BACKBONE_ATOMS)
            coords1.extend(c1[mask1])
            coords2.extend(c2[mask1])
            valid1.extend(v1[mask1])
            valid2.extend(v2[mask1])

    if not coords1:
        return (
            np.empty((0, 3)),
            np.empty((0, 3)),
            np.empty(0, dtype=bool),
            np.empty(0, dtype=bool),
        )
    return np.array(coords1), np.array(coords2), np.array(valid1), np.array(valid2)


def compute_dockq_for_pair(
    ref_struct: Structure,
    model_struct: Structure,
    ref_chain1: str,
    ref_chain2: str,
    ref_to_model_chain_map: dict[str, str],
    align_atoms: bool = True,
) -> dict[str, Any]:
    """
    Computes DockQ for a pair of chains following official implementation.

    Args:
        ref_struct: Reference structure.
        model_struct: Model structure to evaluate.
        ref_chain1: Chain ID of the first chain in reference.
        ref_chain2: Chain ID of the second chain in reference.
        ref_to_model_chain_map: Mapping from reference chain IDs to model chain IDs.
        align_atoms: Whether to align atoms by name within residues. Defaults to True.

    Returns:
        Dictionary containing DockQ score and other metrics (fnat, iRMSD, LRMSD, etc.).
        Returns empty dictionary if chains are not found or not polymers.
    """
    model_chain1 = ref_to_model_chain_map.get(ref_chain1)
    model_chain2 = ref_to_model_chain_map.get(ref_chain2)

    if model_chain1 is None or model_chain2 is None:
        return {}

    # Skip if either chain is not a polymer (e.g., small molecule)
    chain_id_to_entity_id = ref_struct.get_chain_id_to_entity_id()

    def is_polymer(chain_id):
        entity_id = chain_id_to_entity_id.get(chain_id)
        return ref_struct.entity_poly_type.get(entity_id) in POLYMER

    if not is_polymer(ref_chain1) or not is_polymer(ref_chain2):
        return {}

    # 1. Native distances and nat_total
    ref_res1_all = _get_residue_atoms(ref_struct, ref_chain1)
    ref_res2_all = _get_residue_atoms(ref_struct, ref_chain2)
    if not ref_res1_all or not ref_res2_all:
        return {}

    # Distance matrix between all heavy atoms of residues
    ref_dist_mat_all = _compute_residue_distances(ref_res1_all, ref_res2_all)
    nat_total = np.sum(ref_dist_mat_all < FNAT_THRESHOLD**2)
    if nat_total == 0:
        return {}

    # 2. Aligned residues
    def get_all_residues(struct, chain):
        mask = struct.uni_chain_id == chain
        starts = struct.get_residue_starts(add_exclusive_stop=True)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return []
        first, last = indices[0], indices[-1]
        res_list = []
        for start, stop in zip(starts[:-1], starts[1:]):
            if start >= first and stop <= last + 1:
                res_list.append((start, stop))
        return res_list

    ref_all1 = get_all_residues(ref_struct, ref_chain1)
    model_all1 = get_all_residues(model_struct, model_chain1)
    ref_all2 = get_all_residues(ref_struct, ref_chain2)
    model_all2 = get_all_residues(model_struct, model_chain2)

    if align_atoms:
        ref_aln1, model_aln1 = _get_aligned_residues(
            ref_struct, ref_chain1, model_struct, model_chain1
        )
        ref_aln2, model_aln2 = _get_aligned_residues(
            ref_struct, ref_chain2, model_struct, model_chain2
        )
    else:
        # Assume residues are already aligned in order
        ref_aln1 = ref_all1
        model_aln1 = model_all1
        ref_aln2 = ref_all2
        model_aln2 = model_all2

    if not ref_aln1 or not ref_aln2 or not model_aln1 or not model_aln2:
        return {}

    # fnat statistics on aligned residues
    ref_aln1_coords = [ref_struct.atom_array.coord[s:e] for s, e in ref_aln1]
    ref_aln2_coords = [ref_struct.atom_array.coord[s:e] for s, e in ref_aln2]
    model_aln1_coords = [model_struct.atom_array.coord[s:e] for s, e in model_aln1]
    model_aln2_coords = [model_struct.atom_array.coord[s:e] for s, e in model_aln2]

    # Identify valid residues in model (residues that have at least one valid atom)
    model_res1_valid = np.array(
        [np.any(model_struct.valid_mask[s:e]) for s, e in model_aln1]
    )
    model_res2_valid = np.array(
        [np.any(model_struct.valid_mask[s:e]) for s, e in model_aln2]
    )
    # A contact is only valid if both residues are valid
    model_res_contact_mask = model_res1_valid[:, None] & model_res2_valid[None, :]

    ref_aln_dist = _compute_residue_distances(ref_aln1_coords, ref_aln2_coords)
    model_aln_dist = _compute_residue_distances(model_aln1_coords, model_aln2_coords)

    ref_contacts = ref_aln_dist < FNAT_THRESHOLD**2
    model_contacts = (model_aln_dist < FNAT_THRESHOLD**2) & model_res_contact_mask

    nat_correct = np.sum(ref_contacts * model_contacts)
    model_total = np.sum(model_contacts)
    nonnat_count = np.sum(model_contacts * (1 - ref_contacts))

    fnat = nat_correct / nat_total
    fnonnat = nonnat_count / model_total if model_total > 0 else 0.0

    # 3. iRMSD
    # Residues involved in native contacts (using 10A threshold)
    interacting_mask = ref_aln_dist < INTERFACE_THRESHOLD**2
    int_indices1, int_indices2 = np.where(interacting_mask)
    int_res_indices1 = sorted(list(set(int_indices1)))
    int_res_indices2 = sorted(list(set(int_indices2)))

    # Paired backbone atoms for interface residues of both chains
    ref_int_aln1 = [ref_aln1[i] for i in int_res_indices1]
    model_int_aln1 = [model_aln1[i] for i in int_res_indices1]
    (
        ref_int_coords1,
        model_int_coords1,
        ref_int_valid1,
        model_int_valid1,
    ) = _get_paired_backbone_coords(
        ref_struct, model_struct, ref_int_aln1, model_int_aln1, align_atoms=align_atoms
    )

    ref_int_aln2 = [ref_aln2[i] for i in int_res_indices2]
    model_int_aln2 = [model_aln2[i] for i in int_res_indices2]
    (
        ref_int_coords2,
        model_int_coords2,
        ref_int_valid2,
        model_int_valid2,
    ) = _get_paired_backbone_coords(
        ref_struct, model_struct, ref_int_aln2, model_int_aln2, align_atoms=align_atoms
    )

    ref_interface_atoms = np.concatenate([ref_int_coords1, ref_int_coords2])
    model_interface_atoms = np.concatenate([model_int_coords1, model_int_coords2])
    ref_interface_valid = np.concatenate([ref_int_valid1, ref_int_valid2])
    model_interface_valid = np.concatenate([model_int_valid1, model_int_valid2])

    if len(ref_interface_atoms) == 0:
        irmsd = 0.0
    else:
        interface_valid = model_interface_valid & ref_interface_valid
        if not np.any(interface_valid):
            irmsd = np.inf
        else:
            rot, trans = align_src_to_tar(
                model_interface_atoms, ref_interface_atoms, atom_mask=interface_valid
            )
            irmsd = rmsd(
                apply_transform(model_interface_atoms, rot, trans),
                ref_interface_atoms,
                valid_mask1=model_interface_valid,
                valid_mask2=ref_interface_valid,
            )

    # 4. LRMSD
    # Assign receptor and ligand by size (DockQ convention)
    if len(ref_all1) > len(ref_all2):
        receptor_ref, receptor_model = ref_aln1, model_aln1
        ligand_ref, ligand_model = ref_aln2, model_aln2
        class1, class2 = "receptor", "ligand"
    else:
        receptor_ref, receptor_model = ref_aln2, model_aln2
        ligand_ref, ligand_model = ref_aln1, model_aln1
        class1, class2 = "ligand", "receptor"

    # Paired backbone atoms for entire chains
    (
        ref_rec_coords,
        model_rec_coords,
        ref_rec_valid,
        model_rec_valid,
    ) = _get_paired_backbone_coords(
        ref_struct, model_struct, receptor_ref, receptor_model, align_atoms=align_atoms
    )
    (
        ref_lig_coords,
        model_lig_coords,
        ref_lig_valid,
        model_lig_valid,
    ) = _get_paired_backbone_coords(
        ref_struct, model_struct, ligand_ref, ligand_model, align_atoms=align_atoms
    )

    # Align on receptor
    if len(ref_rec_coords) == 0:
        raise ValueError(
            "No common backbone atoms found in receptor for LRMSD calculation."
        )
    else:
        receptor_valid = model_rec_valid & ref_rec_valid
        if not np.any(receptor_valid):
            lrmsd = np.inf
        else:
            rot_rec, trans_rec = align_src_to_tar(
                model_rec_coords, ref_rec_coords, atom_mask=receptor_valid
            )
            # Apply to ligand and compute RMSD
            if len(ref_lig_coords) == 0:
                raise ValueError(
                    "No common backbone atoms found in ligand for LRMSD calculation."
                )
            else:
                model_lig_rotated = apply_transform(
                    model_lig_coords, rot_rec, trans_rec
                )
                lrmsd = rmsd(
                    model_lig_rotated,
                    ref_lig_coords,
                    valid_mask1=model_lig_valid,
                    valid_mask2=ref_lig_valid,
                )

    # DockQ Score formula
    def rms_scaled(rms, d):
        return 1.0 / (1.0 + (rms / d) ** 2)

    dockq = (fnat + rms_scaled(irmsd, 1.5) + rms_scaled(lrmsd, 8.5)) / 3.0

    def get_chain_len(struct, chain_id):
        mask = struct.uni_chain_id == chain_id
        return len(np.unique(struct.atom_array.res_id[mask]))

    return {
        "DockQ": dockq,
        "fnat": fnat,
        "F1": (
            2 * nat_correct / (nat_correct + nonnat_count + nat_total)
            if (nat_correct + nonnat_count + nat_total) > 0
            else 0.0
        ),
        "iRMSD": irmsd,
        "LRMSD": lrmsd,
        "fnonnat": fnonnat,
        "nonnat_count": int(nonnat_count),
        "nat_correct": int(nat_correct),
        "nat_total": int(nat_total),
        "model_total": int(model_total),
        "chain1": model_chain1,
        "chain2": model_chain2,
        "chain_map": ref_to_model_chain_map,
        "len1": get_chain_len(ref_struct, ref_chain1),
        "len2": get_chain_len(ref_struct, ref_chain2),
        "class1": class1,
        "class2": class2,
        "is_het": False,
    }


def _filter_hetatm_atoms(
    ref_struct: Structure,
    model_struct: Structure,
    ref_to_model_chain_map: dict[str, str],
) -> tuple[Structure, Structure]:
    """
    Filter out HETATM atoms from both reference and model structures.

    Some PDB structures label non-standard residues in protein chains as HETATM (e.g., 9CY4 and 8J8V).
    During DockQ calculation, these HETATM residues are often excluded while ATOM residues in the same
    chain are included. To maintain consistency with the native DockQ logic, this function filters
    out HETATM atoms from both reference and model structures based on the reference's HETATM labels.

    Args:
        ref_struct: Reference structure.
        model_struct: Model structure to evaluate.
        ref_to_model_chain_map: Mapping from reference chain IDs to model chain IDs.

    Returns:
        tuple: (filtered_ref_struct, filtered_model_struct)
    """
    if "hetero" in ref_struct.atom_array.get_annotation_categories() and np.any(
        ref_struct.atom_array.hetero
    ):
        ref_atoms = ref_struct.atom_array
        hetero_mask = ref_atoms.hetero

        # Map ref chains to model chains for key consistency
        ref_mapped_chains = np.array(
            [ref_to_model_chain_map.get(c, "") for c in ref_struct.uni_chain_id]
        )
        valid_map_mask = ref_mapped_chains != ""

        # Construct keys for HETATM atoms in ref (mapped to model chains)
        ref_keys = np.array(
            [
                f"{c}_{rid}_{rn}_{an}"
                for c, rid, rn, an in zip(
                    ref_mapped_chains,
                    ref_atoms.res_id,
                    ref_atoms.res_name,
                    ref_atoms.atom_name,
                )
            ]
        )
        hetero_keys_set = set(ref_keys[hetero_mask & valid_map_mask])

        # Filter ref_struct
        ref_struct = ref_struct.select_substructure(~hetero_mask)

        # Filter model_struct
        model_atoms = model_struct.atom_array
        model_keys = [
            f"{c}_{rid}_{rn}_{an}"
            for c, rid, rn, an in zip(
                model_struct.uni_chain_id,
                model_atoms.res_id,
                model_atoms.res_name,
                model_atoms.atom_name,
            )
        ]

        # Use list comprehension with set lookup for efficient filtering
        model_keep = np.array([k not in hetero_keys_set for k in model_keys])
        model_struct = model_struct.select_substructure(model_keep)

    return ref_struct, model_struct


def compute_dockq(
    ref_struct: Structure,
    model_struct: Structure,
    ref_to_model_chain_map: dict[str, str],
    align_atoms: bool = True,
    exclude_hetatms: bool = True,
) -> dict[str, dict[str, Any]]:
    """
    Computes DockQ scores for all interfaces in a reference structure.

    Args:
        ref_struct: Reference structure.
        model_struct: Model structure to evaluate.
        ref_to_model_chain_map: Mapping from reference chain IDs to model chain IDs.
        align_atoms: Whether to align atoms by name within residues. Defaults to True.
        exclude_hetatms: Whether to exclude HETATM atoms based on ref_struct. Defaults to True.

    Returns:
        Dictionary mapping interface keys (e.g., "A:B") to their respective DockQ results.
    """
    if exclude_hetatms:
        # Filter out HETATM based on ref_struct to maintain native DockQ consistency
        ref_struct, model_struct = _filter_hetatm_atoms(
            ref_struct, model_struct, ref_to_model_chain_map
        )

    _, interfaces = ref_struct.get_chains_and_interfaces(
        interface_radius=INTERFACE_THRESHOLD
    )

    dockq_result_dict = {}
    for c1, c2 in interfaces:
        res = compute_dockq_for_pair(
            ref_struct,
            model_struct,
            c1,
            c2,
            ref_to_model_chain_map,
            align_atoms=align_atoms,
        )
        if res:
            key = f"{res['chain1']}:{res['chain2']}"
            dockq_result_dict[key] = res

    return dockq_result_dict
