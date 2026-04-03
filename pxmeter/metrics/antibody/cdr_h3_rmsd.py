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

import numpy as np

from pxmeter.constants import PROTEIN
from pxmeter.data.struct import Structure
from pxmeter.metrics.antibody.annotation import AntibodyAnnotator
from pxmeter.metrics.rmsd import partially_aligned_rmsd


def calc_cdr_h3_bb_rmsd(
    ref_struct: Structure,
    model_struct: Structure,
) -> dict[str, float]:
    """
    Calculate the RMSD of CDR-H3 backbone regions for all antibody heavy chains.

    The calculation assumes ref_struct and model_struct have strictly aligned atom_arrays
    (i.e. model_struct.atom_array[i] corresponds to ref_struct.atom_array[i]).

    It identifies CDR-H3 regions on the reference structure sequence, aligns the
    Framework (FR) regions using backbone atoms (N, C, CA, O), and computes RMSD for CDR-H3 using backbone atoms.

    Args:
        ref_struct (Structure): The reference structure (ground truth).
        model_struct (Structure): The model structure (prediction).

    Returns:
        dict[str, float]: A dictionary mapping chain IDs to their CDR-H3 RMSD values.
    """
    annotator = AntibodyAnnotator()

    results = {}

    # Get unique chain IDs from reference structure (PROTEIN only)
    protein_mask = ref_struct.get_mask_for_given_entity_types(PROTEIN)

    if np.sum(protein_mask) == 0:
        return {}

    # 1. Identify unique protein entities and their sequences
    protein_entity_ids = np.unique(ref_struct.atom_array.label_entity_id[protein_mask])

    unique_sequences = []
    entity_id_to_seq_idx = {}

    for entity_id in protein_entity_ids:
        seq = ref_struct.entity_poly_seq.get(entity_id, "")
        if seq:
            try:
                seq_idx = unique_sequences.index(seq)
            except ValueError:
                unique_sequences.append(seq)
                seq_idx = len(unique_sequences) - 1

            entity_id_to_seq_idx[entity_id] = seq_idx

    if not unique_sequences:
        return {}

    annotation_results = annotator.annotate(unique_sequences)

    ref_chain_ids = np.unique(ref_struct.uni_chain_id[protein_mask])

    for chain_id in ref_chain_ids:

        chain_atom_mask = ref_struct.uni_chain_id == chain_id
        if not np.any(chain_atom_mask):
            continue

        chain_start_idx = np.argmax(chain_atom_mask)  # Finds first True
        entity_id = ref_struct.atom_array.label_entity_id[chain_start_idx]

        if entity_id not in entity_id_to_seq_idx:
            continue

        # Get annotation result
        seq_idx = entity_id_to_seq_idx[entity_id]
        ref_regions_full, chain_type = annotation_results[seq_idx]

        # Filter: Only Heavy chains with CDR3
        # ANARCII not supported scFv currently (v2.0.2)
        if ("H" not in chain_type) or ("CDR3" not in ref_regions_full):
            continue

        # Extract ALL atoms for this chain
        # Note: ref_struct and model_struct are already cleaned (no H, no water), so this is Heavy Atom
        if not np.any(chain_atom_mask):
            continue

        atom_indices = np.flatnonzero(chain_atom_mask)
        res_ids = ref_struct.atom_array.res_id[atom_indices]
        atom_names = ref_struct.atom_array.atom_name[atom_indices]

        # Map atoms to Regions
        seq_len = len(unique_sequences[seq_idx])
        valid_res_mask = (res_ids >= 1) & (res_ids <= seq_len)

        if not np.any(valid_res_mask):
            continue

        valid_indices = atom_indices[valid_res_mask]
        valid_res_ids = res_ids[valid_res_mask]
        valid_atom_names = atom_names[valid_res_mask]

        # Get region for each valid atom
        atom_regions = np.array([ref_regions_full[rid - 1] for rid in valid_res_ids])

        # Extract aligned Model coordinates
        if np.max(valid_indices) >= len(model_struct.atom_array):
            raise ValueError(f"Model atom indices out of bounds for chain {chain_id}")

        ref_coords_aligned = ref_struct.atom_array.coord[valid_indices]
        model_coords_aligned = model_struct.atom_array.coord[valid_indices]

        # Check for NaNs
        nan_mask = np.isnan(model_coords_aligned).any(axis=1)
        if np.any(nan_mask):
            valid_coords_mask = ~nan_mask
            ref_coords_aligned = ref_coords_aligned[valid_coords_mask]
            model_coords_aligned = model_coords_aligned[valid_coords_mask]
            atom_regions = atom_regions[valid_coords_mask]
            valid_atom_names = valid_atom_names[valid_coords_mask]

        if len(ref_coords_aligned) == 0:
            continue

        # Define masks
        backbone_mask = np.isin(valid_atom_names, ["N", "C", "CA", "O"])

        # Alignment: FR regions AND Backbone atoms (N, C, CA, O)
        fr_regions_mask = np.isin(atom_regions, ["FR1", "FR2", "FR3", "FR4"])
        align_mask = fr_regions_mask & backbone_mask

        # RMSD: CDR3 regions AND Backbone atoms (N, C, CA, O)
        cdr_h3_mask = (atom_regions == "CDR3") & backbone_mask

        if np.sum(align_mask) < 3 or np.sum(cdr_h3_mask) == 0:
            continue

        # Calculate RMSD
        _, cdr_h3_rmsd_val, _, _ = partially_aligned_rmsd(
            src_pose=model_coords_aligned,
            tar_pose=ref_coords_aligned,
            align_mask=align_mask,
            rmsd_mask=cdr_h3_mask,
            eps=1e-12,
        )

        results[chain_id] = float(cdr_h3_rmsd_val)

    return results
