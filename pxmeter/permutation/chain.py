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

from pxmeter.data.struct import Structure
from pxmeter.metrics.rmsd import align_src_to_tar, apply_transform, rmsd


class ChainPermutation:
    """
    Chain permutation algorithm.

    Given a reference structure and a model structure, this class implements the
    chain permutation algorithm to find an optimal mapping between the chains
    of the two structures. The model anchor chain is aligned to the reference anchor chain,
    and subsequent chains are aligned based on their spatial proximity to the
    already aligned chains.

    Args:
            ref_struct (Structure): Reference structure object
            model_struct (Structure): Model structure object
            model_to_ref_entity_id (dict[str, str]): Mapping of model entity IDs
                                    to reference entity IDs
            enumerate_all_anchors (bool): Whether to enumerate all anchor chains.
    """

    def __init__(
        self,
        ref_struct: Structure,
        model_struct: Structure,
        model_to_ref_entity_id: dict[str, str],
        enumerate_all_anchors: bool = True,
    ):
        self.ref_struct = ref_struct
        self.model_struct = model_struct
        self.model_to_ref_entity_id = model_to_ref_entity_id
        self.enumerate_all_anchors = enumerate_all_anchors

        self.ref_to_model_entity_id = {
            v: k for k, v in self.model_to_ref_entity_id.items()
        }
        self.ref_entity_id_to_chain_ids = self.ref_struct.get_entity_id_to_chain_ids()
        self.model_entity_id_to_chain_ids = (
            self.model_struct.get_entity_id_to_chain_ids()
        )

        self.ref_chain_id_to_entity_id = self.ref_struct.get_chain_id_to_entity_id()
        self.model_chain_id_to_entity_id = self.model_struct.get_chain_id_to_entity_id()

    def find_model_anchor_chains(self) -> str:
        """
        Ref: AlphaFold3 SI Chapter 4.2. -> AlphaFold Multimer Chapter 7.3.1
        In the alignment phase, we pick a pair of anchor asyms to align,
        one in the ground truth and one in the prediction.
        The ground truth anchor asym a_gt is chosen to be the least ambiguous possible,
        for example in an A3B2 complex an arbitrary B asym is chosen.
        In the event of a tie e.g. A2B2 stoichiometry, the longest asym is chosen,
        with the hope that in general the longer asyms are likely to have higher confident predictions.
        The prediction anchor asym is chosen from the set {a^pred_m} of all prediction asyms
        with the same sequence as the ground truth anchor asym.

        Priority for selecting anchors:
        - Choose chains with more than 4 residues.
        - Select entities with more than 4 resolved residues in the reference.
        - Prioritize polymer chains.
        - Pick entities with the fewest chains in the reference.
        - Opt for chains with the highest number of residues.
        - Select chains with the smallest chain ID.

        Return:
            list[str]: selected anchor chain IDs of model structure. Sorted by priority.
        """
        # Extracting model info
        model_chain_ids = np.unique(self.model_struct.uni_chain_id)
        model_chain_ids = [
            i
            for i in model_chain_ids
            if self.model_chain_id_to_entity_id[i] in self.model_to_ref_entity_id
        ]

        entity_id_per_model_chain = []
        seq_length_per_model_chain = []
        is_polymer_per_model_chain = []
        for chain_id in model_chain_ids:
            entity_id = self.model_chain_id_to_entity_id[chain_id]
            mask = self.model_struct.uni_chain_id == chain_id

            entity_id_per_model_chain.append(entity_id)

            # If not res_id_alignments, any res not mapped should have
            # already been removed before entering the Chain Permutation process.
            seq_length_per_model_chain.append(
                len(np.unique(self.model_struct.atom_array.res_id[mask]))
            )

            is_polymer = int(entity_id in self.model_struct.entity_poly_type)
            is_polymer_per_model_chain.append(is_polymer)

        # Extracting ref info
        resolved_residue_num_in_ref = []
        chain_num_in_ref_entity = []
        for model_chain_entity in entity_id_per_model_chain:
            ref_entity = self.model_to_ref_entity_id[model_chain_entity]
            ref_entity_mask = self.ref_struct.atom_array.label_entity_id == ref_entity
            ref_resolved_residue_num = len(
                np.unique(self.ref_struct.atom_array.res_id[ref_entity_mask])
            )
            resolved_residue_num_in_ref.append(ref_resolved_residue_num)
            chain_num_in_ref_entity.append(
                len(self.ref_entity_id_to_chain_ids[ref_entity])
            )

        # Priority
        sorted_model_chain_ids = np.argsort(model_chain_ids)
        seq_length_greater_than_4 = [
            1 if seq_length > 4 else 0 for seq_length in seq_length_per_model_chain
        ]
        ref_resolved_res_more_than_4 = [
            1 if resolved_residue_num > 4 else 0
            for resolved_residue_num in resolved_residue_num_in_ref
        ]
        merged_info = list(
            zip(
                seq_length_greater_than_4,
                ref_resolved_res_more_than_4,
                is_polymer_per_model_chain,
                chain_num_in_ref_entity,
                seq_length_per_model_chain,
                sorted_model_chain_ids,
                model_chain_ids,
            )
        )

        # Sort candidates by priority
        pick_priority = sorted(
            merged_info,
            key=lambda x: (
                -x[0],
                -x[1],
                -x[2],
                x[3],
                -x[4],
                x[5],
            ),
        )
        model_anchor_chain_ids = [i[-1] for i in pick_priority]

        high_priority_mask = (
            np.array([i[0] for i in pick_priority], dtype=bool)  # seq_length > 4
            & np.array([i[1] for i in pick_priority], dtype=bool)  # ref resolved > 4
            & np.array([i[2] for i in pick_priority], dtype=bool)  # is polymer
        )
        if np.any(high_priority_mask):
            # Only select high priority chains
            model_anchor_chain_ids = np.array(model_anchor_chain_ids)[
                high_priority_mask
            ].tolist()

        return model_anchor_chain_ids

    @staticmethod
    def _find_min_indices(dist_matrix: np.ndarray) -> tuple[list[int], list[int]]:
        # The num of rows always equal to or greater than num of cols
        # The num of matched pairs should equal the total chains in the smaller set.
        assert dist_matrix.shape[1] <= dist_matrix.shape[0]
        num_cols = dist_matrix.shape[1]

        row_indices = []
        col_indices = []

        for _ in range(num_cols):
            min_pos = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)

            row_indices.append(min_pos[0])
            col_indices.append(min_pos[1])

            # Set the found row and column to np.inf to ignore it
            dist_matrix[min_pos[0], :] = np.inf
            dist_matrix[:, min_pos[1]] = np.inf
        return row_indices, col_indices

    @staticmethod
    def _chain_map_1to2_in_a_entity(
        chain_ids1: list[str],
        chain_ids2: list[str],
        struct1: Structure,
        struct2: Structure,
        coords1: np.ndarray,
        coords2: np.ndarray,
    ) -> dict[str, str]:
        """
        Chain mapping between two structures within the same entity using
        spatial proximity of common atoms.

        Strategy:
        1. Ensure struct1 always contains fewer chains for unidirectional mapping
        2. For each chain in struct1, find closest chain in struct2 by:
            - Identifying intersecting atoms present in both chains
            - Calculating centroid distance of these common atoms
            - Selecting the pair with minimal centroid distance

        Args:
            chain_ids1 (list[str]): Chain IDs from first structure (may become struct2 after swap)
            chain_ids2 (list[str]): Chain IDs from second structure (may become struct1 after swap)
            struct1 (Structure): First structure containing chain metadata
            struct2 (Structure): Second structure containing chain metadata
            coords1 (np.ndarray): Atom coordinates for struct1 (shape: [N, 3])
            coords2 (np.ndarray): Atom coordinates for struct2 (shape: [M, 3])

        Returns:
            dict[str, str]: Mapping of chain IDs from struct1 to struct2.
        """
        if len(chain_ids2) > len(chain_ids1):
            # Swap if struct2 has more chains
            chain_ids1, chain_ids2 = chain_ids2, chain_ids1
            struct1, struct2 = struct2, struct1
            coords1, coords2 = coords2, coords1
            swapped = True
        else:
            swapped = False

        chain_ids1_indices = {cid: i for i, cid in enumerate(chain_ids1)}
        chain_ids2_indices = {cid: i for i, cid in enumerate(chain_ids2)}

        dist_mat = np.full((len(chain_ids1), len(chain_ids2)), np.inf)

        for cid1 in chain_ids1:
            mask1 = struct1.uni_chain_id == cid1
            atoms1 = struct1.uni_atom_id[mask1]

            for cid2 in chain_ids2:
                mask2 = struct2.uni_chain_id == cid2
                atoms2 = struct2.uni_atom_id[mask2]

                (
                    common_atoms1,
                    common_atoms2,
                ) = ChainPermutation._align_uni_atom_id_in_chain(atoms1, atoms2)

                center1 = coords1[mask1][common_atoms1].mean(axis=0)
                center2 = coords2[mask2][common_atoms2].mean(axis=0)
                dist_mat[
                    chain_ids1_indices[cid1], chain_ids2_indices[cid2]
                ] = np.linalg.norm(center1 - center2)

        matched_chains = {}
        row_indices, col_indices = ChainPermutation._find_min_indices(dist_mat)
        for row, col in zip(row_indices, col_indices):
            matched_chains[chain_ids1[row]] = chain_ids2[col]

        return (
            matched_chains if not swapped else {v: k for k, v in matched_chains.items()}
        )

    def greedy_match_for_chains(
        self,
        model_anchor_chain_id: str,
        ref_anchor_chain_id: str,
        aligned_ref_coord: np.ndarray,
    ) -> dict[str, str]:
        """
        Expand chain matching from anchor pair to all chains through entity relationships.

        1. Start with anchor chain mapping
        2. For each entity in model structure:
           - Get corresponding reference entity chains (excluding anchor)
           - Find 1:1 chain mapping within entity using spatial alignment
           - Merge mappings across entities

        Args:
            model_anchor_chain_id: Already matched anchor chain in model structure
            ref_anchor_chain_id: Corresponding anchor chain in reference structure
            aligned_ref_coord: Reference coordinates after optimal alignment

        Returns:
            Complete chain mapping dictionary containing:
            - Initial anchor chain pair
            - All entity-specific chain mappings
        """
        matched_chains = {ref_anchor_chain_id: model_anchor_chain_id}

        for (
            model_entity_id,
            model_chain_ids,
        ) in self.model_entity_id_to_chain_ids.items():
            ref_entity = self.model_to_ref_entity_id.get(model_entity_id)
            if not ref_entity:
                continue

            ref_chain_ids = self.ref_entity_id_to_chain_ids[ref_entity]
            ref_chain_ids = [i for i in ref_chain_ids if i != ref_anchor_chain_id]
            model_chain_ids = [i for i in model_chain_ids if i != model_anchor_chain_id]
            matched_chains_in_curr_entity = self._chain_map_1to2_in_a_entity(
                chain_ids1=ref_chain_ids,
                chain_ids2=model_chain_ids,
                struct1=self.ref_struct,
                struct2=self.model_struct,
                coords1=aligned_ref_coord,
                coords2=self.model_struct.atom_array.coord,
            )
            matched_chains.update(matched_chains_in_curr_entity)
        return matched_chains

    def _compute_rmsd_by_matched_chains(
        self, ref_to_model_matched_chains: dict[str, str], aligned_ref_coord: np.ndarray
    ) -> float:
        """
        Calculate average RMSD across matched chain pairs after alignment.

        1. For each matched chain pair:
           - Identify intersecting atoms using unique atom IDs and validity masks
           - Extract aligned reference coordinates and original model coordinates
           - Compute chain-level RMSD using common atoms
        2. Return mean RMSD across all matched pairs

        Args:
            ref_to_model_matched_chains (dict[str, str]): Mapping of reference
                                        chain IDs to model chain IDs
            aligned_ref_coord (np.ndarray): Reference structure coordinates after optimal alignment
                              (shape: [N, 3])

        Returns:
            float: Mean RMSD across all matched chain pairs
        """
        ref_coords = []
        model_coords = []
        for ref_chain_id, model_chain_id in ref_to_model_matched_chains.items():
            ref_chain_mask = self.ref_struct.uni_chain_id == ref_chain_id
            model_chain_mask = self.model_struct.uni_chain_id == model_chain_id

            (
                ref_chain_indices,
                model_chain_indices,
            ) = ChainPermutation._align_uni_atom_id_in_chain(
                self.ref_struct.uni_atom_id[ref_chain_mask],
                self.model_struct.uni_atom_id[model_chain_mask],
            )

            ref_chain_coord = aligned_ref_coord[ref_chain_mask][ref_chain_indices]
            model_chain_coord = self.model_struct.atom_array.coord[model_chain_mask][
                model_chain_indices
            ]
            ref_coords.append(ref_chain_coord)
            model_coords.append(model_chain_coord)

        ref_coords = np.concatenate(ref_coords)
        model_coords = np.concatenate(model_coords)

        rmsd_value = rmsd(ref_coords, model_coords)
        return rmsd_value

    @staticmethod
    def _align_uni_atom_id_in_chain(
        ref_uni_atom_id: np.ndarray, model_uni_atom_id: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Align unique atom IDs between reference and model structures.

        This function finds the common unique atom IDs between
        the reference and model structures, then sorts the indices of
        these common atoms in both structures based on the order
        in the reference structure.

        Args:
            ref_uni_atom_id (np.ndarray): Unique atom IDs in reference structure
            model_uni_atom_id (np.ndarray): Unique atom IDs in model structure

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - Sorted reference atom indices
                - Sorted model atom indices
        """
        ref_mask = np.isin(ref_uni_atom_id, model_uni_atom_id)

        assert np.any(ref_mask), "No common atoms between ref and model chains"

        ref_indices = np.where(ref_mask)[0]

        ref_uid_filtered = ref_uni_atom_id[ref_indices]

        assert len(np.unique(ref_uid_filtered)) == len(
            ref_uid_filtered
        ), "Duplicate atoms in ref chain"

        uid_index_map = {value: idx for idx, value in enumerate(ref_uid_filtered)}

        model_mask = np.isin(model_uni_atom_id, ref_uid_filtered)
        model_indices_unsorted = np.where(model_mask)[0]

        # Sort model indices based on ref indices
        model_uid_filtered = model_uni_atom_id[model_indices_unsorted]

        assert len(np.unique(model_uid_filtered)) == len(
            model_uid_filtered
        ), "Duplicate atoms in model chain"

        model_indices = model_indices_unsorted[
            np.argsort(np.vectorize(uid_index_map.get)(model_uid_filtered))
        ]
        return ref_indices, model_indices

    def find_ref_to_model_optimal_chain_mapping(
        self, model_anchor_chain_id: str, ref_anchor_candidates: list[str]
    ) -> tuple[dict[str, str], float, dict[str, str]]:
        """
        Find optimal chain mapping between reference and model structures by:
        1. Testing each candidate reference anchor chain
        2. Performing structure alignment using common atoms
        3. Greedily matching remaining chains
        4. Selecting the mapping with lowest overall RMSD

        Args:
            model_anchor_chain_id (str): Fixed anchor chain in model structure
            ref_anchor_candidates (list[str]): Candidate anchor chains in reference structure
                                  to test for optimal alignment

        Returns:
            tuple[dict[str, str], float, dict[str, str]]:
                 - Dictionary mapping reference chain IDs to model chain IDs that
                 - gives the lowest RMSD after full structure alignment,
                 - The lowest RMSD value.
                 - Anchors used for alignment.
        """
        model_chain_mask = self.model_struct.uni_chain_id == model_anchor_chain_id
        model_anchor_coord = self.model_struct.atom_array.coord[model_chain_mask]
        model_anchor_uni_atom_id = self.model_struct.uni_atom_id[model_chain_mask]

        # Find best match
        best_rmsd = float("inf")
        ref_to_model_optimal_mapping = None
        anchors = {}

        for ref_anchor_chain_id in ref_anchor_candidates:
            # Find atoms in ref chain to match atoms in model chain
            ref_chain_mask = self.ref_struct.uni_chain_id == ref_anchor_chain_id
            ref_anchor_coord = self.ref_struct.atom_array.coord[ref_chain_mask]

            ref_anchor_uni_atom_id = self.ref_struct.uni_atom_id[ref_chain_mask]

            (
                ref_chain_indices,
                model_chain_indices,
            ) = ChainPermutation._align_uni_atom_id_in_chain(
                ref_anchor_uni_atom_id, model_anchor_uni_atom_id
            )

            rot, trans = align_src_to_tar(
                src_pose=ref_anchor_coord[ref_chain_indices],
                tar_pose=model_anchor_coord[model_chain_indices],
            )

            # Transform all ref coordinates according to the aligment results
            aligned_ref_coord = apply_transform(
                self.ref_struct.atom_array.coord, rot, trans
            )

            # Greedily matches all remaining chains
            ref_to_model_matched_chains = self.greedy_match_for_chains(
                model_anchor_chain_id, ref_anchor_chain_id, aligned_ref_coord
            )

            # Calculate RMSD
            total_rmsd = self._compute_rmsd_by_matched_chains(
                ref_to_model_matched_chains, aligned_ref_coord
            )

            if total_rmsd < best_rmsd:
                best_rmsd = total_rmsd
                ref_to_model_optimal_mapping = ref_to_model_matched_chains
                anchors["ref"] = ref_anchor_chain_id
                anchors["model"] = model_anchor_chain_id
        return ref_to_model_optimal_mapping, best_rmsd, anchors

    def get_permuted_indices(
        self, ref_to_model_mapping: dict[str, str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert chain mapping to atom indices while preserving reference chain order.

        Args:
            ref_to_model_mapping: Dictionary mapping reference chain IDs to model chain IDs

        Returns:
            tuple: (np.ndarray, np.ndarray) where:
                - ref_indices: Array of reference structure atom indices ordered by:
                    1. Original reference chain order
                    2. Within-chain atom order
                - model_indices: Corresponding model structure atom indices aligned
                               to reference chain order
        """

        # Get original order in reference structure
        ref_chain_ids, idx = np.unique(self.ref_struct.uni_chain_id, return_index=True)
        sorted_ref_chain_ids = ref_chain_ids[np.argsort(idx)]

        ref_indices = []
        model_indices = []
        for ref_chain_id in sorted_ref_chain_ids:
            if ref_chain_id not in ref_to_model_mapping:
                continue
            model_chain_id = ref_to_model_mapping[ref_chain_id]

            ref_chain_indices = np.where(self.ref_struct.uni_chain_id == ref_chain_id)[
                0
            ]
            model_chain_indices = np.where(
                self.model_struct.uni_chain_id == model_chain_id
            )[0]

            (
                ref_chain_aligned_indices,
                model_chain_aligned_indices,
            ) = ChainPermutation._align_uni_atom_id_in_chain(
                self.ref_struct.uni_atom_id[ref_chain_indices],
                self.model_struct.uni_atom_id[model_chain_indices],
            )

            ref_indices.extend(ref_chain_indices[ref_chain_aligned_indices])
            model_indices.extend(model_chain_indices[model_chain_aligned_indices])

        ref_indices = np.array(ref_indices)
        model_indices = np.array(model_indices)

        assert np.all(
            self.ref_struct.uni_atom_id[ref_indices]
            == self.model_struct.uni_atom_id[model_indices]
        ), "Unique atom id not match after chain permutation."
        return ref_indices, model_indices

    def get_heurisitic_chain_mapping(self) -> tuple[dict[str, str], dict[str, str]]:
        """
        Chain permutation heuristic algorithm.

        Implements the core workflow:
        1. Identify model anchor chain using entity/residue prioritization
        2. Find corresponding reference entity chains as candidates
        3. Compute optimal chain mapping through alignment and RMSD evaluation

        Returns:
            tuple[dict[str, str], dict[str, str]]:
                - Final chain mapping from reference to model chains
                  that minimizes overall RMSD
                - Anchors used for alignment.
        """
        model_anchor_chain_ids = self.find_model_anchor_chains()
        if not self.enumerate_all_anchors:
            # Only use the first anchor chain
            model_anchor_chain_ids = [model_anchor_chain_ids[0]]

        ref_to_model_optimal_mapping = None
        best_rmsd = float("inf")
        best_anchors = None
        for model_anchor_chain_id in model_anchor_chain_ids:
            model_anchor_entity_id = self.model_chain_id_to_entity_id[
                model_anchor_chain_id
            ]

            ref_entity = self.model_to_ref_entity_id.get(model_anchor_entity_id)
            if not ref_entity:
                continue

            ref_anchor_candidates = self.ref_entity_id_to_chain_ids[ref_entity]
            (
                mapping_i,
                best_rmsd_i,
                anchors,
            ) = self.find_ref_to_model_optimal_chain_mapping(
                model_anchor_chain_id, ref_anchor_candidates
            )
            if best_rmsd_i < best_rmsd:
                best_rmsd = best_rmsd_i
                ref_to_model_optimal_mapping = mapping_i
                best_anchors = anchors
        return ref_to_model_optimal_mapping, best_anchors
