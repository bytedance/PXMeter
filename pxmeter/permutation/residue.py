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

from typing import Optional

import networkx as nx
import numpy as np

from pxmeter.data.struct import Structure
from pxmeter.data.utils import get_res_graph_matches
from pxmeter.metrics.rmsd import align_src_to_tar, apply_transform, rmsd


class ResiduePermutation:
    """
    Generating and applying residue permutations based on a reference structure.

    Args:
        ref_struct (Structure): The reference structure used for permutation generation.
        model_struct (Structure): The model structure used for permutation application.
    """

    def __init__(self, ref_struct: Structure, model_struct: Structure):
        self.ref_struct = ref_struct
        self.model_struct = model_struct

    @staticmethod
    def _calc_residue_centers(
        res_ids: np.ndarray, coords: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the geometric center (mean coordinate) for each unique residue ID.

        Args:
            res_ids (np.ndarray): Array of residue IDs, shape (N_atom,)
            coords (np.ndarray): Cartesian coordinates, shape (N_atom, 3)

        Returns:
            Tuple[np.ndarray, np.ndarray]: (uniq_res_ids, centers) where
            uniq_res_ids shape (N_res,), centers shape (N_res, 3) in the same order.
        """
        res_ids = np.asarray(res_ids)
        coords = np.asarray(coords)
        if (
            res_ids.ndim != 1
            or coords.ndim != 2
            or coords.shape[1] != 3
            or len(res_ids) != len(coords)
        ):
            raise ValueError("Shape mismatch: res_ids (N,), coords (N, 3) required.")

        uniq_ids, inv = np.unique(res_ids, return_inverse=True)
        centers = np.zeros((len(uniq_ids), 3), dtype=float)
        counts = np.bincount(inv).astype(float)
        for i in range(3):
            centers[:, i] = np.bincount(inv, weights=coords[:, i])
        centers /= counts[:, None]
        return uniq_ids, centers

    @staticmethod
    def _get_branch_residue_permutations(
        struct: Structure, chain_id: str
    ) -> Optional[np.ndarray]:
        """
        Detect branch-like connectivity within a chain using non-adjacent residue bonds and,
        if the induced residue-level graph is a single tree, return residue permutations
        corresponding to its graph automorphisms.

        The procedure:
        1) Filter atoms by ``chain_id`` and collect inter-residue bonds from the chain's
            atom-level ``BondList``.
        2) If any inter-residue bond connects residues whose numeric IDs differ by more than 1
            (``|res_id_i - res_id_j| > 1``), mark the chain as having a branch-like connection.
        3) Lift inter-residue bonds to a residue-level undirected graph G (nodes = ``res_id``,
            edges = covalent connections between residues).
        4) If G is disconnected or contains cycles, return ``None`` (only tree-shaped branches
            are supported).
        5) Annotate nodes with ``res_name`` and enumerate automorphisms via subgraph isomorphism
            of G onto itself (constrained by equal ``res_name``). Return their induced permutations.

        Args:
            struct (Structure): A structure object exposing ``atom_array`` with fields
                ``res_id``, ``res_name``, ``uni_chain_id``, and ``bonds``; and where
                ``atom_array.bonds[mask].as_array()`` yields an ``(n_bond, 2)`` integer array
                of atom index pairs for the selected chain.
            chain_id (str): The target chain identifier matched against ``uni_chain_id``.

        Returns:
            np.ndarray | None: ``None`` if no branch-like non-adjacent residue bond is detected,
            or if the residue graph is not a single tree. Otherwise an integer array of shape
            ``(K, N)`` where each row encodes one automorphism as a permutation of the ``N``
            residue nodes (ordered by ascending source node id). ``K`` is the number of
            automorphisms found (capped internally at 1000).
        """
        mask = struct.uni_chain_id == chain_id
        arr = struct.atom_array

        if not np.any(mask):
            return

        if arr.bonds is None:
            return

        bond_arr = arr.bonds[mask].as_array()

        res_id_i = arr.res_id[mask][bond_arr[:, 0]]
        res_id_j = arr.res_id[mask][bond_arr[:, 1]]

        res_id_pairs = set(tuple(zip(res_id_i, res_id_j)))
        has_branch = False
        nodes_adj = set()
        for i, j in res_id_pairs:
            if i == j:
                continue
            nodes_adj.add((i, j))

            if abs(i - j) > 1:
                has_branch = True

        if has_branch:
            G = nx.Graph()
            G.add_edges_from(nodes_adj)
            if (
                nx.number_connected_components(G) > 1
                or len(nx.cycle_basis(G)) > 0
                or (1 not in G.nodes)
            ):
                return

            attrs = {}
            for node in G.nodes:
                node_res_name = arr.res_name[mask][arr.res_id[mask] == node][0]
                node_atom_names = "_".join(
                    sorted(arr.atom_name[mask][arr.res_id[mask] == node])
                )

                if node == 1:
                    # Do not permute the root residue
                    node_res_name += "_root"

                attrs[node] = {"res_name": node_res_name, "atom_names": node_atom_names}
            nx.set_node_attributes(G, attrs)
            matches = get_res_graph_matches(G, G, max_matches=1000)

            perm = []
            for match in matches:
                sorted_result = sorted(match.items(), key=lambda x: x[0])
                match_values = [i[1] for i in sorted_result]
                if match_values[0] != 1:
                    continue
                perm.append(match_values)

            if len(perm) > 1:
                perm = np.array(perm)
                return perm

    def _get_optimal_perm_ids_for_chain(self, chain_id: str) -> Optional[np.ndarray]:
        """
        Compute the residue-ID permutation for a branch-like chain that best aligns
        the model to the reference, measured by centroid RMSD.

        Steps:
            1) Detects residue-level graph automorphisms for the chain (if the chain
                exhibits non-adjacent inter-residue bonds indicating a branch-like tree).
            2) Treats residue 1 as fixed (root) to define the rigid alignment between
                the model and the reference using root-atom coordinates.
            3) For each candidate permutation of residue IDs, applies the rigid
                transform to model residue centroids and computes RMSD to the reference
                residue centroids.
            4) Returns the residue-ID permutation that minimizes this RMSD.

        Args:
        chain_id (str): Target chain identifier to evaluate.

        Returns:
            np.ndarray | None: If the chain has a valid branch-like tree and at least
                one non-trivial automorphism, returns an integer array of shape (N,)
                containing the residue IDs in the selected order (1-based, matching the
                original residue numbering). Returns ``None`` if no branch-like structure
                is detected or no valid permutations are found.
        """
        perm = self._get_branch_residue_permutations(self.model_struct, chain_id)
        if perm is None:
            return

        chain_mask = self.model_struct.uni_chain_id == chain_id

        # Use the residue 1 as the root
        model_root_idx = np.where(
            chain_mask & (self.model_struct.atom_array.res_id == 1)
        )[0]
        ref_root_idx = np.where(chain_mask & (self.ref_struct.atom_array.res_id == 1))[
            0
        ]

        model_names = self.model_struct.atom_array.atom_name[model_root_idx]
        ref_names = self.ref_struct.atom_array.atom_name[ref_root_idx]

        # Get common atoms and align them by atom name to remove order dependency.
        # np.intersect1d returns sorted common_names, so the coordinates extracted
        # using model_comm_idx and ref_comm_idx will follow the same atom-name order.
        common_names, model_comm_idx, ref_comm_idx = np.intersect1d(
            model_names, ref_names, return_indices=True
        )

        # For a stable 3D rigid body alignment (rotation + translation),
        # at least 3 atoms are required.
        if len(common_names) < 3:
            return

        model_root = self.model_struct.atom_array.coord[model_root_idx[model_comm_idx]]
        ref_root = self.ref_struct.atom_array.coord[ref_root_idx[ref_comm_idx]]

        rot, trans = align_src_to_tar(model_root, ref_root)

        _ref_ids, ref_centers = self._calc_residue_centers(
            self.ref_struct.atom_array.res_id[chain_mask],
            self.ref_struct.atom_array.coord[chain_mask],
        )
        model_ids, model_centers = self._calc_residue_centers(
            self.model_struct.atom_array.res_id[chain_mask],
            self.model_struct.atom_array.coord[chain_mask],
        )
        model_pos = {rid: i for i, rid in enumerate(model_ids)}

        best_perm = None
        best_rmsd = np.inf
        for ids in perm:
            ordered = np.array([model_pos[i] for i in ids], dtype=int)
            model_mat = model_centers[ordered]

            transformed = apply_transform(model_mat, rot, trans)

            vmask1 = None
            vmask2 = None

            # Compute residue-level valid mask if available
            if self.model_struct.valid_mask is not None:
                # ids is the permuted indices for model
                # chain_mask is the mask for the chain in model_struct
                model_valid = self.model_struct.valid_mask[chain_mask]
                res_ids_for_mask = self.model_struct.atom_array.res_id[chain_mask]
                uniq_ids_v, inv_v = np.unique(res_ids_for_mask, return_inverse=True)
                vmask1 = np.bincount(inv_v, weights=model_valid) > 0

            if self.ref_struct.valid_mask is not None:
                ref_valid = self.ref_struct.valid_mask[chain_mask]
                res_ids_for_mask = self.ref_struct.atom_array.res_id[chain_mask]
                uniq_ids_v, inv_v = np.unique(res_ids_for_mask, return_inverse=True)
                vmask2 = np.bincount(inv_v, weights=ref_valid) > 0

            # Note: since p permutes the residue sequence, we need to permute vmask1
            if vmask1 is not None:
                vmask1_p = vmask1[ordered]
            else:
                vmask1_p = None

            v = rmsd(transformed, ref_centers, valid_mask1=vmask1_p, valid_mask2=vmask2)

            if v < best_rmsd:
                best_rmsd = v
                best_perm = ids

        return best_perm

    def run(self):
        """
        Reorder model atoms within non-polymer chains according to the
        RMSD-optimal residue-ID permutation per chain.

        For each non-polymer entity and its chains:
            - Detect branch-like residue graphs and enumerate automorphisms.
            - Select the permutation of residue IDs that minimizes centroid RMSD
                to the reference (via rigid alignment anchored at residue 1).
            - Stably reorder atom indices of that chain so atoms follow the selected
                residue-ID order (preserving within-residue atom order).

        Returns:
            np.ndarray: A 1-D integer array of length,
                representing the remapped atom indices.
        """
        model_index = np.arange(len(self.model_struct.atom_array))
        model_entity_id_to_chain_ids = self.model_struct.get_entity_id_to_chain_ids()
        for entity_id, chain_ids in model_entity_id_to_chain_ids.items():
            if entity_id in self.model_struct.entity_poly_type:
                # Skip polymer
                continue

            for chain_id in chain_ids:
                optimal_perm_ids = self._get_optimal_perm_ids_for_chain(chain_id)
                if optimal_perm_ids is None:
                    continue
                chain_mask = self.model_struct.uni_chain_id == chain_id
                model_chain_index = model_index[chain_mask]
                sorted_atom_index = np.concatenate(
                    [
                        model_chain_index[
                            self.model_struct.atom_array.res_id[model_chain_index] == i
                        ]
                        for i in optimal_perm_ids
                    ]
                )
                model_index[chain_mask] = sorted_atom_index
        return model_index
