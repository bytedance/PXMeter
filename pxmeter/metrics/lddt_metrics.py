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
from scipy.spatial import KDTree

from pxmeter.constants import DNA, RNA
from pxmeter.data.struct import Structure


class LDDT:
    """
    LDDT base metrics

    Args:
            ref_struct (Structure): reference Structure object.
            model_struct (Structure): model Structure object.
            is_nucleotide_threshold (float): Threshold distance for
                                    nucleotide atoms. Defaults to 30.0.
            is_not_nucleotide_threshold (float): Threshold distance for
                                        non-nucleotide atoms. Defaults to 15.0.
            eps (float): epsilon for numerical stability. Defaults to 1e-10.
    """

    def __init__(
        self,
        ref_struct: Structure,
        model_struct: Structure,
        is_nucleotide_threshold=30.0,
        is_not_nucleotide_threshold=15.0,
        eps: float = 1e-10,
    ):
        self.ref_struct = ref_struct
        self.model_struct = model_struct

        self.is_nucleotide_threshold = is_nucleotide_threshold
        self.is_not_nucleotide_threshold = is_not_nucleotide_threshold
        self.eps = eps

        self.lddt_atom_pair = self.compute_lddt_atom_pair()

    @staticmethod
    def _get_pair_from_kdtree(
        kdtree: KDTree,
        ref_coords: np.ndarray,
        radius: float,
        subset_mask: np.ndarray,
    ) -> set[tuple[int, int]]:
        result_pairs = []
        subset_index = np.nonzero(subset_mask)[0]
        indices = kdtree.query_ball_point(ref_coords[subset_mask], r=radius)
        for idx, j_list in enumerate(indices):
            i = subset_index[idx]
            for j in j_list:
                if i != j:
                    result_pairs.append((i, j))
        result_pairs = set(result_pairs)
        return result_pairs

    def compute_lddt_atom_pair(self) -> tuple[np.ndarray]:
        """
        Calculate the atom pair mask with the bespoke radius

        Returns:
            np.ndarray: index of atom pairs [N_pair_sparse, 2]
        """
        ref_coords = self.ref_struct.atom_array.coord
        nuc_entities = [
            k for k, v in self.ref_struct.entity_poly_type.items() if v in (DNA, RNA)
        ]
        is_nuc = np.isin(self.ref_struct.atom_array.label_entity_id, nuc_entities)

        # Restrict to bespoke inclusion radius
        kdtree = KDTree(ref_coords)
        atom_pairs = set()

        if np.any(is_nuc):
            nuc_pairs = LDDT._get_pair_from_kdtree(
                kdtree,
                ref_coords,
                radius=self.is_nucleotide_threshold,
                subset_mask=is_nuc,
            )
            atom_pairs |= nuc_pairs

        if np.any(~is_nuc):
            non_nuc_pairs = LDDT._get_pair_from_kdtree(
                kdtree,
                ref_coords,
                radius=self.is_not_nucleotide_threshold,
                subset_mask=~is_nuc,
            )
            atom_pairs |= non_nuc_pairs

        assert atom_pairs, "No atom pairs found for LDDT calculation."
        atom_pairs = np.array(list(atom_pairs))
        return atom_pairs

    def _calc_lddt(
        self, model_dist_sparse_lm: np.ndarray, ref_dist_sparse_lm: np.ndarray
    ) -> float:
        """
        Calculate LDDT scores from predicted and true atom pair distances.

        Args:
            model_dist_sparse_lm: [N_pair_sparse] Distances between atom pairs in model structure.
            ref_dist_sparse_lm: [N_pair_sparse] Distances between atom pairs in reference structure.

        Returns:
            float: LDDT scores ranging 0-1
        """
        distance_error_l1 = np.abs(
            model_dist_sparse_lm - ref_dist_sparse_lm
        )  # [N_pair_sparse]

        thresholds = [0.5, 1.0, 2.0, 4.0]
        sparse_pair_lddt = np.mean(
            np.stack([distance_error_l1 < t for t in thresholds], axis=-1), axis=-1
        )  # [N_pair_sparse]

        # Compute mean
        lddt = np.mean(sparse_pair_lddt)
        return lddt

    def _calc_sparse_dist(
        self,
        l_index: np.ndarray,
        m_index: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate pairwise distances between selected atom
        in predicted and true structures.

        Args:
            l_index: Atom indices for first group [N_pair_sparse]
            m_index: Atom indices for second group [N_pair_sparse]

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - Model distances between l/m groups [N_pair_sparse],
                - Reference distances between l/m groups [N_pair_sparse]
        """
        # [N_atom_sparse_l, 3]
        model_coords_l = self.model_struct.atom_array.coord[l_index]
        # [N_atom_sparse_m, 3]
        model_coords_m = self.model_struct.atom_array.coord[m_index]
        # [N_atom_sparse_l, 3]
        ref_coords_l = self.ref_struct.atom_array.coord[l_index]
        # [N_atom_sparse_m, 3]
        ref_coords_m = self.ref_struct.atom_array.coord[m_index]

        # [N_pair_sparse]
        model_dist_sparse_lm = np.linalg.norm(
            model_coords_l - model_coords_m, axis=-1, ord=2
        )
        ref_dist_sparse_lm = np.linalg.norm(ref_coords_l - ref_coords_m, axis=-1, ord=2)

        return model_dist_sparse_lm, ref_dist_sparse_lm

    def _get_lddt_atom_pair_for_chain_mask(
        self,
        chain_1_mask: np.ndarray,
        chain_2_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Get atom pair mask for chain interface evaluation.
        If the evaluation is for a single chain, the chain_2_mask is the same as the chain_1_mask.

        Args:
            chain_1_mask (np.ndarray): [N_atom] Atom mask for chain 1.

            chain_2_mask (np.ndarray): [N_atom] Atom mask for chain 2.

        Returns:
            np.ndarray: Atom pair mask for chain / interface evaluation [N_pair_sparse]
        """
        chain_1_index = np.nonzero(chain_1_mask)[0]
        chain_2_index = np.nonzero(chain_2_mask)[0]

        mask1 = np.isin(self.lddt_atom_pair[:, 0], chain_1_index) & np.isin(
            self.lddt_atom_pair[:, 1], chain_2_index
        )
        mask2 = np.isin(self.lddt_atom_pair[:, 0], chain_2_index) & np.isin(
            self.lddt_atom_pair[:, 1], chain_1_index
        )
        assert np.any(
            mask1 | mask2
        ), "No atom pair found for chain / interface evaluation."
        interface_atom_pair = self.lddt_atom_pair[mask1 | mask2]
        return interface_atom_pair

    def run(
        self,
        chain_1_masks: np.ndarray = None,
        chain_2_masks: np.ndarray = None,
    ) -> float | list[float]:
        """
        Run LDDT calculation for complex / chain / interface evaluation.
        If the evaluation is for whole complex, the chain_1_mask and chain_2_mask are None.
        If the evaluation is for a single chain, the chain_2_mask is the same as the chain_1_mask.

        Args:
            chain_1_mask (np.ndarray, optional): [N_atom] Atom mask for chain 1. Defaults to None.
            chain_2_mask (np.ndarray, optional): [N_atom] Atom mask for chain 2. Defaults to None.

        Returns:
            np.ndarray: LDDT scores. If evaluating chain interfaces, the shape is [N_eval].
                        Otherwise, the shape is [1].
        """
        eval_chain_interface = chain_1_masks is not None and chain_2_masks is not None

        if not eval_chain_interface:
            l_index = self.lddt_atom_pair[:, 0]
            m_index = self.lddt_atom_pair[:, 1]
            model_dist_sparse_lm, ref_dist_sparse_lm = self._calc_sparse_dist(
                l_index, m_index
            )
            lddt_value = self._calc_lddt(model_dist_sparse_lm, ref_dist_sparse_lm)
        else:
            n_eval = chain_1_masks.shape[0]
            lddt_value = []  # [N_eval]
            for i in range(n_eval):
                interface_atom_pair = self._get_lddt_atom_pair_for_chain_mask(
                    chain_1_masks[i], chain_2_masks[i]
                )
                l_index = interface_atom_pair[:, 0]
                m_index = interface_atom_pair[:, 1]

                model_dist_sparse_lm, ref_dist_sparse_lm = self._calc_sparse_dist(
                    l_index, m_index
                )
                lddt_value_i = self._calc_lddt(
                    model_dist_sparse_lm,
                    ref_dist_sparse_lm,
                )
                lddt_value.append(lddt_value_i)
        return lddt_value
