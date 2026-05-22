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

from typing import Optional, Sequence, Union

import numpy as np
from scipy.spatial import KDTree

from pxmeter.constants import DNA, DNA_RNA_HYBRID, POLYMER, PROTEIN_D, RNA
from pxmeter.data.struct import Structure
from pxmeter.metrics.stereochemistry.check import StereoChemValidator


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
        stereochecks: bool = False,
        lddt_thresholds: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
    ):
        self.ref_struct = ref_struct
        self.model_struct = model_struct

        self.is_nucleotide_threshold = is_nucleotide_threshold
        self.is_not_nucleotide_threshold = is_not_nucleotide_threshold
        self.eps = eps
        self.lddt_thresholds = lddt_thresholds

        # When stereochecks is enabled, we also cache the underlying violation
        # DataFrames for downstream reporting.
        self.stereo_violation_dfs = None
        self.model_atom_mask = (
            self._get_model_stereo_valid_atom_mask() if stereochecks else None
        )

        # Add valid_mask from model_struct to model_atom_mask
        if self.model_struct.valid_mask is not None:
            if self.model_atom_mask is not None:
                self.model_atom_mask = (
                    self.model_atom_mask & self.model_struct.valid_mask
                )
            else:
                self.model_atom_mask = self.model_struct.valid_mask.copy()

        self.lddt_atom_pair = self.compute_lddt_atom_pair()

        model_dist_all, ref_dist_all = self._calc_sparse_dist(
            self.lddt_atom_pair[:, 0], self.lddt_atom_pair[:, 1]
        )
        dist_err_all = np.abs(model_dist_all - ref_dist_all)
        # [N_pairs, 4]
        per_thr_scores_all = np.stack(
            [dist_err_all < t for t in self.lddt_thresholds], axis=-1
        ).astype(float)

        if self.model_atom_mask is not None:
            l_idx = self.lddt_atom_pair[:, 0]
            m_idx = self.lddt_atom_pair[:, 1]
            pair_valid_mask_all = (
                self.model_atom_mask[l_idx] & self.model_atom_mask[m_idx]
            )
            per_thr_scores_all *= pair_valid_mask_all[:, None]

        self.per_pair_lddt_all = np.mean(per_thr_scores_all, axis=-1)

    def _get_model_stereo_valid_atom_mask(self) -> np.ndarray:
        checker = StereoChemValidator(
            struct=self.model_struct, ref_struct=self.ref_struct
        )
        model_atom_mask = checker.get_valid_atom_mask()
        # Populated by `StereoChemValidator.get_valid_atom_mask()`.
        self.stereo_violation_dfs = getattr(checker, "_last_violation_dfs", None)
        return model_atom_mask

    @staticmethod
    def _get_pair_from_kdtree(
        kdtree: KDTree,
        ref_coords: np.ndarray,
        radius: float,
        subset_mask: np.ndarray,
    ) -> np.ndarray:
        subset_index = np.nonzero(subset_mask)[0]
        indices = kdtree.query_ball_point(ref_coords[subset_mask], r=radius)

        lens = [len(j_list) for j_list in indices]
        i_indices = np.repeat(subset_index, lens)

        if len(indices) == 0 or sum(lens) == 0:
            return np.zeros((0, 2), dtype=int)

        j_indices = np.concatenate(indices)
        mask = i_indices != j_indices
        return np.stack([i_indices[mask], j_indices[mask]], axis=1)

    def compute_lddt_atom_pair(self) -> np.ndarray:
        """
        Calculate the atom pair mask with the bespoke radius

        Returns:
            np.ndarray: index of atom pairs [N_pair_sparse, 2]
        """
        ref_coords = self.ref_struct.atom_array.coord
        is_nuc = self.ref_struct.get_mask_for_given_entity_types([DNA, RNA])

        # Restrict to bespoke inclusion radius
        kdtree = KDTree(ref_coords)

        all_pairs = []
        if np.any(is_nuc):
            all_pairs.append(
                LDDT._get_pair_from_kdtree(
                    kdtree,
                    ref_coords,
                    radius=self.is_nucleotide_threshold,
                    subset_mask=is_nuc,
                )
            )

        if np.any(~is_nuc):
            all_pairs.append(
                LDDT._get_pair_from_kdtree(
                    kdtree,
                    ref_coords,
                    radius=self.is_not_nucleotide_threshold,
                    subset_mask=~is_nuc,
                )
            )

        atom_pairs = np.concatenate(all_pairs, axis=0)
        assert atom_pairs.shape[0] > 0, "No atom pairs found for LDDT calculation."

        return atom_pairs

    def _calc_lddt(
        self,
        pair_indices: np.ndarray,
    ) -> float:
        """
        Calculate LDDT scores from pre-calculated per-pair scores.
        """
        return np.mean(self.per_pair_lddt_all[pair_indices])

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

    def _get_lddt_atom_pair_indices_for_chain_mask(
        self,
        chain_1_mask: np.ndarray,
        chain_2_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Get atom pair indices for chain interface evaluation.
        If the evaluation is for a single chain, the chain_2_mask is the same as the chain_1_mask.

        Args:
            chain_1_mask (np.ndarray): [N_atom] Atom mask for chain 1.

            chain_2_mask (np.ndarray): [N_atom] Atom mask for chain 2.

        Returns:
            np.ndarray: Atom pair indices for chain / interface evaluation [N_pair_interface]
        """
        # Optimized: use boolean indexing
        idx_i = self.lddt_atom_pair[:, 0]
        idx_j = self.lddt_atom_pair[:, 1]

        mask1 = chain_1_mask[idx_i] & chain_2_mask[idx_j]
        mask2 = chain_2_mask[idx_i] & chain_1_mask[idx_j]

        return np.where(mask1 | mask2)[0]

    def _apply_atom_mask_to_pair_indices(
        self,
        pair_indices: np.ndarray,
        atom_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Apply an atom-level mask to atom pair indices.
        """
        if atom_mask is None:
            return pair_indices

        l_index = self.lddt_atom_pair[pair_indices, 0]
        m_index = self.lddt_atom_pair[pair_indices, 1]
        pair_subset = atom_mask[l_index] & atom_mask[m_index]
        return pair_indices[pair_subset]

    def calc_lddt_pli(
        self,
        ref_lig_label_asym_id: Union[str, list[str]],
        inclusion_radius: float = 6.0,
    ) -> float:
        """
        Calculate LDDT-PLI score.

        Args:
            ref_lig_label_asym_id (str|list[str]): Ligand chain ID(s).
            inclusion_radius (float): Radius to define binding site residues and
                                      include atom pairs for LDDT calculation.

        Returns:
            float: LDDT-PLI score.
        """
        if isinstance(ref_lig_label_asym_id, str):
            ref_lig_label_asym_id = [ref_lig_label_asym_id]

        ref_struct = self.ref_struct

        # Identify Ligand Atoms
        ligand_mask = np.isin(ref_struct.uni_chain_id, ref_lig_label_asym_id)
        if not np.any(ligand_mask):
            return float("nan")

        # Identify Pocket Residues
        ligand_coords = ref_struct.atom_array.coord[ligand_mask]
        kdtree = KDTree(ref_struct.atom_array.coord)

        nearby_indices = np.unique(
            np.concatenate(kdtree.query_ball_point(ligand_coords, r=inclusion_radius))
        )

        is_polymer = ref_struct.get_mask_for_given_entity_types(
            POLYMER + [PROTEIN_D, DNA_RNA_HYBRID]
        )
        pocket_atom_indices = nearby_indices[is_polymer[nearby_indices]]

        if pocket_atom_indices.size == 0:
            return float("nan")

        starts = ref_struct.get_residue_starts(add_exclusive_stop=True)
        pocket_res_indices = np.unique(
            np.searchsorted(starts, pocket_atom_indices, side="right") - 1
        )

        pocket_mask = np.zeros(len(ref_struct.atom_array), dtype=bool)
        for res_idx in pocket_res_indices:
            s = starts[res_idx]
            e = starts[res_idx + 1]
            pocket_mask[s:e] = True

        pocket_mask = pocket_mask & (~ligand_mask)

        if not np.any(pocket_mask):
            return float("nan")

        # Filter pairs for LDDT-PLI
        pair_indices = self._get_lddt_atom_pair_indices_for_chain_mask(
            ligand_mask, pocket_mask
        )

        if pair_indices.size == 0:
            return float("nan")

        # Apply distance filter (inclusion_radius)
        l_idx = self.lddt_atom_pair[pair_indices, 0]
        m_idx = self.lddt_atom_pair[pair_indices, 1]

        ref_coords_l = ref_struct.atom_array.coord[l_idx]
        ref_coords_m = ref_struct.atom_array.coord[m_idx]
        ref_dists = np.linalg.norm(ref_coords_l - ref_coords_m, axis=-1)

        dist_mask = ref_dists < inclusion_radius
        final_indices = pair_indices[dist_mask]

        if final_indices.size == 0:
            return float("nan")

        return self._calc_lddt(final_indices)

    def run(
        self,
        chain_1_masks: Optional[np.ndarray] = None,
        chain_2_masks: Optional[np.ndarray] = None,
        atom_mask: Optional[np.ndarray] = None,
    ) -> Union[float, list[float]]:
        """
        Run LDDT calculation for complex / chain / interface evaluation.
        If the evaluation is for whole complex, the chain_1_masks and chain_2_masks are None.
        If the evaluation is for a single chain, the chain_2_masks is the same as the chain_1_masks.

        Args:
            chain_1_masks (np.ndarray, optional): [N_eval, N_atom] Atom mask for chain 1.
                                                  Defaults to None.
            chain_2_masks (np.ndarray, optional): [N_eval, N_atom] Atom mask for chain 2.
                                                  Defaults to None.
            atom_mask (np.ndarray, optional): [N_atom] Boolean mask. Only atom pairs where
                both atoms are True will be used for LDDT calculation. Defaults to None.

        Returns:
            np.ndarray: LDDT scores. If evaluating chain interfaces, the shape is [N_eval].
                        Otherwise, the shape is [1].
        """
        eval_chain_interface = chain_1_masks is not None and chain_2_masks is not None

        # Apply user-provided atom_mask as a hard filter on atom pairs.
        #
        # Note: When `stereochecks=True`, stereochemistry validity is handled by
        # `self.per_pair_lddt_all` (pairs touching invalid atoms contribute 0),
        # and we intentionally do NOT drop those pairs here.
        n_atom = self.model_struct.atom_array.coord.shape[0]
        combined_atom_mask = None
        if atom_mask is not None:
            atom_mask = np.asarray(atom_mask, dtype=bool)
            assert (
                atom_mask.shape[0] == n_atom
            ), f"atom_mask shape mismatch: expected ({n_atom}), got {atom_mask.shape}"
            combined_atom_mask = atom_mask

        if not eval_chain_interface:
            pair_indices = np.arange(len(self.lddt_atom_pair))

            # If we have an atom-level mask, restrict to pairs where both atoms are valid.
            pair_indices = self._apply_atom_mask_to_pair_indices(
                pair_indices, combined_atom_mask
            )

            # If no pairs remain after applying the atom_mask, return NaN.
            if pair_indices.size == 0:
                return float("nan")

            lddt_value = self._calc_lddt(pair_indices)
        else:
            n_eval = chain_1_masks.shape[0]
            lddt_value = []  # [N_eval]
            for i in range(n_eval):
                pair_indices = self._get_lddt_atom_pair_indices_for_chain_mask(
                    chain_1_masks[i], chain_2_masks[i]
                )

                pair_indices = self._apply_atom_mask_to_pair_indices(
                    pair_indices, combined_atom_mask
                )

                if pair_indices.size == 0:
                    # No valid pairs for this interface after applying atom_mask.
                    lddt_value_i = float("nan")
                    lddt_value.append(lddt_value_i)
                    continue

                lddt_value_i = self._calc_lddt(pair_indices)
                lddt_value.append(lddt_value_i)
        return lddt_value
