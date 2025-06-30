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

from pxmeter.data.ccd import get_ccd_perm_info
from pxmeter.data.struct import Structure
from pxmeter.metrics.rmsd import align_src_to_tar, apply_transform, rmsd


class AtomPermutation:
    """
    Generating and applying atom permutations based on a reference structure.

    Args:
            ref_struct (Structure): The reference structure used for permutation generation.
            model_struct (Structure): The model structure used for permutation application.
    """

    def __init__(self, ref_struct: Structure, model_struct: Structure):
        self.ref_struct = ref_struct
        self.model_struct = model_struct

    @staticmethod
    def get_atom_perm_list(
        struct: Structure,
    ) -> list[np.ndarray]:
        """
        Generate a list of atom permutation indices for each residue in the given Structure.
        This function processes an array of atoms, grouped by residues, and generates a list of
        permutation indices for each residue based on the permutation information retrieved from
        a CCD. If no permutation information is available for a
        residue, each atom in the residue is assigned a unique index.

        Args:
            struct (Structure): The Structure object containing atom array.

        Returns:
            list[np.ndarray]: A list of array, where each inner array contains permutation indices
                              for the atoms in a residue. (N_res, N_atom, N_perm)
        """
        atom_array = struct.atom_array

        starts = struct.get_residue_starts(add_exclusive_stop=True)
        atom_perm_list = []
        for start, stop in zip(starts[:-1], starts[1:]):
            curr_res_atom_idx = list(range(stop - start))
            perm_dict = get_ccd_perm_info(ccd_code=atom_array.res_name[start])

            if not perm_dict:
                atom_perm_list.extend([[i] for i in curr_res_atom_idx])
                continue

            perm_array = perm_dict["perm_array"]  # [N_atoms, N_perm]
            perm_atom_idx_in_res_order = [
                perm_dict["atom_map"][i] for i in atom_array.atom_name[start:stop]
            ]
            perm_idx_to_present_atom_idx = dict(
                zip(perm_atom_idx_in_res_order, curr_res_atom_idx)
            )

            precent_row_mask = np.isin(
                np.arange(len(perm_array)), perm_atom_idx_in_res_order
            )
            perm_array_row_filtered = perm_array[precent_row_mask]

            precent_col_mask = np.isin(
                perm_array_row_filtered, perm_atom_idx_in_res_order
            ).all(axis=0)
            perm_array_filtered = perm_array_row_filtered[:, precent_col_mask]

            # Replace the elem in new_perm_array according to the perm_idx_to_present_atom_idx dict
            new_perm_array = np.vectorize(perm_idx_to_present_atom_idx.get)(
                perm_array_filtered
            )

            assert (
                new_perm_array.shape[1] <= 1000
                and new_perm_array.shape[1] <= perm_array.shape[1]
            )
            assert new_perm_array.shape[0] == stop - start, (
                f"Number of atoms in residue ({stop - start})"
                f"does not match the number of permutations ({new_perm_array.shape[0]})"
            )
            # Drop duplicate permutations
            new_perm_array_wo_dup = np.unique(new_perm_array, axis=-1)
            atom_perm_list.append(new_perm_array_wo_dup)
        return atom_perm_list

    def _find_unsymm_atoms(self, atom_perm_list: list[np.ndarray]) -> np.ndarray:
        """
        Identify non-symmetric atoms in the structure based on atom permutation indices.

        Args:
            atom_perm_list (list[np.ndarray]): A list of arrays, where each array contains
                permutation indices for the atoms in a residue. Shape: (N_res, N_atom, N_perm)

        Returns:
            np.ndarray: A boolean mask indicating non-symmetric atoms across all residues.
        """
        unsymm_atom_mask = []
        for res in atom_perm_list:
            atom_num = res.shape[0]
            res_unsymm_atom_mask = np.all(
                res == np.arange(atom_num)[:, np.newaxis], axis=1
            )
            unsymm_atom_mask.append(res_unsymm_atom_mask)
        unsymm_atom_mask = np.concatenate(unsymm_atom_mask)
        return unsymm_atom_mask

    def _prepare_coord_and_mask_for_perm(
        self, atom_perm_list: list[np.ndarray], transformed_model_coord: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """
        Prepare reference and model coordinates, as well as RMSD masks,
        for all permutations of each residue.

        Args:
            atom_perm_list (list[np.ndarray]): A list of arrays,
                           where each array contains permutation indices
                           for the atoms in a residue. (N_res, N_atom, N_perm)
            transformed_model_coord (np.ndarray): The model structure's
                                    coordinates after alignment to the reference structure.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                - ref_coord_for_perm: Reference coordinates for all permutations.
                                      (N_res, N_max_perm, N_atoms, 3)
                - model_coord_for_perm: Model coordinates for all permutations.
                                        (N_res, N_max_perm, N_atoms, 3)
                - rmsd_mask: RMSD masks for all permutations.
                             (N_res, N_max_perm, N_atoms)
        """
        ref_coord_for_perm = []  # (N_res, N_max_perm, N_atoms, 3)
        model_coord_for_perm = []  # (N_res, N_max_perm, N_atoms, 3)
        rmsd_mask = []  # (N_res, N_max_perm, N_atoms)

        max_perm = max([i.shape[1] for i in atom_perm_list])
        max_atom = max([i.shape[0] for i in atom_perm_list])

        starts = self.ref_struct.get_residue_starts(add_exclusive_stop=True)
        for res_idx, (res_start, res_stop) in enumerate(zip(starts[:-1], starts[1:])):
            res_perm_array = atom_perm_list[res_idx]  # N_res_atom, N_perm

            ref_res_perm_coord = []  # (N_max_perm, N_max_atoms, 3)
            model_res_perm_coord = []  # (N_max_perm, N_max_atoms, 3)
            ref_res_rmsd_mask = []  # (N_max_perm, N_max_atoms)
            ref_res_coord = self.ref_struct.atom_array.coord[res_start:res_stop]
            ref_res_atom_num = ref_res_coord.shape[0]

            padded_ref_res_coord = np.zeros((max_atom, 3), dtype=float)
            padded_ref_res_coord[:ref_res_atom_num, :] = ref_res_coord
            padded_ref_mask = np.zeros(max_atom, dtype=bool)

            # Valid atoms for rmsd
            padded_ref_mask[:ref_res_atom_num] = True

            padded_transformed_model_coord = np.zeros((max_atom, 3), dtype=float)
            padded_transformed_model_coord[
                :ref_res_atom_num, :
            ] = transformed_model_coord[res_start:res_stop]

            for i in range(max_perm):
                # Copy ref coords for each permutation
                ref_res_perm_coord.append(padded_ref_res_coord)
                ref_res_rmsd_mask.append(padded_ref_mask)
                if i < res_perm_array.shape[1]:
                    perm_index = res_perm_array.T[i]

                    # Copy for prevent from being overwritten
                    padded_transformed_model_coord_copy = (
                        padded_transformed_model_coord.copy()
                    )
                    padded_transformed_model_coord_copy[
                        :ref_res_atom_num, :
                    ] = transformed_model_coord[res_start:res_stop][perm_index]
                    model_res_perm_coord.append(padded_transformed_model_coord_copy)
                else:
                    model_res_perm_coord.append(padded_transformed_model_coord)

            ref_coord_for_perm.append(ref_res_perm_coord)
            model_coord_for_perm.append(model_res_perm_coord)
            rmsd_mask.append(ref_res_rmsd_mask)

        ref_coord_for_perm = np.array(ref_coord_for_perm)
        model_coord_for_perm = np.array(model_coord_for_perm)
        rmsd_mask = np.array(rmsd_mask)
        return ref_coord_for_perm, model_coord_for_perm, rmsd_mask

    def run(self):
        """
        Generate and apply optimal atom permutations to the
        model structure based on the reference structure.

        This method performs the following steps:
        1. Generate atom permutation indices for each residue in the model structure.
        2. Identify non - symmetric atoms in the model structure.
        3. Align the model structure to the reference structure using non - symmetric atoms.
        4. Prepare coordinates and masks for RMSD calculation across all permutations.
        5. Calculate RMSD for each residue and permutation combination.
        6. Select the best permutation for each residue based on the minimum RMSD.
        7. Return the concatenated optimal permutation indices.

        Returns:
            np.ndarray: Concatenated optimal permutation indices for
                        all residues in the model structure.
        """
        atom_perm_list = self.get_atom_perm_list(self.model_struct)
        unsymm_atom_mask = self._find_unsymm_atoms(atom_perm_list)

        rot, trans = align_src_to_tar(
            self.model_struct.atom_array.coord,
            self.ref_struct.atom_array.coord,
            atom_mask=unsymm_atom_mask,
        )
        transformed_model_coord = apply_transform(
            self.model_struct.atom_array.coord, rot, trans
        )

        (
            ref_coord_for_perm,
            model_coord_for_perm,
            rmsd_mask,
        ) = self._prepare_coord_and_mask_for_perm(
            atom_perm_list, transformed_model_coord
        )
        rmsd_for_res_and_perm = rmsd(
            model_coord_for_perm,
            ref_coord_for_perm,
            mask=rmsd_mask,
            reduce=False,
            eps=1e-6,
        )

        curr_atom_idx = 0
        permuted_indices = []
        for res_idx, res_rmsd in enumerate(rmsd_for_res_and_perm):
            res_perm_array = atom_perm_list[res_idx]  # (N_atom, N_perm)
            valid_perm_num = res_perm_array.shape[1]
            best_perm_idx = np.argmin(res_rmsd[:valid_perm_num])
            permuted_indices.append(res_perm_array.T[best_perm_idx] + curr_atom_idx)
            curr_atom_idx += res_perm_array.shape[0]
        return np.concatenate(permuted_indices)
