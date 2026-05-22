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


from typing import Union

import numpy as np
from scipy.spatial import KDTree

from pxmeter.constants import POLYMER
from pxmeter.data.struct import Structure
from pxmeter.metrics.rmsd import partially_aligned_rmsd


class RMSDMetrics:
    """
    Calculating RMSD metrics between a reference structure and a model structure.

    Args:
            ref_struct (Structure): Reference structure containing correct conformation
            model_struct (Structure): Model structure to evaluate against reference
            ref_lig_label_asym_id (str|list[str]): Chain ID(s) identifying ligand(s)
                in reference structure that define binding pockets
    """

    def __init__(
        self,
        ref_struct: Structure,
        model_struct: Structure,
        ref_lig_label_asym_id: Union[str, list[str]],
    ):
        self.ref_struct = ref_struct
        self.model_struct = model_struct
        self.ref_lig_label_asym_id = ref_lig_label_asym_id

    @staticmethod
    def _find_ligand_and_pocket_by_lig_id(
        struct: Structure,
        lig_label_asym_id: Union[str, list[str]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Identify ligand atoms and their nearby receptor backbone (“pocket”) atoms by ligand label_asym_id.

        Given one or more ligand chain identifiers (label_asym_id), this function:
            1) Builds a KD-tree on all atom coordinates.
            2) For each ligand, finds all atoms within 10 Å of any ligand atom.
            3) Among polymer chains, selects the “primary” receptor chain as the one with the
                most backbone atoms (protein CA / nucleic-acid C1′) in that 10 Å neighborhood.
            4) Returns two boolean masks per ligand: the ligand-atom mask and the pocket mask
                (backbone atoms from the selected primary chain within 10 Å).

        Args:
            struct (Structure): The structure containing the AtomArray.
            lig_label_asym_id (str or list[str]): The label_asym_id of the ligand of interest.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple of ligand pocket mask
                                           and pocket mask (dtype=bool).
        """
        is_polymer = struct.get_mask_for_given_entity_types(POLYMER)

        if isinstance(lig_label_asym_id, str):
            lig_label_asym_ids = [lig_label_asym_id]
        else:
            lig_label_asym_ids = list(lig_label_asym_id)

        # Get backbone mask
        polymer_backbone = (
            is_polymer & np.isin(struct.atom_array.atom_name, ["CA", "C1'"])
        ).astype(bool)

        kdtree = KDTree(struct.atom_array.coord)

        ligand_mask_list = []
        pocket_mask_list = []
        for lig_label_asym_id in lig_label_asym_ids:
            assert np.isin(
                lig_label_asym_id, struct.uni_chain_id
            ), f"{lig_label_asym_id} is not in the label_asym_id of the AtomArray."

            ligand_mask = struct.uni_chain_id == lig_label_asym_id
            lig_pos = struct.atom_array.coord[ligand_mask]

            # Get atoms in 10 Angstrom radius
            near_atom_indices = np.unique(
                np.concatenate(kdtree.query_ball_point(lig_pos, r=10.0))
            )
            near_atoms = [
                True if i in near_atom_indices else False
                for i in range(len(struct.atom_array))
            ]

            # Get primary chain (polymer backone in 10 Angstrom radius)
            primary_chain_candidates = near_atoms & polymer_backbone
            primary_chain_candidates_uni_chain_id = struct.uni_chain_id[
                primary_chain_candidates
            ]

            max_atom = 0
            primary_chain_id = None
            for chain_id in np.unique(primary_chain_candidates_uni_chain_id):
                n_atoms = np.sum(primary_chain_candidates_uni_chain_id == chain_id)
                if n_atoms > max_atom:
                    max_atom = n_atoms
                    primary_chain_id = chain_id
            assert (
                primary_chain_id is not None
            ), f"No primary chain found for ligand ({lig_label_asym_id=})."

            pocket_mask = primary_chain_candidates & (
                struct.uni_chain_id == primary_chain_id
            )

            num_pocket_atoms = np.sum(pocket_mask)
            num_ligand_atoms = np.sum(ligand_mask)
            assert (
                num_ligand_atoms >= 1
            ), f"No ligand atoms found. (lig chain: {lig_label_asym_id})"
            assert num_pocket_atoms >= 3, (
                "Pocket not found or less than 3 backbone atoms "
                f"(lig chain: {lig_label_asym_id}, found {num_pocket_atoms} "
                f"backbone atoms of pocket chain {primary_chain_id})."
            )

            ligand_mask_list.append(ligand_mask)
            pocket_mask_list.append(pocket_mask)

        ligand_mask_by_pockets = np.array(ligand_mask_list).astype(bool)
        pocket_mask_by_pockets = np.array(pocket_mask_list).astype(bool)
        return ligand_mask_by_pockets, pocket_mask_by_pockets

    def calc_pocket_aligned_rmsd(self):
        """
        Calculate RMSD between model and reference structure after pocket alignment.

        Returns:
            dict: Nested dictionary containing RMSD metrics for each ligand chain:
                - Key: Reference ligand chain ID
                - Value: Dictionary with:
                    * 'ref_pocket_chain': Corresponding pocket chain ID
                    * 'lig_rmsd': Ligand RMSD without reflection
                    * 'pocket_rmsd': Pocket RMSD without reflection
                    (All RMSD values in Angstroms)
        """
        (
            ligand_mask_by_pockets,
            pocket_mask_by_pockets,
        ) = RMSDMetrics._find_ligand_and_pocket_by_lig_id(
            self.ref_struct,
            lig_label_asym_id=self.ref_lig_label_asym_id,
        )

        rmsd_result_dict = {}
        for pocket_mask, ligand_mask in zip(
            pocket_mask_by_pockets, ligand_mask_by_pockets
        ):
            ref_lig_chain_id = self.ref_struct.uni_chain_id[ligand_mask][0]
            ref_pocket_chain_id = self.ref_struct.uni_chain_id[pocket_mask][0]
            pocket_rmsd, lig_rmsd, _, _ = partially_aligned_rmsd(
                self.model_struct.atom_array.coord,
                self.ref_struct.atom_array.coord,
                align_mask=pocket_mask,
                rmsd_mask=ligand_mask,
                reduce=True,
                allow_reflection=False,
                src_valid_mask=self.model_struct.valid_mask,
                tar_valid_mask=self.ref_struct.valid_mask,
            )
            rmsd_result_dict[ref_lig_chain_id] = {
                "ref_pocket_chain": ref_pocket_chain_id,
                "lig_rmsd": lig_rmsd,
                "pocket_rmsd": pocket_rmsd,
            }
        return rmsd_result_dict
