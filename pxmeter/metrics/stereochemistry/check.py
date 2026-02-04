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

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from biotite.structure.bonds import BondType
from biotite.structure.info.radii import vdw_radius_single
from scipy.spatial import KDTree

from pxmeter.constants import DNA, NUC_BACKBONE, PROTEIN, PROTEIN_BACKBONE, RNA
from pxmeter.data.struct import Structure
from pxmeter.metrics.stereochemistry.params import (
    ANGLE_DATA,
    BOND_DATA,
    INTER_RES_ANGLE_DATA,
    INTER_RES_BOND_DATA,
)


class StereoChemValidator:
    """
    This class performs preprocessing steps required for downstream
    stereochemical checks, including grouping atoms by residue and
    classifying inter-residue connectivity types. The processed metadata is
    stored for later use by bond-length, bond-angle, and clash validation
    routines.

    Attributes:
        struct (Structure): The original input structure.
        ref_struct (Structure | None): Optional reference structure for
            clash detection. If provided, its atoms must be in strict
            one-to-one order correspondence with those in the query
            structure (same length and indexing), and its bond graph will
            be added to the query bond graph to detect clashes.
        atom_array (AtomArray): Parsed atom array containing coordinates
            and residue annotations.
        entity_poly_type (dict[str, str]): Mapping from entity ID to polymer
            category (e.g., protein, DNA, RNA).
        res_groups (dict): Preprocessed residue-level groupings derived from
            the atom array.
        inter_res_types (dict): Classification of inter-residue bond types
            used to guide stereochemical validation.
    """

    def __init__(
        self,
        struct: Structure,
        ref_struct: Optional[Structure] = None,
    ):
        self.struct = struct
        self.ref_struct = ref_struct
        self.atom_array = struct.atom_array
        self.entity_poly_type = struct.entity_poly_type

        # Cache common fields to avoid repeated property access
        self._uni_chain_id = struct.uni_chain_id
        self._res_id = self.atom_array.res_id
        self._res_name = self.atom_array.res_name
        self._atom_name = self.atom_array.atom_name

        # Cache per-atom entity/polymer type to avoid repeated Python loops
        label_entity_ids = self.atom_array.label_entity_id
        self._entity_types = np.fromiter(
            (self.entity_poly_type.get(eid, "OTHER") for eid in label_entity_ids),
            dtype=object,
            count=len(label_entity_ids),
        )

        self.res_groups = self._preprocess_residue_groups()

        # Some model mmCIFs omit key inter-residue covalent connectivity.
        # - Peptide bonds (C-N) are often not present in `_struct_conn`.
        # - Phosphodiester backbone bonds (O3'-P) are often not present in `_struct_conn`.
        # - Disulfide links may be missing as well.
        # We infer these so that covalently bonded pairs are excluded from
        # clash checks and become eligible for inter-res bond validation.
        self._infer_and_add_peptide_bonds(max_c_n_dist=1.8)
        self._infer_and_add_na_backbone_bonds(max_p_o3_dist=2.1)
        self._infer_and_add_disulfide_bonds(max_sg_sg_dist=2.2)

        # Depends on bond graph (e.g. PRO cis/trans classification)
        self.inter_res_types = self._classify_inter_residue_types()

        # Precompute VDW radii and bond hashes for clash detection
        self._vdw_radii = self._calculate_vdw_radii()
        self._bond_hashes = self._calculate_bond_hashes()

    def _infer_and_add_disulfide_bonds(self, *, max_sg_sg_dist: float = 2.2) -> None:
        """
        Infer missing CYS SG-SG bonds and add them to the bond graph.

        This is a conservative heuristic for models that do not encode
        disulfide connectivity.
        """

        atom_array = self.atom_array
        if atom_array.bonds is None:
            return

        res_names = self._res_name
        atom_names = self._atom_name
        is_cys_sg = (res_names == "CYS") & (atom_names == "SG")
        sg_idx = np.where(is_cys_sg)[0]
        if sg_idx.size < 2:
            return

        coords_sg = atom_array.coord[sg_idx]
        tree = KDTree(coords_sg)
        pairs_local = tree.query_pairs(max_sg_sg_dist, output_type="ndarray")
        if pairs_local.size == 0:
            return

        # Map local KDTree indices back to atom indices
        i_global = sg_idx[pairs_local[:, 0]]
        j_global = sg_idx[pairs_local[:, 1]]

        # Exclude within the same residue
        chain_ids = self._uni_chain_id
        res_ids = self._res_id
        diff_res = (chain_ids[i_global] != chain_ids[j_global]) | (
            res_ids[i_global] != res_ids[j_global]
        )
        i_global = i_global[diff_res]
        j_global = j_global[diff_res]
        if i_global.size == 0:
            return

        dists = np.linalg.norm(
            atom_array.coord[i_global] - atom_array.coord[j_global], axis=1
        )
        order = np.argsort(dists)
        i_global = i_global[order]
        j_global = j_global[order]

        used = set()
        bonds = atom_array.bonds
        for i, j in zip(i_global.tolist(), j_global.tolist()):
            if i in used or j in used:
                continue

            # Skip if already bonded
            neigh, _ = bonds.get_bonds(i)
            if j in neigh:
                used.add(i)
                used.add(j)
                continue

            bonds.add_bond(i, j, bond_type=BondType.SINGLE)
            used.add(i)
            used.add(j)

    def _infer_and_add_peptide_bonds(self, *, max_c_n_dist: float = 1.8) -> None:
        """
        Infer missing peptide C-N bonds and add them to the bond graph.

        Many predicted/model mmCIFs do not encode polymer backbone bonds in
        `_struct_conn`, hence `biotite.structure.io.pdbx.get_structure()` will
        not populate these inter-residue bonds.

        We conservatively connect consecutive protein residues within the same
        chain when their backbone C (prev residue) and N (next residue) are
        within a distance threshold.
        """

        atom_array = self.atom_array
        if atom_array.bonds is None:
            return

        atom_name = self._atom_name
        chain_id = self._uni_chain_id
        res_id = self._res_id

        # Determine per-atom entity/polymer types
        entity_types = self._entity_types

        # Fast early-outs
        if not np.any(entity_types == PROTEIN):
            return

        # Map residue key -> backbone atom index (use one atom per residue)
        c_indices = np.where(atom_name == "C")[0]
        n_indices = np.where(atom_name == "N")[0]
        if c_indices.size == 0 or n_indices.size == 0:
            return
        c_map = {(chain_id[i], res_id[i]): i for i in c_indices}
        n_map = {(chain_id[i], res_id[i]): i for i in n_indices}

        # Residues are expected to be contiguous in AtomArray.
        # Compute residue starts in O(N) without `np.unique(axis=0)` to avoid
        # dtype upcasting and extra allocations.
        n_atoms = len(chain_id)
        if n_atoms < 2:
            return
        change = np.ones(n_atoms, dtype=bool)
        change[1:] = (chain_id[1:] != chain_id[:-1]) | (res_id[1:] != res_id[:-1])
        first_pos = np.where(change)[0]

        # Build ordered residue keys and their entity types
        res_keys = [(chain_id[i], res_id[i]) for i in first_pos.tolist()]
        res_ent = entity_types[first_pos]

        if res_ent.size < 2:
            return

        bonds = atom_array.bonds
        coords = atom_array.coord

        for k0, k1, t0, t1 in zip(
            res_keys[:-1], res_keys[1:], res_ent[:-1], res_ent[1:]
        ):
            # Only connect within the same chain and protein polymer
            if k0[0] != k1[0]:
                continue
            if not (t0 == PROTEIN and t1 == PROTEIN):
                continue

            c0 = c_map.get(k0)
            n1 = n_map.get(k1)
            if c0 is None or n1 is None:
                continue

            # Avoid linking across chain breaks / missing segments
            if np.linalg.norm(coords[c0] - coords[n1]) > max_c_n_dist:
                continue

            neigh, _ = bonds.get_bonds(c0)
            if n1 in neigh:
                continue
            bonds.add_bond(c0, n1, bond_type=BondType.SINGLE)

    def _infer_and_add_na_backbone_bonds(self, *, max_p_o3_dist: float = 2.1) -> None:
        """
        Infer missing nucleic-acid backbone bonds (O3'-P) and add them.

        Similar to peptide bonds, many predicted/model mmCIFs omit nucleic-acid
        inter-residue bonds in `_struct_conn`, leading to O3'-P pairs being
        treated as non-bonded contacts (and potentially flagged as clashes).

        We conservatively connect consecutive nucleic-acid residues within the
        same chain when their O3' (prev residue) and P (next residue) are within
        a distance threshold.
        """

        atom_array = self.atom_array
        if atom_array.bonds is None:
            return

        atom_name = self._atom_name
        chain_id = self._uni_chain_id
        res_id = self._res_id

        entity_types = self._entity_types

        # Fast early-outs
        if not np.any((entity_types == DNA) | (entity_types == RNA)):
            return

        # Backbone atoms
        p_indices = np.where(atom_name == "P")[0]
        o3_mask = (atom_name == "O3'") | (atom_name == "O3*")
        o3_indices = np.where(o3_mask)[0]
        if p_indices.size == 0 or o3_indices.size == 0:
            return

        p_map = {(chain_id[i], res_id[i]): i for i in p_indices}
        o3_map = {(chain_id[i], res_id[i]): i for i in o3_indices}

        n_atoms = len(chain_id)
        if n_atoms < 2:
            return
        change = np.ones(n_atoms, dtype=bool)
        change[1:] = (chain_id[1:] != chain_id[:-1]) | (res_id[1:] != res_id[:-1])
        first_pos = np.where(change)[0]

        res_keys = [(chain_id[i], res_id[i]) for i in first_pos.tolist()]
        res_ent = entity_types[first_pos]

        bonds = atom_array.bonds
        coords = atom_array.coord

        is_na = (res_ent == DNA) | (res_ent == RNA)
        if is_na.sum() < 2:
            return
        for k0, k1, na0, na1 in zip(res_keys[:-1], res_keys[1:], is_na[:-1], is_na[1:]):
            if k0[0] != k1[0]:
                continue
            if not (na0 and na1):
                continue

            o3 = o3_map.get(k0)
            p1 = p_map.get(k1)
            if o3 is None or p1 is None:
                continue

            if np.linalg.norm(coords[o3] - coords[p1]) > max_p_o3_dist:
                continue

            neigh, _ = bonds.get_bonds(o3)
            if p1 in neigh:
                continue
            bonds.add_bond(o3, p1, bond_type=BondType.SINGLE)

    def _calculate_vdw_radii(self) -> np.ndarray:
        elements = self.atom_array.element
        unique_elements = np.unique(elements)
        vdw_map = {e: vdw_radius_single(e) or 1.7 for e in unique_elements}
        return np.array([vdw_map[e] for e in elements])

    def _calculate_bond_hashes(self) -> np.ndarray:
        n_atoms = len(self.atom_array)
        bonds = self.atom_array.bonds.copy()
        if self.ref_struct is not None:
            assert len(self.ref_struct.atom_array) == n_atoms
            bonds += self.ref_struct.atom_array.bonds
        bond_array = bonds.as_array()[:, :2]
        b_idx1 = np.minimum(bond_array[:, 0], bond_array[:, 1])
        b_idx2 = np.maximum(bond_array[:, 0], bond_array[:, 1])
        return np.sort(np.unique(b_idx1.astype(np.int64) * n_atoms + b_idx2))

    def _preprocess_residue_groups(
        self,
    ) -> dict[str, dict[str, np.ndarray]]:
        """
        Precompute residue-wise atom groups for bond/angle checks.
        Optimized to avoid string concatenation and reduce Python loops.
        """
        res_names = self._res_name
        chain_ids = self._uni_chain_id
        res_ids = self._res_id

        # Map chain IDs to integers for faster grouping
        _unique_chains, chain_ints = np.unique(chain_ids, return_inverse=True)

        unique_res_types = np.unique(res_names)
        res_groups: dict[str, dict[str, np.ndarray]] = {}

        for rname in unique_res_types:
            idx_res = np.where(res_names == rname)[0]
            if idx_res.size == 0:
                continue

            # Group by (chain_int, res_id) using np.unique on integers
            group_keys = np.stack([chain_ints[idx_res], res_ids[idx_res]], axis=1)
            _, group_ids_res = np.unique(group_keys, axis=0, return_inverse=True)
            n_groups = int(group_ids_res.max()) + 1 if group_ids_res.size > 0 else 0

            res_groups[rname] = {
                "idx_res": idx_res,
                "group_ids": group_ids_res,
                "n_groups": n_groups,
            }

        return res_groups

    @staticmethod
    def _compute_dihedral(
        p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
    ) -> float:
        """
        Return dihedral angle (radians) for four points p0-p1-p2-p3.
        """
        b0 = p0 - p1
        b1 = p2 - p1
        b2 = p3 - p2

        # normalize b1 so that it does not influence magnitude of vector
        b1 /= np.linalg.norm(b1)

        # components orthogonal to b1
        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1

        v_norm = np.linalg.norm(v)
        w_norm = np.linalg.norm(w)
        if v_norm < 1e-8 or w_norm < 1e-8:
            return np.nan

        v /= v_norm
        w /= w_norm

        x = np.dot(v, w)
        y = np.dot(np.cross(b1, v), w)
        return np.arctan2(y, x)

    def _classify_inter_residue_types(self) -> np.ndarray:
        entity_mapping = {PROTEIN: "PEPTIDE", DNA: "NA", RNA: "NA"}

        entity_types = self._entity_types

        inter_res_types = np.array(
            [entity_mapping.get(t, "OTHER") for t in entity_types], dtype=object
        )

        # Update GLY types
        gly_mask = (self._res_name == "GLY") & (entity_types == PROTEIN)
        inter_res_types[gly_mask] = "GLY"

        # Update PRO_CIS/TRANS types
        pro_mask = (self._res_name == "PRO") & (entity_types == PROTEIN)
        if not np.any(pro_mask):
            return inter_res_types

        # Optimization: Pre-index atoms for Proline classification
        atom_name = self._atom_name
        chain_id = self._uni_chain_id
        res_id = self._res_id
        coords = self.atom_array.coord

        # Build a fast residue index on integer keys (avoid object np.unique)
        unique_chains, chain_ints = np.unique(chain_id, return_inverse=True)
        combined_res_keys = np.stack([chain_ints, res_id], axis=1)
        _, res_indices = np.unique(combined_res_keys, axis=0, return_inverse=True)

        # Find CA, N indices once
        ca_mask = atom_name == "CA"
        n_mask = atom_name == "N"

        # Map (chain, res_id) -> atom indices for CA, N
        def get_atom_map(mask):
            indices = np.where(mask)[0]
            return {(chain_id[i], res_id[i]): i for i in indices}

        ca_map = get_atom_map(ca_mask)
        n_map = get_atom_map(n_mask)

        # Find all Proline residues
        pro_indices = np.where(pro_mask)[0]
        if pro_indices.size == 0:
            return inter_res_types

        # Unique proline residues (avoid Python set/sort over atoms)
        pro_keys = np.stack([chain_ints[pro_indices], res_id[pro_indices]], axis=1)
        pro_res_unique = np.unique(pro_keys, axis=0)

        for c_pro_int, r_pro in pro_res_unique.tolist():
            c_pro = unique_chains[c_pro_int]
            # Need prev CA (ca0), prev C (c0), current N (n1), current CA (ca1)
            # Find previous residue in the same chain
            # Since we don't know the exact residue ID of the previous one (might not be r_pro - 1)
            # but usually it is. Let's find the max resid < r_pro in the same chain.
            # Actually, to be safe and fast, we can use the fact that it's a peptide bond.
            # The current N (n1) must be bonded to a C (c0) of the previous residue.

            n1_idx = n_map.get((c_pro, r_pro))
            if n1_idx is None:
                continue

            # Use bond information to find the previous C
            bonded_to_n1, _ = self.atom_array.bonds.get_bonds(n1_idx)
            c0_idx = None
            for b_idx in bonded_to_n1:
                if atom_name[b_idx] == "C" and (chain_id[b_idx], res_id[b_idx]) != (
                    c_pro,
                    r_pro,
                ):
                    c0_idx = b_idx
                    break

            if c0_idx is None:
                continue

            # Now we have c0 and n1. We need ca0 (bonded to c0) and ca1 (bonded to n1)
            ca1_idx = ca_map.get((c_pro, r_pro))
            prev_res_key = (chain_id[c0_idx], res_id[c0_idx])
            ca0_idx = ca_map.get(prev_res_key)

            if ca0_idx is None or ca1_idx is None:
                continue

            p_ca0, p_c0, p_n1, p_ca1 = coords[[ca0_idx, c0_idx, n1_idx, ca1_idx]]
            omega = self._compute_dihedral(p_ca0, p_c0, p_n1, p_ca1)

            if not np.isnan(omega):
                pro_type = "PRO_CIS" if abs(omega) < 1.57 else "PRO_TRANS"
                inter_res_types[res_indices == res_indices[n1_idx]] = pro_type

        return inter_res_types

    def find_bad_bonds(
        self,
        bond_data: Optional[dict[str, dict[str, tuple[float, float]]]] = None,
        inter_res_bond_data: Optional[dict[str, dict[str, list[float]]]] = None,
        z_thresh: float = 12.0,
    ) -> pd.DataFrame:
        """
        Detect all bond length outliers (intra- and inter-residue).

        This is a convenience wrapper that runs both intra-residue and
        inter-residue bond validation and concatenates the results into a single
        table. Bond lengths are compared against reference statistics and those
        whose Z-scores exceed the given threshold are reported.

        Args:
            bond_data (dict[str, dict[str, tuple[float, float]]], optional):
                Reference statistics for intra-residue bond lengths. The outer key
                is typically a residue name (e.g. ``"ALA"``), and the inner key is a
                bond identifier of the form ``"ATOM1_ATOM2"``. Each value is a
                ``(ideal, sigma)`` tuple giving the ideal bond length (Å) and its
                standard deviation. If ``None``, defaults to ``BOND_DATA``.
            inter_res_bond_data (dict[str, dict[str, list[float]]], optional):
                Reference statistics for inter-residue bond lengths. The outer key
                usually describes the inter-residue bond type, and the inner key
                is a bond identifier. Each value is a list of two floats
                ``[ideal, sigma]`` representing the ideal bond length (Å) and its
                standard deviation. If ``None``, defaults to ``INTER_RES_BOND_DATA``.
            z_thresh (float, optional):
                Absolute Z-score threshold above which a bond is considered an
                outlier. Bonds with ``abs(z_score) > z_thresh`` are reported.
                Defaults to ``12.0``.

        Returns:
            pandas.DataFrame:
                A concatenated table of intra- and inter-residue bond length
                outliers. The schema matches the outputs of
                :meth:`find_bad_intra_res_bonds` and
                :meth:`find_bad_inter_res_bonds`, and typically includes residue
                identifiers, atom names, bond identifiers, ideal/observed lengths,
                Z-scores, and atom indices (``idx1``, ``idx2``).

                If no out-of-range bonds are found, an empty ``DataFrame`` is
                returned.
        """
        if bond_data is None:
            bond_data = BOND_DATA
        if inter_res_bond_data is None:
            inter_res_bond_data = INTER_RES_BOND_DATA

        bad_intra_res_bonds = self.find_bad_intra_res_bonds(bond_data, z_thresh)
        bad_inter_res_bonds = self.find_bad_inter_res_bonds(
            inter_res_bond_data, z_thresh
        )
        bad_bonds = pd.concat([bad_intra_res_bonds, bad_inter_res_bonds])
        return bad_bonds

    def find_bad_inter_res_bonds(
        self,
        inter_res_bond_data: Optional[dict[str, dict[str, tuple[float, float]]]] = None,
        z_thresh: float = 12.0,
    ) -> pd.DataFrame:
        """
        Detect inter-residue bond-length outliers.
        """
        if inter_res_bond_data is None:
            inter_res_bond_data = INTER_RES_BOND_DATA

        atom_array = self.atom_array
        coords = atom_array.coord
        atom_names = self._atom_name
        res_names = self._res_name
        chain_ids = self._uni_chain_id
        res_ids = self._res_id
        inter_types = self.inter_res_types

        # Get all bond pairs (idx1, idx2)
        bond_array = atom_array.bonds.as_array()
        idx1 = bond_array[:, 0]
        idx2 = bond_array[:, 1]

        # Filter: Different residues
        # Fast residue comparison using chain_id and res_id
        # We can use a hash or just compare both fields
        diff_res_mask = (chain_ids[idx1] != chain_ids[idx2]) | (
            res_ids[idx1] != res_ids[idx2]
        )
        idx1, idx2 = idx1[diff_res_mask], idx2[diff_res_mask]

        if idx1.size == 0:
            return pd.DataFrame()

        bad_dfs = []
        for group, params in inter_res_bond_data.items():
            # Filter bonds where both atoms belong to this group
            group_mask = (inter_types[idx1] == group) & (inter_types[idx2] == group)
            g_idx1, g_idx2 = idx1[group_mask], idx2[group_mask]
            if g_idx1.size == 0:
                continue

            for bond_key, (ideal, sigma) in params.items():
                name_a, name_b = bond_key.split("_")

                # Check both directions
                mask_ab = (atom_names[g_idx1] == name_a) & (
                    atom_names[g_idx2] == name_b
                )
                mask_ba = (atom_names[g_idx1] == name_b) & (
                    atom_names[g_idx2] == name_a
                )

                final_mask = mask_ab | mask_ba
                if not np.any(final_mask):
                    continue

                # Use mask_ba to swap idx1/idx2 if needed to match name_a/name_b order
                ia = np.where(mask_ab, g_idx1, g_idx2)[final_mask]
                ib = np.where(mask_ab, g_idx2, g_idx1)[final_mask]

                # Compute lengths
                lengths = np.linalg.norm(coords[ia] - coords[ib], axis=1)
                z_scores = (lengths - ideal) / sigma

                bad_mask = np.abs(z_scores) > z_thresh
                if not np.any(bad_mask):
                    continue

                ia_bad = ia[bad_mask]
                ib_bad = ib[bad_mask]

                bad_dfs.append(
                    pd.DataFrame(
                        {
                            "group": group,
                            "bond_key": bond_key,
                            "idx1": ia_bad,
                            "idx2": ib_bad,
                            "res_name1": res_names[ia_bad],
                            "res_name2": res_names[ib_bad],
                            "chain_id1": chain_ids[ia_bad],
                            "chain_id2": chain_ids[ib_bad],
                            "res_id1": res_ids[ia_bad],
                            "res_id2": res_ids[ib_bad],
                            "atom_name1": atom_names[ia_bad],
                            "atom_name2": atom_names[ib_bad],
                            "ideal": ideal,
                            "sigma": sigma,
                            "length": lengths[bad_mask],
                            "z_score": z_scores[bad_mask],
                        }
                    )
                )

        if not bad_dfs:
            return pd.DataFrame()

        return pd.concat(bad_dfs, ignore_index=True)

    def find_bad_intra_res_bonds(
        self,
        bond_data: Optional[dict[str, dict[str, tuple[float, float]]]] = None,
        z_thresh: float = 12.0,
    ) -> pd.DataFrame:
        """
        Detect intra-residue bond length outliers.

        This method evaluates all defined intra-residue bonds for each residue
        type, compares observed bond lengths to reference statistics, and reports
        those whose Z-score exceeds a given threshold. Within each residue
        instance, at most one atom of each name is used per bond (i.e., one
        main-atom A/B per residue group).

        Args:
            bond_data (dict[str, dict[str, tuple[float, float]]], optional):
                Reference statistics for intra-residue bond lengths. The outer key
                is typically a residue name (e.g. ``"ALA"``), and the inner key is a
                bond identifier of the form ``"ATOM1_ATOM2"`` (e.g. ``"N_CA"``).
                Each value is a ``(ideal, sigma)`` tuple giving the ideal bond
                length (in Å) and its standard deviation. If ``None``, defaults to
                ``BOND_DATA``.
            z_thresh (float, optional):
                Absolute Z-score threshold above which a bond is considered an
                outlier. Bonds with ``abs(z_score) > z_thresh`` are reported.
                Defaults to ``12.0``.

        Returns:
            pandas.DataFrame:
                A table of intra-residue bond length outliers. Each row
                corresponds to a single bond that violates the Z-score threshold
                and includes at least the following columns:

                * ``"group"``       - parameter group name (e.g. ``"ALA"``,
                    ``"GLN"``).
                * ``"bond_key"``    - bond key string used in the parameter
                    table.
                * ``"idx1"``        - global atom index of atom 1
                    (the lower index in the pair).
                * ``"idx2"``        - global atom index of atom 2
                    (the higher index in the pair).
                * ``"res_name1"``   - residue name of atom 1.
                * ``"res_name2"``   - residue name of atom 2.
                * ``"chain_id1"``   - chain ID of atom 1.
                * ``"chain_id2"``   - chain ID of atom 2.
                * ``"res_id1"``     - residue ID of atom 1.
                * ``"res_id2"``     - residue ID of atom 2.
                * ``"atom_name1"``  - atom name of atom 1.
                * ``"atom_name2"``  - atom name of atom 2.
                * ``"ideal"``       - ideal bond length used for this bond.
                * ``"sigma"``       - standard deviation used for this bond.
                * ``"length"``      - observed bond length in Å.
                * ``"z_score"``     - z-score of the observed bond length.

                If no out-of-range bonds are found, an empty ``DataFrame`` is
                returned.
        """
        if bond_data is None:
            bond_data = BOND_DATA

        res_groups = self.res_groups

        atom_name = self._atom_name
        chain_id = self._uni_chain_id
        res_id = self._res_id
        coords = self.atom_array.coord  # (N, 3)

        bad_records: list[dict[str, np.ndarray]] = []

        for rname, bonds in bond_data.items():
            if rname not in res_groups:
                continue

            idx_res = res_groups[rname]["idx_res"]
            group_ids_res = res_groups[rname]["group_ids"]
            n_groups = res_groups[rname]["n_groups"]

            atom_names_res = atom_name[idx_res]

            for bond_key, (ideal, sigma) in bonds.items():
                a_name, b_name = bond_key.split("_")

                mask_a = atom_names_res == a_name
                mask_b = atom_names_res == b_name
                if not (mask_a.any() and mask_b.any()):
                    continue

                idx_a_atoms = idx_res[mask_a]
                gid_a = group_ids_res[mask_a]

                idx_b_atoms = idx_res[mask_b]
                gid_b = group_ids_res[mask_b]

                # Keep at most one A/B atom per residue group (main-atom only)
                idx_a_group = np.full(n_groups, -1, dtype=np.int64)
                idx_b_group = np.full(n_groups, -1, dtype=np.int64)
                idx_a_group[gid_a] = idx_a_atoms
                idx_b_group[gid_b] = idx_b_atoms

                valid = (idx_a_group >= 0) & (idx_b_group >= 0)
                if not valid.any():
                    continue

                ia = idx_a_group[valid]
                ib = idx_b_group[valid]

                vec = coords[ia] - coords[ib]
                length = np.linalg.norm(vec, axis=1)

                # Compute z-score and select out-of-range bonds
                z = (length - ideal) / sigma
                bad = np.abs(z) > z_thresh
                if not bad.any():
                    continue

                ia_bad = ia[bad]
                ib_bad = ib[bad]
                length_bad = length[bad]
                z_bad = z[bad]

                res_name_arr = np.full_like(length_bad, rname, dtype=object)
                atom_name1_arr = np.full_like(length_bad, a_name, dtype=object)
                atom_name2_arr = np.full_like(length_bad, b_name, dtype=object)

                bad_records.append(
                    {
                        "group": np.full_like(length_bad, rname, dtype=object),
                        "res_name1": res_name_arr,
                        "res_name2": res_name_arr,
                        "chain_id1": chain_id[ib_bad],  # Use atom2 (B) as reference
                        "chain_id2": chain_id[ib_bad],
                        "res_id1": res_id[ib_bad],
                        "res_id2": res_id[ib_bad],
                        "atom_name1": atom_name1_arr,
                        "atom_name2": atom_name2_arr,
                        "bond_key": np.full_like(
                            length_bad,
                            bond_key,
                            dtype=object,
                        ),
                        "ideal": np.full_like(length_bad, ideal, dtype=float),
                        "sigma": np.full_like(length_bad, sigma, dtype=float),
                        "length": length_bad,
                        "z_score": z_bad,
                        "idx1": ia_bad,
                        "idx2": ib_bad,
                    }
                )

        if not bad_records:
            # No out-of-range bonds found
            return pd.DataFrame()

        out: dict[str, np.ndarray] = {}
        for key in bad_records[0].keys():
            out[key] = np.concatenate([r[key] for r in bad_records])

        return pd.DataFrame(out)

    def find_bad_inter_res_angles(
        self,
        inter_res_angle_data: Optional[
            dict[str, dict[str, tuple[float, float]]]
        ] = None,
        z_thresh: float = 12.0,
    ) -> pd.DataFrame:
        """
        Detect inter-residue backbone angle outliers.
        Vectorized version to avoid Python loops over atoms.
        """
        if inter_res_angle_data is None:
            inter_res_angle_data = INTER_RES_ANGLE_DATA

        atom_array = self.atom_array
        coords = atom_array.coord
        atom_names = self._atom_name
        res_names = self._res_name
        chain_ids = self._uni_chain_id
        res_ids = self._res_id
        inter_types = self.inter_res_types

        # Get all bonds and build adjacency
        bond_array = atom_array.bonds.as_array()[:, :2]
        all_bonds = np.concatenate([bond_array, bond_array[:, ::-1]], axis=0)

        def _join_ab_bc(
            bonds_ab: np.ndarray, bonds_bc: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Join (A,B) and (B,C) edges into (A,B,C) triplets.

            This replaces a per-angle `pandas.merge()` with a light-weight
            NumPy join. It still loops over shared B values, but each B has a
            tiny neighborhood in typical bond graphs.
            """

            ab_a = bonds_ab[:, 0]
            ab_b = bonds_ab[:, 1]
            bc_b = bonds_bc[:, 0]
            bc_c = bonds_bc[:, 1]

            ab_order = np.argsort(ab_b, kind="mergesort")
            bc_order = np.argsort(bc_b, kind="mergesort")

            ab_b_s = ab_b[ab_order]
            bc_b_s = bc_b[bc_order]

            ab_u, ab_start, ab_cnt = np.unique(
                ab_b_s, return_index=True, return_counts=True
            )
            bc_u, bc_start, bc_cnt = np.unique(
                bc_b_s, return_index=True, return_counts=True
            )

            ia_parts: list[np.ndarray] = []
            ib_parts: list[np.ndarray] = []
            ic_parts: list[np.ndarray] = []

            i = j = 0
            while i < ab_u.size and j < bc_u.size:
                b_ab = ab_u[i]
                b_bc = bc_u[j]
                if b_ab == b_bc:
                    a_slice = ab_a[ab_order][ab_start[i] : ab_start[i] + ab_cnt[i]]
                    c_slice = bc_c[bc_order][bc_start[j] : bc_start[j] + bc_cnt[j]]
                    if a_slice.size and c_slice.size:
                        ia_parts.append(np.repeat(a_slice, c_slice.size))
                        ib_parts.append(
                            np.full(a_slice.size * c_slice.size, b_ab, dtype=np.int64)
                        )
                        ic_parts.append(np.tile(c_slice, a_slice.size))
                    i += 1
                    j += 1
                elif b_ab < b_bc:
                    i += 1
                else:
                    j += 1

            if not ia_parts:
                return (
                    np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=np.int64),
                )

            return (
                np.concatenate(ia_parts),
                np.concatenate(ib_parts),
                np.concatenate(ic_parts),
            )

        bad_dfs = []
        for group, params in inter_res_angle_data.items():
            for angle_key, (ideal, sigma) in params.items():
                name_a, name_b, name_c = angle_key.split("_")

                # For each B, find neighbors A and C
                # This part is tricky to vectorize fully without a join.
                # Let's use the join approach:
                # Find all bonds (A, B) where A has name_a and B has name_b and both in group
                # Find all bonds (B, C) where B has name_b and C has name_c and both in group

                mask_ab = (
                    (atom_names[all_bonds[:, 0]] == name_a)
                    & (atom_names[all_bonds[:, 1]] == name_b)
                    & (inter_types[all_bonds[:, 0]] == group)
                    & (inter_types[all_bonds[:, 1]] == group)
                )

                mask_bc = (
                    (atom_names[all_bonds[:, 0]] == name_b)
                    & (atom_names[all_bonds[:, 1]] == name_c)
                    & (inter_types[all_bonds[:, 0]] == group)
                    & (inter_types[all_bonds[:, 1]] == group)
                )

                bonds_ab = all_bonds[mask_ab]  # (idx_a, idx_b)
                bonds_bc = all_bonds[mask_bc]  # (idx_b, idx_c)

                if bonds_ab.size == 0 or bonds_bc.size == 0:
                    continue

                ia, ib, ic = _join_ab_bc(bonds_ab, bonds_bc)
                if ia.size == 0:
                    continue

                # Filter A != C
                neq = ia != ic
                ia, ib, ic = ia[neq], ib[neq], ic[neq]
                if ia.size == 0:
                    continue

                # Filter: At least two distinct residues among A, B, C
                # (chain_id, res_id) comparison
                is_intra = (
                    (chain_ids[ia] == chain_ids[ib])
                    & (res_ids[ia] == res_ids[ib])
                    & (chain_ids[ib] == chain_ids[ic])
                    & (res_ids[ib] == res_ids[ic])
                )

                ia, ib, ic = ia[~is_intra], ib[~is_intra], ic[~is_intra]
                if ia.size == 0:
                    continue

                # Compute angles
                v1 = coords[ia] - coords[ib]
                v2 = coords[ic] - coords[ib]

                norm1 = np.linalg.norm(v1, axis=1)
                norm2 = np.linalg.norm(v2, axis=1)

                mask_nonzero = (norm1 > 1e-8) & (norm2 > 1e-8)
                if not np.any(mask_nonzero):
                    continue

                ia, ib, ic = ia[mask_nonzero], ib[mask_nonzero], ic[mask_nonzero]
                v1, v2 = v1[mask_nonzero], v2[mask_nonzero]
                norm1, norm2 = norm1[mask_nonzero], norm2[mask_nonzero]

                cos_theta = np.einsum("ij,ij->i", v1, v2) / (norm1 * norm2)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                angles_deg = np.degrees(np.arccos(cos_theta))

                z_scores = (angles_deg - ideal) / sigma
                bad_mask = np.abs(z_scores) > z_thresh
                if not np.any(bad_mask):
                    continue

                ia_bad, ib_bad, ic_bad = ia[bad_mask], ib[bad_mask], ic[bad_mask]

                bad_dfs.append(
                    pd.DataFrame(
                        {
                            "group": group,
                            "angle_key": angle_key,
                            "idx_a": ia_bad,
                            "idx_b": ib_bad,
                            "idx_c": ic_bad,
                            "res_name_a": res_names[ia_bad],
                            "res_name_b": res_names[ib_bad],
                            "res_name_c": res_names[ic_bad],
                            "chain_id_a": chain_ids[ia_bad],
                            "chain_id_b": chain_ids[ib_bad],
                            "chain_id_c": chain_ids[ic_bad],
                            "res_id_a": res_ids[ia_bad],
                            "res_id_b": res_ids[ib_bad],
                            "res_id_c": res_ids[ic_bad],
                            "atom_name_a": atom_names[ia_bad],
                            "atom_name_b": atom_names[ib_bad],
                            "atom_name_c": atom_names[ic_bad],
                            "ideal": ideal,
                            "sigma": sigma,
                            "angle": angles_deg[bad_mask],
                            "z_score": z_scores[bad_mask],
                        }
                    )
                )

        if not bad_dfs:
            return pd.DataFrame()

        return pd.concat(bad_dfs, ignore_index=True)

    def find_bad_intra_res_angles(
        self,
        angle_data: Optional[dict[str, dict[str, tuple[float, float]]]] = None,
        z_thresh: float = 12.0,
    ) -> pd.DataFrame:
        """
        Detect bond-angle outliers based on residue-specific statistics.

        For each residue type and each defined angle key in
        ``angle_data``, this method finds the corresponding atom
        triplets within each residue instance and computes the angle at
        the central atom ``B`` for A-B-C. An angle is considered
        out-of-range if its z-score exceeds ``z_thresh`` in absolute
        value::

            z = (angle_deg - ideal) / sigma

        Only angles defined in ``angle_data`` are checked; any atom
        triplets or residues not covered by the definitions are
        implicitly treated as having no violations and are therefore
        omitted from the result.

        Args:
            angle_data: Mapping from residue name (e.g. ``"ALA"``) to a
                mapping ``angle_key -> (ideal, sigma)``, where:

                * ``angle_key`` is a string of the form
                  ``"ATOM_A_ATOM_B_ATOM_C"``, e.g. ``"CA_N_H"`` where
                  ``ATOM_B`` is the central atom.
                * ``ideal`` is the ideal bond angle in degrees.
                * ``sigma`` is the standard deviation used for z-score
                  computation.

                If ``None``, the default :data:`ANGLE_DATA` will be
                used.
            z_thresh: Z-score threshold above which an angle is
                considered an outlier. Defaults to ``12.0``.

        Returns:
            pandas.DataFrame: A table describing only the angles that exceed the
            z-score threshold. If no out-of-range angles are found, an empty
            DataFrame is returned.

            The DataFrame contains the following columns:

            * ``"group"``        - parameter group name (e.g. ``"ALA"``,
              ``"GLN"``).
            * ``"angle_key"``    - angle key string used in the parameter
              table.
            * ``"idx_a"``        - global atom index of atom A.
            * ``"idx_b"``        - global atom index of atom B (central).
            * ``"idx_c"``        - global atom index of atom C.
            * ``"res_name_a"``   - residue name of atom A.
            * ``"res_name_b"``   - residue name of atom B.
            * ``"res_name_c"``   - residue name of atom C.
            * ``"chain_id_a"``   - chain ID of atom A.
            * ``"chain_id_b"``   - chain ID of atom B.
            * ``"chain_id_c"``   - chain ID of atom C.
            * ``"res_id_a"``     - residue ID of atom A.
            * ``"res_id_b"``     - residue ID of atom B.
            * ``"res_id_c"``     - residue ID of atom C.
            * ``"atom_name_a"``  - atom name of A.
            * ``"atom_name_b"``  - atom name of B (central).
            * ``"atom_name_c"``  - atom name of C.
            * ``"ideal"``        - ideal angle (degrees) used for this
              triplet.
            * ``"sigma"``        - standard deviation used for this
              angle.
            * ``"angle"``        - observed angle in degrees.
            * ``"z_score"``      - z-score of the observed angle.
        """
        if angle_data is None:
            angle_data = ANGLE_DATA

        res_groups = self.res_groups

        atom_name_arr = self._atom_name
        chain_id_arr = self._uni_chain_id
        res_id_arr = self._res_id
        coords = self.atom_array.coord

        bad_records: list[dict[str, np.ndarray]] = []

        for rname, angles in angle_data.items():
            if rname not in res_groups:
                continue

            idx_res = res_groups[rname]["idx_res"]
            group_ids_res = res_groups[rname]["group_ids"]
            n_groups = res_groups[rname]["n_groups"]

            atom_names_res = atom_name_arr[idx_res]

            for angle_key, (ideal, sigma) in angles.items():
                a_name, b_name, c_name = angle_key.split("_")

                mask_a = atom_names_res == a_name
                mask_b = atom_names_res == b_name
                mask_c = atom_names_res == c_name
                if not (mask_a.any() and mask_b.any() and mask_c.any()):
                    continue

                idx_a_atoms = idx_res[mask_a]
                gid_a = group_ids_res[mask_a]

                idx_b_atoms = idx_res[mask_b]
                gid_b = group_ids_res[mask_b]

                idx_c_atoms = idx_res[mask_c]
                gid_c = group_ids_res[mask_c]

                idx_a_group = np.full(n_groups, -1, dtype=np.int64)
                idx_b_group = np.full(n_groups, -1, dtype=np.int64)
                idx_c_group = np.full(n_groups, -1, dtype=np.int64)

                idx_a_group[gid_a] = idx_a_atoms
                idx_b_group[gid_b] = idx_b_atoms
                idx_c_group[gid_c] = idx_c_atoms

                valid = (idx_a_group >= 0) & (idx_b_group >= 0) & (idx_c_group >= 0)
                if not valid.any():
                    continue

                ia = idx_a_group[valid]
                ib = idx_b_group[valid]
                ic = idx_c_group[valid]

                A = coords[ia]
                B = coords[ib]
                C = coords[ic]

                v1 = A - B
                v2 = C - B
                v1_norm = np.linalg.norm(v1, axis=1)
                v2_norm = np.linalg.norm(v2, axis=1)

                eps = 1e-8
                len_valid = (v1_norm > eps) & (v2_norm > eps)
                if not len_valid.any():
                    continue

                cos_theta = np.empty_like(v1_norm)
                cos_theta[:] = np.nan

                dot = np.einsum("ij,ij->i", v1[len_valid], v2[len_valid])
                cos_theta[len_valid] = dot / (v1_norm[len_valid] * v2_norm[len_valid])
                cos_theta = np.clip(cos_theta, -1.0, 1.0)

                angle = np.degrees(np.arccos(cos_theta))
                z = (angle - ideal) / sigma
                bad = np.abs(z) > z_thresh

                if not bad.any():
                    continue

                ia_bad = ia[bad]
                ib_bad = ib[bad]
                ic_bad = ic[bad]
                angle_bad = angle[bad]
                z_bad = z[bad]

                res_name_arr = np.full_like(angle_bad, rname, dtype=object)

                bad_records.append(
                    {
                        "group": np.full_like(angle_bad, rname, dtype=object),
                        "res_name_a": res_name_arr,
                        "res_name_b": res_name_arr,
                        "res_name_c": res_name_arr,
                        "chain_id_a": chain_id_arr[ib_bad],
                        "chain_id_b": chain_id_arr[ib_bad],
                        "chain_id_c": chain_id_arr[ib_bad],
                        "res_id_a": res_id_arr[ib_bad],
                        "res_id_b": res_id_arr[ib_bad],
                        "res_id_c": res_id_arr[ib_bad],
                        "atom_name_a": np.full_like(angle_bad, a_name, dtype=object),
                        "atom_name_b": np.full_like(angle_bad, b_name, dtype=object),
                        "atom_name_c": np.full_like(angle_bad, c_name, dtype=object),
                        "angle_key": np.full_like(
                            angle_bad,
                            angle_key,
                            dtype=object,
                        ),
                        "ideal": np.full_like(angle_bad, ideal, dtype=float),
                        "sigma": np.full_like(angle_bad, sigma, dtype=float),
                        "angle": angle_bad,
                        "z_score": z_bad,
                        "idx_a": ia_bad,
                        "idx_b": ib_bad,
                        "idx_c": ic_bad,
                    }
                )

        if not bad_records:
            return pd.DataFrame()

        out: dict[str, np.ndarray] = {}
        for key in bad_records[0].keys():
            out[key] = np.concatenate([r[key] for r in bad_records])

        return pd.DataFrame(out)

    def find_bad_angles(
        self,
        angle_data: Optional[dict[str, dict[str, tuple[float, float]]]] = None,
        inter_res_angle_data: Optional[
            dict[str, dict[str, tuple[float, float]]]
        ] = None,
        z_thresh: float = 12.0,
    ) -> pd.DataFrame:
        """
        Detect intra- and inter-residue bond angle outliers.

        This method scans all defined bond angles in the structure and flags those
        whose deviation from the reference mean exceeds a given Z-score threshold.
        Intra-residue and inter-residue angles are evaluated separately using
        the corresponding reference tables and then concatenated into a single
        output table.

        Args:
            angle_data (dict[str, dict[str, tuple[float, float]]], optional):
                Reference statistics for intra-residue bond angles. The outer key
                typically corresponds to the residue name (e.g. ``"ALA"``) and the
                inner key to an angle identifier (e.g. ``"N-CA-C"``). Each value is
                a ``(mean, std)`` tuple in degrees. If ``None``, defaults to
                ``ANGLE_DATA``.
            inter_res_angle_data (dict[str, dict[str, tuple[float, float]]], optional):
                Reference statistics for inter-residue bond angles, using the same
                ``(mean, std)`` convention as ``angle_data``. If ``None``, defaults
                to ``INTER_RES_ANGLE_DATA``.
            z_thresh (float, optional):
                Absolute Z-score threshold above which an angle is considered
                an outlier. Defaults to ``12.0``.

        Returns:
            pandas.DataFrame:
                A table of bond angle outliers. Each row corresponds to a single
                intra- or inter-residue angle whose Z-score exceeds ``z_thresh``.
                The exact schema is defined by
                :meth:`find_bad_intra_res_angles` and
                :meth:`find_bad_inter_res_angles`, and typically includes residue
                identifiers, atom names, the observed angle, reference mean,
                Z-score, and a flag indicating intra- vs inter-residue origin.
        """
        if angle_data is None:
            angle_data = ANGLE_DATA
        if inter_res_angle_data is None:
            inter_res_angle_data = INTER_RES_ANGLE_DATA
        bad_intra_res_angles = self.find_bad_intra_res_angles(angle_data, z_thresh)
        bad_inter_res_angles = self.find_bad_inter_res_angles(
            inter_res_angle_data, z_thresh
        )
        return pd.concat([bad_intra_res_angles, bad_inter_res_angles])

    def find_clashes(
        self,
        query_mask: Optional[Sequence[bool]] = None,
        vdw_scale_factor: Optional[float] = None,
        tolerance: Optional[float] = 1.5,
        disulfide_clash_tolerance: float = 1.0,
        cutoff: float = 3.0,
    ) -> pd.DataFrame:
        """
        Identify steric clashes based on van der Waals radii.
        """
        atom_array = self.atom_array
        n_atoms = len(atom_array)

        if not ((vdw_scale_factor is None) ^ (tolerance is None)):
            raise ValueError(
                "Exactly one of `vdw_scale_factor` or `tolerance` must be provided."
            )

        if query_mask is None:
            query_mask = np.ones(n_atoms, dtype=bool)
        else:
            query_mask = np.asarray(query_mask, dtype=bool)

        coords = atom_array.coord
        tree = KDTree(coords)

        # 1. Get all candidate pairs within cutoff (idx1 < idx2)
        pairs = tree.query_pairs(cutoff, output_type="ndarray")
        if pairs.size == 0:
            return pd.DataFrame()

        idx1 = pairs[:, 0]
        idx2 = pairs[:, 1]

        # 2. Filter by query_mask: at least one atom must be in query_mask
        q_filter = query_mask[idx1] | query_mask[idx2]
        idx1, idx2 = idx1[q_filter], idx2[q_filter]
        if idx1.size == 0:
            return pd.DataFrame()

        # 3. Filter bonded atoms using precomputed bond hashes
        pair_hashes = idx1.astype(np.int64) * n_atoms + idx2

        # Fast filtering using np.searchsorted
        bond_hashes = self._bond_hashes
        search_idx = np.searchsorted(bond_hashes, pair_hashes)
        is_bonded = np.zeros(len(pair_hashes), dtype=bool)
        valid_search = search_idx < len(bond_hashes)
        is_bonded[valid_search] = (
            bond_hashes[search_idx[valid_search]] == pair_hashes[valid_search]
        )

        idx1, idx2 = idx1[~is_bonded], idx2[~is_bonded]
        if idx1.size == 0:
            return pd.DataFrame()

        # 4. Compute distances
        distances = np.linalg.norm(coords[idx1] - coords[idx2], axis=1)

        # 5. Compute contact limits
        r1 = self._vdw_radii[idx1]
        r2 = self._vdw_radii[idx2]

        if vdw_scale_factor is not None:
            contact_limits = vdw_scale_factor * (r1 + r2)
        else:
            # Special handling for CYS SG
            contact_limits = (r1 + r2) - tolerance

            # Identify CYS SG pairs
            res_names = self._res_name
            atom_names = self._atom_name
            is_cys_sg = (res_names == "CYS") & (atom_names == "SG")
            cys_sg_pair = is_cys_sg[idx1] & is_cys_sg[idx2]

            if np.any(cys_sg_pair):
                # Make sure SG-SG is not treated *more strictly* than generic
                # contacts. If the caller wants extra leniency they can pass a
                # value > `tolerance`.
                effective_tol = max(tolerance, disulfide_clash_tolerance)
                contact_limits[cys_sg_pair] = (
                    r1[cys_sg_pair] + r2[cys_sg_pair]
                ) - effective_tol

        # 6. Final clash filter
        clash_mask = distances < contact_limits
        if not np.any(clash_mask):
            return pd.DataFrame()

        idx1, idx2 = idx1[clash_mask], idx2[clash_mask]
        dist_clash = distances[clash_mask]
        limit_clash = contact_limits[clash_mask]

        elements = atom_array.element
        res_names = self._res_name
        out_df = pd.DataFrame(
            {
                "idx1": idx1,
                "idx2": idx2,
                "res_name1": res_names[idx1],
                "chain_id1": self._uni_chain_id[idx1],
                "res_id1": self._res_id[idx1],
                "res_name2": res_names[idx2],
                "chain_id2": self._uni_chain_id[idx2],
                "res_id2": self._res_id[idx2],
                "atom_name1": self._atom_name[idx1],
                "atom_name2": self._atom_name[idx2],
                "element1": elements[idx1],
                "element2": elements[idx2],
                "distance": dist_clash,
                "contact_limit": limit_clash,
                "overlap": limit_clash - dist_clash,
            }
        )
        return out_df

    def find_all_violations(
        self,
        clash_tolerance: float = 1.5,
        disulfide_clash_tolerance: float = 1.0,
        z_thresh: float = 12.0,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run clash, bond, and angle checks and return all violations.

        Args:
            clash_tolerance: Distance tolerance (Å) used for clash detection.
            disulfide_clash_tolerance: Distance tolerance used in disulfide clash detection.
            z_thresh: Absolute Z-score threshold for bond and angle outliers.

        Returns:
            Tuple of three DataFrames: (clash_df, bad_bond_df, bad_angle_df).
        """
        clash_df = self.find_clashes(
            tolerance=clash_tolerance,
            disulfide_clash_tolerance=disulfide_clash_tolerance,
        )
        bad_bond_df = self.find_bad_bonds(z_thresh=z_thresh)
        bad_angle_df = self.find_bad_angles(z_thresh=z_thresh)

        return clash_df, bad_bond_df, bad_angle_df

    def _get_is_backbone_atoms_mask(
        self,
        protein_chains: list[str],
        nuc_chains: list[str],
        chain_ids: list[str],
        atom_names: list[str],
    ) -> np.ndarray:
        is_bb_atom_mask = (
            np.isin(chain_ids, protein_chains) & np.isin(atom_names, PROTEIN_BACKBONE)
        ) | (np.isin(chain_ids, nuc_chains) & np.isin(atom_names, NUC_BACKBONE))
        return is_bb_atom_mask

    def get_valid_atom_mask(
        self,
        clash_tolerance: float = 1.5,
        disulfide_clash_tolerance: float = 1.0,
        z_thresh: float = 12.0,
    ) -> np.ndarray:
        """
        Compute a per-atom validity mask based on stereochemical violations.
        Optimized version using vectorized grouping.
        """
        # Collect all violations
        clash_df, bad_bond_df, bad_angle_df = self.find_all_violations(
            clash_tolerance=clash_tolerance,
            disulfide_clash_tolerance=disulfide_clash_tolerance,
            z_thresh=z_thresh,
        )

        # Cache for downstream consumers (e.g. stereochemistry-aware metrics)
        # NOTE: indices (idx*) refer to `self.atom_array`.
        self._last_violation_dfs = (clash_df, bad_bond_df, bad_angle_df)

        n_atoms = len(self.atom_array)
        issue_mask = np.zeros(n_atoms, dtype=bool)

        # Fill issue_mask from DataFrames
        for df, keys in [
            (clash_df, ["idx1", "idx2"]),
            (bad_bond_df, ["idx1", "idx2"]),
            (bad_angle_df, ["idx_a", "idx_b", "idx_c"]),
        ]:
            if not df.empty:
                for k in keys:
                    issue_mask[df[k].to_numpy(dtype=int)] = True

        valid_atom_mask = ~issue_mask
        if not issue_mask.any():
            return valid_atom_mask

        # Per-atom basic info
        all_res_ids = self._res_id
        all_atom_names = self._atom_name
        all_chain_ids = self._uni_chain_id

        # Determine which chains are protein vs nucleotide chains
        entity_id_to_chain_ids = self.struct.get_entity_id_to_chain_ids()
        protein_chains = []
        nuc_chains = []
        for k, v in self.struct.entity_poly_type.items():
            if v == PROTEIN:
                protein_chains.extend(entity_id_to_chain_ids[k])
            elif v in {DNA, RNA}:
                nuc_chains.extend(entity_id_to_chain_ids[k])

        polymer_chain_set = set(protein_chains + nuc_chains)

        # Backbone / sidechain classification at atom level
        all_is_bb = self._get_is_backbone_atoms_mask(
            protein_chains=protein_chains,
            nuc_chains=nuc_chains,
            chain_ids=all_chain_ids,
            atom_names=all_atom_names,
        )
        all_is_side_chain = ~all_is_bb

        # Map each atom to a unique residue index
        combined_res_keys = np.stack([all_chain_ids, all_res_ids], axis=1)
        _, res_indices = np.unique(combined_res_keys, axis=0, return_inverse=True)
        n_residues = res_indices.max() + 1

        # Identify residues with issues
        res_has_bb_issue = np.zeros(n_residues, dtype=bool)
        res_has_sc_issue = np.zeros(n_residues, dtype=bool)

        res_has_bb_issue[res_indices[issue_mask & all_is_bb]] = True
        res_has_sc_issue[res_indices[issue_mask & all_is_side_chain]] = True

        # Identify which residues are polymer
        # We can do this by checking the first atom of each residue
        # (Since all atoms of a residue belong to the same chain)
        first_atom_idx = np.unique(res_indices, return_index=True)[1]
        res_is_polymer = np.isin(all_chain_ids[first_atom_idx], list(polymer_chain_set))

        # Invalidation rules:
        # 1. If backbone issue -> invalidate entire residue
        res_to_invalidate_entirely = res_is_polymer & res_has_bb_issue
        # 2. If only sidechain issue -> invalidate only sidechain
        res_to_invalidate_sidechain = (
            res_is_polymer & res_has_sc_issue & (~res_has_bb_issue)
        )

        # Map back to atoms
        invalidate_mask = res_to_invalidate_entirely[res_indices] | (
            res_to_invalidate_sidechain[res_indices] & all_is_side_chain
        )

        # For non-polymer residues, we already have issue_mask[atom] = True if it has an issue.
        # So we just combine them.
        final_valid_mask = ~(
            invalidate_mask | (issue_mask & (~res_is_polymer[res_indices]))
        )

        return final_valid_mask


def stereochem_check_to_csv(
    model_cif: Union[str, Path],
    output_csv: Union[str, Path],
    *,
    include_bonds: bool = True,
    clash_tolerance: float = 1.5,
    disulfide_clash_tolerance: float = 1.0,
    z_thresh: float = 12.0,
) -> pd.DataFrame:
    """
    Run stereochemistry checks for a single model CIF and write a CSV report.

    Notes:
    - This API only depends on `model_cif` and does not require a reference CIF.
      Therefore, clash detection only uses the model's own bond graph.
    - The output is a single CSV that concatenates all violations with a
      `violation_type` column (clash/bond/angle).

    Returns:
        pandas.DataFrame: The merged report DataFrame (also written to `output_csv`).
            None if no violations found.
    """

    model_cif = Path(model_cif)
    output_csv = Path(output_csv)

    model_struct = Structure.from_mmcif(model_cif, include_bonds=include_bonds)
    checker = StereoChemValidator(struct=model_struct, ref_struct=None)

    clash_df = checker.find_clashes(
        tolerance=clash_tolerance,
        disulfide_clash_tolerance=disulfide_clash_tolerance,
    )
    bad_bond_df = checker.find_bad_bonds(z_thresh=z_thresh)
    bad_angle_df = checker.find_bad_angles(z_thresh=z_thresh)

    def _format_atom_series(
        df: pd.DataFrame,
        *,
        chain_col: str,
        res_name_col: str,
        res_id_col: str,
        atom_name_col: str,
    ) -> pd.Series:
        """Format single-atom identifier as `{chain}_{res}{res_id}_{atom}`."""
        res_id = df[res_id_col]
        if pd.api.types.is_numeric_dtype(res_id):
            # Avoid `12.0` when a numeric column is upcasted
            res_id_str = res_id.astype("Int64").astype(str)
        else:
            res_id_str = res_id.astype(str)
        return (
            df[chain_col].astype(str)
            + "_"
            + df[res_name_col].astype(str)
            + res_id_str
            + "_"
            + df[atom_name_col].astype(str)
        )

    dfs: list[pd.DataFrame] = []
    if clash_df is not None and (not clash_df.empty):
        a1 = _format_atom_series(
            clash_df,
            chain_col="chain_id1",
            res_name_col="res_name1",
            res_id_col="res_id1",
            atom_name_col="atom_name1",
        )
        a2 = _format_atom_series(
            clash_df,
            chain_col="chain_id2",
            res_name_col="res_name2",
            res_id_col="res_id2",
            atom_name_col="atom_name2",
        )
        clash_df = clash_df.assign(site=a1 + "..." + a2).drop(
            columns=[
                "chain_id1",
                "res_name1",
                "res_id1",
                "atom_name1",
                "chain_id2",
                "res_name2",
                "res_id2",
                "atom_name2",
            ],
            errors="ignore",
        )
        dfs.append(clash_df.assign(violation_type="clash"))
    if bad_bond_df is not None and (not bad_bond_df.empty):
        b1 = _format_atom_series(
            bad_bond_df,
            chain_col="chain_id1",
            res_name_col="res_name1",
            res_id_col="res_id1",
            atom_name_col="atom_name1",
        )
        b2 = _format_atom_series(
            bad_bond_df,
            chain_col="chain_id2",
            res_name_col="res_name2",
            res_id_col="res_id2",
            atom_name_col="atom_name2",
        )
        bad_bond_df = bad_bond_df.assign(site=b1 + "-" + b2).drop(
            columns=[
                "chain_id1",
                "res_name1",
                "res_id1",
                "atom_name1",
                "chain_id2",
                "res_name2",
                "res_id2",
                "atom_name2",
            ],
            errors="ignore",
        )
        dfs.append(bad_bond_df.assign(violation_type="bond"))
    if bad_angle_df is not None and (not bad_angle_df.empty):
        c1 = _format_atom_series(
            bad_angle_df,
            chain_col="chain_id_a",
            res_name_col="res_name_a",
            res_id_col="res_id_a",
            atom_name_col="atom_name_a",
        )
        c2 = _format_atom_series(
            bad_angle_df,
            chain_col="chain_id_b",
            res_name_col="res_name_b",
            res_id_col="res_id_b",
            atom_name_col="atom_name_b",
        )
        c3 = _format_atom_series(
            bad_angle_df,
            chain_col="chain_id_c",
            res_name_col="res_name_c",
            res_id_col="res_id_c",
            atom_name_col="atom_name_c",
        )
        bad_angle_df = bad_angle_df.assign(site=c1 + "-" + c2 + "-" + c3).drop(
            columns=[
                "chain_id_a",
                "res_name_a",
                "res_id_a",
                "atom_name_a",
                "chain_id_b",
                "res_name_b",
                "res_id_b",
                "atom_name_b",
                "chain_id_c",
                "res_name_c",
                "res_id_c",
                "atom_name_c",
                "idx_a",
                "idx_b",
                "idx_c",
            ],
            errors="ignore",
        )
        dfs.append(bad_angle_df.assign(violation_type="angle"))

    if len(dfs) == 0:
        report_df = pd.DataFrame(columns=["violation_type"])
    else:
        report_df = pd.concat(dfs, axis=0, ignore_index=True, sort=False)

    if report_df.empty:
        # No violations found
        return

    # Put violation_type first if present
    if "violation_type" in report_df.columns:
        first_cols = [c for c in ["violation_type", "site"] if c in report_df.columns]
        rest_cols = [c for c in report_df.columns if c not in set(first_cols)]
        report_df = report_df[first_cols + rest_cols]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_csv, index=False)
    return report_df
