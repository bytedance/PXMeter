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
from rdkit.Chem import GetPeriodicTable

from pxmeter.metrics.pb_valid.data.rdkit_radii import RDKIT_RADII

_periodic_table = GetPeriodicTable()


def _get_radii(atoms_elements, radius_type="vdw"):
    def _lookup(e):
        symbol = e.capitalize()
        if symbol in RDKIT_RADII:
            if radius_type == "vdw":
                return RDKIT_RADII[symbol]["vdw_radius"]
            elif radius_type == "covalent":
                return RDKIT_RADII[symbol]["covalent_radius"]

        # Fallback to RDKit
        try:
            if radius_type == "vdw":
                return _periodic_table.GetRvdw(symbol)
            elif radius_type == "covalent":
                return _periodic_table.GetRcovalent(symbol)
        except Exception:
            pass
        return 1.7

    return np.array([_lookup(e) for e in atoms_elements])


def check_internal_clash(mol, threshold_clash=0.3):
    """
    Reimplementing PoseBusters internal clash check logic.
    Follows PoseBusters default: ignore all atom pairs where at least one atom is H.
    """
    from rdkit.Chem.rdDistGeom import GetMoleculeBoundsMatrix

    if mol.GetNumAtoms() <= 1:
        return True, 0, np.nan

    bounds = GetMoleculeBoundsMatrix(
        mol, set15bounds=True, scaleVDW=True, doTriangleSmoothing=True
    )
    conf = mol.GetConformer()
    coords = conf.GetPositions()

    num_atoms = mol.GetNumAtoms()
    lower_triangle_idcs = np.tril_indices(num_atoms, k=-1)

    # distance matrix
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)

    obs_dist = dist_matrix[lower_triangle_idcs]
    lower_bound = bounds[lower_triangle_idcs]

    # We need to filter out bonds and angles
    bond_indices = set()
    for bond in mol.GetBonds():
        i, j = sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        bond_indices.add((i, j))

    angle_indices = set()
    for i in range(num_atoms):
        neighbors = [a.GetIdx() for a in mol.GetAtomWithIdx(i).GetNeighbors()]
        for idx1 in range(len(neighbors)):
            for idx2 in range(idx1 + 1, len(neighbors)):
                a1, a2 = sorted((neighbors[idx1], neighbors[idx2]))
                angle_indices.add((a1, a2))

    is_clash_candidate = []
    for i, j in zip(*lower_triangle_idcs):
        # lower_triangle_idcs has j < i
        pair = (int(j), int(i))
        if pair in bond_indices or pair in angle_indices:
            is_clash_candidate.append(False)
        else:
            # PoseBusters default ignore_hydrogens=True:
            # Skip if either atom is Hydrogen (atomic number 1)
            if (
                mol.GetAtomWithIdx(pair[0]).GetAtomicNum() == 1
                or mol.GetAtomWithIdx(pair[1]).GetAtomicNum() == 1
            ):
                is_clash_candidate.append(False)
            else:
                is_clash_candidate.append(True)

    is_clash_candidate = np.array(is_clash_candidate)
    if not any(is_clash_candidate):
        return True, 0, np.nan

    relevant_dist = obs_dist[is_clash_candidate]
    relevant_lb = lower_bound[is_clash_candidate]

    bpe = (relevant_dist - relevant_lb) / relevant_lb
    num_clashes = np.sum(bpe < -threshold_clash)
    no_internal_clash = num_clashes == 0
    closest_noncov = np.min(relevant_dist / relevant_lb)

    return no_internal_clash, int(num_clashes), closest_noncov


def check_intermolecular_distance(
    mol_pred,
    cond_atom_array,
    clash_cutoff=0.75,
    max_distance=5.0,
    ignore_types=None,
    inorganic_mask=None,
    ligand_radius_type="vdw",
):
    """
    mol_pred: RDKit Mol
    cond_atom_array: Biotite AtomArray
    inorganic_mask: boolean mask for atoms in cond_atom_array that should use covalent radius
    """
    if ignore_types is None:
        ignore_types = {"H"}
    coords_ligand = mol_pred.GetConformer().GetPositions()
    elements_ligand = [a.GetSymbol() for a in mol_pred.GetAtoms()]

    # Filter hydrogens from ligand
    mask_lig = np.array([e != "H" for e in elements_ligand])
    coords_ligand = coords_ligand[mask_lig]
    elements_ligand = np.array(elements_ligand)[mask_lig]

    if len(coords_ligand) == 0:
        return True, True, 0, np.nan

    # Filter cond_atom_array
    mask_cond = np.ones(len(cond_atom_array), dtype=bool)
    if "H" in ignore_types or "hydrogens" in ignore_types:
        mask_cond &= cond_atom_array.element != "H"

    coords_cond = cond_atom_array.coord[mask_cond]
    elements_cond = cond_atom_array.element[mask_cond]
    if inorganic_mask is not None:
        inorg_mask_filtered = inorganic_mask[mask_cond]
    else:
        inorg_mask_filtered = np.zeros(len(coords_cond), dtype=bool)

    if len(coords_cond) == 0:
        return True, True, 0, np.nan

    radii_ligand = _get_radii(elements_ligand, ligand_radius_type)

    # Get radii for cond atoms: inorganic use covalent, others use vdw
    radii_cond = np.zeros(len(coords_cond))
    if any(inorg_mask_filtered):
        radii_cond[inorg_mask_filtered] = _get_radii(
            elements_cond[inorg_mask_filtered], "covalent"
        )
    if any(~inorg_mask_filtered):
        radii_cond[~inorg_mask_filtered] = _get_radii(
            elements_cond[~inorg_mask_filtered], "vdw"
        )

    # Distance calculation
    from scipy.spatial import cKDTree

    tree = cKDTree(coords_cond)
    dists, _ = tree.query(coords_ligand, k=1)

    # smallest distance
    min_dist = np.min(dists)
    not_too_far = min_dist <= max_distance

    # Clashes
    # PoseBusters logic: dist / (r1 + r2) < clash_cutoff
    max_r_sum = np.max(radii_ligand) + np.max(radii_cond)
    pairs = tree.query_ball_point(coords_ligand, r=max_r_sum * clash_cutoff)

    num_clashes = 0
    for i, neighbors in enumerate(pairs):
        r_i = radii_ligand[i]
        for j in neighbors:
            r_j = radii_cond[j]
            d = np.linalg.norm(coords_ligand[i] - coords_cond[j])
            if d / (r_i + r_j) < clash_cutoff:
                num_clashes += 1

    no_clashes = num_clashes == 0
    return not_too_far, no_clashes, num_clashes, min_dist
