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
from rdkit import Chem
from rdkit.Chem.rdDistGeom import GetMoleculeBoundsMatrix


def check_bond_and_angle_violations(
    mol, threshold_bad_bond_length=0.25, threshold_bad_angle=0.25
):
    if mol.GetNumAtoms() <= 1:
        return True, True, 0, 0, np.nan, np.nan, np.nan

    # DG bounds
    bounds = GetMoleculeBoundsMatrix(
        mol, set15bounds=True, scaleVDW=True, doTriangleSmoothing=True
    )
    conf = mol.GetConformer()
    coords = conf.GetPositions()

    num_atoms = mol.GetNumAtoms()

    # Get bonds
    bond_pairs = []
    for bond in mol.GetBonds():
        i, j = sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        # Ignore hydrogens as per PoseBusters default
        if (
            mol.GetAtomWithIdx(i).GetAtomicNum() == 1
            or mol.GetAtomWithIdx(j).GetAtomicNum() == 1
        ):
            continue
        bond_pairs.append((i, j))

    # Get angles (1-3 atom pairs)
    angle_pairs = []  # Store as (atom1, atom3) where atom2 is the shared atom
    for i in range(num_atoms):
        if mol.GetAtomWithIdx(i).GetAtomicNum() == 1:
            continue
        neighbors = [
            a.GetIdx()
            for a in mol.GetAtomWithIdx(i).GetNeighbors()
            if a.GetAtomicNum() != 1
        ]
        for idx1 in range(len(neighbors)):
            for idx2 in range(idx1 + 1, len(neighbors)):
                a1, a3 = sorted((neighbors[idx1], neighbors[idx2]))
                angle_pairs.append((a1, a3))

    # Check bonds
    num_bad_bonds = 0
    bond_rel_lengths = []
    for i, j in bond_pairs:
        dist = np.linalg.norm(coords[i] - coords[j])
        lb = bounds[j, i]  # lower bound
        ub = bounds[i, j]  # upper bound

        # PoseBusters bpe
        if dist < lb:
            pe = (dist - lb) / lb
        elif dist > ub:
            pe = (dist - ub) / ub
        else:
            pe = 0.0

        if abs(pe) > threshold_bad_bond_length:
            num_bad_bonds += 1

        bond_rel_lengths.append((dist / lb, dist / ub))

    shortest_bond = (
        np.min([x[0] for x in bond_rel_lengths]) if bond_rel_lengths else np.nan
    )
    longest_bond = (
        np.max([x[1] for x in bond_rel_lengths]) if bond_rel_lengths else np.nan
    )

    # Check angles
    num_bad_angles = 0
    angle_rel_dist = []
    for i, k in angle_pairs:
        dist = np.linalg.norm(coords[i] - coords[k])
        lb = bounds[k, i]
        ub = bounds[i, k]

        if dist < lb:
            ape = abs((dist - lb) / lb)
        elif dist > ub:
            ape = abs((dist - ub) / ub)
        else:
            ape = 0.0

        if ape > threshold_bad_angle:
            num_bad_angles += 1

        angle_rel_dist.append((dist / lb, dist / ub))

    if angle_rel_dist:
        lb_extreme = 2 - np.min([x[0] for x in angle_rel_dist])
        ub_extreme = np.max([x[1] for x in angle_rel_dist])
        extremest_angle = max(lb_extreme, ub_extreme)
    else:
        extremest_angle = np.nan

    return (
        (num_bad_bonds == 0),
        (num_bad_angles == 0),
        num_bad_bonds,
        num_bad_angles,
        shortest_bond,
        longest_bond,
        extremest_angle,
    )


def check_flatness(
    mol,
    threshold_flatness=0.1,
    check_rings=True,
    check_double_bonds=True,
    check_sp2_centers=False,
    check_amides=False,
    check_nonflat_rings=False,
    check_nonflat_sp3_centers=False,
):
    """
    Checks aromatic ring flatness, double bond flatness, sp2 center planarity,
    amide group planarity, or ring non-flatness, sp3 center non-flatness.
    """
    flat_systems = {}
    if check_rings:
        flat_systems["aromatic_5"] = "[ar5^2]1[ar5^2][ar5^2][ar5^2][ar5^2]1"
        flat_systems["aromatic_6"] = "[ar6^2]1[ar6^2][ar6^2][ar6^2][ar6^2][ar6^2]1"
    if check_double_bonds:
        flat_systems["double_bond"] = "[C;X3;^2](*)(*)=[C;X3;^2](*)(*)"
    if check_sp2_centers:
        flat_systems["sp2_C"] = "[*]~[#6;D3;^2](~[*])~[*]"
    if check_amides:
        flat_systems["amides"] = "[CX3](=[OX1])[NX3][*]"

    exclude_atoms = set()
    if check_nonflat_rings:
        flat_systems = {
            "non-aromatic_5_membered_rings": "[C,O,S,N;R1]~1[C,O,S,N;R1][C,O,S,N;R1][C,O,S,N;R1][C,O,S,N;R1]1",
            "non-aromatic_6_membered_rings": "[C,O,S,N;R1]~1[C,O,S,N;R1][C,O,S,N;R1][C,O,S,N;R1][C,O,S,N;R1][C,O,S,N;R1]1",
            "non-aromatic_6_membered_rings_db03_0": "[C;R1]~1[C;R1][C,O,S,N;R1]~[C,O,S,N;R1][C;R1][C;R1]1",
            "non-aromatic_6_membered_rings_db03_1": "[C;R1]~1[C;R1][C;R1]~[C;R1][C,O,S,N;R1][C;R1]1",
            "non-aromatic_6_membered_rings_db02_0": "[C;R1]~1[C;R1][C;R1][C,O,S,N;R1]~[C,O,S,N;R1][C;R1]1",
            "non-aromatic_6_membered_rings_db02_1": "[C;R1]~1[C;R1][C,O,S,N;R1][C;R1]~[C;R1][C;R1]1",
        }
        exclude_systems = {
            # O2U and AWJ
            "non-aromatic_5_membered_rings_3_adj_sp2": "[r5&^2]@[r5&^2]@[r5&^2]",
            # BR8 and W9F; O9U is not flat in CCD ideal coords but is flat in PDB 7FWM
            "non-aromatic_6_membered_rings_4_adj_sp2": "[r6&^2]@[r6&^2]@[r6&^2]@[r6&^2]",
        }
        for smarts in exclude_systems.values():
            pat = Chem.MolFromSmarts(smarts)
            if pat:
                for m in mol.GetSubstructMatches(pat):
                    m_set = set(m)
                    for ring in mol.GetRingInfo().AtomRings():
                        if m_set.issubset(ring):
                            exclude_atoms.update(ring)

    if check_nonflat_sp3_centers:
        flat_systems = {"nonflat_sp3_C_N": "[*][#6,#7;^3]([*])[*]"}

    planar_groups = []
    for smarts in flat_systems.values():
        match = Chem.MolFromSmarts(smarts)
        if match:
            matches = mol.GetSubstructMatches(match)
            if check_nonflat_rings and exclude_atoms:
                for group in matches:
                    if not set(group).issubset(exclude_atoms):
                        planar_groups.append(group)
            else:
                planar_groups.extend(matches)

    if not planar_groups:
        return True, None

    conf = mol.GetConformer()
    max_dist_all = []
    all_pass = True

    for group in planar_groups:
        coords = np.array([conf.GetAtomPosition(i) for i in group])
        # SVD for plane fitting
        centered = coords - coords.mean(axis=0)
        try:
            _, _, vh = np.linalg.svd(centered)
            normal = vh[-1]
            dists = np.abs(np.dot(centered, normal))
            max_d = np.max(dists)

            max_dist_all.append(max_d)

            if not (check_nonflat_rings or check_nonflat_sp3_centers):
                # Normal flatness check: distance should be small
                if max_d > threshold_flatness:
                    all_pass = False
            else:
                # Non-flatness check: distance should be large (at least one atom out of plane)
                if max_d < threshold_flatness:
                    all_pass = False

        except Exception:
            # SVD might fail for very few atoms or linear systems
            all_pass = False

    if not max_dist_all:
        return True, None

    if not (check_nonflat_rings or check_nonflat_sp3_centers):
        return all_pass, max(max_dist_all)
    else:
        # For non-flatness, we return the MINIMUM deviation found (closest to being flat)
        # If any ring is "too flat", it fails.
        return all_pass, min(max_dist_all)
