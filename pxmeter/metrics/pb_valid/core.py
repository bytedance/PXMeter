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

import biotite.structure as struc
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdShapeHelpers import ShapeTverskyIndex
from scipy.spatial import KDTree

from .chemistry import (
    check_all_atoms_connected,
    check_identity,
    check_radicals,
    check_sanitization,
)
from .clashes import check_intermolecular_distance, check_internal_clash
from .energy import check_energy_ratio
from .geometry import check_bond_and_angle_violations, check_flatness

_inorganic_cofactor_elements = {
    "Li",
    "Be",
    "Na",
    "Mg",
    "Cl",
    "K",
    "Ca",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Br",
    "Rb",
    "Mo",
    "Cd",
}
_inorganic_cofactor_ccd_codes = {"FES", "MOS", "PO3", "PO4", "PPK", "SO3", "SO4", "VO4"}
_water_ccd_codes = {"HOH"}


def biotite_to_rdkit_fast(atom_array):
    """
    Convert Biotite AtomArray to a simple RDKit Mol (no bonds, just for shape/distance checks).
    """
    mol = Chem.RWMol()
    conf = Chem.Conformer(len(atom_array))
    for i, atom in enumerate(atom_array):
        try:
            rd_atom = Chem.Atom(atom.element.capitalize())
        except Exception:
            # Fallback to Carbon if element is invalid (e.g. "X")
            rd_atom = Chem.Atom("C")
        idx = mol.AddAtom(rd_atom)
        # Convert to list or float64 to avoid RDKit type extraction issues with float32
        conf.SetAtomPosition(idx, atom.coord.astype(float))
    mol.AddConformer(conf)
    return mol.GetMol()


def run_pb_valid(
    mol_pred: Chem.Mol,
    mol_true: Chem.Mol,
    mol_cond: struc.AtomArray,
    mol_cond_valid_mask: Optional[np.ndarray] = None,
    mol_pred_valid_mask: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Run pose-busting valid check.

    Args:
        mol_pred (Chem.Mol): Predicted ligand RDKit molecule
        mol_true (Chem.Mol): True ligand RDKit molecule
        mol_cond (AtomArray): Conditioner receptor Biotite AtomArray
        mol_cond_valid_mask (np.ndarray): Boolean mask indicating valid atoms in mol_cond
        mol_pred_valid_mask (np.ndarray): Boolean mask indicating valid atoms in mol_pred

    Returns:
        pd.DataFrame: DataFrame containing pose-busting valid results
    """
    # handle unmapped coordinates in mol_cond (from penalize_unmapped_reference_atoms)
    if mol_cond_valid_mask is not None:
        if not np.all(mol_cond_valid_mask):
            mol_cond = mol_cond[mol_cond_valid_mask]

    coords_pred = mol_pred.GetConformer().GetPositions()
    if mol_pred_valid_mask is not None:
        if not np.all(mol_pred_valid_mask):
            # Return invalid PB result if the ligand has missing atoms
            cols = [
                "mol_pred_loaded",
                "mol_true_loaded",
                "mol_cond_loaded",
                "sanitization",
                "inchi_convertible",
                "all_atoms_connected",
                "molecular_formula",
                "molecular_bonds",
                "double_bond_stereochemistry",
                "tetrahedral_chirality",
                "protein-ligand_maximum_distance",
                "minimum_distance_to_protein",
                "minimum_distance_to_organic_cofactors",
                "minimum_distance_to_inorganic_cofactors",
                "minimum_distance_to_waters",
                "internal_steric_clash",
                "aromatic_ring_flatness",
                "double_bond_flatness",
                "internal_energy",
            ]
            return pd.DataFrame({k: [False] for k in cols})

    # Suppress RDKit logs locally to handle multi-processing workers
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    results = {}

    # 1. Loading (Assume loaded)
    results["mol_pred_loaded"] = mol_pred is not None
    results["mol_true_loaded"] = mol_true is not None
    results["mol_cond_loaded"] = mol_cond is not None

    if mol_pred is None:
        return pd.DataFrame([results])

    if mol_cond is None:
        return pd.DataFrame([results])

    # Classify cond atoms
    inorganic_mask = np.array(
        [
            (e.capitalize() in _inorganic_cofactor_elements)
            or (r in _inorganic_cofactor_ccd_codes)
            for e, r in zip(mol_cond.element, mol_cond.res_name)
        ]
    )
    water_mask = np.array([r in _water_ccd_codes for r in mol_cond.res_name])
    protein_mask = struc.filter_amino_acids(mol_cond) | struc.filter_nucleotides(
        mol_cond
    )
    organic_mask = ~(protein_mask | inorganic_mask | water_mask)

    # 2. Chemistry
    # Perceive stereochemistry from 3D coordinates.
    # We add Hydrogens back based on coordinates to make stereo perception robust.
    # We also clear existing chiral tags to ensure perception is purely from 3D.
    def prepare_for_stereo(mol):
        if mol is None:
            return None
        m = Chem.Mol(mol)
        m.ClearComputedProps()
        for atom in m.GetAtoms():
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        for bond in m.GetBonds():
            bond.SetStereo(Chem.BondStereo.STEREONONE)

        try:
            # AddHs with addCoords=True uses 3D geometry to place Hs correctly
            m = Chem.AddHs(m, addCoords=True)
            Chem.AssignStereochemistryFrom3D(m)
        except Exception:
            pass
        return m

    mol_pred = prepare_for_stereo(mol_pred)
    mol_true = prepare_for_stereo(mol_true)

    results["sanitization"] = check_sanitization(mol_pred)
    results["no_radicals"] = check_radicals(mol_pred)
    try:
        inchi_val = Chem.MolToInchi(mol_pred)
        results["inchi_convertible"] = inchi_val != ""
    except Exception:
        results["inchi_convertible"] = False

    results["all_atoms_connected"] = check_all_atoms_connected(mol_pred)

    formula_match, connections_match, dbond_match, tetra_match = check_identity(
        mol_pred, mol_true
    )
    results["molecular_formula"] = formula_match
    results["molecular_bonds"] = connections_match
    results["double_bond_stereochemistry"] = dbond_match
    results["tetrahedral_chirality"] = tetra_match

    # 3. Geometry
    (
        bonds_ok,
        angles_ok,
        n_bad_bonds,
        n_bad_angles,
        shortest_bond,
        longest_bond,
        extremest_angle,
    ) = check_bond_and_angle_violations(mol_pred)
    results["bond_lengths"] = bonds_ok
    results["bond_angles"] = angles_ok
    results["number_outlier_bonds"] = n_bad_bonds
    results["number_outlier_angles"] = n_bad_angles
    results["shortest_bond_relative_length"] = shortest_bond
    results["longest_bond_relative_length"] = longest_bond
    results["most_extreme_relative_angle"] = extremest_angle

    internal_clash_ok, n_internal_clashes, closest_noncov = check_internal_clash(
        mol_pred
    )
    results["internal_steric_clash"] = internal_clash_ok
    results["number_clashes"] = n_internal_clashes
    results["shortest_noncovalent_relative_distance"] = closest_noncov

    # Ring flatness
    flat_rings_ok, max_dist_ring = check_flatness(
        mol_pred, threshold_flatness=0.25, check_rings=True, check_double_bonds=False
    )
    results["aromatic_ring_flatness"] = flat_rings_ok
    results["aromatic_ring_maximum_distance_from_plane"] = max_dist_ring

    # Ring non-flatness (check that non-aromatic rings are NOT flat)
    # PoseBusters threshold is 0.05, and check_nonflat=True
    nonflat_rings_ok, min_dist_nonflat = check_flatness(
        mol_pred,
        threshold_flatness=0.05,
        check_rings=False,
        check_double_bonds=False,
        check_nonflat_rings=True,
    )
    results["non_aromatic_ring_non_flatness"] = nonflat_rings_ok
    results["non_aromatic_ring_maximum_distance_from_plane"] = min_dist_nonflat

    # Double bond flatness
    flat_db_ok, max_dist_db = check_flatness(
        mol_pred, threshold_flatness=0.25, check_rings=False, check_double_bonds=True
    )
    results["double_bond_flatness"] = flat_db_ok
    results["double_bond_maximum_distance_from_plane"] = max_dist_db

    # sp2 flatness
    flat_sp2_ok, max_dist_sp2 = check_flatness(
        mol_pred,
        threshold_flatness=0.25,
        check_rings=False,
        check_double_bonds=False,
        check_sp2_centers=True,
    )
    results["sp2_center_flatness"] = flat_sp2_ok
    results["sp2_center_maximum_distance_from_plane"] = max_dist_sp2

    # Amide flatness
    flat_amide_ok, max_dist_amide = check_flatness(
        mol_pred,
        threshold_flatness=0.25,
        check_rings=False,
        check_double_bonds=False,
        check_amides=True,
    )
    results["amide_flatness"] = flat_amide_ok
    results["amide_maximum_distance_from_plane"] = max_dist_amide

    # sp3 center non-flatness
    nonflat_sp3_ok, min_dist_sp3 = check_flatness(
        mol_pred,
        threshold_flatness=0.05,
        check_rings=False,
        check_double_bonds=False,
        check_nonflat_sp3_centers=True,
    )
    results["sp3_center_non_flatness"] = nonflat_sp3_ok
    results["sp3_center_maximum_distance_from_plane"] = min_dist_sp3

    # 4. Energy
    energy_ok, energy_ratio, pred_e, avg_e = check_energy_ratio(
        mol_pred, threshold_energy_ratio=100.0
    )
    results["internal_energy"] = energy_ok
    results["energy_ratio"] = energy_ratio
    results["mol_pred_energy"] = pred_e
    results["ensemble_avg_energy"] = avg_e

    # 5. Intermolecular
    # Protein
    if protein_mask.any():
        (
            not_too_far,
            no_clash_prot,
            n_clash_prot,
            min_dist_prot,
        ) = check_intermolecular_distance(
            mol_pred,
            mol_cond[protein_mask],
            max_distance=5.0,
            clash_cutoff=0.75,
            inorganic_mask=inorganic_mask[protein_mask],
        )
        results["protein-ligand_maximum_distance"] = not_too_far
        results["minimum_distance_to_protein"] = no_clash_prot
        results["num_pairwise_clashes_protein"] = n_clash_prot
        results["smallest_distance_protein"] = min_dist_prot
    else:
        results["protein-ligand_maximum_distance"] = True
        results["minimum_distance_to_protein"] = True
        results["num_pairwise_clashes_protein"] = 0
        results["smallest_distance_protein"] = np.nan

    # Organic cofactors
    if organic_mask.any():
        (_, no_clash_org, n_clash_org, min_dist_org,) = check_intermolecular_distance(
            mol_pred,
            mol_cond[organic_mask],
            clash_cutoff=0.75,
            inorganic_mask=inorganic_mask[organic_mask],
        )
        results["minimum_distance_to_organic_cofactors"] = no_clash_org
    else:
        results["minimum_distance_to_organic_cofactors"] = True

    # Inorganic cofactors
    if inorganic_mask.any():
        (
            _,
            no_clash_inorg,
            n_clash_inorg,
            min_dist_inorg,
        ) = check_intermolecular_distance(
            mol_pred,
            mol_cond[inorganic_mask],
            clash_cutoff=0.75,
            inorganic_mask=inorganic_mask[inorganic_mask],
            ligand_radius_type="covalent",
        )
        results["minimum_distance_to_inorganic_cofactors"] = no_clash_inorg
    else:
        results["minimum_distance_to_inorganic_cofactors"] = True

    # Waters
    if water_mask.any():
        (
            _,
            no_clash_water,
            n_clash_water,
            min_dist_water,
        ) = check_intermolecular_distance(
            mol_pred,
            mol_cond[water_mask],
            clash_cutoff=0.75,
            inorganic_mask=inorganic_mask[water_mask],
        )
        results["minimum_distance_to_waters"] = no_clash_water
    else:
        results["minimum_distance_to_waters"] = True

    # 6. Volume Overlap
    def get_overlap(mol_pred, mask, vdw_scale, search_distance=6.0):
        if not mask.any():
            return True, None
        try:
            # Filter conditioning atoms by proximity to the ligand to avoid expensive overlap computation
            # on atoms that are far away and cannot contribute to overlap.
            cond_atoms = mol_cond[mask]

            # Drop hydrogens early (ShapeTverskyIndex is run with ignoreHs=True)
            try:
                cond_heavy_mask = np.char.upper(cond_atoms.element.astype(str)) != "H"
                cond_atoms = cond_atoms[cond_heavy_mask]
            except Exception:
                pass

            if len(cond_atoms) == 0:
                return True, None

            try:
                lig_coords = mol_pred.GetConformer().GetPositions()
                lig_elems = np.array([a.GetSymbol() for a in mol_pred.GetAtoms()])
                lig_heavy = lig_coords[lig_elems != "H"]
                if lig_heavy.shape[0] == 0:
                    lig_heavy = lig_coords

                tree = KDTree(lig_heavy)
                dists, _ = tree.query(cond_atoms.coord, k=1)
                keep = dists <= (search_distance * vdw_scale)
                cond_atoms = cond_atoms[keep]
            except Exception:
                # If KDTree filtering fails, fall back to using all atoms
                pass

            if len(cond_atoms) == 0:
                return True, None

            mol_rd = biotite_to_rdkit_fast(cond_atoms)
            overlap = ShapeTverskyIndex(
                mol_pred, mol_rd, alpha=1, beta=0, vdwScale=vdw_scale, ignoreHs=True
            )
            return overlap <= 0.075, overlap
        except Exception:
            return True, None

    (
        results["volume_overlap_with_protein"],
        results["volume_overlap_protein"],
    ) = get_overlap(mol_pred, protein_mask, 0.8)
    (
        results["volume_overlap_with_organic_cofactors"],
        results["volume_overlap_organic_cofactors"],
    ) = get_overlap(mol_pred, organic_mask, 0.8)
    (
        results["volume_overlap_with_inorganic_cofactors"],
        results["volume_overlap_inorganic_cofactors"],
    ) = get_overlap(mol_pred, inorganic_mask, 0.5)
    (
        results["volume_overlap_with_waters"],
        results["volume_overlap_waters"],
    ) = get_overlap(mol_pred, water_mask, 0.5)

    # 7. RMSD
    try:
        from rdkit.Chem import rdMolAlign

        # Use copies to avoid potential in-place coordinate modification by RDKit
        m_pred = Chem.Mol(mol_pred)
        m_true = Chem.Mol(mol_true)

        # 1. Centroid distance
        c1 = m_pred.GetConformer().GetPositions().mean(axis=0)
        c2 = m_true.GetConformer().GetPositions().mean(axis=0)
        centroid_dist = float(np.linalg.norm(c1 - c2))

        # 2. RMSD (unaligned, symmetry-unaware as per PoseBusters CalcRMS default)
        rmsd = rdMolAlign.CalcRMS(m_pred, m_true)

        # 3. Kabsch RMSD (aligned, symmetry-aware as per PoseBusters GetBestRMS default)
        kabsch_rmsd = rdMolAlign.GetBestRMS(m_pred, m_true)

        results["rmsd"] = rmsd
        results["kabsch_rmsd"] = kabsch_rmsd
        results["centroid_distance"] = centroid_dist
        results["rmsd_≤_2å"] = rmsd <= 2.0
    except Exception:
        results["rmsd"] = None
        results["kabsch_rmsd"] = None
        results["centroid_distance"] = None
        results["rmsd_≤_2å"] = False

    return pd.DataFrame([results])
