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

from rdkit import Chem


def check_sanitization(mol):
    try:
        mol_copy = Chem.Mol(mol)
        res = Chem.SanitizeMol(mol_copy)
        return res == Chem.SanitizeFlags.SANITIZE_NONE
    except Exception:
        return False


def check_radicals(mol):
    """Check that a molecule has no radicals. (Matches PoseBusters logic)"""
    try:
        # Check radicals before sanitization (PoseBusters logic)
        radicals_before = any(a.GetNumRadicalElectrons() for a in mol.GetAtoms())

        # We need to sanitize to catch some implicit valence issues that manifest as radicals,
        # but we must work on a copy to avoid modifying the input molecule state.
        mol_copy = Chem.Mol(mol)
        Chem.SanitizeMol(mol_copy, catchErrors=True)
        radicals_after = any(a.GetNumRadicalElectrons() for a in mol_copy.GetAtoms())

        return not radicals_after
    except Exception:
        return False


def check_all_atoms_connected(mol):
    return len(Chem.GetMolFrags(mol)) <= 1


def check_identity(pred_mol, true_mol):
    """
    Check if the molecular formula and connections are the same.
    Follows PoseBusters logic for separating connectivity and stereochemistry.
    """
    if pred_mol is None or true_mol is None:
        return False, False, False, False

    # Formula
    pred_formula = Chem.rdMolDescriptors.CalcMolFormula(pred_mol)
    true_formula = Chem.rdMolDescriptors.CalcMolFormula(true_mol)
    formula_match = pred_formula == true_formula

    def get_inchi_layers(mol):
        try:
            # Use /FixedH to match PoseBusters default behavior for consistent H handling
            inchi = Chem.MolToInchi(mol, options="/FixedH")
            layers = {}
            parts = inchi.split("/")
            for part in parts:
                if len(part) > 0:
                    layers[part[0]] = part[1:]
            return layers
        except Exception:
            return None

    pred_layers = get_inchi_layers(pred_mol)
    true_layers = get_inchi_layers(true_mol)

    if pred_layers is None or true_layers is None:
        return formula_match, False, False, False

    # Connections: InChI layer /c
    connections_match = pred_layers.get("c") == true_layers.get("c")

    # Double bond stereochemistry: InChI layer /b
    dbond_match = pred_layers.get("b") == true_layers.get("b")

    # Tetrahedral chirality: InChI layers /t, /m, /s
    t_match = pred_layers.get("t") == true_layers.get("t")
    m_match = pred_layers.get("m") == true_layers.get("m")
    s_match = pred_layers.get("s") == true_layers.get("s")
    tetra_match = t_match and m_match and s_match

    return formula_match, connections_match, dbond_match, tetra_match
