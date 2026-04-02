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

from functools import lru_cache

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdDistGeom import EmbedMultipleConfs, ETKDGv3
from rdkit.Chem.rdForceFieldHelpers import (
    UFFGetMoleculeForceField,
    UFFHasAllMoleculeParams,
    UFFOptimizeMoleculeConfs,
)


def get_conf_energy(mol, conf_id=-1):
    try:
        mol_with_hs = Chem.AddHs(mol, addCoords=True)
        if not UFFHasAllMoleculeParams(mol_with_hs):
            return np.nan
        uff = UFFGetMoleculeForceField(mol_with_hs, confId=conf_id)
        if uff:
            return uff.CalcEnergy()
    except Exception:
        pass
    return np.nan


@lru_cache(maxsize=128)
def get_average_energy(inchi, n_confs=50):
    try:
        mol = Chem.MolFromInchi(inchi)
        if not mol:
            return np.nan

        mol = Chem.AddHs(mol)
        if not UFFHasAllMoleculeParams(mol):
            return np.nan

        etkdg = ETKDGv3()
        etkdg.randomSeed = 42
        etkdg.useRandomCoords = True

        cids = EmbedMultipleConfs(mol, n_confs, etkdg)
        if not cids:
            return np.nan

        # Energy minimization
        res = UFFOptimizeMoleculeConfs(mol)
        energies = [v[1] for v in res]
        return sum(energies) / len(energies)
    except Exception:
        return np.nan


def check_energy_ratio(mol, threshold_energy_ratio=100.0, n_confs=50):
    """
    mol: RDKit Mol (pred)
    """
    conf_energy = get_conf_energy(mol)
    if np.isnan(conf_energy):
        return True, np.nan, np.nan, np.nan  # Assume pass if cannot calculate

    try:
        inchi = Chem.MolToInchi(mol)
    except Exception:
        return True, np.nan, conf_energy, np.nan

    avg_energy = get_average_energy(inchi, n_confs=n_confs)
    if np.isnan(avg_energy) or avg_energy == 0:
        return True, np.nan, conf_energy, avg_energy

    ratio = conf_energy / avg_energy
    passes = ratio <= threshold_energy_ratio
    return passes, ratio, conf_energy, avg_energy
