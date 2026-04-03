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


# "subset_label" accepts a single label or multiple labels
# separated by semicolons, e.g. "[antibody-protein];[peptide-interface]".
# When multiple labels are provided, their corresponding masks
# on the metrics DataFrame are intersected first, then
# "inverse_subset" is applied (if True).
DATASET_METRICS_CONFIG = {
    "RecentPDB": [
        {
            "subset": "All",
            "subset_label": None,
            "inverse_subset": False,
            "eval_type": None,
            "metric": ["dockq", "lddt", "rmsd", "cdr_h3_bb_rmsd", "lddt_pli"],
        },
        {
            "subset": "Antibody=True",
            "subset_label": "[antibody-protein]",
            "inverse_subset": False,
            "eval_type": ["Protein-Protein"],
            "metric": ["dockq", "lddt"],
        },
        {
            "subset": "Antibody=False",
            "subset_label": "[antibody-protein]",
            "inverse_subset": True,
            "eval_type": ["Protein-Protein"],
            "metric": ["dockq", "lddt"],
        },
        {
            "subset": "Monomer",
            "subset_label": "[protein_monomer]",
            "inverse_subset": False,
            "eval_type": ["Intra-Protein"],
            "metric": ["lddt"],
        },
        {
            "subset": "Peptide",
            "subset_label": "[peptide-interface]",
            "inverse_subset": False,
            "eval_type": ["Protein-Protein", "RNA-Protein", "DNA-Protein"],
            "metric": ["lddt"],
        },
        {
            "subset": "Cyclic-Peptide",
            "subset_label": "[cyclic_peptide-interface]",
            "inverse_subset": False,
            "eval_type": ["Protein-Protein", "RNA-Protein", "DNA-Protein"],
            "metric": ["lddt"],
        },
    ],
    "PoseBusters": [
        {
            "subset": "All",
            "subset_label": None,
            "inverse_subset": False,
            "eval_type": None,
            "metric": ["rmsd"],
        },
    ],
    "AF3-AB": [
        {
            "subset": "All",
            "subset_label": None,
            "inverse_subset": False,
            "eval_type": ["Protein-Protein"],
            "metric": ["dockq", "lddt"],
        },
    ],
    "RNA-Protein": [
        {
            "subset": "All",
            "subset_label": None,
            "inverse_subset": False,
            "eval_type": ["RNA-Protein"],
            "metric": ["lddt"],
        },
    ],
    "DNA-Protein": [
        {
            "subset": "All",
            "subset_label": None,
            "inverse_subset": False,
            "eval_type": ["DNA-Protein"],
            "metric": ["lddt"],
        },
    ],
    "Custom": [
        {
            "subset": "All",
            "subset_label": None,
            "inverse_subset": False,
            "eval_type": None,
            "metric": ["lddt", "rmsd", "dockq", "cdr_h3_bb_rmsd", "lddt_pli"],
        },
    ],
}
