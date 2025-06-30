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

import os

from ml_collections import ConfigDict

PXM_EVAL_DATA_ROOT_PATH = os.environ.get("PXM_EVAL_DATA_ROOT_PATH", "evaluation")

_TMP_EVAL_RESULTS = {
    "boltz-1": {
        "model": "boltz",
        "seeds": [101, 102, 103, 104, 105],
        "dataset_path": {
            "RecentPDB": "eval_results/boltz-1/RecentPDB",
            "PoseBusters": "eval_results/boltz-1/PoseBusters",
            "AF3-AB": "eval_results/boltz-1/AF3_AB",
        },
    },
    "chai-1": {
        "model": "chai",
        "seeds": [101, 102, 103, 104, 105],
        "dataset_path": {
            "RecentPDB": "eval_results/chai-1/RecentPDB",
            "PoseBusters": "eval_results/chai-1/PoseBusters",
            "AF3-AB": "eval_results/chai-1/AF3_AB",
        },
    },
    "protenix": {
        "model": "protenix",
        "seeds": [101, 102, 103, 104, 105],
        "dataset_path": {
            "RecentPDB": "eval_results/protenix/RecentPDB",
            "PoseBusters": "eval_results/protenix/PoseBusters",
            "AF3-AB": "eval_results/protenix/AF3_AB",
            "RecentPDB-NA": "eval_results/protenix/RecentPDB_NA",
        },
    },
}

_TMP_SUPPORTED_DATA = {
    "true_dir": "supported_data/mmcif",
    "pb_true_dir": "supported_data/posebusters_mmcif",
    "pb_info_csv": "supported_data/posebusters_lig_info.csv",
    "pdb_cluster_file": "supported_data/clusters-by-entity-40.txt",
    "recentpdb_low_homology_cluster": "supported_data/RecentPDB_low_homology_cluster_info.csv",
    "recentpdb_low_homology": "supported_data/RecentPDB_protein_low_homology.csv",
    "recentpdb_low_homology_entity_type_count": "supported_data/RecentPDB_low_homology_entity_types_count.csv",
    "recentpdb_na_cluster": "supported_data/RecentPDB_NA_cluster_info.csv",
    "recentpdb_na_low_homology": "supported_data/RecentPDB_NA_low_homology.csv",
    "af3_ab_metadata": "supported_data/af3_metadata_antibody_antigen.csv",
}

_TMP_SRC_DATA = {
    "pdb_meta_info": "supported_data/pdb_meta_info.csv",
    "recentpdb_chain_interface_csv": "supported_data/RecentPDB_chain_interface.csv",
    "test_to_train_entity_homo_json": "supported_data/test_to_train_entity_homo.json",
}

# Add PXM_EVAL_DATA_ROOT_PATH to the dataset_path
EVAL_RESULTS = {}
for k, v in _TMP_EVAL_RESULTS.items():
    EVAL_RESULTS[k] = {
        "model": v["model"],
        "seeds": v["seeds"],
        "dataset_path": {
            k: os.path.join(PXM_EVAL_DATA_ROOT_PATH, v["dataset_path"][k])
            for k in v["dataset_path"]
        },
    }

SUPPORTED_DATA = ConfigDict(
    {
        k: os.path.join(PXM_EVAL_DATA_ROOT_PATH, v)
        for k, v in _TMP_SUPPORTED_DATA.items()
    }
)


SRC_DATA = ConfigDict(
    {k: os.path.join(PXM_EVAL_DATA_ROOT_PATH, v) for k, v in _TMP_SRC_DATA.items()}
)
