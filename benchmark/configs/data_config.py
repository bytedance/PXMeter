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
from pathlib import Path

from ml_collections import ConfigDict

PXM_EVAL_DATA_ROOT_PATH = Path(
    os.environ.get(
        "PXM_EVAL_DATA_ROOT_PATH",
        "evaluation",
    )
)

SUPPORT_DATA_DIR = PXM_EVAL_DATA_ROOT_PATH / "supported_data"
SRC_DATA_DIR = PXM_EVAL_DATA_ROOT_PATH / "src_data"


_SUPPORTED_DATA_NAME = {
    "true_dir": "mmcif",
    "recentpdb_low_homology_cluster": "RecentPDB_low_homology_cluster_info.csv",
    "recentpdb_low_homology": "RecentPDB_low_homology.csv",
    "lig_info_csv": "RecentPDB_low_homology_lig_info.csv",
    "af3_ab_metadata": "af3_metadata_antibody_antigen.csv",
    "recentpdb_low_homology_entity_homo_parquet": "RecentPDB_low_homology_entity_homo.parquet",
}

_SRC_DATA_NAME = {
    "pdb_meta_info": "pdb_meta_info.csv",
    "pdb_seq_csv": "pdb_seqs.csv",
    "recentpdb_chain_interface_csv": "RecentPDB_chain_interface.csv",
    "recentpdb_low_homology_entity_type_count": "RecentPDB_low_homology_entity_types_count.csv",
    "ccd_to_similar_ccds": "ccd_to_similar_ccds.json",
    "test_to_train_entity_homo_parquet": "test_to_train_entity_homo.parquet",
    "sabdab_summary_file": "sabdab_summary_all.tsv",
}

SUPPORTED_DATA = ConfigDict(
    {k: SUPPORT_DATA_DIR / v for k, v in _SUPPORTED_DATA_NAME.items()}
)


SRC_DATA = ConfigDict({k: SRC_DATA_DIR / v for k, v in _SRC_DATA_NAME.items()})


PXM_MMCIF_DIR = Path(os.environ.get("PXM_MMCIF_DIR", SUPPORTED_DATA.true_dir))
