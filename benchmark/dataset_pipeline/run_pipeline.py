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

import argparse
from pathlib import Path

from benchmark.configs.data_config import (
    PXM_MMCIF_DIR,
    SRC_DATA,
    SUPPORT_DATA_DIR,
    SUPPORTED_DATA,
)
from benchmark.dataset_pipeline.step1_filter_for_recentpdb import filter_recentpdb_entry
from benchmark.dataset_pipeline.step2_make_lowh_file import make_lowh_parquet_file
from benchmark.dataset_pipeline.step3_filter_to_lowh import filter_lowh
from benchmark.dataset_pipeline.step4_make_cluster_csv import (
    get_recentpdb_lowh_cluster_csv,
)
from benchmark.dataset_pipeline.step5_find_monomer import (
    find_protein_monomer_for_recentpdb_lowh,
)
from benchmark.dataset_pipeline.step6_add_subset_to_lowh import identify_subset
from benchmark.dataset_pipeline.step7_copy_true_cif import copy_true_cif
from benchmark.dataset_pipeline.step8_get_homology_for_lowh import get_homology_for_lowh
from benchmark.dataset_pipeline.step9_analysis_dataset import run_data_analysis


def run(mmcif_dir: Path, after_date: str, before_date: str, n_cpu: int = -1):
    """
    Execute the complete dataset pipeline for processing recent PDB (Protein Data Bank) entries.
    This function orchestrates multiple steps to filter, analyze, and cluster PDB data.

    Args:
        mmcif_dir (Path): Directory path containing mmCIF files of PDB entries.
        after_date (str): The start date in "YYYY-MM-DD" format for filtering PDB entries.
        before_date (str): The end date in "YYYY-MM-DD" format for filtering PDB entries.
        n_cpu (int, optional): Number of CPU cores to use for parallel processing.
                               A value of -1 indicates using all available cores. Defaults to -1.
    """
    filter_recentpdb_entry(
        mmcif_dir,
        output_meta_csv=SRC_DATA.pdb_meta_info,
        output_chain_interface_csv=SRC_DATA.recentpdb_chain_interface_csv,
        output_seq_csv=SRC_DATA.pdb_seq_csv,
        after_date=after_date,
        before_date=before_date,
        n_cpu=n_cpu,
    )

    make_lowh_parquet_file(
        seqs_csv=SRC_DATA.pdb_seq_csv,
        lowh_parquet=SRC_DATA.test_to_train_entity_homo_parquet,
        ccd_similairty_file=SRC_DATA.ccd_to_similar_ccds,
        after_date=after_date,
        before_date=before_date,
    )

    filter_lowh(
        chain_interface_csv=SRC_DATA.recentpdb_chain_interface_csv,
        test_to_train_entity_homo_parquet=SRC_DATA.test_to_train_entity_homo_parquet,
        output_lowh_csv=SUPPORTED_DATA.recentpdb_low_homology,
        mmcif_dir=mmcif_dir,
        n_cpu=n_cpu,
    )

    get_recentpdb_lowh_cluster_csv(
        lowh_csv=SUPPORTED_DATA.recentpdb_low_homology,
        seqs_csv=SRC_DATA.pdb_seq_csv,
        output_cluster_csv=SUPPORTED_DATA.recentpdb_low_homology_cluster,
    )

    find_protein_monomer_for_recentpdb_lowh(
        mmcif_dir,
        lowh_csv=SUPPORTED_DATA.recentpdb_low_homology,
        output_csv=SRC_DATA.recentpdb_low_homology_entity_type_count,
        n_cpu=n_cpu,
    )

    identify_subset(
        lowh_csv=SUPPORTED_DATA.recentpdb_low_homology,
        sabdab_summary_file=SRC_DATA.sabdab_summary_file,
        entity_type_count_csv=SRC_DATA.recentpdb_low_homology_entity_type_count,
        mmcif_dir=mmcif_dir,
        n_cpu=n_cpu,
    )

    copy_true_cif(
        input_dir=mmcif_dir,
        csv_w_entry_id=SUPPORTED_DATA.recentpdb_low_homology,
        output_dir=SUPPORTED_DATA.true_dir,
        n_cpu=n_cpu,
        symlink=False,
        copy_all_cif=False,
    )

    get_homology_for_lowh(
        lowh_csv=SUPPORTED_DATA.recentpdb_low_homology,
        seqs_csv=SRC_DATA.pdb_seq_csv,
        after_date=after_date,
        lowh_homo_parquet=SUPPORTED_DATA.recentpdb_low_homology_entity_homo_parquet,
    )

    run_data_analysis(
        pdb_meta_info_csv=SRC_DATA.pdb_meta_info,
        lowh_csv=SUPPORTED_DATA.recentpdb_low_homology,
        cluster_csv=SUPPORTED_DATA.recentpdb_low_homology_cluster,
        output_dir=(SUPPORT_DATA_DIR / "stat_data"),
        after_date=after_date,
        before_date=before_date,
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-m", "--mmcif_dir", type=Path, default=PXM_MMCIF_DIR)
    argparser.add_argument(
        "-a", "--after_date", type=str, required=True, help="YYYY-MM-DD"
    )
    argparser.add_argument(
        "-b", "--before_date", type=str, required=True, help="YYYY-MM-DD"
    )
    argparser.add_argument("-n", "--n_cpu", type=int, default=-1)

    args = argparser.parse_args()
    run(args.mmcif_dir, args.after_date, args.before_date, args.n_cpu)
