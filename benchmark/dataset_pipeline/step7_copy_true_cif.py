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
import os
import shutil
from pathlib import Path

import pandas as pd
from joblib import delayed, Parallel
from tqdm import tqdm

from benchmark.configs.data_config import PXM_MMCIF_DIR, SUPPORTED_DATA


def copy_true_cif(
    input_dir: Path,
    csv_w_entry_id: Path,
    output_dir: Path,
    n_cpu: int = -1,
    symlink: bool = True,
    copy_all_cif: bool = True,
):
    """
    Copy true CIF files to the output directory.

    Args:
        input_dir (Path): The directory where the true CIF files are stored.
        csv_w_entry_id (Path): The path to the CSV file containing the entry IDs.
        output_dir (Path): The directory where the true CIF files will be copied.
        n_cpu (int, optional): The number of CPUs to use for parallel copying. Defaults to -1.
    """

    # Only create directories when we actually need to write into them,
    # instead of eagerly creating them at import time via configuration.
    if copy_all_cif:
        # For copy_all_cif, the copying/symlinking utilities create the final
        # output_dir; we only need to ensure its parent exists.
        output_dir.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_w_entry_id, dtype={"entry_id": str})

    tasks = []
    for entry_id in df["entry_id"].unique():
        input_path = input_dir / f"{entry_id}.cif"
        output_path = output_dir / f"{entry_id}.cif"
        tasks.append((input_path, output_path))

    def symlink_exist_ok(src: str, dst: str) -> None:
        try:
            os.symlink(src, dst)
        except FileExistsError:
            pass

    if copy_all_cif:
        if symlink:
            symlink_exist_ok(input_dir, output_dir)
        else:
            shutil.copytree(input_dir, output_dir)

    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        if symlink:
            cp_func = symlink_exist_ok
        else:
            cp_func = shutil.copy
        _results = [
            r
            for r in tqdm(
                Parallel(n_jobs=n_cpu, return_as="generator_unordered")(
                    delayed(cp_func)(
                        src,
                        tar,
                    )
                    for src, tar in tasks
                ),
                total=len(tasks),
                desc="Symlinking ground truth CIFs",
            )
        ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--cif_dir",
        type=Path,
        help="The directory where the true CIF files are stored.",
        default=PXM_MMCIF_DIR,
    )
    parser.add_argument(
        "-e",
        "--csv_w_entry_id",
        type=Path,
        default=SUPPORTED_DATA.recentpdb_low_homology,
        help="The path to the CSV file containing the entry IDs.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default=SUPPORTED_DATA.true_dir,
        help="The directory where the true CIF files will be copied.",
    )
    parser.add_argument(
        "-n",
        "--n_cpu",
        type=int,
        default=-1,
        help="The number of CPUs to use for parallel copying. Defaults to all cores",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Symlink the true CIF files instead of copying them.",
    )
    parser.add_argument(
        "--copy_all_cif",
        action="store_true",
        help="Copy all CIF files in the input directory to the output directory.",
    )
    args = parser.parse_args()

    copy_true_cif(
        input_dir=args.cif_dir,
        csv_w_entry_id=args.csv_w_entry_id,
        output_dir=args.output_dir,
        n_cpu=args.n_cpu,
        symlink=args.symlink,
        copy_all_cif=args.copy_all_cif,
    )
