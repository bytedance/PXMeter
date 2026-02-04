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
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from benchmark.configs.data_config import PXM_MMCIF_DIR
from benchmark.utils import get_eval_result_json_path, get_infer_cif_path


def extract_cases(
    df: pd.DataFrame,
    infer_result_dir: Path,
    eval_result_dir: Path,
    model: str,
    output_dir: Path,
    gt_dir: Path = PXM_MMCIF_DIR,
    sample_num: int = -1,
    name: str | None = None,
    seed_col: str | None = "seed",
    sample_col: str | None = "sample",
):
    """
    Extract evaluation cases and organize them into a directory structure.

    This function reads a CSV file containing case details, optionally
    subsamples a fixed number of rows, and then for each ``entry_id``:

    * Creates an ``entry_id`` directory under ``output_dir``.
    * Symlinks the ground-truth mmCIF file into ``{output_dir}/{entry_id}/{entry_id}.cif``.
    * For each unique pair of (``chain_id_1``, ``chain_id_2``) within the entry:
      - Creates a case-specific subdirectory, whose name depends on ``type``:
        * ``type == "chain"``      → ``{entry_id}_{chain_id_1}``
        * ``type == "interface"``  → ``{entry_id}_{chain_id_1}_{chain_id_2}``
        * other (e.g. ``"complex"``) → ``{entry_id}``
      - Symlinks the inferred structure CIF and evaluation result JSON into
        the case directory as ``{entry_id}.cif`` and ``{entry_id}.json``.

    The input CSV is expected to contain at least the following columns:
    ``entry_id``, ``chain_id_1``, ``chain_id_2``, ``type``, ``seed``,
    and ``sample``.

    Args:
        df: DataFrame containing case details.
        infer_result_dir: Root directory that stores model inference results.
        eval_result_dir: Root directory that stores evaluation result JSON files.
        model: Model name used to resolve inference result paths.
        output_dir: Root directory where extracted cases and symlinks are written.
        gt_dir: Directory containing ground-truth mmCIF files, named
            as ``{entry_id}.cif``. Defaults to ``PXM_MMCIF_DIR``.
        sample_num: If > 0, randomly subsample this many rows from the CSV
            (with ``random_state=42``). If <= 0, use all rows.
        name: If not None, filter the DataFrame to include only rows
            where ``name == name``. Defaults to None.
        seed_col: Column name for seed values. Defaults to "seed".
        sample_col: Column name for sample values. Defaults to "sample".
    """
    if name and "name" in df.columns:
        df = df[df["name"] == name]

    if sample_num > 0:
        df = df.sample(n=min(sample_num, len(df)), random_state=42)

    for entry_id, sub_df in tqdm(
        df.groupby("entry_id", observed=True), total=len(df["entry_id"].unique())
    ):
        gt_cif = gt_dir / f"{entry_id}.cif"

        entry_output_dir = output_dir / entry_id
        entry_output_dir.mkdir(parents=True, exist_ok=True)

        output_gt_cif = entry_output_dir / f"{entry_id}.cif"
        if output_gt_cif.is_symlink() or output_gt_cif.exists():
            output_gt_cif.unlink()
        os.symlink(gt_cif, output_gt_cif)

        sub_df = sub_df.drop_duplicates(subset=["chain_id_1", "chain_id_2"])

        for _, row in sub_df.iterrows():
            dir_name = entry_id
            if not pd.isna(row["chain_id_1"]):
                dir_name += f"_{row['chain_id_1']}"

            if not pd.isna(row["chain_id_2"]):
                dir_name += f"_{row['chain_id_2']}"

            if not name:
                name = model

            case_output_dir = entry_output_dir / name / dir_name
            case_output_dir.mkdir(parents=True, exist_ok=True)

            infer_cif = get_infer_cif_path(
                infer_result_dir,
                model=model,
                entry_id=entry_id,
                seed=row[seed_col],
                sample=row[sample_col],
            )
            eval_result_json = get_eval_result_json_path(
                eval_result_dir,
                entry_id=entry_id,
                seed=row[seed_col],
                sample=row[sample_col],
            )

            output_cif = (
                case_output_dir
                / f"{model}_seed{row[seed_col]}_sample{row[sample_col]}.cif"
            )
            output_json = (
                case_output_dir
                / f"{model}_seed{row[seed_col]}_sample{row[sample_col]}.json"
            )

            if output_cif.is_symlink() or output_cif.exists():
                output_cif.unlink()
            os.symlink(infer_cif, output_cif)

            if output_json.is_symlink() or output_json.exists():
                output_json.unlink()
            os.symlink(eval_result_json, output_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--details_csv", type=Path, required=True)
    parser.add_argument("-i", "--infer_result_dir", type=Path, required=True)
    parser.add_argument("-e", "--eval_result_dir", type=Path, required=True)
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=Path, required=True)
    parser.add_argument("-g", "--gt_dir", type=Path, default=PXM_MMCIF_DIR)
    parser.add_argument("--seed_col", type=str, default="seed")
    parser.add_argument("--sample_col", type=str, default="sample")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("-s", "--sample_num", type=int, default=-1)
    args = parser.parse_args()

    df = pd.read_csv(args.details_csv)

    extract_cases(
        df=df,
        infer_result_dir=args.infer_result_dir,
        eval_result_dir=args.eval_result_dir,
        model=args.model,
        output_dir=args.output_dir,
        gt_dir=args.gt_dir,
        name=args.name,
        sample_num=args.sample_num,
        seed_col=args.seed_col,
        sample_col=args.sample_col,
    )
