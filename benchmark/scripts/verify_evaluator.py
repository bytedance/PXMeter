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
import logging
import random
import shutil
import tempfile
from pathlib import Path

from benchmark.configs.data_config import PXM_MMCIF_DIR
from benchmark.evaluators import MODEL_TO_EVALUATOR


def verify_model_evaluator(
    model_name: str, input_dir: Path, true_dir: Path, limit: int = 2
):
    """
    Verifies the implementation of a specific model evaluator.

    This function checks if the model is registered, attempts to instantiate its evaluator,
    scans for tasks in the input directory, and runs a limited number of evaluation trials
    to ensure the end-to-end logic works correctly.

    Args:
        model_name (str): The name of the model to verify.
        input_dir (Path): The directory containing predicted structures.
        true_dir (Path): The directory containing ground truth structures.
        limit (int, optional): The maximum number of tasks to run for verification. Defaults to 2.

    Returns:
        bool: True if verification succeeds for all samples up to the limit, False otherwise.
    """
    logging.info("Starting verification for model: %s", model_name)

    # 1. Check if model is registered
    if model_name not in MODEL_TO_EVALUATOR:
        logging.error("Model '%s' not found in MODEL_TO_EVALUATOR.", model_name)
        logging.info("Available models: %s", list(MODEL_TO_EVALUATOR.keys()))
        return False

    evaluator_cls = MODEL_TO_EVALUATOR[model_name]
    logging.info("Found evaluator class: %s", evaluator_cls.__name__)

    # 2. Setup temporary output directory
    tmp_output = Path(tempfile.mkdtemp(prefix="pxm_verify_"))
    logging.info("Using temporary output directory: %s", tmp_output)

    # 3. Quick sample of subdirectories to avoid full scan of input_dir
    logging.info("Sampling subdirectories for quick verification...")
    sampled_pdb_ids = []
    # We sample a bit more than the limit in case some directories are invalid for the model
    for pdb_dir in input_dir.iterdir():
        if pdb_dir.is_dir():
            sampled_pdb_ids.append(pdb_dir.name)
        if len(sampled_pdb_ids) >= limit * 3:
            break

    if not sampled_pdb_ids:
        logging.error("No subdirectories found in input_dir: %s", input_dir)
        return False

    try:
        # 4. Instantiate evaluator with the restricted PDB list
        evaluator = evaluator_cls(
            true_dir=true_dir,
            pred_dir=input_dir,
            output_dir=tmp_output,
            num_cpu=1,
            overwrite=True,
            pdb_ids_list=sampled_pdb_ids,
        )

        # 5. Try to load tasks
        logging.info("Scanning for CIF and confidence files in sampled directories...")
        all_tasks = evaluator.load_all_cif_and_confidence()

        if not all_tasks:
            logging.error(
                "No tasks found! Please check your input_dir and _get_info_from_each_pdb_dir implementation."
            )
            return False

        num_to_run = min(len(all_tasks), limit)
        # Randomly sample tasks to run
        tasks_to_run = random.sample(all_tasks, num_to_run)
        logging.info(
            "Found %d tasks. Will attempt to run a random selection of %d tasks.",
            len(all_tasks),
            num_to_run,
        )

        # 5. Run evaluation for a few samples
        for i, task in enumerate(tasks_to_run):
            _name, pdb_id, seed, sample = task[:4]
            logging.info(
                "[%d/%d] Testing evaluation for %s (seed=%s, sample=%s)...",
                i + 1,
                num_to_run,
                pdb_id,
                seed,
                sample,
            )
            try:
                evaluator.run_eval(task)
                logging.info("Successfully evaluated %s.", pdb_id)
            except Exception as e:
                logging.error("Failed to evaluate %s: %s", pdb_id, e, exc_info=True)
                return False

        logging.info("Verification completed successfully!")
        return True

    finally:
        # Cleanup
        if tmp_output.exists():
            shutil.rmtree(tmp_output)
            logging.info("Cleaned up temporary output directory.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Verify a model evaluator implementation."
    )
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Model name to verify."
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing prediction results.",
    )
    parser.add_argument(
        "-t",
        "--true_dir",
        type=Path,
        default=PXM_MMCIF_DIR,
        help="Directory containing reference (true) mmCIF files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2,
        help="Maximum number of samples to test (default: 2).",
    )

    args = parser.parse_args()

    success = verify_model_evaluator(
        model_name=args.model,
        input_dir=args.input_dir,
        true_dir=args.true_dir,
        limit=args.limit,
    )

    if not success:
        exit(1)
