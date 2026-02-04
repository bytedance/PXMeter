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
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def run_integrity_check(metrics_csv: Path, output_csv: Path = None):
    """
    Check the integrity of the seeds and samples in the metrics csv file.

    Args:
        metrics_csv: The path to the metrics csv file.
        output_csv: The path to the output csv file.
                    If None, the output will be printed to the console.
    """
    if metrics_csv.suffix == ".parquet":
        df = pd.read_parquet(metrics_csv)
    elif metrics_csv.suffix == ".csv":
        df = pd.read_csv(metrics_csv, dtype={"entry_id": str})
    else:
        raise ValueError(f"Unsupported metrics csv file suffix: {metrics_csv.suffix}")

    all_set = set(df["entry_id"])
    results = []

    tasks = []
    for seed in df["seed"].unique():
        for sample in df["sample"].unique():
            tasks.append((seed, sample))

    for seed, sample in tqdm(tasks, desc="Check seeds and samples"):
        failed = all_set - set(
            df["entry_id"][(df["seed"] == seed) & (df["sample"] == sample)]
        )
        if failed:
            results.append((seed, sample, ";".join(failed)))

    if results:
        df = pd.DataFrame(results, columns=["seed", "sample", "failed"])
        if output_csv is None:
            for seed, sample, failed in results:
                logging.info("Seed %s, sample %s, failed: %s", seed, sample, failed)
        else:
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_csv, index=False)
    else:
        logging.info("All Seeds are complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--metrics_csv", type=Path)
    parser.add_argument("-o", "--output_csv", type=Path, default=None)
    args = parser.parse_args()

    run_integrity_check(args.metrics_csv, args.output_csv)
