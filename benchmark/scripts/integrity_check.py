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
import itertools
import logging
from pathlib import Path

import pandas as pd


def run_integrity_check(metrics_file: Path, output_csv: Path = None):
    """
    Check the integrity of the seeds and samples in the metrics file.

    Args:
        metrics_file: The path to the metrics file.
        output_csv: The path to the output csv file.
                    If None, the output will be printed to the console.
    """
    if metrics_file.suffix == ".parquet":
        df = pd.read_parquet(metrics_file)
    elif metrics_file.suffix == ".csv":
        df = pd.read_csv(metrics_file, dtype={"entry_id": str})
    else:
        raise ValueError(f"Unsupported metrics csv file suffix: {metrics_file.suffix}")

    all_set = set(df["entry_id"])
    all_seeds = df["seed"].unique()
    all_samples = df["sample"].unique()

    logging.info("# Unique seeds: %s", len(all_seeds))
    logging.info("# Unique samples: %s", len(all_samples))

    grouped = df.groupby(["seed", "sample"])["entry_id"].apply(set)

    results = []
    for seed, sample in itertools.product(all_seeds, all_samples):
        present_entries = grouped.get((seed, sample), set())
        failed = all_set - present_entries

        if failed:
            results.append((seed, sample, ";".join(failed)))

    if results:
        df_results = pd.DataFrame(results, columns=["seed", "sample", "failed"])
        if output_csv is None:
            for seed, sample, failed in results:
                logging.info("Seed %s, sample %s, failed: %s", seed, sample, failed)
        else:
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            df_results.to_csv(output_csv, index=False)
    else:
        logging.info("All Seeds are complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--metrics_file", type=Path)
    parser.add_argument("-o", "--output_csv", type=Path, default=None)
    args = parser.parse_args()

    run_integrity_check(args.metrics_file, args.output_csv)
