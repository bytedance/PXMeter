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

import time
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.configs.data_config import SUPPORTED_DATA
from benchmark.show_intersection_results import (
    get_low_homology_subset,
    get_recent_pdb_results_df,
)
from pxmeter.constants import PROTEIN

TEST_DATA_DIR = Path(__file__).absolute().parent / "files"


class TestCalcResults(unittest.TestCase):
    """
    Test class for calculate results.
    """

    def setUp(self) -> None:
        self._start_time = time.time()
        super().setUp()

    def tearDown(self):
        elapsed_time = time.time() - self._start_time
        print(f"Test {self.id()} took {elapsed_time:.6f}s")

    def test_calc_results(self):
        """
        Test the calculation of DockQ scores for protein-protein interfaces.

        This test method loads sample metrics data, filters it to get low homology
        protein-protein interfaces, calculates the best and median DockQ scores per cluster,
        and then compares the results with those obtained from `get_recent_pdb_results_df`.
        """
        metric_csv = TEST_DATA_DIR / "RecentPDB_metrics_sample50.csv"
        df = pd.read_csv(
            metric_csv,
            dtype={
                "entry_id": str,
                "entity_id_1": str,
                "entity_id_2": str,
                "seed": str,
                "sample": str,
            },
            low_memory=False,
        )
        lowh_df = get_low_homology_subset(df, SUPPORTED_DATA.recentpdb_low_homology)

        prot_prot_df = lowh_df[
            (lowh_df["type"] == "interface")
            & (lowh_df["entity_type_1"] == PROTEIN)
            & (lowh_df["entity_type_2"] == PROTEIN)
        ].copy()

        prot_prot_df.dropna(subset=["dockq"], inplace=True, how="all", axis=0)

        cluster_id_to_best_scores = defaultdict(list)
        cluster_id_to_median_scores = defaultdict(list)
        group_by_key = ["entry_id", "chain_id_1", "chain_id_2"]
        for _group_id, group_df in prot_prot_df.groupby(by=group_by_key):
            cluster_id = group_df["cluster_id"].iloc[0]
            best = group_df.loc[group_df["dockq"].idxmax()]["dockq"]
            median = group_df.loc[
                (group_df["dockq"] - group_df["dockq"].median()).abs().idxmin()
            ]["dockq"]

            cluster_id_to_best_scores[cluster_id].append(best)
            cluster_id_to_median_scores[cluster_id].append(median)

        cluster_id_to_mean_best = {}
        for cluster_id, scores in cluster_id_to_best_scores.items():
            avg_dockq_sr = np.mean(np.array(scores) > 0.23)
            cluster_id_to_mean_best[cluster_id] = avg_dockq_sr

        cluster_id_to_mean_median = {}
        for cluster_id, scores in cluster_id_to_median_scores.items():
            avg_dockq_sr = np.mean(np.array(scores) > 0.23)
            cluster_id_to_mean_median[cluster_id] = avg_dockq_sr

        final_mean_best_score = np.mean(list(cluster_id_to_mean_best.values()))
        final_mean_median_score = np.mean(list(cluster_id_to_mean_median.values()))

        dockq_results_df, _lddt_results_df, _dockq_details_df, _lddt_details_df = (
            get_recent_pdb_results_df(df)
        )

        self.assertAlmostEqual(
            dockq_results_df[
                (dockq_results_df["eval_type"] == "Protein-Protein")
                & (dockq_results_df["ranker"] == "best")
            ].iloc[0]["avg_dockq_sr_avg_sr"],
            final_mean_best_score,
            delta=1e-6,
        )

        self.assertAlmostEqual(
            dockq_results_df[
                (dockq_results_df["eval_type"] == "Protein-Protein")
                & (dockq_results_df["ranker"] == "median")
            ].iloc[0]["avg_dockq_sr_avg_sr"],
            final_mean_median_score,
            delta=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
