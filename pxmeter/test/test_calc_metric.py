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

import logging
import time
import unittest

from DockQ.DockQ import run_on_all_native_interfaces

from pxmeter.calc_metric import load_PDB
from pxmeter.configs.run_config import RUN_CONFIG
from pxmeter.eval import evaluate
from pxmeter.test.test_utils import TEST_DATA_DIR


class TestCalcMetric(unittest.TestCase):
    """
    Test class for calculating metrics and comparing results.
    """

    def setUp(self) -> None:
        self._start_time = time.time()
        super().setUp()

    def tearDown(self):
        elapsed_time = time.time() - self._start_time
        logging.info(f"Test {self.id()} took {elapsed_time:.6f}s")

    @staticmethod
    def _calc_dockq(ref_cif, model_cif, ref_to_model_chain_map):
        model = load_PDB(str(model_cif), small_molecule=False)
        native = load_PDB(str(ref_cif), small_molecule=False)

        native_chains = [c.id for c in native]
        model_chains = [c.id for c in model]
        valid_ref_to_model_chain_map = {}
        for k, v in ref_to_model_chain_map.items():
            if k in native_chains:
                valid_ref_to_model_chain_map[k] = v
                assert v in model_chains

        dockq_result_dict, _total_dockq = run_on_all_native_interfaces(
            model, native, chain_map=valid_ref_to_model_chain_map
        )
        return dockq_result_dict

    def test_metric(self):
        """
        Test the evaluation function with DockQ and LDDT metrics.
        """

        ref_cif = TEST_DATA_DIR / "7n0a_ref.cif"
        model_cif = TEST_DATA_DIR / "7n0a_model.cif"

        result = evaluate(
            ref_cif=ref_cif,
            model_cif=model_cif,
            run_config=RUN_CONFIG,
        )
        result_dict = result.to_json_dict()

        # Check DockQ result
        dockq_result = TestCalcMetric._calc_dockq(
            ref_cif,
            model_cif,
            ref_to_model_chain_map={
                "A": "B0",
                "B": "C0",
                "C": "A0",
            },
        )

        self.assertAlmostEqual(
            dockq_result["AB"]["DockQ"],
            result_dict["interface"]["B,C"]["dockq"],
            delta=1e-2,
        )
        self.assertAlmostEqual(
            dockq_result["AC"]["DockQ"],
            result_dict["interface"]["A,B"]["dockq"],
            delta=1e-2,
        )
        self.assertAlmostEqual(
            dockq_result["BC"]["DockQ"],
            result_dict["interface"]["A,C"]["dockq"],
            delta=1e-2,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
