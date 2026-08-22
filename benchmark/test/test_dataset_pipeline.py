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

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from benchmark.dataset_pipeline import step2_make_lowh_file


class TestStep2MakeLowhFile(unittest.TestCase):
    def test_mmseqs_short_sequence_keeps_all_exact_matches(self):
        db_df = pd.DataFrame(
            [
                {"seq": "AAA", "entry_id": "A", "entity_id": "1"},
                {"seq": "AAA", "entry_id": "B", "entity_id": "2"},
            ]
        )
        query_df = pd.DataFrame(
            [{"seq": "AAA", "entry_id": "Q", "entity_id": "1"}]
        )

        def create_empty_mmseqs_output(command, **_kwargs):
            temp_dir = Path(command.split(";", 1)[0].removeprefix("cd "))
            (temp_dir / "test_vs_train.tsv").touch()

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            step2_make_lowh_file, "SRC_DATA_DIR", Path(temp_dir)
        ), patch.object(
            step2_make_lowh_file.sp,
            "run",
            side_effect=create_empty_mmseqs_output,
        ):
            result = step2_make_lowh_file.calc_mmseqs_seq_identity(
                db_df,
                query_df,
                min_seq_length=25,
            )

        self.assertEqual(result["query_id"].tolist(), ["Q_1", "Q_1"])
        self.assertEqual(result["db_id"].tolist(), ["A_1", "B_2"])
        self.assertEqual(result["similarity"].tolist(), [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
