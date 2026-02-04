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

import unittest

import numpy as np
import pandas as pd

from benchmark.utils import (
    divide_list_into_chunks,
    int_to_letters,
    nested_dict_to_sorted_list,
    shrink_dataframe,
)


class TestUtils(unittest.TestCase):
    """
    Test class for benchmark.utils functions.
    """

    def test_int_to_letters(self):
        """Test conversion of integers to Excel-style letters."""
        self.assertEqual(int_to_letters(1), "A")
        self.assertEqual(int_to_letters(26), "Z")
        self.assertEqual(int_to_letters(27), "AA")
        self.assertEqual(int_to_letters(52), "AZ")
        self.assertEqual(int_to_letters(53), "BA")
        self.assertEqual(int_to_letters(702), "ZZ")
        self.assertEqual(int_to_letters(703), "AAA")

    def test_divide_list_into_chunks(self):
        """Test dividing a list into approximately equal-sized chunks."""
        lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # Equal division
        chunks = divide_list_into_chunks(lst, 2)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0], [1, 2, 3, 4, 5])
        self.assertEqual(chunks[1], [6, 7, 8, 9, 10])

        # Unequal division
        chunks = divide_list_into_chunks(lst, 3)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(len(chunks[0]), 4)  # 10 // 3 = 3, remainder 1
        self.assertEqual(len(chunks[1]), 3)
        self.assertEqual(len(chunks[2]), 3)

        # Empty list
        chunks = divide_list_into_chunks([], 3)
        self.assertEqual(chunks, [[], [], []])

    def test_nested_dict_to_sorted_list(self):
        """Test converting a nested dictionary into a sorted list."""
        # Simple dict with integer keys
        data = {"2": "B", "1": "A", "10": "C"}
        result = nested_dict_to_sorted_list(data)
        self.assertEqual(result, ["A", "B", "C"])

        # Nested dict
        data = {"2": {"2": "B2", "1": "B1"}, "1": {"1": "A1", "2": "A2"}}
        result = nested_dict_to_sorted_list(data)
        self.assertEqual(result, [["A1", "A2"], ["B1", "B2"]])

        # Non-integer keys
        data = {"b": "val_b", "a": "val_a"}
        result = nested_dict_to_sorted_list(data)
        self.assertEqual(result, ["val_a", "val_b"])

        # Non-dict input
        self.assertEqual(nested_dict_to_sorted_list(123), 123)
        self.assertEqual(nested_dict_to_sorted_list([1, 2]), [1, 2])

    def test_shrink_dataframe_bool(self):
        """Test boolean casting for 0/1 values."""
        df = pd.DataFrame(
            {
                "bool_col": [0, 1, 0, 1],
                # If it's already float, bool_cast skips it by design to avoid false positives.
                # We use dtype=object to simulate data read from CSV that could be boolean.
                "bool_nan_obj": pd.Series([0, 1, np.nan, 1], dtype=object),
            }
        )
        # Note: columns with NaNs and only 0/1 are object or float by default in pandas
        # shrink_dataframe should convert them to bool or boolean
        shrunk_df, _ = shrink_dataframe(df, bool_cast=True)

        self.assertEqual(shrunk_df["bool_col"].dtype, bool)
        self.assertEqual(shrunk_df["bool_nan_obj"].dtype, "boolean")

    def test_shrink_dataframe_numeric(self):
        """Test downcasting for float and integer types."""
        df = pd.DataFrame(
            {"float_col": [1.1, 2.2, 3.3], "int_col": [1, 2, 100]}  # float64  # int64
        )
        shrunk_df, _ = shrink_dataframe(df, downcast_float=True, downcast_int=True)

        self.assertEqual(shrunk_df["float_col"].dtype, np.float32)
        # downcast="integer" usually picks the smallest signed int
        self.assertEqual(shrunk_df["int_col"].dtype, np.int8)

    def test_shrink_dataframe_nullable_int_strict(self):
        """
        Test the newly implemented strict nullable integer conversion.
        Only object columns where all non-null values are integers should convert.
        """
        df = pd.DataFrame(
            {
                "pure_int_str": ["1", "2", "3", np.nan],  # Should convert to Int8
                "mixed_str": [
                    "1",
                    "2.5",
                    np.nan,
                    "4",
                ],  # Should NOT convert (has float-like)
                "garbage_str": [
                    "1",
                    "abc",
                    np.nan,
                    "2",
                ],  # Should NOT convert (has non-numeric)
                "large_int_str": ["1", "40000", np.nan, "2"],  # Should convert to Int32
            }
        )
        shrunk_df, _ = shrink_dataframe(df, use_nullable_int=True)

        self.assertEqual(shrunk_df["pure_int_str"].dtype, "Int8")
        # mixed_str and garbage_str should NOT become Int types.
        self.assertFalse(str(shrunk_df["mixed_str"].dtype).startswith("Int"))
        self.assertFalse(str(shrunk_df["garbage_str"].dtype).startswith("Int"))
        self.assertEqual(shrunk_df["large_int_str"].dtype, "Int32")

    def test_shrink_dataframe_category(self):
        """Test conversion of low-cardinality object columns to category."""
        df = pd.DataFrame(
            {
                "cat_col": ["A", "B", "A", "B"] * 100,  # Low cardinality
                "str_col": [str(i) for i in range(400)],  # High cardinality
            }
        )
        shrunk_df, _ = shrink_dataframe(df, cat_threshold=10)

        self.assertIsInstance(shrunk_df["cat_col"].dtype, pd.CategoricalDtype)
        # str_col cardinality is high, so it shouldn't be category
        self.assertNotIsInstance(shrunk_df["str_col"].dtype, pd.CategoricalDtype)

    def test_shrink_dataframe_exclude(self):
        """Test that excluded columns are not modified."""
        df = pd.DataFrame({"float_col": [1.1, 2.2, 3.3], "exclude_me": [1.1, 2.2, 3.3]})
        shrunk_df, _ = shrink_dataframe(df, downcast_float=True, exclude=["exclude_me"])

        self.assertEqual(shrunk_df["float_col"].dtype, np.float32)
        self.assertEqual(shrunk_df["exclude_me"].dtype, np.float64)


if __name__ == "__main__":
    unittest.main()
