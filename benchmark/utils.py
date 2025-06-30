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

from pathlib import Path
from typing import Any


def divide_list_into_chunks(lst: list, n: int) -> list[list]:
    """
    Divide a Sequence into n approximately equal-sized chunks.

    Args:
        lst (list[Any]): The list to be divided.
        n (int): The number of chunks to create.

    Returns:
        list[list[Any]]: A list of n chunks, where each chunk is a sublist of lst.
    """
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def int_to_letters(n: int) -> str:
    """
    Convert int to letters.
    Useful for converting chain index to label_asym_id.

    Args:
        n (int): int number
    Returns:
        str: letters. e.g. 1 -> A, 2 -> B, 27 -> AA, 28 -> AB
    """
    result = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        result = chr(65 + remainder) + result
    return result


def nested_dict_to_sorted_list(data: dict | Any) -> list | Any:
    """
    Convert a nested dictionary into a sorted list.

    This function takes a nested dictionary and converts it into a sorted list.
    If the input is a dictionary, it sorts the keys and recursively processes the values.
    If the input is not a dictionary, it returns the value directly.

    Args:
        data (dict or Any): The input data, which can be a dictionary or any other type.

    Returns:
        list or Any: The sorted list or the original value if the input is not a dictionary.
    """
    if isinstance(data, dict):
        # If the input is a dictionary, sort the keys and recursively process the values
        try:
            for i in data.keys():
                int(i)
            key_type = int
        except ValueError:
            key_type = str
        return [
            nested_dict_to_sorted_list(data[key])
            for key in sorted(data.keys(), key=key_type)
        ]
    else:
        # If the input is not a dictionary, return the value directly
        return data


def get_infer_cif_path(
    infer_output_dir: Path, model: str, entry_id: str, seed: str, sample: str
) -> Path:
    """
    Get the path to the inferred CIF file based on the model name.

    Args:
        infer_output_dir (Path): The directory where inference outputs are stored.
        model (str): The name of the model used for inference.
        entry_id (str): The identifier for the entry.
        seed (str): The seed value used in the inference process.
        sample (str): The sample identifier.

    Returns:
        Path: The path to the inferred CIF file.

    Raises:
        NotImplementedError: If the provided model name is not recognized.
    """
    if model == "protenix":
        cif_path = (
            infer_output_dir
            / entry_id
            / entry_id
            / f"seed_{seed}"
            / "predictions"
            / f"{entry_id}_sample_{sample}.cif"
        )
    elif model == "chai":
        cif_path = infer_output_dir / entry_id / seed / f"pred.model_idx_{sample}.cif"
    elif model == "boltz":
        cif_path = (
            infer_output_dir
            / entry_id
            / f"seed_{seed}"
            / f"boltz_results_{entry_id}"
            / "predictions"
            / entry_id
            / f"{entry_id}_model_{sample}.cif"
        )
    else:
        raise NotImplementedError(f"Unknown model: {model}")
    return cif_path


def get_eval_result_json_path(
    eval_result_dir: Path, entry_id: str, seed: str, sample: str
) -> Path:
    """
    Get the path to the evaluation result JSON file.

    This function constructs the path to the JSON file that contains the evaluation results
    based on the provided evaluation result directory, entry ID, seed, and sample identifier.

    Args:
        eval_result_dir (Path): The directory where evaluation results are stored.
        entry_id (str): The identifier for the entry.
        seed (str): The seed value used in the evaluation process.
        sample (str): The sample identifier.

    Returns:
        Path: The path to the evaluation result JSON file.
    """
    return eval_result_dir / entry_id / str(seed) / f"sample_{sample}_metrics.json"
