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
import os

from biotite.structure.info import set_ccd_path
from ml_collections.config_dict import ConfigDict

RUN_CONFIG = ConfigDict(
    {
        "mapping": {
            "mapping_polymer": True,
            "mapping_ligand": True,
            "res_id_alignments": True,
            "enumerate_all_anchors": True,
            "auto_fix_model_entities": True,
        },
        "metric": {
            "calc_clashes": True,
            "calc_lddt": True,
            "calc_dockq": True,
            "calc_rmsd": True,
            "calc_pb_valid": True,
            "calc_cdr_h3_bb_rmsd": False,
            "penalize_unmapped_reference_atoms": False,
            "lddt": {
                "eps": 1e-6,
                "nucleotide_threshold": 30.0,
                "non_nucleotide_threshold": 15.0,
                "stereochecks": False,
                "calc_backbone_lddt": True,
                "calc_lddt_pli": True,
            },
            "dockq": {
                "exclude_hetatms": True,
            },
            "clashes": {
                "vdw_scale_factor": 0.5,
            },
        },
    }
)


# Set the path to the CCD file from the environment variable.
pxm_ccd_file = os.environ.get("PXM_CCD_FILE")
if pxm_ccd_file:
    logging.info("Using PXM_CCD_FILE: %s", pxm_ccd_file)
    set_ccd_path(pxm_ccd_file)


def set_cfg(path: list[str], value):
    """
    Set the value of a config item.

    Args:
        path: A list of keys to the config item.
        value: The value to set.

    Example:
        set_cfg(["metric", "calc_dockq"], False)
    """
    node = RUN_CONFIG
    for key in path[:-1]:
        node = node.get(key)
        assert node is not None, f"Config item {key} not found ({'.'.join(path)}). "
    node[path[-1]] = value


def _cast_value_like(old_value, raw: str):
    """
    Cast a raw string to the same type as an existing config value.

    This helper is used to convert command-line string values into the
    appropriate Python types based on the type of the corresponding
    value in ``RUN_CONFIG``.

    Casting rules:
      * bool:
          - Accepted truthy values (case-insensitive): "1", "true",
            "yes", "y", "on"
          - Accepted falsy values (case-insensitive): "0", "false",
            "no", "n", "off"
          - Any other value will raise a ValueError.
      * int (but not bool): cast via ``int(raw)``.
      * float: cast via ``float(raw)``.
      * All other types: returned as the original string.

    Args:
        old_value: The existing value in the config whose type should be
            used as a reference for casting.
        raw: The raw string value provided from the command line.

    Returns:
        The converted value with a type consistent with ``old_value``,
        or the original string if no specific casting rule applies.

    Raises:
        ValueError: If the value is expected to be boolean but cannot be
            interpreted as a valid boolean literal.
    """
    # Boolean: accept several common textual representations.
    if isinstance(old_value, bool):
        v = raw.strip().lower()
        if v in ("1", "true", "yes", "y", "on"):
            return True
        if v in ("0", "false", "no", "n", "off"):
            return False
        raise ValueError(f"Cannot parse boolean from {raw!r}")

    # Integer (excluding bool which is a subclass of int).
    if isinstance(old_value, int) and not isinstance(old_value, bool):
        return int(raw)

    # Floating point.
    if isinstance(old_value, float):
        return float(raw)

    # Fallback: keep as string for all other types.
    return raw


def apply_run_config_overrides(overrides: list[str]) -> None:
    """
    Apply command-line overrides to the global RUN_CONFIG.

    Each override must follow the format:
        key1.key2.key3=VALUE

    The dotted path must correspond to an existing hierarchy in RUN_CONFIG.
    The VALUE part will be automatically cast based on the original value's type:
        - bool   → accepts: true/false/1/0/yes/no/on/off (case-insensitive)
        - int    → cast to integer
        - float  → cast to float
        - others → kept as string

    Examples:
        -C metric.lddt.eps=1e-5
        -C mapping.mapping_ligand=false

    Args:
        overrides (list[str]): A list of key-value override expressions
        provided via the command line.
    """
    for item in overrides:
        # Validate format: must contain "="
        if "=" not in item:
            raise ValueError(
                f"Invalid config override {item!r}. Expected: key1.key2=VALUE"
            )

        key_path, raw_value = item.split("=", 1)
        path = key_path.split(".")

        # Traverse RUN_CONFIG to locate the existing value for type casting
        node = RUN_CONFIG
        for key in path[:-1]:
            node = node.get(key)
            if node is None:
                raise ValueError(
                    f"Unknown config path segment {key!r} in override {item!r}"
                )

        leaf_key = path[-1]
        if leaf_key not in node:
            raise ValueError(f"Unknown config key {leaf_key!r} in override {item!r}")

        old_value = node[leaf_key]

        # Type casting
        try:
            new_value = _cast_value_like(old_value, raw_value)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f"Failed to parse value {raw_value!r} for key {key_path!r}: {exc}"
            ) from exc

        # Apply the assignment using the existing helper
        set_cfg(path, new_value)
