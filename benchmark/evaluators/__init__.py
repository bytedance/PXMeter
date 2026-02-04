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

import importlib
import pkgutil
from pathlib import Path

from .base import BaseEvaluator

MODEL_TO_EVALUATOR = {}
MODEL_TO_RANKER_KEYS = {
    "nan": {
        "complex": [],
        "chain": [],
        "interface": [],
    },
}

# Dynamically search for all sub-packages in the current directory
package_dir = str(Path(__file__).parent)
for _, name, is_pkg in pkgutil.iter_modules([package_dir]):
    if not is_pkg:
        continue

    # 1. Try to load RANKER_KEYS from {model}/config.py
    try:
        config_module = importlib.import_module(f".{name}.config", package=__name__)
        if hasattr(config_module, "RANKER_KEYS"):
            MODEL_TO_RANKER_KEYS[name] = config_module.RANKER_KEYS
    except ImportError:
        pass

    # 2. Try to load Evaluator class from {model}/evaluator.py
    try:
        evaluator_module = importlib.import_module(
            f".{name}.evaluator", package=__name__
        )
        for attr_name in dir(evaluator_module):
            attr = getattr(evaluator_module, attr_name)
            # Find classes that inherit from BaseEvaluator
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseEvaluator)
                and attr is not BaseEvaluator
            ):
                MODEL_TO_EVALUATOR[name] = attr
                break
    except ImportError:
        pass

__all__ = [
    "BaseEvaluator",
    "MODEL_TO_EVALUATOR",
    "MODEL_TO_RANKER_KEYS",
]
