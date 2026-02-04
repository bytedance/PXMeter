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

RANKER_KEYS = {
    "complex": [
        ("plddt", False),
        ("gpde", True),
        ("ranking_score", False),
        ("ptm", False),
        ("iptm", False),
        ("has_clash", False),
        ("disorder", False),
    ],
    "chain": [
        ("chain_ptm", False),
        ("chain_plddt", False),
        ("chain_iptm", False),
        ("chain_gpde", True),
    ],
    "interface": [
        ("chain_pair_iptm", False),
        ("chain_pair_plddt", False),
        ("chain_pair_iptm_global", False),
        ("chain_pair_gpde", True),
    ],
}
