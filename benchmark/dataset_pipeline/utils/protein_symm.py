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

import pandas as pd
import requests
from tqdm import tqdm

ASYMMETRIC = ["C1"]
CYCLIC = [
    "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11",
    "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C21", "C22",
    "C23", "C24", "C26", "C27", "C30", "C31", "C32", "C33", "C34",
    "C35", "C38", "C39", "C52"
]  # fmt:skip
DIHEDRAL = [
    "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11",
    "D12", "D14", "D16", "D17", "D18", "D26", "D39", "D44", "D48"
]  # fmt:skip

HELICAL = ["H"]
ICOSAHEDRAL = ["I"]
OCTAHEDRAL = ["O"]
TETRAHEDRAL = ["T"]


SYMMETRY_LABELS = (
    ASYMMETRIC + CYCLIC + DIHEDRAL + HELICAL + ICOSAHEDRAL + OCTAHEDRAL + TETRAHEDRAL
)


SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
DATA_REST_BASE = "https://data.rcsb.org/rest/v1/core"


QUERY_TEMPLATE = {
    "query": {
        "type": "group",
        "logical_operator": "and",
        "nodes": [
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_struct_symmetry.kind",
                    "operator": "exact_match",
                    "value": "Global Symmetry",
                },
            },
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_struct_symmetry.symbol",
                    "operator": "exact_match",
                    "value": "C1",
                },
            },
        ],
    },
    "return_type": "assembly",
    "request_options": {"return_all_hits": True},
}


def get_global_symmetry_by_rcsb(output_csv=None) -> pd.DataFrame:
    """
    Query protein assemblies from RCSB PDB annotated with specific global symmetry
    labels (e.g., Cn, Dn, T, O, I, H, C1), and return the results as a DataFrame.

    This function iterates over a predefined set of symmetry labels, sends REST API
    queries to the RCSB Search service, and collects all assemblies matching the
    given global symmetry. Each record includes the symmetry type, the PDB entry ID,
    and the assembly ID. Optionally, the results can be exported to a CSV file.

    Args:
        output_csv (str or Path, optional):
            Path to save the query results as a CSV file. If None, the results
            are not written to disk. Default is None.

    Returns:
        pd.DataFrame:
            A DataFrame with the following columns:
                - protein_global_symmetry (str): The global symmetry label (e.g., "C2", "D3").
                - entry_id (str): The PDB entry ID (lowercase, e.g., "4hhb").
                - assembly_id (str): The assembly identifier within the entry (e.g., "1").
    """
    results = []
    for label in tqdm(
        SYMMETRY_LABELS,
        total=len(SYMMETRY_LABELS),
        desc="Querying protein symmetry labels from RCSB PDB",
    ):
        query = QUERY_TEMPLATE.copy()
        query["query"]["nodes"][1]["parameters"]["value"] = label

        resp = requests.post(SEARCH_URL, json=query, timeout=60)
        resp.raise_for_status()
        for r in resp.json().get("result_set", []):
            hit = r["identifier"].split("-")  # e.g. "4HHB-1"
            entry_id = hit[0].lower()
            assembly_id = hit[1]
            results.append([label, entry_id, assembly_id])

        time.sleep(0.05)

    df = pd.DataFrame(
        results, columns=["protein_global_symmetry", "entry_id", "assembly_id"]
    )
    if output_csv is not None:
        df.to_csv(output_csv, index=False)
    return df
