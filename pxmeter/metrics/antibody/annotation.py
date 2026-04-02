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


import contextlib
import io

import anarcii

# Defined as (upper_bound, region_name)
REGION_DEFINITION = {
    "imgt": [
        (26, "FR1"),
        (38, "CDR1"),
        (55, "FR2"),
        (65, "CDR2"),
        (104, "FR3"),
        (117, "CDR3"),
        (128, "FR4"),
    ]
}


class AntibodyAnnotator:
    """
    Antibody region annotator using ANARCI.

    Args:
            scheme (str, optional): The numbering scheme to use. Defaults to "imgt".
            ncpu (int, optional): Number of CPUs to use for ANARCI. Defaults to 1.
    """

    def __init__(self, scheme: str = "imgt", seq_type: str = "unknown", ncpu: int = 1):
        self.scheme = scheme
        if scheme != "imgt":
            raise NotImplementedError(
                f"Region mapping for scheme {scheme} is not implemented."
            )
        self.model = anarcii.Anarcii(seq_type=seq_type, ncpu=ncpu, cpu=True)

    def _get_imgt_region(self, pos: int) -> str:
        regions = REGION_DEFINITION[self.scheme]
        for threshold, region in regions:
            if pos <= threshold:
                return region
        return "Unknown"

    def annotate(self, seqs: list[str]) -> list[tuple[list[str], str]]:
        """
        Annotate antibody regions for a batch of sequences.

        Args:
            seqs (list[str]): A list of amino acid sequences.

        Returns:
            list[tuple[list[str], str]]: A list of tuples, where each tuple contains:
                - regions (list[str]): A list of region strings for each residue.
                - chain_type (str): The antibody chain type.
        """
        if not seqs:
            return []

        results = self._run_anarcii(seqs)

        output = []
        for i, seq in enumerate(seqs):
            annotation = self._process_single_sequence(i, seq, results)
            output.append(annotation)

        return output

    def _run_anarcii(self, seqs: list[str]) -> dict:
        try:
            # Pass sequence as a list. Results are keyed by 'Sequence 1', 'Sequence 2', etc.
            with contextlib.redirect_stdout(io.StringIO()):
                results = self.model.number(seqs, scfv=True)
            return results
        except Exception:
            return {}

    def _process_single_sequence(
        self, index: int, seq: str, results: dict
    ) -> tuple[list[str], str]:
        # ANARCI keys are 1-based indices: 'Sequence 1', 'Sequence 2', ...
        # But can also be 'Sequence 1-1', 'Sequence 1-2' for multi-domain sequences
        base_key = f"Sequence {index + 1}"
        keys_to_process = []

        if base_key in results:
            keys_to_process.append(base_key)

        # Check for split domains
        split_idx = 1
        while True:
            split_key = f"{base_key}-{split_idx}"
            if split_key in results:
                keys_to_process.append(split_key)
                split_idx += 1
            else:
                break

        if not keys_to_process:
            return ["-"] * len(seq), "Unknown"

        result_regions = ["-"] * len(seq)
        collected_chain_types = set()
        found_valid = False

        for key in keys_to_process:
            info = results.get(key)
            if not info or info.get("error"):
                continue

            found_valid = True
            chain_type = info.get("chain_type", "Unknown")
            if chain_type and chain_type != "Unknown":
                collected_chain_types.add(chain_type)

            numbering = info.get("numbering", [])
            query_start = info.get("query_start", 0)

            # query_start is the index in seq where alignment starts (0-based)
            current_seq_idx = query_start

            for num_tuple, residue in numbering:
                if residue != "-":
                    # This corresponds to a residue in the input sequence
                    if current_seq_idx < len(seq):
                        # Map position to region
                        pos = num_tuple[0]
                        region = self._get_imgt_region(pos)
                        result_regions[current_seq_idx] = region

                    current_seq_idx += 1
                else:
                    # Gap in sequence relative to HMM (deletion)
                    pass

        if not found_valid:
            return ["-"] * len(seq), "Unknown"

        final_chain_type = (
            "_".join(sorted(collected_chain_types))
            if collected_chain_types
            else "Unknown"
        )
        return result_regions, final_chain_type
