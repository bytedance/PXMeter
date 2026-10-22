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

from unittest import TestCase

from pxmeter.constants import PROTEIN
from pxmeter.input_builder.model_inputs.protenix import ProtenixInput
from pxmeter.input_builder.seq import Bond, PolymerChainSequence, Sequences


class TestProtenixCovalentBondSerialization(TestCase):
    @staticmethod
    def _source_with_copy_pairs(
        copy_pairs: list[tuple[int, int]],
    ) -> ProtenixInput:
        sequences = (
            PolymerChainSequence("AA", PROTEIN, ori_chain_id="A1"),
            PolymerChainSequence("AA", PROTEIN, ori_chain_id="A2"),
            PolymerChainSequence("GG", PROTEIN, ori_chain_id="B1"),
            PolymerChainSequence("GG", PROTEIN, ori_chain_id="B2"),
        )
        bonds = tuple(
            Bond(
                chain_index_1=copy_id_1 - 1,
                res_id_1=1,
                atom_name_1="CA",
                chain_index_2=copy_id_2 + 1,
                res_id_2=1,
                atom_name_2="N",
            )
            for copy_id_1, copy_id_2 in copy_pairs
        )
        return ProtenixInput.from_sequences(
            Sequences("copy-pair-serialization", sequences, bonds), seeds=[0]
        )

    def test_crossed_copy_pairs_remain_explicit_after_roundtrip(self):
        source = self._source_with_copy_pairs([(1, 2), (2, 1)])

        encoded = source.to_json()
        encoded_pairs = [
            (bond["copy1"], bond["copy2"]) for bond in encoded["covalent_bonds"]
        ]
        restored = ProtenixInput.from_json(encoded)
        restored_pairs = sorted(
            (bond.copy_id_1, bond.copy_id_2) for bond in restored.bonds
        )

        self.assertEqual(encoded_pairs, [(1, 2), (2, 1)])
        self.assertEqual(restored_pairs, [(1, 2), (2, 1)])

    def test_complete_identity_copy_pairs_remain_compressed(self):
        source = self._source_with_copy_pairs([(1, 1), (2, 2)])

        encoded = source.to_json()
        self.assertEqual(len(encoded["covalent_bonds"]), 1)
        self.assertNotIn("copy1", encoded["covalent_bonds"][0])
        self.assertNotIn("copy2", encoded["covalent_bonds"][0])

        restored = ProtenixInput.from_json(encoded)
        restored_pairs = sorted(
            (bond.copy_id_1, bond.copy_id_2) for bond in restored.bonds
        )
        self.assertEqual(restored_pairs, [(1, 1), (2, 2)])
