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

import json
import logging
import tempfile
import unittest
from pathlib import Path

from pxmeter.constants import PROTEIN
from pxmeter.input_builder.gen_input import run_gen_input
from pxmeter.input_builder.seq import Bond, PolymerChainSequence, Sequences
from pxmeter.test.test_utils import TEST_DATA_DIR


class TestInputBuilder(unittest.TestCase):
    """Test suite for pxmeter.input_builder module."""

    def setUp(self) -> None:
        super().setUp()
        self.ref_cif = TEST_DATA_DIR / "7n0a_ref.cif"
        assert self.ref_cif.exists(), "Test CIF file 7n0a_ref.cif must exist."

    def test_sequences_from_mmcif(self):
        """Sequences.from_mmcif should load sequences and bonds from CIF."""
        seqs = Sequences.from_mmcif(self.ref_cif)
        self.assertIsInstance(seqs, Sequences)
        self.assertGreater(len(seqs.sequences), 0)

        # All sequences should report a non-negative token count
        total_tokens = seqs.get_num_tokens()
        self.assertGreater(total_tokens, 0)

    def test_run_gen_input_cif_to_af3_single_file(self):
        """run_gen_input should generate a valid AF3 JSON from a CIF file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = Path(tmpdir) / "af3_input.json"

            run_gen_input(
                input_path=self.ref_cif,
                output_path=out_json,
                input_type="cif",
                output_type="af3",
                seeds=[0, 1],
                num_seeds=None,
                assembly_id=None,
                num_cpu=1,
            )

            self.assertTrue(out_json.exists(), "AF3 JSON output file must be created.")

            with out_json.open("r", encoding="utf-8") as f:
                data = json.load(f)

            # Basic structural checks on AF3 JSON
            self.assertIn("name", data)
            self.assertIn("modelSeeds", data)
            self.assertEqual(data["modelSeeds"], [0, 1])
            self.assertIn("sequences", data)
            self.assertGreater(len(data["sequences"]), 0)

    def test_cif_to_af3_roundtrip_sequences(self):
        """CIF -> AF3 JSON -> Sequences should preserve sequences and bonds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            af3_json = Path(tmpdir) / "af3_input.json"

            # First generate AF3 JSON from CIF
            run_gen_input(
                input_path=self.ref_cif,
                output_path=af3_json,
                input_type="cif",
                output_type="af3",
                seeds=[0],
                num_seeds=None,
                assembly_id=None,
                num_cpu=1,
            )

            # Load AF3 JSON back to AlpahFold3Input, then to Sequences
            from pxmeter.input_builder.model_inputs.alphafold3 import AlpahFold3Input

            af3_input = AlpahFold3Input.from_json_file(af3_json)
            seqs_from_af3 = af3_input.to_sequences()
            seqs_from_cif = Sequences.from_mmcif(self.ref_cif)

            # Compare basic properties
            self.assertEqual(len(seqs_from_cif.sequences), len(seqs_from_af3.sequences))
            self.assertEqual(
                seqs_from_cif.get_num_tokens(), seqs_from_af3.get_num_tokens()
            )

            # Bonds should be preserved as well
            self.assertEqual(len(seqs_from_cif.bonds), len(seqs_from_af3.bonds))

    def test_alphafold3_memory_json_roundtrip(self):
        """Use real CIF to test in-memory AF3 JSON roundtrip.

        Sequences (from CIF) -> AlpahFold3Input -> JSON dict -> AlpahFold3Input
        -> Sequences, basic topology should be preserved.
        """
        from pxmeter.input_builder.model_inputs.alphafold3 import AlpahFold3Input

        seqs_from_cif = Sequences.from_mmcif(self.ref_cif)
        af3_input = AlpahFold3Input.from_sequences(seqs_from_cif, seeds=[0, 1])

        json_dict = af3_input.to_json()
        af3_reloaded = AlpahFold3Input.from_json(json_dict)
        seqs_roundtrip = af3_reloaded.to_sequences()

        self.assertEqual(len(seqs_from_cif.sequences), len(seqs_roundtrip.sequences))
        self.assertEqual(
            seqs_from_cif.get_num_tokens(), seqs_roundtrip.get_num_tokens()
        )
        self.assertEqual(len(seqs_from_cif.bonds), len(seqs_roundtrip.bonds))

    def test_protenix_memory_json_roundtrip(self):
        """Use real CIF to test in-memory Protenix JSON roundtrip.

        Sequences (from CIF) -> ProtenixInput -> JSON dict -> ProtenixInput
        -> Sequences, basic topology should be preserved.
        """
        from pxmeter.input_builder.model_inputs.protenix import ProtenixInput

        seqs_from_cif = Sequences.from_mmcif(self.ref_cif)
        px_input = ProtenixInput.from_sequences(seqs_from_cif, seeds=[0, 1])

        json_dict = px_input.to_json()
        px_reloaded = ProtenixInput.from_json(json_dict)
        seqs_roundtrip = px_reloaded.to_sequences()

        self.assertEqual(len(seqs_from_cif.sequences), len(seqs_roundtrip.sequences))
        self.assertEqual(
            seqs_from_cif.get_num_tokens(), seqs_roundtrip.get_num_tokens()
        )
        self.assertEqual(len(seqs_from_cif.bonds), len(seqs_roundtrip.bonds))

    def test_sequences_to_protenix_roundtrip(self):
        """Sequences -> ProtenixInput -> Sequences should preserve topology."""
        from pxmeter.input_builder.model_inputs.protenix import ProtenixInput

        seqs_from_cif = Sequences.from_mmcif(self.ref_cif)
        px_input = ProtenixInput.from_sequences(seqs_from_cif, seeds=[0])
        seqs_roundtrip = px_input.to_sequences()

        # Same number of chains
        self.assertEqual(len(seqs_from_cif.sequences), len(seqs_roundtrip.sequences))
        # Token count should be preserved
        self.assertEqual(
            seqs_from_cif.get_num_tokens(), seqs_roundtrip.get_num_tokens()
        )
        # Bond count should be preserved
        self.assertEqual(len(seqs_from_cif.bonds), len(seqs_roundtrip.bonds))

    def test_protenix_maps_non_contiguous_duplicate_chain_to_existing_entity(self):
        """A repeated sequence should retain its original entity ID in bonds."""
        from pxmeter.input_builder.model_inputs.protenix import ProtenixInput

        first_a = PolymerChainSequence(
            sequence="AA", entity_type=PROTEIN, ori_chain_id="A"
        )
        chain_b = PolymerChainSequence(
            sequence="GG", entity_type=PROTEIN, ori_chain_id="B"
        )
        second_a = PolymerChainSequence(
            sequence="AA", entity_type=PROTEIN, ori_chain_id="C"
        )
        sequences = Sequences(
            name="non_contiguous_duplicate",
            sequences=(first_a, chain_b, second_a),
            bonds=(
                Bond(
                    chain_index_1=2,
                    res_id_1=1,
                    atom_name_1="CA",
                    chain_index_2=0,
                    res_id_2=1,
                    atom_name_2="N",
                ),
            ),
        )

        px_input = ProtenixInput.from_sequences(sequences, seeds=[0])

        self.assertEqual(px_input.sequences[0].entity_id, 1)
        self.assertEqual(px_input.sequences[0].count, 2)
        self.assertEqual(px_input.bonds[0].entity_id_1, 1)
        self.assertEqual(px_input.bonds[0].copy_id_1, 2)
        self.assertEqual(px_input.bonds[0].entity_id_2, 1)
        self.assertEqual(px_input.bonds[0].copy_id_2, 1)

    def test_boltz_memory_yaml_roundtrip(self):
        """Use real CIF to test in-memory Boltz YAML roundtrip.

        Sequences (from CIF) -> BoltzInput -> YAML dict -> BoltzInput
        -> Sequences. Boltz does not support ligands with multiple CCD codes,
        so this test focuses on polymer chains being preserved.
        """
        from pxmeter.input_builder.model_inputs.boltz import BoltzInput

        seqs_from_cif = Sequences.from_mmcif(self.ref_cif)
        boltz_input = BoltzInput.from_sequences(seqs_from_cif)

        yaml_dict = boltz_input.to_yaml()
        boltz_reloaded = BoltzInput.from_yaml(yaml_dict, name=boltz_input.name)
        seqs_roundtrip = boltz_reloaded.to_sequences()

        # Boltz skips ligands with multiple CCD codes, so total chain count
        # may decrease. Polymer chains should still be preserved.
        orig_polymer_seqs = [s for s in seqs_from_cif.sequences if s.is_polymer()]
        rt_polymer_seqs = [s for s in seqs_roundtrip.sequences if s.is_polymer()]

        self.assertEqual(len(orig_polymer_seqs), len(rt_polymer_seqs))
        self.assertEqual(
            sum(s.get_num_tokens() for s in orig_polymer_seqs),
            sum(s.get_num_tokens() for s in rt_polymer_seqs),
        )

    def test_long_chain_roundtrip_cif_af3_protenix_boltz(self):
        """CIF -> AF3 -> Sequences -> Protenix -> Sequences -> Boltz -> Sequences.

        This long conversion chain should preserve basic topology information
        such as chain count, token count and bond count when projected back
        to the Sequences abstraction.
        """
        from pxmeter.input_builder.model_inputs.alphafold3 import AlpahFold3Input
        from pxmeter.input_builder.model_inputs.protenix import ProtenixInput
        from pxmeter.input_builder.model_inputs.boltz import BoltzInput

        # Start from CIF as the canonical source
        seqs_from_cif = Sequences.from_mmcif(self.ref_cif)

        with tempfile.TemporaryDirectory() as tmpdir:
            af3_json = Path(tmpdir) / "af3_input.json"

            # CIF -> AF3 JSON
            run_gen_input(
                input_path=self.ref_cif,
                output_path=af3_json,
                input_type="cif",
                output_type="af3",
                seeds=[0],
                num_seeds=None,
                assembly_id=None,
                num_cpu=1,
            )

            # AF3 JSON -> Sequences
            af3_input = AlpahFold3Input.from_json_file(af3_json)
            seqs_from_af3 = af3_input.to_sequences()

        # Sequences (from AF3) -> Protenix -> Sequences
        px_input = ProtenixInput.from_sequences(seqs_from_af3, seeds=[0])
        seqs_from_protenix = px_input.to_sequences()

        # Sequences (from Protenix) -> Boltz -> Sequences
        boltz_input = BoltzInput.from_sequences(seqs_from_protenix)
        seqs_from_boltz = boltz_input.to_sequences()

        # All representations should agree on basic topology metrics
        self.assertEqual(len(seqs_from_cif.sequences), len(seqs_from_boltz.sequences))
        self.assertEqual(
            seqs_from_cif.get_num_tokens(), seqs_from_boltz.get_num_tokens()
        )
        self.assertEqual(len(seqs_from_cif.bonds), len(seqs_from_boltz.bonds))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
