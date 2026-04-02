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

import re
from pathlib import Path

from biotite.structure.info import get_from_ccd

from pxmeter.constants import (
    DNA,
    DNA_ONE_TO_THREE,
    DNA_STD_RESIDUES_ONE_LETTER,
    PRO_ONE_TO_THREE,
    PRO_STD_RESIDUES_ONE_LETTER,
    PROTEIN,
    RNA,
    RNA_ONE_TO_THREE,
    RNA_STD_RESIDUES_ONE_LETTER,
)
from pxmeter.input_builder.constants import VALID_INPUT_TYPES, VALID_OUTPUT_TYPES
from pxmeter.input_builder.model_inputs.alphafold3 import AlpahFold3Input
from pxmeter.input_builder.model_inputs.boltz import BoltzInput
from pxmeter.input_builder.model_inputs.openfold3 import OpenFold3Input
from pxmeter.input_builder.model_inputs.protenix import ProtenixInput
from pxmeter.input_builder.seq import (
    Bond,
    LigandChainSequence,
    PolymerChainSequence,
    Sequences,
)


def run_interactive_gen():
    """
    Interactively create a model-input file by asking the user for
    sequences, bonds, and output type.
    """
    print("\n" + "=" * 40)
    print("   Interactive Model Input Generator")
    print("=" * 40)

    def ask_choice(question, options, default=None):
        """
        Helper to ask a choice from a list of options using numbers.
        options: list of (display_name, value)
        """
        print(f"\n{question}")
        default_idx = None
        for i, (name, val) in enumerate(options, 1):
            is_default = default and val == default
            if is_default:
                default_idx = i
            default_mark = "*" if is_default else ""
            print(f"  {i}. {name} {default_mark}")

        while True:
            prompt_suffix = f" [default: {default_idx}]: " if default_idx else ": "
            choice = input(
                f"Select an option (1-{len(options)})" + prompt_suffix
            ).strip()
            if not choice and default:
                return default
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    return options[idx][1]
            except ValueError:
                pass
            print(
                f"Invalid selection. Please enter a number between 1 and "
                f"{len(options)}."
            )

    def ask_yes_no(question, default="n"):
        """Helper for simple y/n questions."""
        prompt = f"\n{question} (y/n, default: {default}): "
        while True:
            val = input(prompt).strip().lower()
            if not val:
                val = default
            if val in ["y", "yes"]:
                return True
            if val in ["n", "no"]:
                return False
            print("Invalid input. Please enter 'y' or 'n'.")

    def get_ccd_info_biotite(ccd_code):
        """Get CCD existence and non-H atom list using biotite's get_from_ccd."""
        try:
            # Check existence
            _ = get_from_ccd("chem_comp", ccd_code, "id").as_item()
            # Get atom names and elements
            atom_names = get_from_ccd("chem_comp_atom", ccd_code, "atom_id").as_array()
            elements = get_from_ccd(
                "chem_comp_atom", ccd_code, "type_symbol"
            ).as_array()
            # Filter H/D
            valid_atoms = sorted(
                [
                    name
                    for name, elem in zip(atom_names, elements)
                    if elem not in ["H", "D"]
                ]
            )
            return True, valid_atoms
        except Exception:
            return False, []

    def validate_ccd(ccd_code):
        """Check if CCD exists in biotite and ask user if not."""
        exists, _ = get_ccd_info_biotite(ccd_code)
        if exists:
            return True

        return ask_yes_no(
            f"  Warning: CCD code '{ccd_code}' was not found in the local CCD database. "
            "Force add anyway?",
            default="n",
        )

    output_type = ask_choice(
        "Select output type:",
        [(t.upper(), t) for t in VALID_OUTPUT_TYPES],
        default="af3",
    )

    name = (
        input("\nEnter job name (default: 'interactive_job'): ").strip()
        or "interactive_job"
    )

    all_seqs = []
    all_bonds = []

    if ask_yes_no("Do you want to load components from an existing file?", default="n"):
        while True:
            file_path = Path(input("  Enter file path to load: ").strip())
            if not file_path.exists():
                print(f"  Error: File {file_path} does not exist.")
                if not ask_yes_no("Try another file?", default="y"):
                    break
                continue

            in_type = ask_choice(
                "Select input file type:",
                [(t.upper(), t) for t in VALID_INPUT_TYPES],
                default=(
                    file_path.suffix[1:]
                    if file_path.suffix[1:] in VALID_INPUT_TYPES
                    else "cif"
                ),
            )

            try:
                if in_type == "cif":
                    a_id = (
                        input("  Enter Assembly ID (default: None): ").strip() or None
                    )
                    loaded_seqs_obj = Sequences.from_mmcif(file_path, assembly_id=a_id)
                elif in_type == "af3":
                    loaded_seqs_obj = AlpahFold3Input.from_json_file(
                        file_path
                    ).to_sequences()
                elif in_type == "protenix":
                    loaded_seqs_obj = ProtenixInput.from_json_file(
                        file_path
                    ).to_sequences()
                elif in_type == "boltz":
                    loaded_seqs_obj = BoltzInput.from_yaml_file(
                        file_path
                    ).to_sequences()
                elif in_type == "openfold3":
                    loaded_seqs_obj = OpenFold3Input.from_json_file(
                        file_path
                    ).to_sequences()
                else:
                    print(f"  Error: Unsupported input type {in_type}")
                    continue

                all_seqs = list(loaded_seqs_obj.sequences)
                all_bonds = list(loaded_seqs_obj.bonds)
                if not name or name == "interactive_job":
                    name = loaded_seqs_obj.name
                print(
                    f"  Successfully loaded {len(all_seqs)} components and {len(all_bonds)} bonds."
                )
                break
            except Exception as e:
                print(f"  Error loading file: {e}")
                if not ask_yes_no("Try another file?", default="y"):
                    break

    def print_current_components(seqs):
        if not seqs:
            print("\n>>> No components added yet.")
            return

        type_display_map = {PROTEIN: "Protein", DNA: "DNA", RNA: "RNA"}

        print("\n" + "-" * 20)
        print("Current components (0-indexed):")
        for i, s in enumerate(seqs):
            if isinstance(s, PolymerChainSequence):
                # Apply modifications to display
                display_parts = list(s.sequence)
                for pos, mod_type in s.modifications:
                    if 1 <= pos <= len(display_parts):
                        display_parts[pos - 1] = f"({mod_type})"

                display_seq = "".join(display_parts)
                if len(display_seq) > 100:
                    display_seq = display_seq[:100] + "..."

                type_name = type_display_map.get(s.entity_type, s.entity_type)
                desc = f"{type_name}, len={len(s.sequence)}, seq: {display_seq}"
            else:
                if s.ccd_codes:
                    lig_info = "-".join(s.ccd_codes)
                    num_res = len(s.ccd_codes)
                else:
                    lig_info = s.smiles or str(s.file_path)
                    num_res = 1
                desc = f"ligand ({lig_info}), contains {num_res} residue(s)"
            print(f"  {i}: {desc}")
        print("-" * 20)

    def print_current_bonds(bonds):
        if not bonds:
            print("\n>>> No covalent bonds added yet.")
            return
        print("\n" + "-" * 20)
        print("Current covalent bonds (0-indexed):")
        for i, b in enumerate(bonds):
            print(
                f"  {i}: Chain {b.chain_index_1}[Res {b.res_id_1}, "
                f"Atom {b.atom_name_1}] <-> Chain {b.chain_index_2}"
                f"[Res {b.res_id_2}, Atom {b.atom_name_2}]"
            )
        print("-" * 20)

    print("\n[Step 1: Add components (sequences or ligands)]")
    while True:
        print_current_components(all_seqs)

        comp_action = ask_choice(
            "What would you like to do?",
            [
                ("Add Polymer (Protein/DNA/RNA)", "p"),
                ("Add Ligand", "l"),
                ("Remove Component", "r"),
                ("Done adding components", "q"),
            ],
        )

        if comp_action == "q":
            if not all_seqs:
                print("Error: At least one component is required.")
                continue
            break

        if comp_action == "r":
            if not all_seqs:
                print("Error: No components to remove.")
                continue
            try:
                rem_idx = int(
                    input(
                        f"Enter Component index to remove (0-{len(all_seqs)-1}): "
                    ).strip()
                )
                if 0 <= rem_idx < len(all_seqs):
                    # Remove the component
                    all_seqs.pop(rem_idx)

                    # Update and filter bonds
                    new_bonds = []
                    for b in all_bonds:
                        if b.chain_index_1 == rem_idx or b.chain_index_2 == rem_idx:
                            continue  # Remove bond involving deleted component

                        # Shift indices for components that were after the deleted one
                        c1 = (
                            b.chain_index_1 - 1
                            if b.chain_index_1 > rem_idx
                            else b.chain_index_1
                        )
                        c2 = (
                            b.chain_index_2 - 1
                            if b.chain_index_2 > rem_idx
                            else b.chain_index_2
                        )

                        new_bonds.append(
                            Bond(
                                chain_index_1=c1,
                                res_id_1=b.res_id_1,
                                atom_name_1=b.atom_name_1,
                                chain_index_2=c2,
                                res_id_2=b.res_id_2,
                                atom_name_2=b.atom_name_2,
                            )
                        )
                    all_bonds = new_bonds
                    print(
                        f"Successfully removed component {rem_idx} and "
                        "updated affected bonds."
                    )
                else:
                    print("Error: Index out of range.")
            except ValueError:
                print("Error: Please enter a valid integer.")
            continue

        if comp_action == "p":
            p_type = ask_choice(
                "Select polymer type:",
                [("Protein", "protein"), ("DNA", "dna"), ("RNA", "rna")],
                default="protein",
            )

            while True:
                seq_str = input("Enter sequence string: ").strip()
                if not seq_str:
                    print("  Error: Sequence cannot be empty.")
                    continue
                if not re.match("^[A-Z]+$", seq_str):
                    print(
                        "  Error: Sequence must contain only uppercase "
                        "English letters (A-Z)."
                    )
                    continue

                # Specific alphabet validation
                allowed_alphabet = {
                    "protein": set(PRO_STD_RESIDUES_ONE_LETTER),
                    "dna": set(DNA_STD_RESIDUES_ONE_LETTER),
                    "rna": set(RNA_STD_RESIDUES_ONE_LETTER),
                }[p_type]

                invalid_chars = sorted(list(set(seq_str) - allowed_alphabet))
                if invalid_chars:
                    print(
                        f"  Error: Invalid characters for {p_type}: "
                        f"{' '.join(invalid_chars)}"
                    )
                    print(
                        f"  Allowed alphabet: "
                        f"{' '.join(sorted(list(allowed_alphabet)))}"
                    )
                    continue
                break

            try:
                num_copies = int(
                    input("Enter number of copies (default: 1): ").strip() or 1
                )
            except ValueError:
                num_copies = 1

            modifications = []
            while True:
                # Show current sequence state with modifications
                display_parts = list(seq_str)
                for pos, mod_type in modifications:
                    if 1 <= pos <= len(display_parts):
                        display_parts[pos - 1] = f"({mod_type})"
                curr_display = "".join(display_parts)
                if len(curr_display) > 100:
                    curr_display = curr_display[:100] + "..."
                print(f"\n  Current sequence: {curr_display}")

                if not ask_yes_no("Add a modification for this polymer?", default="n"):
                    break
                try:
                    pos = int(input("  Enter position (1-indexed): ").strip())
                    while True:
                        m_type = (
                            input(
                                "  Enter modification CCD type (e.g. SEP, HYP, 5MC): "
                            )
                            .strip()
                            .upper()
                        )
                        if not m_type:
                            print("  Error: CCD type cannot be empty.")
                            continue
                        if not re.match("^[A-Z0-9]+$", m_type):
                            print(
                                "  Error: CCD type must contain only uppercase "
                                "letters and numbers."
                            )
                            continue

                        if not validate_ccd(m_type):
                            continue
                        break
                    modifications.append((pos, m_type))
                except ValueError:
                    print("  Invalid position. Skipping this modification.")

            type_map = {"protein": PROTEIN, "dna": DNA, "rna": RNA}

            for _ in range(num_copies):
                all_seqs.append(
                    PolymerChainSequence(
                        sequence=seq_str,
                        entity_type=type_map[p_type],
                        modifications=tuple(modifications),
                    )
                )

        elif comp_action == "l":
            while True:
                l_input_type = ask_choice(
                    "Select ligand input method:",
                    [
                        ("CCD Codes (e.g. NAG, NAG, FUC)", "c"),
                        ("SMILES String", "s"),
                        ("File Path (e.g. .sdf, .mol)", "f"),
                    ],
                )

                # Validation for file_path
                if l_input_type == "f" and output_type in ["af3", "boltz", "openfold3"]:
                    print(
                        f"\nError: {output_type.upper()} does not support "
                        "ligand input via file_path."
                    )
                    continue
                break

            ccd_codes = None
            smiles = None
            file_path = None

            if l_input_type == "c":
                while True:
                    codes_str = input("  Enter CCD codes (comma separated): ").strip()
                    codes = [
                        c.strip().upper() for c in codes_str.split(",") if c.strip()
                    ]
                    if not codes:
                        print("  Error: CCD codes cannot be empty.")
                        continue

                    if not all(re.match("^[A-Z0-9]+$", c) for c in codes):
                        print(
                            "  Error: CCD codes must contain only uppercase "
                            "letters and numbers."
                        )
                        continue

                    # Validation for Boltz (only 1 CCD code)
                    if output_type == "boltz" and len(codes) > 1:
                        print(
                            "  Error: Boltz only supports a single CCD code "
                            "for ligands."
                        )
                        continue

                    # Validation for CCD existence
                    all_valid = True
                    for c in codes:
                        if not validate_ccd(c):
                            all_valid = False
                            break
                    if not all_valid:
                        continue

                    ccd_codes = tuple(codes)
                    break
            elif l_input_type == "s":
                smiles = input("  Enter SMILES string: ").strip()
                while not smiles:
                    smiles = input(
                        "  SMILES cannot be empty. Enter SMILES string: "
                    ).strip()
            elif l_input_type == "f":
                file_path = Path(input("  Enter file path: ").strip())

            try:
                num_copies = int(
                    input("Enter number of copies (default: 1): ").strip() or 1
                )
            except ValueError:
                num_copies = 1

            for _ in range(num_copies):
                all_seqs.append(
                    LigandChainSequence(
                        ccd_codes=ccd_codes,
                        smiles=smiles,
                        file_path=file_path,
                    )
                )

    # Bonds
    print("\n[Step 2: Add covalent bonds (optional)]")

    def get_valid_bond_atom(prompt_prefix, all_seqs):
        while True:
            try:
                user_input = input(
                    f"\nEnter Chain Index for {prompt_prefix} (or 'q' to cancel): "
                ).strip()
                if user_input.lower() == "q":
                    return None
                c_idx = int(user_input)
                if not (0 <= c_idx < len(all_seqs)):
                    print(
                        f"  Error: Chain index {c_idx} out of range "
                        f"(0-{len(all_seqs)-1})."
                    )
                    continue

                s = all_seqs[c_idx]
                if isinstance(s, PolymerChainSequence):
                    max_res = len(s.sequence)
                else:
                    max_res = len(s.ccd_codes) if s.ccd_codes else 1

                user_input = input(
                    f"Enter Residue ID for {prompt_prefix} (1-{max_res}, or 'q' to cancel): "
                ).strip()
                if user_input.lower() == "q":
                    return None
                r_id = int(user_input)
                if not (1 <= r_id <= max_res):
                    print(
                        f"  Error: Residue ID {r_id} out of range for "
                        f"chain {c_idx} (1-{max_res})."
                    )
                    continue

                # Determine CCD code for validation
                ccd_code = ""
                if isinstance(s, PolymerChainSequence):
                    mod_map = dict(s.modifications)
                    if r_id in mod_map:
                        ccd_code = mod_map[r_id]
                    else:
                        one_letter = s.sequence[r_id - 1]
                        if s.entity_type == PROTEIN:
                            ccd_code = PRO_ONE_TO_THREE[one_letter]
                        elif s.entity_type == DNA:
                            ccd_code = DNA_ONE_TO_THREE[one_letter]
                        elif s.entity_type == RNA:
                            ccd_code = RNA_ONE_TO_THREE[one_letter]
                else:
                    # Ligand
                    if s.ccd_codes:
                        ccd_code = s.ccd_codes[r_id - 1]
                    else:
                        ccd_code = None

                # Fetch valid atoms from biotite
                _, valid_atoms = (
                    get_ccd_info_biotite(ccd_code) if ccd_code else (False, [])
                )

                while True:
                    a_name = (
                        input(
                            f"Enter Atom Name for {prompt_prefix} (or 'q' to cancel): "
                        )
                        .strip()
                        .upper()
                    )
                    if a_name.lower() == "q":
                        return None
                    if not a_name:
                        print("  Error: Atom name cannot be empty.")
                        continue

                    if valid_atoms and a_name not in valid_atoms:
                        print(
                            f"  Error: '{a_name}' is not a valid atom for "
                            f"residue {ccd_code}."
                        )
                        print(f"  Valid atoms are: {', '.join(valid_atoms)}")
                        print(
                            f"  Hint: You can check the valid atom names at: "
                            f"https://www.rcsb.org/ligand/{ccd_code}"
                        )
                        continue
                    break

                return c_idx, r_id, a_name
            except ValueError:
                print("  Error: Please enter a valid integer or 'q' to cancel.")

    while True:
        print_current_components(all_seqs)
        print_current_bonds(all_bonds)
        bond_action = ask_choice(
            "What would you like to do?",
            [
                ("Add Covalent Bond", "a"),
                ("Remove Covalent Bond", "r"),
                ("Done with bonds", "q"),
            ],
            default="q",
        )

        if bond_action == "q":
            break

        if bond_action == "r":
            if not all_bonds:
                print("Error: No bonds to remove.")
                continue
            try:
                rem_idx = int(
                    input(
                        f"Enter Bond index to remove (0-{len(all_bonds)-1}): "
                    ).strip()
                )
                if 0 <= rem_idx < len(all_bonds):
                    all_bonds.pop(rem_idx)
                    print(f"Successfully removed bond {rem_idx}.")
                else:
                    print("Error: Index out of range.")
            except ValueError:
                print("Error: Please enter a valid integer.")
            continue

        if bond_action == "a":
            print_current_components(all_seqs)

            res1 = get_valid_bond_atom("Atom 1", all_seqs)
            if res1 is None:
                continue
            c1, r1, a1 = res1

            res2 = get_valid_bond_atom("Atom 2", all_seqs)
            if res2 is None:
                continue
            c2, r2, a2 = res2

            if c1 == c2 and r1 == r2 and a1 == a2:
                print("Error: Cannot bond an atom to itself.")
                continue

            new_bond = Bond(
                chain_index_1=c1,
                res_id_1=r1,
                atom_name_1=a1,
                chain_index_2=c2,
                res_id_2=r2,
                atom_name_2=a2,
            )

            if new_bond in all_bonds:
                print("Error: This covalent bond already exists.")
            else:
                all_bonds.append(new_bond)

    print("\n[Step 3: Finalize]")
    output_f = input("Enter output file path (e.g. input.json): ").strip()
    while not output_f:
        output_f = input("File path cannot be empty: ").strip()
    output_f = Path(output_f)

    seeds = None
    if output_type not in ["boltz", "openfold3"]:
        while True:
            seeds_str = input(
                "Enter model seeds (comma separated, default: None): "
            ).strip()
            if not seeds_str:
                seeds = None
                break
            try:
                seeds = [int(x.strip()) for x in seeds_str.split(",") if x.strip()]
                if not seeds:
                    seeds = None
                break
            except ValueError:
                print(
                    "  Error: Invalid seeds format. Please enter integers separated by commas (e.g., 0, 1, 2)."
                )

    seqs_obj = Sequences(name=name, sequences=tuple(all_seqs), bonds=tuple(all_bonds))

    output_f.parent.mkdir(parents=True, exist_ok=True)
    if seeds is None:
        seeds = [42]

    if output_type == "af3":
        AlpahFold3Input.from_sequences(seqs_obj, seeds).to_json_file(output_f)
    elif output_type == "protenix":
        ProtenixInput.from_sequences(seqs_obj, seeds).to_json_file(output_f)
    elif output_type == "boltz":
        BoltzInput.from_sequences(seqs_obj).to_yaml_file(output_f)
    elif output_type == "openfold3":
        OpenFold3Input.from_sequences(seqs_obj).to_json_file(output_f)

    print("\n" + "=" * 40)
    print(f"Successfully generated {output_type.upper()} input file at:")
    print(f"  {output_f.absolute()}")
    print("=" * 40)
