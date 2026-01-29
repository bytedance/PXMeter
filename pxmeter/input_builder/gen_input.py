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

import argparse
import logging
import random
import traceback
from pathlib import Path
from typing import Optional

from joblib import Parallel, delayed
from tqdm import tqdm

from pxmeter.input_builder.constants import VALID_INPUT_TYPES, VALID_OUTPUT_TYPES
from pxmeter.input_builder.interactive import run_interactive_gen
from pxmeter.input_builder.model_inputs.alphafold3 import AlpahFold3Input
from pxmeter.input_builder.model_inputs.boltz import BoltzInput
from pxmeter.input_builder.model_inputs.protenix import ProtenixInput
from pxmeter.input_builder.seq import Sequences
from pxmeter.utils import str_to_none


def gen_one(
    input_f: Path,
    output_f: Path,
    input_type: str,
    output_type: str,
    seeds: list[int],
    assembly_id: Optional[str] = None,
):
    """
    Generate a single model-input file from one source file.

    Args:
        input_f (Path): Input file path (CIF / AF3 JSON / Protenix JSON).
        output_f (Path): Output file path to write the converted input.
        input_type (str): Input type, e.g. "cif", "af3", or "protenix".
        output_type (str): Output type, e.g. "af3" or "protenix".
        seeds (list[int]): List of model seeds to encode in the output.
            Required for AF3, optional or unused for other formats.
        assembly_id (str | None, optional): Assembly ID for CIF input. Defaults to None.
    """
    output_f.parent.mkdir(parents=True, exist_ok=True)
    if input_type == "cif":
        seqs = Sequences.from_mmcif(input_f, assembly_id=assembly_id)
    elif input_type == "af3":
        seqs = AlpahFold3Input.from_json_file(input_f).to_sequences()
    elif input_type == "protenix":
        seqs = ProtenixInput.from_json_file(input_f).to_sequences()
    elif input_type == "boltz":
        seqs = BoltzInput.from_yaml_file(input_f).to_sequences()
    else:
        raise ValueError(f"Unsupported input type: {input_type}")

    if output_type == "af3":
        AlpahFold3Input.from_sequences(seqs, seeds).to_json_file(output_f)
    elif output_type == "protenix":
        ProtenixInput.from_sequences(seqs, seeds).to_json_file(output_f)
    elif output_type == "boltz":
        BoltzInput.from_sequences(seqs).to_yaml_file(output_f)
    else:
        raise ValueError(f"Unsupported output type: {output_type}")


def gen_batch(
    input_and_output_files: list[tuple[Path, Path]],
    input_type: str,
    output_type: str,
    seeds: list[int],
    assembly_id: Optional[str] = None,
    num_cpu: int = -1,
):
    """
    Generate model-input files for a batch of inputs in parallel.

    Args:
        input_files (list[Path]): List of input file paths.
        output_files (list[Path]): List of output file paths, same length as input_files.
        input_type (str): Input type for all files ("cif", "af3", or "protenix").
        output_type (str): Output type for all files ("af3" or "protenix").
        seeds (list[int]): List of model seeds to encode in each output.
            Required for AF3, optional or unused for other formats.
        assembly_id (str | None, optional): Assembly ID for CIF input. Defaults to None.
        num_cpu (int, optional): Number of worker processes for parallelism.
            Defaults to -1 (all available cores).
    """

    def try_gen_one(inp, out, input_type, output_type, seeds, assembly_id):
        try:
            gen_one(inp, out, input_type, output_type, seeds, assembly_id)
            return 1
        except Exception as e:
            logging.error(
                "Failed to generate %s from %s: %s\n%s",
                out,
                inp,
                e,
                traceback.format_exc(),
            )
            return 0

    if input_type == "cif":
        # Shuffle CIFs to reduce the chance of memory spikes from many large files in a row
        random.shuffle(input_and_output_files)

    results = [
        r
        for r in tqdm(
            Parallel(n_jobs=num_cpu, return_as="generator_unordered")(
                delayed(try_gen_one)(
                    inp, out, input_type, output_type, seeds, assembly_id
                )
                for inp, out in input_and_output_files
            ),
            total=len(input_and_output_files),
            desc="Generating model inputs",
        )
    ]

    success_count = sum(results)
    total = len(results)

    success_rate = int(success_count * 10000 / total) / 100
    logging.info(
        "Success: %d, total: %d, success rate: %.2f%%",
        success_count,
        total,
        success_rate,
    )


def run_gen_input(
    input_path: Path,
    output_path: Path,
    input_type: str,
    output_type: str,
    seeds: list[int] = None,
    num_seeds: int = None,
    assembly_id: Optional[str] = None,
    num_cpu: int = -1,
    pdb_ids: Optional[str] = None,
):
    """
    Entry point for generating model inputs from files or directories.

    This function handles single-file and directory modes, infers suffixes
    from input/output types, and dispatches to single-file or batched
    generation.

    Args:
        input_path (Path): Input file or directory path.
        output_path (Path): Output file or directory path.
        input_type (str): Input type ("cif", "af3", or "protenix").
        output_type (str): Output type ("af3" or "protenix").
        seeds (list[int], optional): Explicit list of seeds. For AF3 output, 
            exactly one of `seeds` or `num_seeds` must be provided.
        num_seeds (int, optional): Number of seeds to generate as 0..num_seeds-1.
            Required for AF3 if `seeds` is not provided.
        assembly_id (str | None, optional): Assembly ID for CIF input. Defaults to None.
        num_cpu (int, optional): Number of CPUs for parallel batch generation.
            Defaults to -1 (all available cores).
        pdb_ids (str | None, optional): Optional list of PDB IDs.
            It supports either a comma-separated string (e.g.``"7n0a,7rss"``)
            or a path to a text file containing one PDB ID per line.
            After parsing, the PDB IDs are passed here as a list of
            strings and are used to restrict which input/output files are
            generated when ``input_path`` is a directory.
    """
    pdb_ids_lst = []
    if pdb_ids:
        pdb_ids_path = Path(pdb_ids)
        if pdb_ids_path.is_file():
            with pdb_ids_path.open("r", encoding="utf-8") as f:
                pdb_ids_lst = [line.strip() for line in f if line.strip()]
        else:
            pdb_ids_lst = [x.strip() for x in pdb_ids.split(",") if x.strip()]

    input_type = input_type.strip().lower()
    output_type = output_type.strip().lower()

    input_is_dir = input_path.is_dir()
    if output_path.exists():
        output_is_dir = output_path.is_dir()
    else:
        output_is_dir = input_is_dir if output_path.suffix == "" else False

    assert input_type in VALID_INPUT_TYPES, f"Unsupported input type: {input_type}"
    assert output_type in VALID_OUTPUT_TYPES, f"Unsupported output type: {output_type}"

    if input_type == output_type:
        logging.warning("Input type and output type are the same")

    assert (input_is_dir and output_is_dir) or (
        not input_is_dir and not output_is_dir
    ), "Input and output should be both directories or both files."

    if output_type == "af3":
        assert (seeds is None) != (
            num_seeds is None
        ), "Either seeds or num_seeds should be provided."

    if seeds is None:
        if num_seeds is None:
            seeds = []
        else:
            seeds = list(range(num_seeds))

    if input_is_dir:
        # Select file suffixes
        if input_type == "cif":
            input_suffixes = ".cif"
        elif input_type in ["af3", "protenix"]:
            # protenix / af3
            input_suffixes = ".json"
        elif input_type == "boltz":
            input_suffixes = ".yaml"
        else:
            raise ValueError(f"Unsupported input type: {input_type}")

        if output_type in ["af3", "protenix"]:
            output_suffixes = ".json"
        elif output_type == "boltz":
            output_suffixes = ".yaml"
        else:
            raise ValueError(f"Unsupported output type: {output_type}")

        if pdb_ids_lst:
            input_and_output_files = [
                (
                    input_path / f"{pdb_id}{input_suffixes}",
                    output_path / f"{pdb_id}{output_suffixes}",
                )
                for pdb_id in pdb_ids_lst
            ]
        else:
            input_and_output_files = []
            for input_f in input_path.iterdir():
                if input_f.is_file() and (input_f.suffix == input_suffixes):
                    input_and_output_files.append(
                        (
                            input_f,
                            output_path / input_f.with_suffix(output_suffixes).name,
                        )
                    )

        if len(input_and_output_files) == 0:
            raise RuntimeError(
                f"No input files with suffixes {input_suffixes} found under "
                f"directory: {input_path}"
            )
        gen_batch(
            input_and_output_files,
            input_type,
            output_type,
            seeds,
            assembly_id,
            num_cpu,
        )

    else:
        gen_one(input_path, output_path, input_type, output_type, seeds, assembly_id)
        logging.info("Generated %s from %s", output_path, input_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate model inputs")
    parser.add_argument(
        "-i", "--input", type=Path, required=False, help="Input file or directory"
    )
    parser.add_argument(
        "-o", "--output", type=Path, required=False, help="Output file or directory"
    )
    parser.add_argument(
        "-it",
        "--input-type",
        dest="input_type",
        type=str,
        required=False,
        help="Input type, choices: " + ", ".join(VALID_INPUT_TYPES),
        choices=VALID_INPUT_TYPES,
    )
    parser.add_argument(
        "-ot",
        "--output-type",
        dest="output_type",
        type=str,
        required=False,
        help="Output type, choices: " + ", ".join(VALID_OUTPUT_TYPES),
        choices=VALID_OUTPUT_TYPES,
    )
    parser.add_argument(
        "-I",
        "--interactive",
        action="store_true",
        help="Run in interactive mode to create an input file.",
    )
    parser.add_argument(
        "-s",
        "--seeds",
        type=str_to_none,
        default=None,
        help=(
            "Comma-separated seeds, e.g. '0,1,2'; required if --num-seeds "
            "is not provided and -ot is 'af3'."
        ),
    )
    parser.add_argument(
        "-ns",
        "--num-seeds",
        type=int,
        default=None,
        help=(
            "Number of seeds; required if --seeds is not provided "
            "and -ot is 'af3'."
        ),
    )
    parser.add_argument(
        "-a",
        "--assembly-id",
        dest="assembly_id",
        type=str_to_none,
        default=None,
        help=(
            "Assembly ID in the input CIF file. Defaults to None. "
            "Ignored for non-CIF input types."
        ),
    )
    parser.add_argument("-n", "--num-cpu", type=int, default=-1, help="Number of CPUs")
    parser.add_argument(
        "-p",
        "--pdb-ids",
        dest="pdb_ids",
        type=str_to_none,
        default=None,
        help=(
            "PDB IDs as a comma-separated string (e.g. '7n0a,7rss') "
            "or a path to a text file containing one PDB ID per line. "
            "This option is only applicable when the input is a directory. "
            "If not provided, all files in the input directory will be "
            "processed."
        ),
    )
    args = parser.parse_args()

    if args.interactive:
        run_interactive_gen()
        exit(0)

    if not all([args.input, args.output, args.input_type, args.output_type]):
        parser.error(
            "The following arguments are required when not in interactive "
            "mode: -i, -o, -it, -ot"
        )

    if args.seeds is not None:
        seeds_lst = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    else:
        seeds_lst = args.seeds

    run_gen_input(
        input_path=args.input,
        output_path=args.output,
        input_type=args.input_type,
        output_type=args.output_type,
        seeds=seeds_lst,
        num_seeds=args.num_seeds,
        assembly_id=args.assembly_id,
        num_cpu=args.num_cpu,
        pdb_ids=args.pdb_ids,
    )
