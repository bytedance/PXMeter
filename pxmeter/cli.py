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
import sys
from pathlib import Path
from typing import Optional

import click
from biotite import setup_ccd

from pxmeter.configs.run_config import apply_run_config_overrides
from pxmeter.eval import evaluate, MetricResult
from pxmeter.input_builder.constants import VALID_INPUT_TYPES, VALID_OUTPUT_TYPES
from pxmeter.input_builder.gen_input import run_gen_input
from pxmeter.input_builder.interactive import run_interactive_gen
from pxmeter.metrics.stereochemistry.check import stereochem_check_to_csv
from pxmeter.utils import read_chain_id_to_mol_from_json, str_to_none

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def run_eval_cif(
    ref_cif: Path,
    model_cif: Path,
    output_json: Path,
    ref_model: int = 1,
    ref_assembly_id: Optional[str] = None,
    ref_altloc: str = "first",
    interested_lig_label_asym_id: Optional[str] = None,
    chain_id_to_mol_json: Optional[Path] = None,
    output_mapped_cif: bool = False,
) -> MetricResult:
    """
    Evaluate the performance of a model CIF file by comparing it to a reference CIF file,
    and save the results in a JSON file.
    """
    if chain_id_to_mol_json is not None:
        chain_id_to_mol = read_chain_id_to_mol_from_json(chain_id_to_mol_json)
    else:
        chain_id_to_mol = None

    if interested_lig_label_asym_id is not None:
        # split by comma
        interested_lig_label_asym_id = interested_lig_label_asym_id.split(",")

    metric_result = evaluate(
        ref_cif=ref_cif,
        model_cif=model_cif,
        ref_model=ref_model,
        ref_assembly_id=ref_assembly_id,
        ref_altloc=ref_altloc,
        interested_lig_label_asym_id=interested_lig_label_asym_id,
        model_chain_id_to_lig_mol=chain_id_to_mol,
    )

    metric_result.to_json(json_file=output_json)

    if output_mapped_cif:
        ref_mapped_cif = str(output_json).replace(".json", "_mapped_ref.cif")
        model_mapped_cif = str(output_json).replace(".json", "_mapped_model.cif")

        # Select valid atoms of model also by in ref structure
        metric_result.ref_struct.to_cif(ref_mapped_cif)
        metric_result.model_struct.to_cif(model_mapped_cif)

    return metric_result


@click.group(
    invoke_without_command=True,
    context_settings=dict(help_option_names=["-h", "--help"]),
)
@click.option("-r", "--ref_cif", type=Path, help="Path to the reference CIF file.")
@click.option("-m", "--model_cif", type=Path, help="Path to the model CIF file.")
@click.option(
    "-o",
    "--output_json",
    type=Path,
    default="./pxm_output.json",
    help="Path to the output JSON file. Defaults to 'pxm_output.json'.",
)
@click.option(
    "--ref_model",
    type=int,
    default=1,
    help="Model number in the reference CIF file to use. Defaults to 1.",
)
@click.option(
    "--ref_assembly_id",
    type=str,
    default=None,
    help="Assembly ID in the reference CIF file. Defaults to None.",
)
@click.option(
    "--ref_altloc",
    type=str,
    default="first",
    help="Altloc ID in the reference CIF file. Defaults to 'first'.",
)
@click.option(
    "-l",
    "--interested_lig_label_asym_id",
    type=str,
    default=None,
    help="The label_asym_id of the ligand of interest in the reference structure (for ligand RMSD metrics). \
        If multiple ligands are present, separate them by comma. Defaults to None.",
)
@click.option(
    "-c",
    "--chain_id_to_mol_json",
    type=Path,
    default=None,
    help="Path to a JSON file containing a mapping of chain IDs to molecular input (SMILES). \
        E.g. {'B': 'c1ccccc1', 'D':'CCCC'}",
)
@click.option(
    "--output_mapped_cif",
    is_flag=True,
    help="Whether to output the mapped CIF file. Defaults to False.",
)
@click.option(
    "-C",
    "--config",
    "config_overrides",
    multiple=True,
    help=(
        "Override run config. Use dotted keys from RUN_CONFIG, e.g. "
        "-C metric.lddt.eps=1e-5 -C mapping.mapping_ligand=false"
    ),
)
@click.pass_context
def cli(
    ctx,
    ref_cif: Path,
    model_cif: Path,
    output_json: Path,
    ref_model: int = 1,
    ref_assembly_id: Optional[str] = None,
    ref_altloc: str = "first",
    interested_lig_label_asym_id: Optional[str] = None,
    chain_id_to_mol_json: Optional[Path] = None,
    output_mapped_cif: bool = False,
    config_overrides: tuple[str, ...] = (),
):
    """
    Evaluate the performance of a model CIF file by comparing it to a reference CIF file,
    and save the results in a JSON file.
    """
    # Handle "None" string from CLI
    ref_assembly_id = str_to_none(ref_assembly_id)
    interested_lig_label_asym_id = str_to_none(interested_lig_label_asym_id)

    if ctx.invoked_subcommand is None:
        if len(sys.argv) == 1:
            click.echo(ctx.get_help())
            ctx.exit()

        if ref_cif is None or model_cif is None:
            click.echo("Error: --ref_cif and --model_cif are required.")
            ctx.exit()

        if config_overrides:
            apply_run_config_overrides(config_overrides)

        run_eval_cif(
            ref_cif,
            model_cif,
            output_json,
            ref_model,
            ref_assembly_id,
            ref_altloc,
            interested_lig_label_asym_id,
            chain_id_to_mol_json,
            output_mapped_cif,
        )


@cli.command(name="stereocheck")
@click.option(
    "-c",
    "--cif",
    "cif",
    type=click.Path(path_type=Path, exists=True),
    required=True,
    help="Path to the model CIF file.",
)
@click.option(
    "-o",
    "--output-csv",
    "output_csv",
    type=click.Path(path_type=Path),
    default=Path("./stereochem_report.csv"),
    show_default=True,
    help="Path to the output CSV report.",
)
def stereochem_cli(
    cif: Path,
    output_csv: Path,
):
    """
    Run stereochemistry checks for a single model CIF and export a CSV report.
    """

    report_df = stereochem_check_to_csv(
        model_cif=cif,
        output_csv=output_csv,
    )
    if report_df is None:
        logging.info("No stereochemistry violations found.")
    else:
        logging.info("Output stereochemistry report to %s", output_csv)


@cli.group(name="ccd")
def ccd_cli():
    """
    CCD Options.
    """
    return


@ccd_cli.command(name="update")
def update():
    """
    Update the CCD database.
    """
    setup_ccd.main()


@cli.command(name="gen-input")
@click.option(
    "-i",
    "--input",
    "input_path",
    type=click.Path(path_type=Path, exists=True),
    required=False,
    help="Input file or directory.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    required=False,
    help="Output file or directory.",
)
@click.option(
    "-it",
    "--input-type",
    type=click.Choice(VALID_INPUT_TYPES, case_sensitive=False),
    required=False,
    help="Input type, choices: " + ", ".join(VALID_INPUT_TYPES),
)
@click.option(
    "-ot",
    "--output-type",
    type=click.Choice(VALID_OUTPUT_TYPES, case_sensitive=False),
    required=False,
    help="Output type, choices: " + ", ".join(VALID_OUTPUT_TYPES),
)
@click.option(
    "-I",
    "--interactive",
    is_flag=True,
    help="Run in interactive mode to create an input file.",
)
@click.option(
    "-p",
    "--pdb-ids",
    type=str,
    default=None,
    help=(
        "PDB IDs as a comma-separated string (e.g. '7n0a,7rss') "
        "or a path to a text file containing one PDB ID per line."
        "This option is only applicable when the input is a directory."
        "If not provided, all files in the input directory will be processed."
    ),
)
@click.option(
    "-s",
    "--seeds",
    type=str,
    default=None,
    help=r'Comma-separated seeds, e.g. "0,1,2"; required if --num-seeds is not provided and -ot is "af3".',
)
@click.option(
    "-ns",
    "--num-seeds",
    type=int,
    default=None,
    help=r'Number of seeds; required if --seeds is not provided and -ot is "af3".',
)
@click.option(
    "-a",
    "--assembly-id",
    type=str,
    default=None,
    help="Assembly ID in the input CIF file. Defaults to None. Ignored for non-CIF input types.",
)
@click.option(
    "-n",
    "--num-cpu",
    type=int,
    default=-1,
    help="Number of CPUs to use. Defaults to -1 (all available).",
)
@click.option(
    "--keep_polymer_crosslinks",
    is_flag=True,
    help="Keep polymer-polymer crosslinks (e.g. disulfide bonds, cyclic-peptides) in the bonds list.",
)
@click.option(
    "-rm",
    "--remove-entity-types",
    type=str,
    default=None,
    help=(
        "Comma-separated list of entity types to remove from the input. "
        "Choices: ligand, ion, glycan, protein, dna, rna, covalent_ligand."
    ),
)
@click.option(
    "--reassign-chain-id",
    is_flag=True,
    help="Reassign chain IDs, ignoring original ones from the input file.",
)
def gen_input_cli(
    input_path: Optional[Path],
    output_path: Optional[Path],
    input_type: Optional[str],
    output_type: Optional[str],
    interactive: bool,
    seeds: Optional[str] = None,
    num_seeds: Optional[int] = None,
    assembly_id: Optional[str] = None,
    pdb_ids: Optional[str] = None,
    num_cpu: int = -1,
    keep_polymer_crosslinks: bool = False,
    remove_entity_types: Optional[str] = None,
    reassign_chain_id: bool = False,
):
    """
    Generate model inputs.
    """
    # Handle "None" string from CLI
    assembly_id = str_to_none(assembly_id)
    pdb_ids = str_to_none(pdb_ids)

    if interactive:
        run_interactive_gen()
        return

    if not all([input_path, output_path, input_type, output_type]):
        raise click.UsageError(
            "The following options are required when not in interactive mode: --input, --output, --input-type, --output-type"
        )

    if seeds is not None:
        seeds_lst = [int(x.strip()) for x in seeds.split(",") if x.strip()]
    else:
        seeds_lst = None

    remove_entity_types_lst = None
    if remove_entity_types:
        remove_entity_types_lst = [
            x.strip().lower() for x in remove_entity_types.split(",") if x.strip()
        ]

    run_gen_input(
        input_path,
        output_path,
        input_type,
        output_type,
        seeds=seeds_lst,
        num_seeds=num_seeds,
        assembly_id=assembly_id,
        pdb_ids=pdb_ids,
        num_cpu=num_cpu,
        keep_polymer_crosslinks=keep_polymer_crosslinks,
        remove_entity_types=remove_entity_types_lst,
        use_ori_chain_id=not reassign_chain_id,
    )
