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
import os
import subprocess as sp
from pathlib import Path

import gemmi

logging.basicConfig(level=logging.INFO)


def download_ccd_cif(output_path: Path):
    """
    Download the CCD CIF file from rcsb.org.

    Args:
        output_path (Path): The output path for saving the downloaded CCD CIF file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Downloading CCD CIF file from rcsb.org ...")

    output_cif_gz = output_path / "components.cif.gz"
    if output_cif_gz.exists():
        logging.info("Remove old zipped CCD CIF file: %s", output_cif_gz)
        output_cif_gz.unlink()

    output_cif = output_cif_gz.with_suffix("")
    if output_cif.exists():
        logging.info("Remove old CCD CIF file: %s", output_cif)
        output_cif.unlink()

    sp.run(
        f"wget https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz -P {output_path}",
        shell=True,
        check=True,
    )

    sp.run(f"gunzip -d {output_cif_gz}", shell=True, check=True)

    logging.info("Download CCD CIF file successfully: %s", output_cif)


def make_one_letter_code_json_from_ccd(components_file: Path, output_json: Path):
    """
    Make a one-letter code JSON file from the CCD CIF file.

    Args:
        components_file (Path): The path to the CCD CIF file.
    """
    ccd_cif = gemmi.cif.read(str(components_file))

    ccd_code_to_one_letter_code = {}
    for block in ccd_cif:
        ccd_code = block.find_value("_chem_comp.id")
        one_letter_code = block.find_value("_chem_comp.one_letter_code")
        if one_letter_code is None or one_letter_code == "?":
            continue
        ccd_code_to_one_letter_code[ccd_code] = one_letter_code

    with open(output_json, "w") as f:
        json.dump(ccd_code_to_one_letter_code, f, indent=4)

    logging.info("Make ONE_LETTER_CODE_JSON successfully: %s", output_json)


# default is <repo_dir>/ccd_cache/components.cif Your path for components file
# You can change this path to your own path for components file by setting the environment variable:
# export PXM_CCD_FILE=<your_path>
repo_dir = Path(__file__).absolute().parent.parent.parent
ccd_file_in_repo = repo_dir / "ccd_cache" / "components.cif"
COMPONENTS_FILE = Path(os.environ.get("PXM_CCD_FILE", ccd_file_in_repo))
ONE_LETTER_CODE_JSON = COMPONENTS_FILE.with_suffix(".json")

if not COMPONENTS_FILE.exists():
    logging.debug(
        "CCD CIF file not found. Downloading CCD CIF file to %s", COMPONENTS_FILE.parent
    )
    download_ccd_cif(output_path=COMPONENTS_FILE.parent)
    make_one_letter_code_json_from_ccd(COMPONENTS_FILE, ONE_LETTER_CODE_JSON)
else:
    logging.debug("Load CCD CIF file from: %s", COMPONENTS_FILE)

if not ONE_LETTER_CODE_JSON.exists():
    make_one_letter_code_json_from_ccd(COMPONENTS_FILE, ONE_LETTER_CODE_JSON)

logging.debug("Load CCD one-letter code from: %s", ONE_LETTER_CODE_JSON)
with open(ONE_LETTER_CODE_JSON, "r") as f:
    CCD_ONE_LETTER_CODE = json.load(f)
