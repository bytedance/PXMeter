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

import copy
import functools
import gzip
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Sequence, Union

import biotite.structure.io.pdbx as pdbx
import numpy as np
import pandas as pd
from biotite.structure import (
    AtomArray,
    AtomArrayStack,
    get_chain_starts,
    get_residue_starts,
)
from biotite.structure.io.pdbx.component import MaskValue
from biotite.structure.io.pdbx.convert import (
    _apply_transformations,
    _get_block,
    _get_transformations,
    _parse_operation_expression,
    concatenate,
    get_structure,
    InvalidFileError,
)

from pxmeter.constants import (
    DNA,
    DNA_STD_RESIDUES,
    PRO_STD_RESIDUES,
    PROTEIN,
    RNA,
    RNA_STD_RESIDUES,
)
from pxmeter.data.ccd import get_ccd_one_letter_code
from pxmeter.data.utils import is_valid_date_format


def get_assembly(
    pdbx_file: Union[pdbx.CIFFile, pdbx.CIFBlock],
    assembly_id: Optional[str] = None,
    model: Optional[int] = None,
    data_block: Optional[str] = None,
    altloc: str = "first",
    extra_fields: Optional[list[str]] = None,
    use_author_fields: bool = True,
    include_bonds: bool = False,
) -> Union[AtomArray, AtomArrayStack]:
    """
    This code is copied from a Biotite fix commit, as Biotite versions between
    v1.2.0 and v1.5.0 lose inter-chain bonds in the get_assembly function.
    https://github.com/biotite-dev/biotite/issues/838

    Build the given biological assembly.

    This function receives the data from the
    ``pdbx_struct_assembly_gen``, ``pdbx_struct_oper_list`` and
    ``atom_site`` categories in the file.
    Consequently, these categories must be present in the file.

    Parameters
    ----------
    pdbx_file : Union[pdbx.CIFFile, pdbx.CIFBlock]
        The file object.
    assembly_id : Optional[str]
        The assembly to build.
        Available assembly IDs can be obtained via
        :func:`list_assemblies()`.
    model : Optional[int]
        If this parameter is given, the function will return an
        :class:`AtomArray` from the atoms corresponding to the given
        model number (starting at 1).
        Negative values are used to index models starting from the last
        model insted of the first model.
        If this parameter is omitted, an :class:`AtomArrayStack`
        containing all models will be returned, even if the structure
        contains only one model.
    data_block : Optional[str]
        The name of the data block.
        Default is the first (and most times only) data block of the
        file.
        If the data block object is passed directly to `pdbx_file`,
        this parameter is ignored.
    altloc : str
        This parameter defines how *altloc* IDs are handled:
            - ``'first'`` - Use atoms that have the first *altloc* ID
              appearing in a residue.
            - ``'occupancy'`` - Use atoms that have the *altloc* ID
              with the highest occupancy for a residue.
            - ``'all'`` - Use all atoms.
              Note that this leads to duplicate atoms.
              When this option is chosen, the ``altloc_id`` annotation
              array is added to the returned structure.
    extra_fields : Optional[list[str]]
        The strings in the list are entry names, that are
        additionally added as annotation arrays.
        The annotation category name will be the same as the PDBx
        subcategory name.
        The array type is always `str`.
        An exception are the special field identifiers:
        ``'atom_id'``, ``'b_factor'``, ``'occupancy'`` and ``'charge'``.
        These will convert the fitting subcategory into an
        annotation array with reasonable type.
    use_author_fields : bool
        Some fields can be read from two alternative sources,
        for example both, ``label_seq_id`` and ``auth_seq_id`` describe
        the ID of the residue.
        While, the ``label_xxx`` fields can be used as official pointers
        to other categories in the file, the ``auth_xxx``
        fields are set by the author(s) of the structure and are
        consistent with the corresponding values in PDB files.
        If `use_author_fields` is true, the annotation arrays will be
        read from the ``auth_xxx`` fields (if applicable),
        otherwise from the the ``label_xxx`` fields.
    include_bonds : bool
        If set to true, a :class:`BondList` will be created for the
        resulting :class:`AtomArray` containing the bond information
        from the file.
        Inter-residue bonds, will be read from the ``struct_conn``
        category.
        Intra-residue bonds will be read from the ``chem_comp_bond``, if
        available, otherwise they will be derived from the Chemical
        Component Dictionary.

    Returns
    -------
    Union[AtomArray, AtomArrayStack]
        The assembly.
        The return type depends on the `model` parameter.
        Contains the `sym_id` annotation, which enumerates the copies of the asymmetric
        unit in the assembly.

    Raises
    ------
    InvalidFileError
        If the file has no 'pdbx_struct_assembly_gen' or 'pdbx_struct_oper_list' category.
    KeyError
        If the Assembly ID is not present in the file.

    Examples
    --------
    >>> import os.path
    >>> file = CIFFile.read(os.path.join(path_to_structures, "1f2n.cif"))
    >>> assembly = get_assembly(file, model=1)
    """
    block = _get_block(pdbx_file, data_block)

    try:
        assembly_gen_category = block["pdbx_struct_assembly_gen"]
    except KeyError:
        raise InvalidFileError("File has no 'pdbx_struct_assembly_gen' category")

    try:
        struct_oper_category = block["pdbx_struct_oper_list"]
    except KeyError:
        raise InvalidFileError("File has no 'pdbx_struct_oper_list' category")

    assembly_ids = assembly_gen_category["assembly_id"].as_array(str)
    if assembly_id is None:
        assembly_id = assembly_ids[0]
    elif assembly_id not in assembly_ids:
        raise KeyError(f"File has no Assembly ID '{assembly_id}'")

    # Calculate all possible transformations
    transformations = _get_transformations(struct_oper_category)

    # Get structure according to additional parameters
    # Include 'label_asym_id' as annotation array
    # for correct asym ID filtering
    extra_fields = [] if extra_fields is None else extra_fields
    if "label_asym_id" in extra_fields:
        extra_fields_and_asym = extra_fields
    else:
        # The operations apply on asym IDs
        # -> they need to be included to select the correct atoms
        extra_fields_and_asym = extra_fields + ["label_asym_id"]
    structure = get_structure(
        pdbx_file,
        model,
        data_block,
        altloc,
        extra_fields_and_asym,
        use_author_fields,
        include_bonds,
    )

    # Get transformations and apply them to the affected asym IDs
    sub_assemblies = []
    sym_id_counter = Counter()
    for id, op_expr, asym_id_expr in zip(
        assembly_gen_category["assembly_id"].as_array(str),
        assembly_gen_category["oper_expression"].as_array(str),
        assembly_gen_category["asym_id_list"].as_array(str),
    ):
        # Find the operation expressions for given assembly ID
        # We already asserted that the ID is actually present
        if id == assembly_id:
            chain_ids = asym_id_expr.split(",")
            operations = _parse_operation_expression(op_expr)
            sub_structure = structure[..., np.isin(structure.label_asym_id, chain_ids)]
            sub_assembly = _apply_transformations(
                sub_structure, transformations, operations
            )

            # Add 'sym_id' annotation, that should increment for each 'oper_expression'
            sub_assembly.set_annotation(
                "sym_id",
                np.repeat(np.arange(len(operations)), sub_structure.array_length()),
            )
            # As different 'oper_expression's may comprise different chains
            # increment the 'sym_id' individually for each chain
            for chain_id in chain_ids:
                chain_mask = sub_assembly.label_asym_id == chain_id
                sub_assembly.sym_id[chain_mask] += sym_id_counter[chain_id]
                sym_id_counter[chain_id] += len(operations)

            sub_assemblies.append(sub_assembly)
    assembly = concatenate(sub_assemblies)

    # Sort AtomArray or AtomArrayStack by 'sym_id'
    max_sym_id = assembly.sym_id.max()
    assembly = concatenate(
        [assembly[..., assembly.sym_id == sym_id] for sym_id in range(max_sym_id + 1)]
    )

    # Remove 'label_asym_id', if it was not included in the original
    # user-supplied 'extra_fields'
    if "label_asym_id" not in extra_fields:
        assembly.del_annotation("label_asym_id")

    return assembly


class MMCIFParser:
    """
    Parsing and extracting data from MMCIF files.

    Args:
        mmcif_file (Path or str): The path to the mmCIF file to be parsed.
                                       It can be either a Path object or a string.
    """

    def __init__(self, mmcif_file: Union[Path, str]):
        self.mmcif_file = mmcif_file
        self.cif = self._parse(mmcif_file)

    def _parse(self, mmcif_file: Union[str, Path]) -> pdbx.CIFFile:
        """
        Parses a given mmCIF file and returns a CIFFile object.

        Args:
            mmcif_file (str or Path): The path to the mmCIF file. The file can be
                                           either a plain text file or a gzip-compressed file.

        Returns:
            pdbx.CIFFile: The parsed CIFFile object.
        """
        mmcif_file = Path(mmcif_file)
        if mmcif_file.suffix == ".gz":
            with gzip.open(mmcif_file, "rt") as f:
                cif_file = pdbx.CIFFile.read(f)
        else:
            with open(mmcif_file, "rt") as f:
                cif_file = pdbx.CIFFile.read(f)
        return cif_file

    def get_category_table(self, name: str) -> Optional[pd.DataFrame]:
        """
        Retrieve a category table from the CIF block as a pandas DataFrame.

        Args:
            name (str): The name of the category to retrieve.

        Returns:
            pd.DataFrame or None: A DataFrame containing the category data if the category exists,
                                    otherwise None.
        """
        if name not in self.cif.block:
            return None
        category = self.cif.block[name]
        category_dict = {k: column.as_array() for k, column in category.items()}
        return pd.DataFrame(category_dict, dtype=str)

    @functools.cached_property
    def entity_poly_type(self) -> dict[str, str]:
        """
        Ref: https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_entity_poly.type.html
        Map entity_id to entity_poly_type.

        Allowed Value:
        · cyclic-pseudo-peptide
        · other
        · peptide nucleic acid
        · polydeoxyribonucleotide
        · polydeoxyribonucleotide/polyribonucleotide hybrid
        · polypeptide(D)
        · polypeptide(L)
        · polyribonucleotide

        Returns:
            Dict: A dict of label_entity_id --> entity_poly_type.
        """
        entity_poly = self.get_category_table("entity_poly")
        if (
            entity_poly is not None
            and "type" in entity_poly.columns
            and "entity_id" in entity_poly.columns
        ):
            return {i: t for i, t in zip(entity_poly.entity_id, entity_poly.type)}

        # Fallback: Infer from atom_site if _entity_poly.type is missing
        atom_site = self.get_category_table("atom_site")
        if atom_site is None:
            return {}

        entity = self.get_category_table("entity")
        assert (
            entity is not None and "type" in entity.columns
        ), "Both _entity_poly.type and _entity.type are missing. Cannot infer entity types."

        entity_types = {}
        if entity is not None and "id" in entity.columns and "type" in entity.columns:
            entity_types = {i: t for i, t in zip(entity.id, entity.type)}

        poly_types = {}
        for entity_id in atom_site["label_entity_id"].unique():
            if entity_types.get(entity_id) == "non-polymer":
                # We don't include non-polymer in entity_poly_type mapping
                continue

            res_names = atom_site[atom_site["label_entity_id"] == entity_id][
                "label_comp_id"
            ].unique()

            pro_count = sum(1 for res in res_names if res in PRO_STD_RESIDUES)
            rna_count = sum(1 for res in res_names if res in RNA_STD_RESIDUES)
            dna_count = sum(1 for res in res_names if res in DNA_STD_RESIDUES)

            max_count = max(pro_count, rna_count, dna_count)
            if max_count == 0:
                continue  # Or infer as other, but usually if 0 it might be ligand or unstandard polymer
            elif max_count == pro_count:
                poly_types[entity_id] = PROTEIN
            elif max_count == rna_count:
                poly_types[entity_id] = RNA
            elif max_count == dna_count:
                poly_types[entity_id] = DNA

        return poly_types

    @functools.cached_property
    def exptl_methods(self) -> list[str]:
        """
        The methods to get the structure.

        Most of the time, methods only has one method, such as 'X-RAY DIFFRACTION'.
        But some entries have multi methods, such as ['X-RAY DIFFRACTION', 'NEUTRON DIFFRACTION'].

        Allowed Values:
        https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_exptl.method.html

        Returns:
            list[str]: such as ['X-RAY DIFFRACTION'], ['ELECTRON MICROSCOPY'], ['SOLUTION NMR', 'THEORETICAL MODEL'],
                ['X-RAY DIFFRACTION', 'NEUTRON DIFFRACTION'], ['ELECTRON MICROSCOPY', 'SOLUTION NMR'], etc.
        """
        if "exptl" not in self.cif.block:
            return []
        else:
            methods = self.cif.block["exptl"]["method"]
            return methods.as_array()

    @functools.cached_property
    def entry_id(self) -> str:
        """
        Retrieves the entry ID from the CIF block.

        Returns:
            str: The entry ID in lowercase if it exists, otherwise an empty string.
        """
        if "entry" not in self.cif.block:
            return ""
        else:
            return self.cif.block["entry"]["id"].as_item().lower()

    @staticmethod
    def _replace_auth_with_label(atom_array: AtomArray) -> AtomArray:
        """
        Replace chain_id with label_chain_id.
        Replace res_id with label_seq_id (ligand res_id is reset by order in chain).

        Args:
            atom_array (AtomArray): Biotite AtomArray object.

        Returns:
            AtomArray: Biotite AtomArray object with reset chain_id and res_id.
        """
        atom_array.chain_id = atom_array.label_asym_id

        # reset ligand res_id
        res_id = copy.deepcopy(atom_array.label_seq_id)
        chain_starts = get_chain_starts(atom_array, add_exclusive_stop=True)
        for chain_start, chain_stop in zip(chain_starts[:-1], chain_starts[1:]):
            if atom_array.label_seq_id[chain_start] != ".":
                continue
            else:
                res_starts = get_residue_starts(
                    atom_array[chain_start:chain_stop], add_exclusive_stop=True
                )
                num = 1
                for res_start, res_stop in zip(res_starts[:-1], res_starts[1:]):
                    res_id[chain_start:chain_stop][res_start:res_stop] = num
                    num += 1

        atom_array.res_id = res_id.astype(int)
        return atom_array

    @staticmethod
    def res_names_to_sequence(res_names: Sequence[str]) -> str:
        """
        Convert res_names to sequences {chain_id: canonical_sequence} based on CCD.add()

        Args:
            res_names (list[str]): list of res_names, e.g. ["CYS", "ALA", "GLY"]

        Return
            str: Canonical sequence.
        """
        seq = ""
        for res_name in res_names:
            one = get_ccd_one_letter_code(res_name)
            one = "X" if one is None or one == "?" or len(one) > 1 else one
            seq += one
        return seq

    def get_poly_res_names(
        self, atom_array: Optional[AtomArray] = None
    ) -> dict[str, list[str]]:
        """
        Get 3-letter residue names by combining mmcif._entity_poly_seq and AtomArray.

        If AtomArray is None: keep first altloc residue of the same res_id based in mmcif._entity_poly_seq
        If AtomArray is provided: keep same residue of AtomArray.

        Returns
            dict[str, list[str]]: label_entity_id of polymers to [res_names], eg: {"1": ["ALA", "GLY", "CYS"]}
        """
        entity_res_names = defaultdict(dict)
        if atom_array is not None:
            # build entity_id -> res_id -> res_name for input atom array
            res_starts = get_residue_starts(atom_array, add_exclusive_stop=False)
            for start in res_starts:
                entity_id = atom_array.label_entity_id[start]
                res_id = atom_array.res_id[start]
                res_name = atom_array.res_name[start]
                entity_res_names[entity_id][res_id] = res_name

        poly_entity_id_to_res_names = {}

        # build reference entity atom array, including missing residues
        entity_poly_seq = self.get_category_table("entity_poly_seq")
        if entity_poly_seq is None:
            # If entity_poly_seq is missing, fallback to entity_res_names inferred from atom_site
            for entity_id, res_dict in entity_res_names.items():
                if entity_id in self.entity_poly_type:
                    # Sort by res_id to keep sequence order
                    sorted_res_ids = sorted(res_dict.keys())
                    poly_entity_id_to_res_names[entity_id] = [
                        res_dict[res_id] for res_id in sorted_res_ids
                    ]
            return poly_entity_id_to_res_names

        polymer_entities = set(entity_poly_seq["entity_id"])
        for entity_id in polymer_entities:
            chain_mask = entity_poly_seq.entity_id == entity_id
            seq_mon_ids = entity_poly_seq.mon_id[chain_mask].to_numpy(dtype=str)

            seq_nums = entity_poly_seq.num[chain_mask].to_numpy(dtype=int)

            if np.unique(seq_nums).size == seq_nums.size:
                # no altloc residues
                poly_entity_id_to_res_names[entity_id] = seq_mon_ids
                continue

            # filter altloc residues, eg: 181 ALA (altloc A); 181 GLY (altloc B)
            select_mask = np.zeros(len(seq_nums), dtype=bool)
            matching_res_id = seq_nums[0]
            for i, res_id in enumerate(seq_nums):
                if res_id != matching_res_id:
                    continue

                res_name_in_atom_array = entity_res_names.get(entity_id, {}).get(res_id)
                if res_name_in_atom_array is None:
                    # res_name is mssing in atom_array,
                    # keep first altloc residue of the same res_id
                    select_mask[i] = True
                else:
                    # keep match residue to atom_array
                    if res_name_in_atom_array == seq_mon_ids[i]:
                        select_mask[i] = True

                if select_mask[i]:
                    matching_res_id += 1

            seq_mon_ids = seq_mon_ids[select_mask]
            seq_nums = seq_nums[select_mask]
            assert len(seq_nums) == max(seq_nums)
            poly_entity_id_to_res_names[entity_id] = seq_mon_ids
        return poly_entity_id_to_res_names

    def get_entity_poly_seq(self, atom_array=None) -> dict:
        """
        Get sequence by combining mmcif._entity_poly_seq and atom_array

        if ref_atom_array is None: keep first altloc residue of the same res_id based in mmcif._entity_poly_seq
        if ref_atom_array is provided: keep same residue of atom_array.

        Return
            Dict{str:str}: label_entity_id --> canonical_sequence
        """
        sequences = {}
        for entity_id, res_names in self.get_poly_res_names(atom_array).items():
            seq = self.res_names_to_sequence(res_names)
            sequences[entity_id] = seq
        return sequences

    @staticmethod
    def filter_first_and_specified_altloc(
        atom_array: AtomArray, spec_altloc: str = "A"
    ) -> AtomArray:
        """
        Filters atoms in an AtomArray based on alternate location identifiers (altloc).

        This function performs two main filtering operations:
        1. Filters out all atoms that do not have an alternate location identifier (altloc).
        2. For each residue, it retains atoms with the first encountered altloc ID. If a specified
           altloc ID is provided and exists within the residue, it retains atoms with that altloc ID
           instead.

        Args:
            atom_array (AtomArray): The array of atoms to be filtered, which includes altloc identifiers.
            spec_altloc (str, optional): The specified alternate location identifier to retain
                                         if it exists within a residue. Default is "A".
        Returns:
            AtomArray: A new AtomArray containing only the atoms that pass the filtering criteria.
        """
        altloc_ids = atom_array.label_alt_id
        # Filter all atoms without altloc code
        altloc_filter = np.in1d(altloc_ids, [".", "?", " ", ""])

        # And filter all atoms for each residue with the first altloc ID
        residue_starts = get_residue_starts(atom_array, add_exclusive_stop=True)
        for start, stop in zip(residue_starts[:-1], residue_starts[1:]):
            letter_altloc_ids = [l for l in altloc_ids[start:stop] if l.isalpha()]
            if len(letter_altloc_ids) > 0:
                first_id = letter_altloc_ids[0]
                if spec_altloc in letter_altloc_ids:
                    altloc_filter[start:stop] |= altloc_ids[start:stop] == spec_altloc
                else:
                    altloc_filter[start:stop] |= altloc_ids[start:stop] == first_id
            else:
                # No altloc ID in this residue -> Nothing to do
                pass

        return atom_array[altloc_filter]

    def get_structure(
        self,
        model: int = 1,
        altloc: str = "first",
        assembly_id: Optional[str] = None,
        include_bonds: bool = True,
    ) -> AtomArray:
        """
        Get an AtomArray created by bioassembly of MMCIF.

        Args:
            model (int): The model number of the structure.
            altloc (str): "all", "first", "occupancy", "A", "B", etc.
            assembly_id (str, optional): The Assembly id of the structure.
            include_bonds (bool): Whether to include bonds in the AtomArray. Defaults to True.

        Returns:
            AtomArray: Biotite AtomArray object created by bioassembly of MMCIF.
        """
        extra_fields = ["label_asym_id", "label_entity_id", "auth_asym_id"]  # Chain
        extra_fields += ["label_seq_id", "auth_seq_id"]  # Residue
        atom_site_fields = {
            "occupancy": "occupancy",
            "pdbx_formal_charge": "charge",
            "B_iso_or_equiv": "b_factor",
            "label_alt_id": "label_alt_id",
        }  # Atom
        for atom_site_name, alt_name in atom_site_fields.items():
            if atom_site_name in self.cif.block["atom_site"]:
                extra_fields.append(alt_name)

        if altloc in ["all", "first", "occupancy"]:
            tmp_altloc = altloc
        else:
            # First obtain all altlocs, then filter them
            tmp_altloc = "all"

        if tmp_altloc == "all":
            logging.warning(
                "Bond computation is not supported with `altloc='all'`. "
                "The include_bonds will be set to False."
            )
            include_bonds = False

        if assembly_id is None:
            atom_array = pdbx.get_structure(
                pdbx_file=self.cif,
                model=model,
                altloc=tmp_altloc,
                extra_fields=extra_fields,
                use_author_fields=True,
                include_bonds=include_bonds,
            )
        else:
            atom_array = get_assembly(
                pdbx_file=self.cif,
                assembly_id=assembly_id,
                model=model,
                altloc=tmp_altloc,
                extra_fields=extra_fields,
                use_author_fields=True,
                include_bonds=include_bonds,
            )

        if altloc not in ["all", "first", "occupancy"]:
            # "A", "B", etc.
            atom_array = self.filter_first_and_specified_altloc(atom_array, altloc)

        # Use label_seq_id to match seq and structure
        atom_array = self._replace_auth_with_label(atom_array)
        return atom_array

    @staticmethod
    def atom_array_to_atom_site(
        array: AtomArray, extra_fields: Optional[list[str]] = None
    ):
        """
        Convert an AtomArray to a PDBx category "atom_site".
        Those codes copied from `biotite.structure.io.pdbx.convert.set_structure`.

        Args:
            array (AtomArray): The array to be converted.
            extra_fields (list[str]): Additional fields to be included in the output.
                                      Defaults to an empty list.
        Returns:
            pdbx.CIFCategory: A PDBx category "atom_site" containing the converted data.
        """
        if extra_fields is None:
            extra_fields = []

        pdbx.convert._check_non_empty(array)

        atom_site = pdbx.CIFCategory(name="atom_site")
        Column = atom_site.subcomponent_class()
        atom_site["group_PDB"] = np.where(array.hetero, "HETATM", "ATOM")
        atom_site["type_symbol"] = np.copy(array.element)
        atom_site["label_atom_id"] = np.copy(array.atom_name)
        atom_site["label_alt_id"] = Column(
            # AtomArrays do not store altloc atoms
            np.full(array.array_length(), "."),
            np.full(array.array_length(), MaskValue.INAPPLICABLE),
        )
        atom_site["label_comp_id"] = np.copy(array.res_name)
        atom_site["label_asym_id"] = np.copy(array.chain_id)
        atom_site["label_entity_id"] = (
            np.copy(array.label_entity_id)
            if "label_entity_id" in array.get_annotation_categories()
            else pdbx.convert._determine_entity_id(array.chain_id)
        )
        atom_site["label_seq_id"] = np.copy(array.res_id)
        atom_site["pdbx_PDB_ins_code"] = Column(
            np.copy(array.ins_code),
            np.where(array.ins_code == "", MaskValue.INAPPLICABLE, MaskValue.PRESENT),
        )
        atom_site["auth_seq_id"] = atom_site["label_seq_id"]
        atom_site["auth_comp_id"] = atom_site["label_comp_id"]
        atom_site["auth_asym_id"] = atom_site["label_asym_id"]
        atom_site["auth_atom_id"] = atom_site["label_atom_id"]

        annot_categories = array.get_annotation_categories()
        if "atom_id" in annot_categories:
            atom_site["id"] = np.copy(array.atom_id)
        if "b_factor" in annot_categories:
            atom_site["B_iso_or_equiv"] = np.copy(array.b_factor)
        if "occupancy" in annot_categories:
            atom_site["occupancy"] = np.copy(array.occupancy)
        if "charge" in annot_categories:
            atom_site["pdbx_formal_charge"] = Column(
                np.array([f"{c:+d}" if c != 0 else "?" for c in array.charge]),
                np.where(array.charge == 0, MaskValue.MISSING, MaskValue.PRESENT),
            )

        # Handle all remaining custom fields
        if len(extra_fields) > 0:
            # ... check to avoid clashes with standard annotations
            _standard_annotations = [
                "hetero",
                "element",
                "atom_name",
                "res_name",
                "chain_id",
                "res_id",
                "ins_code",
                "atom_id",
                "b_factor",
                "occupancy",
                "charge",
            ]
            _reserved_annotation_names = list(atom_site.keys()) + _standard_annotations

            for annot in extra_fields:
                if annot in _reserved_annotation_names:
                    raise ValueError(
                        f"Annotation name '{annot}' is reserved and cannot be written to as extra field. "
                        "Please choose another name."
                    )
                atom_site[annot] = np.copy(array.get_annotation(annot))

        # In case of a single model handle each coordinate
        # simply like a flattened array
        if isinstance(array, AtomArray) or (
            isinstance(array, AtomArrayStack) and array.stack_depth() == 1
        ):
            # 'ravel' flattens coord without copy
            # in case of stack with stack_depth = 1
            atom_site["Cartn_x"] = np.copy(np.ravel(array.coord[..., 0]))
            atom_site["Cartn_y"] = np.copy(np.ravel(array.coord[..., 1]))
            atom_site["Cartn_z"] = np.copy(np.ravel(array.coord[..., 2]))
            atom_site["pdbx_PDB_model_num"] = np.ones(
                array.array_length(), dtype=np.int32
            )
        # In case of multiple models repeat annotations
        # and use model specific coordinates
        else:
            atom_site = pdbx.convert._repeat(atom_site, array.stack_depth())
            coord = np.reshape(
                array.coord, (array.stack_depth() * array.array_length(), 3)
            )
            atom_site["Cartn_x"] = np.copy(coord[:, 0])
            atom_site["Cartn_y"] = np.copy(coord[:, 1])
            atom_site["Cartn_z"] = np.copy(coord[:, 2])
            atom_site["pdbx_PDB_model_num"] = np.repeat(
                np.arange(1, array.stack_depth() + 1, dtype=np.int32),
                repeats=array.array_length(),
            )
        if "atom_id" not in annot_categories:
            # Count from 1
            atom_site["id"] = np.arange(1, len(atom_site["group_PDB"]) + 1)
        return atom_site

    @staticmethod
    def get_release_date(cif_block: dict) -> str:
        """
        Get first release date.

        Args:
            cif_block (dict): mmcif_io.MMCIFParser dictionary.

        Returns:
            str: yyyy-mm-dd
        """

        if "pdbx_audit_revision_history" in cif_block:
            history = cif_block["pdbx_audit_revision_history"]
            # np.str_ is inherit from str, so return is str
            date = history["revision_date"].as_array()[0]
        else:
            # no release date
            date = "9999-12-31"

        valid_date = is_valid_date_format(date)
        assert (
            valid_date
        ), f"Invalid date format: {date}, it should be yyyy-mm-dd format"
        return date

    @staticmethod
    def get_resolution(cif_block: dict) -> float:
        """
        Get resolution for X-ray and cryoEM.
        Some methods don't have resolution, set as -1.0.

        Args:
            cif_block (dict): mmcif_io.MMCIFParser dictionary.

        Returns:
            float: resolution (set to -1.0 if not found)
        """
        resolution_names = [
            "refine.ls_d_res_high",
            "em_3d_reconstruction.resolution",
            "reflns.d_resolution_high",
        ]
        for category_item in resolution_names:
            category, item = category_item.split(".")
            if category in cif_block and item in cif_block[category]:
                try:
                    resolution = cif_block[category][item].as_array(float)[0]
                    # "." will be converted to 0.0, but it is not a valid resolution.
                    if resolution == 0.0:
                        continue
                    return resolution
                except ValueError:
                    # in some cases, resolution_str is "?"
                    continue
            else:
                continue
        return -1.0
