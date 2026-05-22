# PXMeter runtime configuration (RUN_CONFIG)

This document describes the structure and semantics of the global
`RUN_CONFIG` object defined in `pxmeter.configs.run_config`. It explains what
each field controls at runtime and when you may want to change it.

## Table of Contents

- [1. How to override RUN_CONFIG](#1-how-to-override-run-config)
- [2. Mapping configuration (`mapping`)](#2-mapping-configuration-mapping)
- [3. Metric configuration (`metric`)](#3-metric-configuration-metric)
- [4. LDDT-specific configuration (`metric.lddt`)](#4-lddt-specific-configuration-metriclddt)
- [5. Clash-specific configuration (`metric.clashes`)](#5-clash-specific-configuration-metricclashes)
- [6. DockQ-specific configuration (`metric.dockq`)](#6-dockq-specific-configuration-metricdockq)

The default configuration looks like:

```python
RUN_CONFIG = ConfigDict(
    {
        "mapping": {
            "mapping_polymer": True,
            "mapping_ligand": True,
            "res_id_alignments": True,
            "enumerate_all_anchors": True,
            "auto_fix_model_entities": True,
        },
        "metric": {
            "calc_clashes": True,
            "calc_lddt": True,
            "calc_dockq": True,
            "calc_rmsd": True,
            "calc_pb_valid": True,
            "calc_cdr_h3_bb_rmsd": False,
            "penalize_unmapped_reference_atoms": False,
            "lddt": {
                "eps": 1e-6,
                "nucleotide_threshold": 30.0,
                "non_nucleotide_threshold": 15.0,
                "stereochecks": False,
                "calc_backbone_lddt": True,
                "calc_lddt_pli": True,
            },
            "dockq": {
                "exclude_hetatms": True,
            },
            "clashes": {
                "vdw_scale_factor": 0.5,
            },
        },
    }
)
```

You can modify any of these fields either from Python by updating
`RUN_CONFIG` directly, or from the command line via `-C` overrides
(e.g. `-C metric.lddt.eps=1e-5`).

---

## 1. How to override RUN_CONFIG

You can adjust any of the above options from either the command line or
Python.

### 1.1 Command-line overrides (`-C`)

The PXMeter CLI accepts overrides via the `-C` flag:

```bash
pxm -r ref.cif -m model.cif -o result.json \
  -C mapping.res_id_alignments=false \
  -C metric.lddt.eps=1e-5 \
  -C metric.clashes.vdw_scale_factor=0.6
```

Each `-C` argument has the form:

```text
key1.key2.key3=VALUE
```

The key path must correspond to an existing hierarchy in `RUN_CONFIG`.
Values are automatically cast to the type of the existing value
(bool/int/float/string).

### 1.2 Python API

From Python, you can modify `RUN_CONFIG` directly before calling
`evaluate`:

```python
from pxmeter.configs.run_config import RUN_CONFIG
from pxmeter.eval import evaluate

RUN_CONFIG.mapping.res_id_alignments = False
RUN_CONFIG.metric.lddt.nucleotide_threshold = 25.0

result = evaluate(
    ref_cif="ref.cif",
    model_cif="model.cif",
    run_config=RUN_CONFIG,
)
```

This approach is especially useful in batch evaluation scripts, where a
single customised `RUN_CONFIG` can be reused for many evaluations.

---

## 2. Mapping configuration (`mapping`)

The `mapping` section controls how PXMeter builds the correspondence between
reference and model structures: entities, chains, residues, and atoms.

### 2.1 `mapping.mapping_polymer` (bool)

**Default:** `True`

Controls whether PXMeter attempts to map **polymer entities** (proteins,
DNA/RNA, etc.) between reference and model structures.

- When `True`, PXMeter:
  - reads polymer entities and sequences from the mmCIF entity tables,
  - aligns sequences between reference and model entities of the same type,
  - chooses the best mapping based on sequence identity.
- When `False`, PXMeter will not perform entity-level mapping for polymers.
  In most workflows this should stay `True`, otherwise chain matching and
  downstream metrics may become undefined.

**Typical use cases:**

- Keep this enabled unless you have a very specialised pipeline where
  polymer entities are already pre-mapped and you are only interested in
  non-polymer behaviour.

### 2.2 `mapping.mapping_ligand` (bool)

**Default:** `True`

Controls whether PXMeter attempts to map **ligand entities** (non-polymers,
such as small molecules, ions, sugars) between reference and model.

- When `True`, PXMeter:
  - groups non-polymer residues into entities and builds a coarse "CCD
    sequence" for each;
  - first maps ligands via exact CCD sequence matches;
  - then uses chemical similarity (RDKit fingerprints and Tanimoto
    similarity) to map remaining unmatched ligands;
  - constructs atom-level mappings between matched ligand entities.
- When `False`, PXMeter skips ligand entity mapping and atom-level
  ligand alignment. Ligand-centric metrics (e.g. pocket RMSD,
  PoseBusters) may then be unavailable or produce incomplete results.

**Typical use cases:**

- Disable when you know the model has unreliable or intentionally omitted
  ligands and you want to focus purely on polymer geometry.
- Keep enabled for most real-world protein–ligand or protein–cofactor
  evaluations.

### 2.3 `mapping.res_id_alignments` (bool)

**Default:** `True`

Controls how residues are matched between reference and model *within*
corresponding chains.

- When `True` (default):
  - PXMeter matches residues by **residue ID (`res_id`)** directly.
  - This assumes the reference and model sequences are effectively
    aligned already and use consistent numbering.
  - This is stricter but faster and easier to interpret.
- When `False`:
  - PXMeter switches to **sequence alignment–based residue mapping**.
  - Residues are matched according to a global sequence alignment rather
    than exact `res_id` equality.
  - This is more robust when the model uses different numbering or
    contains insertions/deletions relative to the reference.

**Typical use cases:**

- Keep `True` when your model structures are derived by refinement or
  small perturbations from the reference, and residue numbering matches.
- Set to `False` when evaluating models with **different residue
  numbering or truncations**, e.g. de novo predictions, models with
  missing terminal segments, or models built from alternative templates.

### 2.4 `mapping.enumerate_all_anchors` (bool)

**Default:** `True`

Controls how thoroughly PXMeter searches for the best **anchor chain
pair** during chain-level alignment.

PXMeter aligns the reference and model by choosing an anchor chain in
both and using it to initialise the rigid-body transform.

- When `True`:
  - PXMeter enumerates *all compatible reference anchor chains* for the
    chosen model anchor chain.
  - For each candidate anchor pair, it:
    - computes a transform from the reference anchor to the model
      anchor;
    - applies the transform to the full reference structure;
    - evaluates the overall RMSD across mapped chains;
    - picks the anchor that minimises this global RMSD.
  - This is more robust but slightly more expensive.
- When `False`:
  - PXMeter may use a more restricted or heuristic choice of anchors.
  - Alignment may be faster but potentially less stable for ambiguous
    complexes with many similar chains.

**Typical use cases:**

- Keep `True` for general benchmarking where robustness and correctness
  matter more than a small runtime difference.
- Consider `False` only for very large complexes where you need to
  reduce runtime and are confident about the chain mapping.

### 2.5 `mapping.auto_fix_model_entities` (bool)

**Default:** `True`

Controls whether PXMeter attempts to **auto-correct model entity
annotations** when they are inconsistent or suboptimal for mapping.

- When `True`:
  - PXMeter may adjust model entity assignments (e.g. polymer vs
    non-polymer, ligand grouping) to improve compatibility with the
    reference.
  - This helps when model CIFs were generated by tools that assign
    entities in a way that differs from the reference but are still
    chemically equivalent.
- When `False`:
  - PXMeter takes the model entity assignments at face value.
  - This may lead to failed or degraded mapping in the presence of
    inconsistent entity definitions.

**Typical use cases:**

- Keep enabled when working with heterogeneous CIF sources or models
  exported by different toolchains.
- Disable this option only when you are sure the model entity annotations
  are already correct and consistent with the reference (e.g. reference-to-reference self-mapping).

---

## 3. Metric configuration (`metric`)

The `metric` section controls which metrics PXMeter computes and how they
are configured.

### 3.1 Top-level metric switches

These booleans enable or disable whole metric families.

#### 3.1.1 `metric.calc_clashes` (bool)

**Default:** `True`

Enables computation of the **complex-level clash count**, i.e. the
number of atoms in the model involved in severe van der Waals overlaps.

- If `True`, PXMeter:
  - builds a KD-tree over model atoms;
  - finds neighbouring atoms within a cutoff distance;
  - counts atoms participating in overlaps defined by scaled van der
    Waals radii (see `metric.clashes.vdw_scale_factor`).
- If `False`, clash statistics are not computed or reported.

#### 3.1.2 `metric.calc_lddt` (bool)

**Default:** `True`

Enables computation of **LDDT** metrics:

- complex-level LDDT,
- per-chain LDDT,
- per-interface LDDT.

If `False`, all LDDT calculations are skipped, regardless of the
settings in `metric.lddt`.

#### 3.1.3 `metric.calc_dockq` (bool)

**Default:** `True`

Enables computation of **DockQ** interface scores.

- When `True`, PXMeter:
  - writes temporary CIF files for the reference and model with chain IDs
    derived from the internal mapping;
  - calls the DockQ toolkit to compute interface quality metrics;
  - maps DockQ's interface labels back to reference chain IDs.
- When `False`, DockQ is not invoked and no DockQ metrics are reported.

Typical reasons to disable:

- DockQ is not installed or you do not need interface-specific scores.
- You are running in a constrained environment where external tools
  cannot be executed.

#### 3.1.4 `metric.calc_rmsd` (bool)

**Default:** `True`

Enables computation of **RMSD-based metrics**, primarily pocket-aligned
RMSD around ligands when `interested_lig_label_asym_id` is provided.

- When `True`, PXMeter:
  - identifies ligand atoms and nearby backbone atoms (CA/C1′);
  - aligns the model to the reference using pocket atoms;
  - reports ligand and pocket RMSD values.
- When `False`, RMSD calculations for these pockets are skipped.

#### 3.1.5 `metric.calc_pb_valid` (bool)

**Default:** `True`

Enables computation of **PoseBusters-like validity metrics** for selected ligands,
provided that:

- `metric.calc_pb_valid` is `True` *and*
- `interested_lig_label_asym_id` is set when calling `evaluate()`.

- When enabled, PXMeter invokes internal validity checks:
  - builds RDKit molecules for the reference and model ligands;
  - extracts the model environment;
  - runs chemical, geometric, and physical checks and records the per-ligand report.
- When disabled, these checks are not run even if ligand IDs are provided.

Typical reasons to disable:

- You are only interested in geometric metrics (LDDT, RMSD, DockQ,
  clashes) and want to reduce runtime.

#### 3.1.6 `metric.calc_cdr_h3_bb_rmsd` (bool)

**Default:** `False`

Enables computation of **CDR-H3 backbone RMSD** for antibody heavy chains.

- When `True`, PXMeter:
  - annotates protein sequences using ANARCII to identify antibody heavy chains and their CDR-H3 loops;
  - aligns the model heavy chain to the reference using backbone atoms of framework regions (FR1-4);
  - computes the RMSD of backbone atoms in the CDR-H3 loop.
- When `False`, this metric is skipped.

Typical use cases:

- Enable this when benchmarking antibody structure prediction models.
- Keep disabled for general protein complexes to save runtime (avoids ANARCII annotation overhead).

#### 3.1.7 `metric.penalize_unmapped_reference_atoms` (bool)

**Default:** `False`

Controls how unmapped reference atoms (atoms in the reference structure that have no equivalent in the model structure) are treated across various metrics.

- When `True`, PXMeter:
  - Pads the model structure with dummy representations (`0.0` coordinates) for all reference atoms that could not be mapped to the model.
  - Handles these missing atoms in downstream metrics:
    - **LDDT**: Missing atoms fail the distance pair checks, lowering the final LDDT score (as the denominator remains the full reference set).
    - **RMSD (iRMSD/LRMSD)**: Missing atoms are **ignored** during both the structural alignment (superposition) and the final RMSD calculation. This allows you to see the geometric accuracy of the predicted parts without being blocked by `inf` values.
    - **DockQ**: This composite score is penalized via its component **fnat** (Fraction of Native Contacts), where missing residues are treated as having **no contacts**. Consequently, native contacts involving missing residues are marked as "not recovered", lowering the score. Its RMSD components (`iRMSD` and `LRMSD`) follow the same "ignore missing atoms" logic as above.
    - **PoseBusters**: Missing atoms in the **receptor** are ignored (to avoid fake clashes). Missing atoms in the **ligand** cause the ligand to immediately fail all validation checks (`False`).
- When `False`, unmapped reference atoms are simply ignored in all metric calculations. The model is evaluated purely on the subset of atoms it successfully predicted, and the denominators for scores like LDDT/fnat are reduced to only include mapped atoms.

Typical use cases:

- Enable this when you want a strict evaluation that penalizes models for dropping atoms or entire chains from the reference.
- Keep disabled if you only care about the geometric accuracy of the atoms the model *did* produce.

---

## 4. LDDT-specific configuration (`metric.lddt`)

The `metric.lddt` sub-dictionary fine-tunes how LDDT is calculated.

### 4.1 `metric.lddt.eps` (float)

**Default:** `1e-6`

A small numerical epsilon used in internal LDDT computations, typically
as a safeguard in divisions or normalisations to avoid numerical
instabilities.

- In most scenarios you should not need to change this.
- Advanced users may tweak it when experimenting with alternative
  numerical schemes.

### 4.2 `metric.lddt.nucleotide_threshold` (float)

**Default:** `30.0`

Distance inclusion radius (in Å) used when selecting atom pairs involving
**nucleic acid atoms** for LDDT.

- For each nucleic atom in the reference, PXMeter records neighbours
  within this radius.
- Larger values increase the number of pairs (more global context),
  smaller values focus on more local environments.

Typical modifications:

- Decrease for more local LDDT assessments on nucleic acids.
- Increase only for experimental setups where very long-range contacts
  are critical.

### 4.3 `metric.lddt.non_nucleotide_threshold` (float)

**Default:** `15.0`

Distance inclusion radius (in Å) for **non-nucleic atoms** (e.g. protein
atoms) when constructing the LDDT atom pair set.

- For each non-nucleic atom, PXMeter records neighbours within this
  radius.
- As with the nucleotide threshold, larger values consider more
  long-range pairs.

Typical modifications:

- Reduce to emphasise local structural quality for proteins.
- Increase for experiments focused on long-range contacts.

### 4.4 `metric.lddt.stereochecks` (bool)

**Default:** `False`

Controls whether LDDT should **mask out stereochemically invalid atoms**
when computing scores.

- When `False` (default):
  - All atoms are considered for LDDT, regardless of stereochemical
    issues.
- When `True`:
  - PXMeter can use stereochemical validation (via
    `StereoChemValidator`) to exclude atoms that fail basic
    stereochemistry checks from LDDT computations.

Typical use cases:

- Set to `True` when you want LDDT to reflect only stereochemically
  reasonable parts of the model.
- Keep `False` for simpler, fully-inclusive scoring.

### 4.5 `metric.lddt.calc_backbone_lddt` (bool)

**Default:** `True`

Controls whether PXMeter reports an additional **backbone-only LDDT** alongside
the standard all-atom LDDT.

- When `True`:
  - PXMeter computes LDDT on a reduced atom set that focuses on protein and
    nucleic-acid backbones (e.g. Cα for proteins, C3′ for nucleic acids).
  - The resulting scores are exposed under the key `"bb_lddt"` at complex,
    per-chain and per-interface level, in parallel to the all-atom `"lddt"`.
- When `False`:
  - only the standard all-atom LDDT is computed and reported;
  - no `bb_lddt` entries are present in the output.

Typical use cases:

- Keep enabled when you want a backbone-focused quality signal that is less
  sensitive to side-chain modelling.
- Disable when you only care about all-atom LDDT or want to minimise the number
  of reported metrics.

### 4.6 `metric.lddt.calc_lddt_pli` (bool)

**Default:** `True`

Controls whether PXMeter computes the **Protein-Ligand Interface LDDT (LDDT-PLI)** for protein-ligand interactions.

- When `True`:
  - PXMeter computes an LDDT specifically focused on the ligand and its surrounding protein environment.
  - The calculation measures the conservation of interatomic distances between the ligand atoms and nearby protein pocket atoms.
  - The results are reported at the ligand-interface level.
- When `False`:
  - The specific LDDT-PLI calculation is skipped, though standard complex-level or all-atom LDDT computations may still capture ligand atoms based on other settings.

Typical use cases:

- Keep enabled for drug discovery or binding pose evaluations where the local geometry of the ligand binding site is critical.
- Disable if your target is a pure protein/nucleic acid complex and contains no relevant small-molecule ligands, saving computation time.

---

## 5. Clash-specific configuration (`metric.clashes`)

The `metric.clashes` sub-dictionary adjusts the definition of a steric
clash used in the clash metric.

### 5.1 `metric.clashes.vdw_scale_factor` (float)

**Default:** `0.5`

Scaling factor applied to the sum of van der Waals (vdW) radii when
classifying a pair of atoms as clashing.

For a pair of atoms *i* and *j* with vdW radii `r_i` and `r_j`, PXMeter
considers them to be in clash if:

```text
d < vdw_scale_factor * (r_i + r_j)
```

where `d` is the inter-atomic distance.

- Smaller values (e.g. `0.4`) make the criterion **stricter**, flagging
  only more severe overlaps.
- Larger values (e.g. `0.6` or `0.7`) are more permissive and will mark
  more contacts as clashes.

Typical modifications:

- Decrease when you only want to count the most egregious steric
  problems.
- Increase when you want a more sensitive clash count that flags mild
  overlaps as well.

---

## 6. DockQ-specific configuration (`metric.dockq`)

The `metric.dockq` sub-dictionary adjusts how DockQ scores are computed.

### 6.1 `metric.dockq.exclude_hetatms` (bool)

**Default:** `True`

Controls whether HETATM atoms (as defined in the reference mmCIF) are excluded
before computing DockQ.

- When `True` (default):
  - PXMeter filters out all HETATM atoms from both reference and model before
    running the DockQ alignment and scoring.
  - This matches the "native" DockQ behavior. However, note that some non-standard
    residues in peptides (e.g., Chain C of PDB 9CY4) are labeled as HETATM in
    mmCIF and will be excluded, which might not be desired for peptide-protein
    interfaces.
- When `False`:
  - HETATM atoms are kept during the DockQ calculation.
  - This is useful if you want to include non-standard residues or certain
    non-polymer components in the interface assessment (e.g., to correctly
    evaluate interfaces involving peptides with modified residues).
