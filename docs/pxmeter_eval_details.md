# PXMeter evaluation pipeline (runtime logic)

## Table of Contents

- [1. High‑level overview](#1-high-level-overview)
- [2. Inputs and default configuration (what the user provides)](#2-inputs-and-default-configuration-what-the-user-provides)
- [3. Loading the structures](#3-loading-the-structures)
- [4. Normalising and cleaning the structures](#4-normalising-and-cleaning-the-structures)
- [5. Mapping entities and ligands](#5-mapping-entities-and-ligands)
- [6. Aligning chains between reference and model](#6-aligning-chains-between-reference-and-model)
- [7. Resolving symmetry at residue and atom level](#7-resolving-symmetry-at-residue-and-atom-level)
- [8. Computing evaluation metrics](#8-computing-evaluation-metrics)
- [9. Assembling the result and JSON output](#9-assembling-the-result-and-json-output)
- [10. Summary](#10-summary)

> Entry point: `pxmeter.eval.evaluate()`

This document describes what actually happens at runtime when
PXMeter evaluates a prediction against a reference structure, using the default
configuration.

---

## 1. High‑level overview

At a high level, a single call to `evaluate(ref_cif, model_cif, ...)` performs
these stages:

1. **Load the reference and model structures** from mmCIF files.
2. **Normalise and clean** both structures (fix common quirks, remove water,
   hydrogens, crystallisation additives, etc.).
3. **Build a geometrically aligned mapping** between reference and model:
   - map polymer and ligand *entities* (sequence / chemistry level),
   - map *chains* within each entity,
   - resolve *residue‑level* permutations in branched ligands,
   - resolve *atom‑level* permutations for symmetric atoms.
4. **Construct aligned working copies** of the reference and model, containing
   only mutually corresponding atoms in a consistent order.
5. **Compute a set of structural metrics** on these aligned structures:
   - van der Waals clashes (complex level),
   - pocket‑aligned RMSD around selected ligands (optional),
   - LDDT for the full complex, individual chains, and chain interfaces,
   - DockQ for chain interfaces,
   - PoseBusters validity for selected ligands (optional).
6. **Package all results** into a structured object (`MetricResult`) which can
   be converted to JSON.

By default, all mapping steps are enabled. Most metrics are computed, but
ligand‑specific metrics (pocket RMSD, PoseBusters) are only computed when
`interested_lig_label_asym_id` is provided.

---

## 2. Inputs and default configuration (what the user provides)

At runtime the user typically calls `evaluate()` with:

- `ref_cif` (required): path to the **reference** mmCIF file.
- `model_cif` (required): path to the **model** mmCIF file to be evaluated.
- `ref_model` (default `1`): which model to use from the reference mmCIF.
- `ref_assembly_id` (default `None`): if set, a particular biological assembly
  of the reference is constructed; otherwise the asymmetric unit is used.
- `ref_altloc` (default `"first"`): how to handle alternate locations in the
  reference (see below).
- `model_chain_id_to_lig_mol` (default `None`): optional mapping from specific
  model chain IDs to RDKit `Mol` objects for ligands. If not provided, ligand
  molecules are reconstructed from the model mmCIF.
- `interested_lig_label_asym_id` (default `None`): which ligand(s) to treat as
  *ligands of interest*. When this is left `None`, ligand‑specific metrics
  (pocket RMSD, PoseBusters) are *skipped*.
- `run_config` (default `RUN_CONFIG`): global configuration object. By default:
  - entity and ligand mapping are enabled,
  - metric switches are enabled (clashes, LDDT, DockQ, RMSD, PoseBusters),
    but RMSD and PoseBusters are only executed when `interested_lig_label_asym_id`
    is provided,
  - LDDT uses a nucleotide radius of 30 Å and a non‑nucleotide radius of 15 Å,
  - clash detection uses half the sum of van der Waals radii as the clash
    threshold.

Unless otherwise noted, the descriptions below assume all these defaults.

---

## 3. Loading the structures

### 3.1 Parsing mmCIF into atom arrays

For both `ref_cif` and `model_cif`, PXMeter:

1. Opens the mmCIF file (optionally through gzip if the file ends with
   `.cif.gz`).
2. Reads the full CIF block using the biotite `pdbx` parser.
3. Constructs an internal `AtomArray` with, for each atom:
   - element,
   - atom name,
   - residue name (`res_name`),
   - residue ID (`res_id`),
   - chain IDs (both author and label forms),
   - entity IDs, occupancy, B‑factors, optional formal charges,
   - 3D coordinates.
4. Uses **label** identifiers (`label_asym_id`, `label_seq_id`, etc.) as the
   primary coordinate system for chains and residues:
   - chain IDs are set to `label_asym_id`,
   - polymer residues use `label_seq_id` as sequence position,
   - ligand residues (which usually have `label_seq_id = "."`) are
     renumbered sequentially within each chain (1, 2, 3, ...).

In other words, by the end of this stage both reference and model are
represented as consistent atom arrays with label‑space chain and residue IDs.

### 3.2 Deriving sequence and entity information

While reading the CIF, PXMeter also:

- Reads the `_entity_poly` and `_entity_poly_seq` tables and builds, for each
  **polymer entity**:
  - its sequence as a string of one‑letter codes (using CCD definitions),
  - its polymer type (L‑polypeptide, DNA, RNA, etc.).
- Records the experimental methods (X‑ray, EM, NMR, etc.) and entry ID.

This information is later used for entity‑level mapping and filtering.

---

## 4. Normalising and cleaning the structures

After parsing, each structure is cleaned in a series of fixed steps. The goal
is to remove or normalise atoms that would otherwise destabilise mapping and
metric calculations.

### 4.1 Fixing common naming quirks

For each structure separately:

1. **Arginine naming** – ambiguous NH1/NH2 naming is resolved:
   - for every ARG residue, the code identifies CD, NH1 and NH2,
   - the one closer to CD is always treated as NH1, ensuring a consistent
     side‑chain geometry representation.
2. **Selenomethionine (MSE) to methionine (MET)**:
   - residues named MSE are converted to MET,
   - the selenium atom is renamed from SE to SD and its element changed from
     Se to S,
   - the residue is marked as non‑hetero.
3. **ASX/GLX resolution**:
   - ambiguous ASX and GLX residues are mapped to ASP and GLU respectively,
   - ambiguous atoms with element X are mapped to oxygen, and atom names are
     updated accordingly.

These adjustments mean that later logic can treat these residues as standard
amino acids.

### 4.2 Removing irrelevant or problematic atoms

A boolean mask is built over all atoms, starting with “keep everything” and then
applying the following rules:

1. **Water removal (enabled by default)**
   - Atoms in residues named HOH or DOD are removed.

2. **Hydrogen removal (enabled by default)**
   - Atoms whose element is H or D are removed.

3. **Unknown / placeholder residues removal (enabled by default)**
   - Entire residues labelled UNX or UNL are removed.

4. **Crystallisation additive removal (enabled by default)**
   - If the experimental methods indicate a crystallographic technique
     (X‑ray, neutron, fibre, powder, etc.), then:
     - atoms in residues known to be common crystallisation agents (SO4, GOL,
       PEG, etc.) are removed,
     - *but only* if those residues belong to non‑polymer entities (to avoid
       accidentally deleting protein chains).

At the end of this step, a new structure is created containing only atoms that
survived the mask.

The net effect is that both reference and model structures are reduced to
clean, comparable coordinate sets without water, hydrogens, stray X atoms or
obvious crystallisation artefacts.

---

## 5. Mapping entities and ligands

With both structures cleaned, PXMeter next builds a mapping between **entities**
(polymer and non‑polymer) in the reference and model.

### 5.1 Polymer entities (proteins / nucleic acids)

For each polymer entity in the reference:

1. The reference sequence and type are taken from the entity tables.
2. All model polymer entities of the same type are considered candidates.
3. For each pair of reference/model entities of the same type:
   - their sequences are standardised (e.g. uncommon amino acids collapsed
     to X, U treated as C for proteins; U treated as T for nucleic acids),
   - a global alignment is computed using a standard substitution matrix and
     gap penalties,
   - sequence identity is recorded.
4. All candidate pairs are sorted by descending sequence identity.
5. A greedy pass assigns model polymer entities to reference polymer entities
   such that:
   - each entity appears in at most one pair,
   - the overall entity mapping respects sequence similarity as much as
     possible.

By default, this entity mapping step is **enabled**.

#### 5.1.1 Entity Type Fallback

When specific entity types are missing in the model, PXMeter allows fallback mapping:

- Ref `PROTEIN_D` → Model `PROTEIN` (if model has no `PROTEIN_D`).
- Ref `DNA_RNA_HYBRID` → Model `DNA` or `RNA` (if model has no `DNA_RNA_HYBRID`).

### 5.2 Ligand entities (non‑polymers)

Ligand mapping is done in two passes:

1. **Exact CCD sequence matching**
   - For each non‑polymer entity in both structures, a coarse “CCD sequence”
     is produced by listing residue names along a representative chain and
     joining them with underscores.
   - Only non‑polymer entities are considered here (solvents, ions, small
     molecules, sugars, etc.).
   - Exact string matches between model and reference CCD sequences produce
     a first round of ligand entity matches.

2. **Chemical similarity matching**
   - Remaining (unmatched) ligand entities are compared chemically:
     - each ligand entity in the reference is converted to an RDKit `Mol` from
       CCD data,
     - each unmatched ligand entity in the model is converted either from:
       - the user‑provided `model_chain_id_to_lig_mol`, or
       - the model chain atom coordinates via RDKit’s PDB reader, with atom
         properties (names, residue names, residue IDs) set from the CIF.
   - For each model/reference ligand pair, a Morgan fingerprint is computed
     (radius = 2, size = 2048, with chirality), and Tanimoto similarity is
     measured.
   - Pairs are sorted by similarity and tried greedily.
   - Note: the current implementation does not enforce an explicit similarity
     threshold; whether a pair is accepted also depends on whether a consistent
     atom‑level graph match can be found.

By default, ligand mapping is **enabled**.

### 5.3 Ligand atom‑level alignment

Once ligand entities are matched, PXMeter also attempts to align their atoms
more precisely:

- For each matched ligand entity pair:
  1. Both ref and model ligands are converted to NetworkX graphs.
  2. A maximum common subgraph match is computed, constrained to match atoms
     with identical elements and consistent bonding.
  3. From this match, PXMeter derives a mapping between individual ligand
     atoms in the model and reference.
- Atom‑level mapping is then used to *rename* model ligand atoms to match the
  reference where possible:
  - residue names, residue IDs, and atom names of mapped model atoms are
    adjusted to match the reference ligand,
  - atoms that could not be mapped reliably are effectively marked as
    “unusable” and later removed.

The end result of this stage is a **model‑to‑reference entity ID mapping**, plus
optional atom‑level mapping for ligands, that will be used in chain and residue
alignment.

---

## 6. Aligning chains between reference and model

After entities are mapped, PXMeter determines how to pair up **chains** within
those entities.

### 6.0 Intuition: why “aligning chains” is necessary

The key point is that **chain IDs are not guaranteed to be comparable** between
the reference and the prediction:

- Model chain IDs may be arbitrary (e.g. `A/B` vs `X/Y`).
- In homomers (multiple copies of the same entity), chain IDs can be *swapped*
  even when the geometry is correct.

PXMeter therefore treats chain pairing as an **assignment problem**: find the
reference→model chain mapping that yields the most consistent global alignment
and the lowest overall RMSD.

The following schematic shows the kind of ambiguity PXMeter resolves:

```
Reference (ref)                           Model (pred)

Entity E (same sequence, 2 copies)        Entity E (same sequence, 2 copies)
  chain A    chain B                        chain X    chain Y
    |          |                              |          |
    |          |                              |          |
    +---- correct geometry, but IDs can be permuted ----+

Goal: choose a mapping that makes corresponding chains overlap best
  e.g. {A -> Y, B -> X} rather than naively matching IDs by name.

Extra constraint (if present): ligand attachment points
  ref: A has ligand covalently attached at residue 42
  model: only X has that ligand attached at residue 42
  => forbid pairing A<->Y, so A must map to X.
```

At runtime, PXMeter implements this by trying plausible anchor chain pairs,
superposing the reference onto the model using common atoms on the anchors, and
then matching the remaining chains within each entity by spatial proximity (via
linear assignment / Hungarian method), while respecting any forbidden chain
pairs derived from ligand attachment points.

The overall logic can be summarised as a single flow:

```
                 ┌───────────────────────────────────────┐
                 │ Fix one model anchor chain (chosen by │
                 │ the priorities in 6.2)                │
                 └───────────────────────────────────────┘
                                   │
                                   v
                 ┌───────────────────────────────────────┐
                 │ Enumerate compatible reference anchor │
                 │ chains (same entity, not forbidden)   │
                 └───────────────────────────────────────┘
                                   │
                                   v   (for each ref anchor candidate)
┌────────────────────────────────────────────────────────────────────────────┐
│ 1 Find common atoms on the anchor chains -> compute rigid transform T      │
│ 2 Apply T to *all* reference atoms (superpose ref onto model space)        │
│ 3 For each entity, build a chain-pair cost matrix                          │
│    - cost = distance between chain centroids (computed on common atoms)    │
│    - set cost = +∞ for forbidden ref/model chain pairs (ligand constraints)│
│ 4 Solve optimal chain pairing with Hungarian (linear assignment)           │
│ 5 Compute overall RMSD over all matched chain pairs                        │
└───────────────────────────────────────────────────────────────────────── ──┘
                                   │
                                   v
                 ┌───────────────────────────────────────┐
                 │ Pick the ref anchor with *lowest*     │
                 │ overall RMSD                          │
                 └───────────────────────────────────────┘
                                   │
                                   v
                 Output: final ref_chain -> model_chain mapping
```

### 6.1 Using ligand attachment points as constraints

First, PXMeter examines bonds that connect ligand atoms to polymer atoms in
both structures. For each ligand chain:

- it finds which polymer entity it is covalently attached to and at which
  residue number,
- this attachment point is used to construct a set of *forbidden chain pairs*:
  - reference and model chains whose ligand attachments disagree (either
    attachment is to a different polymer entity or at a different residue
    index) are not allowed to be mapped to each other.

This ensures, for example, that a ligand covalently attached to chain A in the
reference is not accidentally mapped onto a model chain where that ligand is
attached to a completely different location.

### 6.2 Choosing anchor chains (default strategy)

To align chains robustly, PXMeter chooses one **anchor chain** in the model
and one **anchor chain** in the reference and uses them to initialise the
alignment.

The model anchor chain is selected using these default priorities:

1. Chains with more than 4 resolved residues are preferred.
2. Chains whose corresponding reference entity also has more than 4 resolved
   residues are preferred.
3. Polymer chains are preferred over non‑polymer chains.
4. Among candidates, entities with fewer chains in the reference are preferred
   (less ambiguity).
5. Among remaining candidates, longer chains are preferred.
6. As a final tie‑breaker, the lexicographically smallest chain ID is used.

### 6.3 Testing reference anchors and aligning the full structure

For the chosen model anchor chain, all compatible reference chains in the
corresponding reference entity are considered as possible **reference anchors**.
For each candidate anchor pair (reference anchor, model anchor):

1. **Find common atoms** between the two anchor chains using their unique
   atom IDs (which encode residue ID, residue name and atom name).
2. From these common atoms, compute a rigid‑body transform (rotation and
   translation) that aligns the reference anchor onto the model anchor.
3. Apply this transform to *all atoms* of the reference structure, not just
   the anchor chain.
4. Within each entity, match the remaining reference and model chains by
   spatial proximity:
   - for each pair of candidate chains, compute centroids of their common
     atoms,
   - build a cost matrix based on centroid distances,
   - use a linear assignment algorithm (Hungarian method) to find an optimal
     chain pairing, while respecting the forbidden chain pairs derived from
     ligand attachment points.
5. For this particular reference anchor, compute an overall RMSD between all
   matched chain pairs.

The reference anchor that yields the **lowest overall RMSD** is selected. This
produces a final mapping from reference chains to model chains.

### 6.4 Deriving atom indices for chain‑level alignment

Given the final chain mapping, PXMeter then:

1. Iterates over reference chains in their original order.
2. For each reference chain, finds the atoms that have a counterpart in the
   mapped model chain (again by matching unique atom IDs).
3. Records the index positions of these atoms in both reference and model
   structures.

These indices define a consistent reordering of reference and model atoms so
that, within each mapped chain, the two structures list corresponding atoms in
exactly the same order.

If less than half of the atoms in a chain can be aligned, a warning is logged,
indicating potentially poor chain matching.

---

## 7. Resolving symmetry at residue and atom level

Even after chains are matched, there may be ambiguity in how residues and
atoms are ordered, especially for branched ligands and symmetric functional
groups. Two additional stages in PXMeter address this.

### 7.1 Residue‑level permutations in non‑polymer chains

For each non‑polymer entity (e.g. a sugar chain or other branched ligand), and
for each of its chains, PXMeter:

1. Looks at the inter‑residue bonds within the chain.
2. Builds a **residue‑level graph**:
   - nodes are residue IDs within the chain,
   - edges connect residues that are covalently bonded.
3. Checks if this graph is a single **tree** (connected and acyclic) and if
   there exists at least one bond that connects non‑adjacent residues (i.e.
   a topological branch).
4. If so, treats this graph as a tree rooted at residue 1 and searches for
   **graph automorphisms** (permutations of residues that preserve the tree
   structure and residue types):
   - nodes carry labels such as residue name and a summary of atom names,
   - only automorphisms that keep residue 1 fixed are considered.
5. For each such residue permutation:
   - compute the geometric centre (centroid) of each residue in both reference
     and model,
   - align model centroids to reference centroids using the fixed root residue
     (residue 1),
   - compute RMSD between the full sets of residue centroids.
6. Choose the permutation that yields the **lowest centroid RMSD**.
7. Reorder all atoms of the chain accordingly, so that residues appear in the
   order dictated by the best permutation while keeping atoms within each
   residue in their original order.

If no non‑trivial automorphisms are found, or if the graph is not a simple
branching tree, the chain is left as‑is.

### 7.2 Atom‑level permutations using CCD symmetry information

Finally, PXMeter addresses **atom‑level symmetry** within each residue:

1. For each residue in the model structure:
   - the residue type (CCD code) is looked up in the CCD database, which
     provides any known atom permutations corresponding to chemical symmetry
     (e.g. equivalent ring carbons, symmetric methyl groups).
   - these permutations effectively say “atoms {A,B,C,...} can be rotated or
     permuted among themselves without changing the chemistry”.
2. For each residue where CCD permutation data exists:
   - a matrix of permutations is derived that corresponds exactly to the
     atoms *present* in this particular residue instance; permutations that
     reference missing atoms are discarded, and duplicates are removed.
3. For residues with no CCD permutation information, the only available
   permutation is the identity (no atom reordering).
4. Across all residues, PXMeter identifies atoms that are **non‑symmetric**
   (they never move under any of the residue permutations). These atoms are
   used to define a **global rigid‑body alignment** between the model and
   reference coordinates:
   - the transform is chosen to minimise RMSD between non‑symmetric atoms.
5. This transform is applied to all model atoms, so the model is roughly in
   the correct orientation relative to the reference.
6. For each residue individually:
   - PXMeter considers each CCD‑derived atom permutation,
   - applies the permutation to the transformed model residue atoms,
   - computes RMSD between the permuted model residue and the reference
     residue (with zero padding for residues with fewer atoms),
   - selects the permutation with **minimum residue‑level RMSD**.
7. The chosen residue‑level permutations are then combined across all residues
   into a single final atom ordering for the model structure.

After this step, reference and model structures are aligned at the level of
entities, chains, residues and atoms, and ambiguity due to symmetry has been
resolved in a way that favours the lowest RMSD.

---

## 8. Computing evaluation metrics

With aligned structures in hand, PXMeter computes a collection of metrics.
All of the following are **enabled by default**, subject to the availability
of the required inputs (e.g. selected ligand IDs).

### 8.1 Clash count (van der Waals overlaps)

Goal: count how many atoms in the model structure are involved in severe
steric clashes, based on default van der Waals radii.

Runtime behaviour:

1. Consider all atoms in the **model** structure as query atoms.
2. For each atom, look up its default van der Waals radius.
3. Build a 3D KD‑tree over all atom coordinates.
4. For each query atom, fetch all neighbours within 3.0 Å.
5. For each neighbour:
   - skip if it is the same atom,
   - skip if the two atoms are covalently bonded (to avoid flagging normal
     bond distances),
   - if either atom’s van der Waals radius is unknown, fall back to a carbon
     radius.
6. Compute the inter‑atomic distance `d` and the threshold
   `0.5 * (r_i + r_j)` (default `vdw_scale_factor = 0.5`).
7. If `d` is smaller than this scaled sum of radii, the pair is considered to
   be clashing.
8. Finally, PXMeter counts how many distinct atoms appear in at least one
   clashing pair and reports this count as the **complex‑level clash metric**.

### 8.2 Pocket‑aligned RMSD around selected ligands

This metric is only computed if `interested_lig_label_asym_id` is provided.
Default behaviour for each selected ligand:

1. Identify all atoms whose **unique chain ID** matches the specified ligand
   ID in the reference structure. These atoms form the **ligand mask**.
2. Using a KD‑tree on all reference atoms, find all atoms within 10.0 Å of any
   ligand atom.
3. Among these neighbours, restrict to polymer **backbone** atoms:
   - CA atoms in protein chains,
   - C1′ atoms in nucleic‑acid chains.
4. Group these backbone atoms by chain, and choose the chain with the largest
   number of backbone atoms within 10 Å as the **primary pocket chain**.
5. The pocket mask is defined as these backbone atoms in the primary pocket
   chain.
6. Using the aligned reference and model structures, PXMeter then:
   - aligns the model to the reference by finding the rigid‑body transform
     that minimises RMSD between pocket atoms (pocket mask),
   - applies this transform to all model atoms,
   - computes RMSD over ligand atoms only (ligand mask) to obtain
     **ligand RMSD**,
   - computes RMSD over pocket atoms to obtain **pocket RMSD**.
7. For each ligand, PXMeter stores:
   - the ID of the pocket chain in the reference,
   - the pocket RMSD,
   - the ligand RMSD.

This yields a per‑ligand assessment of how well the ligand and its immediate
binding pocket are reproduced.

### 8.3 LDDT (complex, per chain, per interface)

LDDT is used to evaluate local distance agreement between reference and model.
The default configuration uses:

- a distance inclusion radius of 30 Å for nucleic‑acid atoms,
- 15 Å for all other atoms,
- distance deviation thresholds of 0.5, 1.0, 2.0 and 4.0 Å.

The computation proceeds in two levels.

#### 8.3.1 Selecting atom pairs

1. In the reference structure, PXMeter classifies atoms as nucleic vs
   non‑nucleic based on their entity types.
2. It builds a KD‑tree over all reference coordinates.
3. For nucleic acids:
   - for each nucleic atom, query neighbours within 30 Å,
   - record all unordered pairs (i, j) where i ≠ j.
4. For all other atoms:
   - query neighbours within 15 Å,
   - record all unordered pairs.
5. The union of these two sets is taken as the **LDDT atom pair set**. There
   must be at least one pair, otherwise an error is raised.

This atom pair set is fixed for the entire evaluation; for chain or interface
LDDT, subsets of these pairs are selected.

#### 8.3.2 Computing LDDT scores

For any given set of atom pairs (full complex, a chain, or an interface):

1. For each pair (i, j), compute in both reference and model:
   - `d_ref = ‖coord_ref[i] − coord_ref[j]‖`,
   - `d_model = ‖coord_model[i] − coord_model[j]‖`.
2. Compute the absolute distance deviation `Δ = |d_model − d_ref|`.
3. For each threshold T in {0.5, 1.0, 2.0, 4.0} Å, assign 1 if `Δ < T`, else 0.
4. For each pair, average these four binary values to obtain its **pair‑level
   LDDT contribution** (between 0 and 1).
5. The final LDDT score for the set is the mean of the pair‑level scores
   across all pairs.

For the **full complex**, the set is simply the full atom pair set.

For **chains and interfaces**:

1. PXMeter first identifies chains and chain pairs that are spatially in
   contact:
   - using a 3D cell list, chains whose atoms are within 5 Å of each other form
     an interface;
   - chains that only contain ions (as per a predefined ion list) are
     discarded from chain and interface LDDT.
2. For each chain or chain pair, PXMeter selects from the global atom pair set
   those that involve atoms in the relevant chains.
3. LDDT is computed for each such subset of pairs as above.

Results are recorded as:

- one complex‑level LDDT score,
- per‑chain LDDT scores,
- per‑interface LDDT scores.

#### 8.3.3 Stereochemistry‑aware LDDT

The `metric.lddt.stereochecks` flag controls whether obvious stereochemical geometry problems in the model can influence the LDDT score.

- With `stereochecks = False` (default):
  - All mapped atoms are treated as equally valid.
  - LDDT only measures how well local distances agree, without checking whether the geometry itself is chemically reasonable.

- With `stereochecks = True`:
  - PXMeter first runs a stereochemical validation on the **model** structure, checking for severe clashes and out‑of‑range bond lengths or angles.
  - These violations are converted into a per‑atom validity mask:
    - For **polymer chains** (proteins and nucleic acids):
      - if any **backbone atom** in a residue is involved in a violation, the whole residue (backbone and side‑chain) is marked invalid for LDDT;
      - if only **side‑chain atoms** are affected, only the side‑chain atoms of that residue are marked invalid, while the backbone stays valid.
    - For **non‑polymer entities** (ligands, ions, etc.), only the atoms directly involved in violations are marked invalid.
  - During LDDT calculation:
    - pairs where **both atoms are valid** contribute normally;
    - pairs that touch any invalid atom still count in the average, but their contribution is forced to zero.

In practice, enabling stereochemical checks means that regions with unrealistic backbone geometry strongly reduce the LDDT, while purely side‑chain problems mainly affect the side‑chain contribution to the score.

#### 8.3.4 Backbone‑only LDDT

Backbone‑only LDDT is a variant of LDDT that focuses on the overall shape of
protein and nucleic‑acid backbones, rather than all atoms.

Conceptually:

- It measures how well the main chain of the model follows the main chain of
  the reference, while ignoring most side‑chain details.
- When backbone LDDT is enabled in the configuration, the evaluation report
  will contain, in addition to the usual LDDT scores:
  - a backbone LDDT for the whole complex (polymer backbone atoms),
  - a backbone LDDT for each protein or nucleic acid chain,
  - a backbone LDDT for each protein–protein or protein–nucleic‑acid interface.
- For chains or interfaces that do not have a meaningful backbone
  (for example, pure ion chains), a backbone LDDT value may simply be omitted
  from the results.

In practice, you can read backbone LDDT as “how good is the backbone geometry”,
complementing the all‑atom LDDT that also reflects side‑chain quality.

When `metric.lddt.calc_backbone_lddt` is enabled, PXMeter first builds a backbone atom mask on the reference structure (for example, representative main‑chain atoms in each polymer residue) and computes LDDT using only distances between these backbone atoms.

If `metric.lddt.stereochecks` is also set to `True`, backbone‑only LDDT still evaluates over backbone atom pairs, but pairs touching stereochemically invalid backbone atoms contribute zero (they are not dropped from the averaging). If a chain/interface has no backbone atom pairs at all (e.g. too few atoms after filtering), the backbone‑only LDDT entry may be omitted from results.

### 8.3.5 LDDT-PLI

`LDDT-PLI` measures the local distance agreement between the ligand and its binding pocket.

- **Binding Pocket Definition**: The binding pocket is defined as the set of all polymer residues (protein or nucleic acid) in the reference structure that have at least one atom within a specified inclusion radius (default **6.0 Å**) of any ligand atom.
- **Atom Pair Selection**: The calculation considers atom pairs between the **ligand** and the **binding pocket**, subject to two constraints:
  1. One atom must belong to the ligand, and the other to the binding pocket.
  2. The distance between the atoms in the reference structure must be less than the inclusion radius (default **6.0 Å**).
- **Scoring**: Standard LDDT thresholds (0.5, 1.0, 2.0, 4.0 Å) are used to compute the score based on these selected pairs.

### 8.4 DockQ (interface quality)

DockQ provides an alternative, interface‑centric quality measure. PXMeter uses a native implementation that follows the official DockQ *formulae*.

Note: residue pairing differs from the official DockQ pipeline. In PXMeter, residues are paired by `(res_id, res_name)` after the upstream mapping/permutation steps, rather than by an explicit sequence alignment.

Runtime behaviour (default):

1. PXMeter identifies all polymer interfaces present in the reference structure (using a 10 Å threshold).
2. For each interface, it computes:
   - **fnat**: The fraction of native residue-residue contacts (heavy atom distance < 5 Å) preserved in the model.
   - **iRMSD**: The RMSD of backbone atoms (C, CA, N, O for proteins; P, C1', etc. for nucleic acids) of interface residues after optimal superposition.
   - **LRMSD**: The RMSD of backbone atoms of the smaller "ligand" chain after superimposing the larger "receptor" chain.
3. The final **DockQ score** is calculated as the average of `fnat`, `rms_scaled(iRMSD, 1.5)`, and `rms_scaled(LRMSD, 8.5)`.
4. Detailed statistics such as the number of native/non-native contacts and F1 score are also recorded.

This gives a complementary, widely‑used view of interface quality.

### 8.5 PoseBusters validity (re-implemented)

This metric is only computed if **both**:

- `calc_pb_valid` is `True` in the config (default), and
- `interested_lig_label_asym_id` is provided.

**Note**: PXMeter now uses an internal Python-only implementation (`pxmeter/metrics/pb_valid`) that replicates the key validity checks of the PoseBusters suite using RDKit and Biotite. It no longer requires an external PoseBusters executable.

For each selected ligand:

1. **Structure preparation**:
   - The predicted ligand is converted to an RDKit molecule (with 3D-based stereo-perception).
   - The reference ligand is also converted to RDKit for identity comparison.
   - The surrounding environment (protein, cofactors, water) is extracted from the model structure.

2. **Metric computation**:
   PXMeter runs a battery of checks directly in memory:
   - **Chemistry**: Sanitization, radicals, connectivity, and identity consistency (formula, bonds, stereochemistry) against the reference.
   - **Geometry**: Bond lengths, bond angles, internal steric clashes, and flatness of aromatic rings/double bonds.
   - **Energy**: Internal energy ratio using the UFF force field.
   - **Intermolecular interactions**: Minimal distances and volume overlaps between the ligand and the protein, organic/inorganic cofactors, and waters.
   - **RMSD**: Standard RMSD, symmetry-corrected (Kabsch) RMSD, and centroid distances.

3. **Output**:
   - A tabular report (DataFrame) containing pass/fail status and values for each check is generated.
   - This report is stored under the corresponding reference ligand chain ID.

This provides a fast, dependency-light assessment of physical plausibility and chemical correctness, matching the logic of the original PoseBusters tool.

### 8.6 Framework-aligned CDR-H3 loop backbone RMSD

This metric is designed specifically for antibody heavy chains to evaluate the structural accuracy of the hypervariable CDR‑H3 loop, which is critical for antigen binding. It is computed when `calc_cdr_h3_bb_rmsd` is enabled in the configuration.

Runtime behaviour:

1. **Identify protein chains**: PXMeter iterates over all unique protein entities in the reference structure.
2. **Sequence annotation (Batch)**:
   - Unique sequences are extracted from the reference protein entities.
   - These sequences are annotated in batch using **ANARCII** (LM).
   - The annotation identifies the chain type (Heavy, Light, etc.) and delineates the variable regions (FR1, CDR1, FR2, CDR2, FR3, CDR3, FR4) according to the **IMGT** numbering scheme.
   - Using `seq_type="unknown"` ensures that T‑cell receptor (TCR) chains are not misclassified as antibody heavy chains by ANARCII.
3. **Per-chain calculation**:
   - The algorithm iterates through each protein chain in the reference.
   - If a chain is identified as a **Heavy chain** and contains a **CDR3** region, it proceeds to calculation.
4. **Feature extraction**:
   - **All atoms** (specifically heavy atoms, as hydrogens are removed during cleaning) are extracted from both the reference and the mapped model structure.
   - The residue IDs of these atoms are mapped to the IMGT regions identified in the annotation step.
5. **Alignment and RMSD**:
   - **Alignment**: The model chain is superimposed onto the reference chain using the backbone atoms (N, C, CA, O) of the framework regions (**FR1, FR2, FR3, FR4**). This minimizes the RMSD of the stable framework.
   - **RMSD calculation**: After alignment, the RMSD is computed for **backbone atoms (N, C, CA, O)** of the **CDR-H3** loop.
6. **Reporting**:
   - The resulting CDR‑H3 RMSD value is recorded for the corresponding chain in the metric results.
   - If a chain is not a heavy chain or lacks a CDR3 loop, this metric is skipped for that chain.

This approach ensures that the loop accuracy is measured relative to a correctly oriented framework, providing a biologically meaningful assessment of antibody modeling performance.

---

## 9. Assembling the result and JSON output

All metrics and metadata are finally packaged into a single
**`MetricResult` object**.

Conceptually, this object contains:

- **Meta information**
  - the reference entry ID,
  - the mapping from reference chain IDs to model chain IDs,
  - basic information about each reference chain (entity ID, entity type),
  - the original model chain IDs before permutation (for traceability).

- **Complex‑level metrics**
  - global LDDT,
  - number of atoms involved in steric clashes.

- **Per‑chain metrics**
  - chain‑level LDDT for all meaningful chains,
  - for selected ligands of interest:
    - the ID of the corresponding pocket chain,
    - ligand RMSD,
    - pocket RMSD.

- **Per‑interface metrics**
  - LDDT scores for each chain pair that forms an interface,
  - DockQ scores for those interfaces, plus selected DockQ sub‑metrics.

- **Per‑ligand PoseBusters metrics** (optional)
  - for each selected ligand in the reference, a detailed PoseBusters report
    (if computed).

The object can be converted to a plain Python dictionary that mirrors this
structure and then serialised to JSON. Interface keys are typically expressed
as strings of the form `"CHAIN1,CHAIN2"`.

---

## 10. Summary

In its default configuration, `pxmeter.eval.evaluate()` implements a
multi‑stage evaluation pipeline that:

1. **Reads and cleans** reference and model structures, normalising common
   quirks in residue naming and removing irrelevant atoms.
2. **Builds a precise mapping** between entities, chains, residues and atoms,
   exploiting sequence information, ligand chemistry and geometric criteria.
3. **Resolves symmetry ambiguities** in branched ligands and symmetric
   residues, always choosing the arrangement that best agrees with the
   reference geometry.
4. **Computes a comprehensive set of metrics** that cover:
   - overall structural agreement (LDDT),
   - local pocket agreement around ligands (pocket‑aligned RMSD),
   - interface quality (DockQ and interface LDDT),
   - steric correctness (clash counts, PoseBusters for ligands).
5. **Packages everything in a structured output** that can be consumed by
   downstream tools or converted to JSON.

This document describes the runtime behaviour of that pipeline, focusing on the
default configuration and on what the user can expect to happen when a
reference and a prediction are passed to `pxmeter` for evaluation.
