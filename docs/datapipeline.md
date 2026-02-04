# Dataset Pipeline Overview

This directory implements a multi-stage pipeline for constructing a **low-homology evaluation dataset** from PDB/mmCIF data.
The goal is to:

1. Filter PDB structures under strict structure-quality and size constraints.
2. Define *train* vs *test* by release date and compute sequence / ligand similarities.
3. Select *low-homology* chains and interfaces (protein / nucleic acid / ligand).
4. Cluster entities, tag biologically meaningful subsets (e.g. antibodies, monomers, peptides).
5. Prepare ground-truth CIFs and dataset statistics for downstream benchmarking.

---

## Table of Contents
- [How to Run](#how-to-run)
  - [Setup](#setup)
  - [Basic Usage](#basic-usage)
  - [Outputs](#outputs)
- [Dataset Construction Steps](#dataset-construction-steps)
  - [Step 1 - Filter for RecentPDB](#step-1---filter-for-recentpdb)
  - [Step 2 - Make low-homology mapping file](#step-2---make-low-homology-mapping-file)
  - [Step 3 - Filter to low-homology subset](#step-3---filter-to-low-homology-subset)
    - [3.1. Build test-train map](#31-build-test-train-map)
    - [3.2. Low-homology condition](#32-low-homology-condition)
    - [3.3. Handling short polymers (“peptides”)](#33-handling-short-polymers-peptides)
    - [3.4. Entity-type-specific filtering](#34-entity-type-specific-filtering)
    - [3.5. High-quality ligand filters](#35-high-quality-ligand-filters)
  - [Step 4 - Cluster low-homology entities](#step-4---cluster-low-homology-entities)
    - [4.1. Collect unique entities from low-homology CSV](#41-collect-unique-entities-from-low-homology-csv)
    - [4.2. Cluster by sequence identity (proteins / DNA / RNA)](#42-cluster-by-sequence-identity-proteins--dna--rna)
    - [4.3. Ligand clustering](#43-ligand-clustering)
    - [4.4. Output](#44-output)
  - [Step 5 - Find monomer / homomer entries](#step-5---find-monomer--homomer-entries)
    - [5.1. Extract entity counts per entry](#51-extract-entity-counts-per-entry)
    - [5.2. Classify monomer / homomer](#52-classify-monomer-homomer)
  - [Step 6 - Add subset labels](#step-6---add-subset-labels)
    - [6.1. Antibody / antibody-protein tags](#61-antibody-antibodyprotein-tags)
    - [6.2. Monomer / homomer tags](#62-monomer-homomer-tags)
    - [6.3. Peptide and cyclic peptide interfaces](#63-peptide-and-cyclic-peptide-interfaces)
    - [6.4. Merge all subset labels](#64-merge-all-subset-labels)
  - [Step 7 - Copy ground-truth CIFs](#step-7---copy-ground-truth-cifs)
  - [Step 8 - Homology for low-homology subset](#step-8---homology-for-low-homology-subset)
    - [8.1. Select train vs low-homology test sequences](#81-select-train-vs-low-homology-test-sequences)
    - [8.2. MMseqs2 with 0.3 identity threshold](#82-mmseqs2-with-03-identity-threshold)
    - [8.3. Output](#83-output)
  - [Step 9 - Dataset analysis](#step-9---dataset-analysis)
    - [9.1. Filter-step statistics](#91-filter-step-statistics)
    - [9.2. Low-homology subset statistics](#92-low-homology-subset-statistics)
    - [9.3. Plots and PDB ID lists](#93-plots-and-pdb-id-lists)

---

## How to Run

The entire dataset construction process is orchestrated by a single entry point:

`benchmark/dataset_pipeline/run_pipeline.py`

This script runs **all stages (Step 1-9)** in sequence, from raw mmCIF structures to the final low-homology dataset, cluster assignments, subset tags, and analysis reports.

---

### Setup

Before running the pipeline, ensure the following prerequisites are completed:

* **Set the environment variable `PXM_EVAL_DATA_ROOT_PATH`**
  This path serves as the root directory for all pipeline outputs and must be writable.

* **Install Python dependencies (Python 3.11+ required)**

  ```bash
  pip install -r requirements.txt
  ```

* **Prepare mmCIF data**
  Download all required mmCIF structures from the PDB and place them under the directory specified by `--mmcif_dir`.

* **Install external tools**

  * **MMseqs2** must be installed and accessible from your system `PATH`.

---

### Basic Usage

Minimal example:

```bash
export PXM_EVAL_DATA_ROOT_PATH="/path/to/output"
python -m benchmark.dataset_pipeline.run_pipeline \
  --mmcif_dir /path/to/mmcif \
  --after_date 2024-01-01 \
  --before_date 2025-01-01 \
  --n_cpu 16
```

Key arguments:

* `--mmcif_dir`
  Path to the directory with all mmCIF files.

* `--after_date`, `--before_date`
  Date cutoffs defining train vs test. Formulate as `YYYY-MM-DD`.
  Entries with `release_date < after_date` are treated as **train**.
  Entries with `after_date <= release_date <= before_date` form the **candidate test set** and are filtered into the low-homology subset.

* `--n_cpu` (optional, default: machine-dependent)
  Number of CPU cores to use for parallelized steps (structure parsing, symmetry checks, MMseqs preprocessing, etc.).

### Outputs
The pipeline outputs the following directory structure under `PXM_EVAL_DATA_ROOT_PATH`:

```
PXM_EVAL_DATA_ROOT_PATH
├── src_data
│   ├── ccd_to_similar_ccds.json
│   ├── pdb_meta_info.csv
│   ├── pdb_seqs.csv
│   ├── RecentPDB_chain_interface.csv
│   ├── RecentPDB_low_homology_entity_types_count.csv
│   ├── sabdab_summary_all.tsv
│   └── test_to_train_entity_homo.parquet
└── supported_data
    ├── mmcif
    ├── RecentPDB_low_homology_cluster_info.csv
    ├── RecentPDB_low_homology.csv
    ├── RecentPDB_low_homology_entity_homo.parquet
    └── stat_data
        ├── figs
        │   ├── ... (other png files)
        │   └── token_num_distribution.png
        ├── pdb_ids
        │   ├── all.txt
        │   ├── ... (other txt files)
        │   └── subset
        │       ├── ... (other txt files)
        │       └── antibody-protein.txt
        └── stat.txt
```

For a detailed description of all generated files and their schemas, please refer to the
[Dataset Output Files Reference](datapipeline_output.md).

---


## Dataset Construction Steps

At a high level, the pipeline does:

1. **Filter Recent PDB structures** → `recentpdb_chain_interface.csv` + metadata
2. **Train-test homology scan** → `test_to_train_entity_homo.parquet`
3. **Select low-homology chains & interfaces** → `RecentPDB_low_homology.csv`
4. **Cluster low-homology entities** → `RecentPDB_low_homology_cluster_info.csv`
5. **Detect monomer / homomer entries** → `RecentPDB_low_homology_entity_types_count.csv`
6. **Add subset labels** → update `RecentPDB_low_homology.csv` (`subset` column)
7. **Prepare ground-truth CIFs** → `true_dir`
8. **Fine-grained homology for low-homology entities** → `RecentPDB_low_homology_entity_homo.parquet`
9. **Dataset statistics and plots** → `stat.txt`, histogram PNGs, PDB ID lists

Below is a detailed description of each step (`step1` - `step9`).

---
### Step 1 - Filter for RecentPDB

**Goal:** Start from all PDB/mmCIF entries, apply strict quality and size filters, and extract chain / interface metadata for assemblies in a given date window.

Logically, the step implements your listed rules:

1. Filter entries by `release_date` within `[after_date, before_date]`.
2. Exclude any structure whose experimental methods include `"SOLID-STATE NMR"` or `"SOLUTION NMR"`.
3. Keep only entries with `resolution < 4.5 Å` (if available).
4. Use **biological Assembly 1** coordinates.
5. Strip hydrogens and waters.
6. For crystallographic methods, remove crystallization aids (buffer / precipitant etc.).
7. Discard entries with token count (sum over all chains) `> 2560`.
8. For polymers, keep only DNA, RNA, and protein chains; drop entries that contain no such chains.
9. Remove entries where any polymer entity has more than 20 chains.
10. Remove polymer chains where *all* residues are unknown.
11. Remove protein chains where any neighboring Cα-Cα distance exceeds 5 Å (chain breaks).
12. Remove polymer chains where

* the number of resolved residues `< 4`, **or**
* resolved residue fraction `< 0.3`.

13. **Ligand chain filters** (excluding glycan/ions):

* PDB experimental method must be uniquely `"X-RAY DIFFRACTION"`.
* Resolution ≤ 2.0 Å.
* All atom `occupancy == 1.0` on that chain.
* Exactly one residue in that ligand chain.
* CCD formula weight in [100, 900] Da.
* CCD element set is subset of `{H, C, O, N, P, S, F, Cl}`.
* At least 3 heavy (non-H) atoms.
* No covalent bonds to other chains.

14. Record remaining chain-level information

* For Ligand/Glycan/Ion chains, only record those in the asymmetric unit.

15. Compute and record **interfaces**: if any atom pair between two chains is within 5 Å (excluding H atoms), that pair is treated as an interface. Only interfaces with at least one chain in the asymmetric unit are kept; for ligand/glycan/ion interfaces, only interfaces where both chains are in the asymmetric unit are kept.

The counts and some of these filters are later re-computed/validated via `stat_filtered_num()` in `step9_analysis_dataset.py` when summarising statistics for a given date window.

---

## Step 2 - Make low-homology mapping file

File: benchmark/dataset_pipeline/step2_make_lowh_file.py

**Script:** `benchmark/dataset_pipeline/step2_make_lowh_file.py`

**Goal:** Given all entity sequences and their release dates, compute **train-test similarity mappings**:

* Train = entities with `release_date < after_date`
* Test  = entities with `after_date <= release_date <= before_date`

The script:

1. Loads a **sequence table** (`pdb_seq_csv`) with columns including
   `entry_id`, `entity_id`, `entity_type` (`PROTEIN`, `DNA`, `RNA`, `LIGAND`), `seq`, `release_date`.

2. Splits by entity type and runs **MMseqs2 sequence search** via `calc_mmseqs_seq_identity()` (defined in `step2_make_lowh_file.py`, wrapping `mmseqs easy-search`).

   * For **proteins**:

     * `--min-seq-id 0.4`
     * `-e 0.1`
     * `--max-seqs 500000`
     * `-s 7.5`
   * For **DNA/RNA**:

     * `--min-seq-id 0.8`
     * `-e 0.1`, `--max-seqs 500000`, `-s 7.5`
     * `--search-type 3` for nucleotide search.

3. For **short sequences** (`len(seq) < min_seq_length`), the helper first does an exact-identity match on the raw string and treats those as identity 1.0 hits, avoiding misbehaviour of MMseqs on very short segments.

4. For **ligands** (CCD codes), it constructs Morgan fingerprints (`radius=2`, `fpSize=2048`) and computes **Tanimoto similarity** between CCDs:

   * `gen_ccd_fp()` builds an RDKit molecule and fingerprint for each CCD entry from the in-memory CCD CIF blocks.
   * `get_ccd_similarity()` precomputes CCD-CCD pairs with Tanimoto similarity ≥ 0.6 and saves them as a JSON mapping `{ccd_code: [[similar_ccd, similarity], ...]}`.
   * `get_lowh_by_ccd_similarity()` maps test ligand entities to all train ligand entities whose CCDs are similar above this threshold.

5. Concatenates all entity-type-specific results into a **single table** with columns
   `query_id`, `db_id`, `similarity`, `aligned_res_num` (for sequences) or `similarity` (for CCDs).

6. Down-casts dtypes using `shrink_dataframe()` and writes to a Parquet file (e.g. `test_to_train_entity_homo.parquet`).

This Parquet file is consumed by **Step 3** to decide which chains/interfaces belong to the *low-homology* test subset.

---

## Step 3 - Filter to low-homology subset

**Script:** `benchmark/dataset_pipeline/step3_filter_to_lowh.py`

**Inputs:**

* `recentpdb_chain_interface_csv`: chain & interface table from Step 1, with per-side `entity_type`, `entity_id`, `seq_length`, etc.
* `test_to_train_entity_homo.parquet`: output from Step 2, mapping test entities to train entities.

**Goal:** Produce `recentpdb_low_homology.csv`, which contains only:

* chains/interfaces whose component entities have **no overlapping train PDB IDs** according to Step 2,
* and satisfy additional length rules for short chains and ligand interfaces.

### 3.1. Build test-train map

The script reads the Parquet file into a DataFrame with columns
`query_id`, `db_id`, `similarity`, `aligned_res_num`, and groups it into a dictionary:

```text
test_to_train = {
    query_id: [db_id1, db_id2, ...],
    ...
}
```

where `query_id` and `db_id` have the form `{entry_id}_{entity_id}`.

### 3.2. Low-homology condition

For a given row in the chain/interface table:

* For a **chain** row:

  * Low-homology if `test_to_train[entry_id_entity_1]` is empty.

* For an **interface** row:

  * Let `train_pdb_ids_1` be PDB IDs of all db entities mapped from entity 1, and likewise `train_pdb_ids_2`.
  * Low-homology if `train_pdb_ids_1 ∩ train_pdb_ids_2` is **empty**.
  * If the two chains map to the same `{PDB, entity}` in train, they are considered to have overlapping homology and are removed.

This logic is implemented in `_check_for_lowh()`.

### 3.3. Handling short polymers (“peptides”)

`_is_short_chain()` defines a “short” polymer as **< 25 residues** for protein/DNA/RNA.

`_filter_short_polymer()` implements the length-based exclusion rules:

* For **chains**: require `seq_length_1 >= 25` to keep (short polymer chains are dropped).
* For **interfaces**:

  * If both sides are long (≥25), keep.
  * If both sides are short, drop.
  * If one side is short and the other is long, keep the interface **only** if the long chain has **no train hits** (i.e. is low-homology); otherwise drop.

This reflects the textual rule you described:

* Remove “short-short” interfaces.
* For “short-polymer”, keep only if the polymer side itself has no similar sequence in the training window.

### 3.4. Entity-type-specific filtering

The script then selects low-homology rows per entity type:

* **Proteins**:

  * select chains and protein-protein interfaces (`entity_type_1 == PROTEIN` and, for interfaces, `entity_type_2 == PROTEIN`),
  * apply `_check_for_lowh` + `_filter_short_polymer`.
  * Result: `lowh_protein_df`.

* **Nucleic acids (DNA/RNA)**:

  * include nucleic acid chains, nuc-protein interfaces, and nuc-nuc interfaces,
  * apply `_check_for_lowh` and then `_filter_short_polymer` to remove short chains and short-short interfaces, plus short-polymer where the polymer is not low-homology.
  * Result: `lowh_nuc_df`.

* **Ligands**:

  * keep ligand chains with `entity_type_1 == LIGAND`,
  * and ligand-polymer interfaces where the polymer side is protein/DNA/RNA with `seq_length >= 25`.
  * apply `_check_for_lowh` to require that in the training set there is no **same-PDB combination** of similar ligand and similar polymer.
  * Result: `lowh_lig_df`.

The three subsets are concatenated into a single DataFrame `merged_df`.

### 3.5. High-quality ligand filters

For ligand chains and ligand-polymer interfaces, `filter_lig_in_chain_interface_df()` from `dataset_pipeline/utils/select_ligand.py` applies **additional structural quality filters**:

1. **RCSB validation report** via GraphQL:

   * Fetch non-polymer instance validation metrics for `(entry_id, chain_id)` pairs.
   * Keep only instances satisfying:

     * `intermolecular_clashes == 0`
     * `is_best_instance == "Y"`
     * `stereo_outliers == 0`
     * `completeness == 1.0`
     * `RSR <= 0.2`
     * `RSCC >= 0.95`.

2. **Symmetry-mate contact check**:

   * Load mmCIF, build a 3×3×3 unit cell using Biotite’s `repeat_box`, and compute neighbors for ligand atoms via a KD-tree.
   * If any ligand atom has neighbors within 5 Å in chains that are not in `(Assembly 1 ∩ Asym Unit)`, the ligand is rejected as potentially crystallographic-artifact-like.
   * Only ligands that **do not** contact symmetry mates are kept.

The final filtered DataFrame is written as `recentpdb_low_homology.csv`.

### 3.6. Save ligand info CSV

`make_lig_info_df()` constructs a DataFrame with columns:

* `entry_id`
* `label_asym_id`

This function extracts all **ligand chains present in the low-homology subset**, and records them in a unified table.
This DataFrame is saved as `RecentPDB_low_homology_lig_info.csv`, which is used in the evaluation pipeline to identify ligand entities that require
**pocket-aligned RMSD computation** and **PoseBusters ligand validity checks**.
Only ligands listed in this file will be included in these ligand-specific quality assessments during benchmarking.

---

## Step 4 - Cluster low-homology entities

**Script:** `benchmark/dataset_pipeline/step4_make_cluster_csv.py`

**Goal:** Cluster the entities (protein/DNA/RNA/ligand) that appear in the low-homology subset and assign **cluster IDs** that will be used:

* for interface clustering, and
* in analysis / sampling logic downstream.

### 4.1. Collect unique entities from low-homology CSV

* Iterate over each row in `recentpdb_low_homology.csv`.
* For `type == "chain"`, record `entry_id`, `entity_id_1`, `entity_type_1`, `seq_length_1`.
* For `type == "interface"`, record both sides `(entry_id, entity_id_1, entity_type_1, seq_length_1)` and `(entry_id, entity_id_2, entity_type_2, seq_length_2)`.
* Deduplicate by `(entry_id, entity_id)`.

### 4.2. Cluster by sequence identity (proteins / DNA / RNA)

The core helper is `cluster_by_seq_identity()`:

* For sequences with length `< min_seq_length` (default 10), uses the **sequence string itself** as `cluster_id` (degenerate cluster).
* For longer sequences, writes them to a FASTA file and runs:

```bash
mmseqs easy-cluster seq.fasta mm mmseqs_tmp \
  --min-seq-id {threshold} -c {coverage} -s 8 --max-seqs 1000 --cluster-mode 1
```

* For proteins: `threshold=0.4`, `coverage=0.8`.
* For DNA/RNA: `threshold=0.8`, `coverage=0.8`.
* Parses `mm_cluster.tsv` to map `{entry_id, entity_id}` to `cluster_id` (cluster center ID).

### 4.3. Ligand clustering

For ligands:

* No MMseqs clustering is performed; the **CCD code** itself is treated as the cluster ID.
* The code prepends `"CCD_"` to the CCD code in the `cluster_id` column to distinguish from polymer clusters.

### 4.4. Output

The final cluster file has columns:

* `entry_id`, `label_entity_id`, `cluster_id`, `entity_type`

and is saved as `RecentPDB_low_homology_cluster_info.csv`.

Interfaces in later analysis use cluster IDs formed as `{chain_1_cluster_id}_{chain_2_cluster_id}`, with special handling to collapse interfaces where one side is ligand/short chain onto the polymer side’s cluster ID.

---

## Step 5 - Find monomer / homomer entries

**Script:** `benchmark/dataset_pipeline/step5_find_monomer.py`

**Goal:** For PDB entries appearing in the low-homology set, detect:

* **Protein monomers**
* **Protein homomers**
* **RNA monomers**

based on **Assembly 1** composition (counts of each entity type).

### 5.1. Extract entity counts per entry

`get_entity_counts_from_cif()`:

* Loads mmCIF with `Structure.from_mmcif(..., assembly_id=assembly_id)` and cleans structure.
* Iterates over `label_entity_id`s, determines `entity_type` (protein, RNA, ligand, etc.), and counts:

  * **Number of chains** per entity type (e.g. `PROTEIN`, `RNA`, `LIGAND` columns).
  * **Number of entities** per type (columns like `"protein entities"`, `"rna entities"`).

`get_entity_counts_from_cif_batch()` runs this in parallel over all PDB IDs in the low-homology CSV and returns a DataFrame with one row per `entry_id`.

### 5.2. Classify monomer / homomer

`find_monomer_and_homomer()` attaches boolean flags:

* `is_protein_monomer`

  * All non-ligand entity-type columns except `PROTEIN` are 0.
  * `PROTEIN == 1`.
* `is_protein_homomer`

  * All `"{entity} entities"` columns except `"protein entities"` are 0.
  * `PROTEIN > 1` and `"protein entities" == 1` (multiple identical protein chains).
* `is_rna_monomer`

  * All non-ligand entity-type columns except `RNA` are 0.
  * `RNA == 1`.

`find_protein_monomer_for_recentpdb_lowh()`:

* Reads `recentpdb_low_homology.csv` to get the set of `entry_id`s,
* runs entity count and classification on Assembly 1,
* writes `recentpdb_low_homology_entity_type_count.csv`.

This CSV is consumed in Step 6 to tag monomer/homomer subsets.

---

## Step 6 - Add subset labels

**Script:** `benchmark/dataset_pipeline/step6_add_subset_to_lowh.py`

**Goal:** For each chain/interface row in `recentpdb_low_homology.csv`, assign one or more **subset labels** describing its biological and structural class, and write them into a `subset` column. A single row can have multiple labels separated by `";"`.

The subsets include (non-exhaustive):

* Antibody-related: `[antibody]`, `[antibody-protein]`, `[antibody_HL]`, `[antibody_HL-protein]`, etc.
* Monomer / homomer: `[protein_monomer]`, `[protein_homomer]`, `[rna_monomer]`.
* Peptide-related interfaces: `[peptide-interface]`, `[peptide-peptide]`, `[peptide-protein]`, etc.
* Cyclic peptide interfaces: `[cyclic_peptide-interface]`, `[cyclic_peptide-protein]`, etc.

### 6.1. Antibody / antibody-protein tags

`identify_antibody_protein()`:

1. Ensures a **SAbDab summary TSV** is available (downloads via `wget` if not).
2. `get_sabdab_ab_chain_to_type()` builds a mapping `{pdb_id_auth_chain_id -> antibody type}` where types are one of `antibody_scFv`, `antibody_HL`, `antibody_H`, `antibody_L`.
3. `get_antibody_antigen_label()` inspects each low-homology row:

   * For **chains**, if `entry_id_auth_chain_id` exists in the map, label `[antibody];[ab_type]`.
   * For **interfaces**, it checks both chains and whether they are proteins, and returns:

     * `[antibody-antibody]` if both chains are antibodies.
     * `[antibody-protein];[{ab_type}-protein]` when one side is antibody and the other is protein.

### 6.2. Monomer / homomer tags

`identify_monomer_and_homomer()` uses `recentpdb_low_homology_entity_type_count.csv` from Step 5:

* Builds sets of `entry_id`s that are protein monomer, protein homomer, RNA monomer.
* For each low-homology row:

  * If the entry is a protein monomer and this row is a **protein chain**, label `[protein_monomer]`.
  * If the entry is a protein homomer, label `[protein_homomer]`.
  * If the entry is an RNA monomer and this row is an **RNA chain**, label `[rna_monomer]`.

### 6.3. Peptide and cyclic peptide interfaces

`identity_peptide()`:

* For **interfaces** only, classify based on `entity_type_{1,2}` and `seq_length_{1,2}` using a `peptide_threshold` (default 25):

  * Both chains short proteins → `[peptide-interface];[peptide-peptide]`.
  * Short protein vs long protein → `[peptide-interface];[peptide-protein]`.
  * Short protein vs DNA/RNA → `[peptide-interface];[peptide-dna]` / `[peptide-interface];[peptide-rna]`.

`identity_cyclic_peptide()`:

1. From rows labelled as peptide-interfaces, collect `entry_id`s.
2. For each mmCIF file:

   * Identify peptide entity IDs (`polypeptide(L)`).
   * For each such chain, if the number of unique residues ≤ `peptide_threshold` and at least one bond connects non-adjacent residue IDs (`|res_i - res_j| > 1`), treat the entity as **cyclic peptide**.
3. Label interfaces where one/both sides are cyclic-peptide entities as:

   * `[cyclic_peptide-interface];[cyclic_peptide-cyclic_peptide]`
   * `[cyclic_peptide-interface];[cyclic_peptide-protein]`
   * `[cyclic_peptide-interface];[cyclic_peptide-dna]` / `[cyclic_peptide-interface];[cyclic_peptide-rna]`.

### 6.4. Merge all subset labels

`identify_subset()`:

* Reads low-homology CSV.
* Computes antibody labels, monomer/homomer labels, peptide labels, and cyclic peptide labels.
* Concatenates them per row and joins non-NA entries with `";"` to form `subset`.
* Writes the updated CSV back to `recentpdb_low_homology.csv` (in-place).

---

## Step 7 - Copy ground-truth CIFs

**Script:** `benchmark/dataset_pipeline/step7_copy_true_cif.py`

**Goal:** Prepare a directory containing **ground-truth mmCIFs** for all entries in the low-homology dataset, to be used by downstream evaluation code.

`copy_true_cif()`:

* Reads the low-homology CSV (or another CSV with `entry_id` column).
* Builds `(input_path, output_path)` pairs for `<entry_id>.cif`.
* Two modes:

  * `copy_all_cif=True` (default):

    * If `symlink=True`, create a single symlink from `input_dir` to `output_dir`.
    * If `symlink=False`, copy the entire tree.
  * `copy_all_cif=False`:

    * Create `output_dir` and copy/symlink **only** the CIFs referenced in the CSV.
* Parallelised with `joblib.Parallel` when copying individual files.

This gives you a canonical `true_dir` that the evaluation pipeline can mount as ground truth.

---

## Step 8 - Homology for low-homology subset

**Script:** `benchmark/dataset_pipeline/step8_get_homology_for_lowh.py`

**Goal:** Although `recentpdb_low_homology.csv` already ensures low homology at certain sequence-identity thresholds, Step 8 recomputes and stores **fine-grained homology scores** between low-homology entities and the older training set, using a lower sequence-identity threshold (0.3) to support **stratification** of the evaluation set by homology level.

### 8.1. Select train vs low-homology test sequences

`get_homology_for_lowh()`:

1. Reads `pdb_seq_csv` into `seq_df` and filters `train_df = seq_df[release_date < after_date]`.
2. Reads `recentpdb_low_homology.csv` and uses `get_lowh_sequences()` to extract **only those sequences** that appear in the low-homology dataset (both sides of interfaces).

### 8.2. MMseqs2 with 0.3 identity threshold

For each entity type (protein/RNA/DNA), it calls `calc_mmseqs_seq_identity()` with:

* `threshold = 0.3`
* `e_value_cutoff = 0.1`
* `sensitivity = 7.5`
* `max_seqs = 500000`
* `cov_mode = 0`, `coverage = 0.0`
* `nuc=True` for nucleic acids.

The helper again:

* handles short sequences with identity-only matching,
* uses `mmseqs easy-search` for the rest,
* returns a DataFrame with `query_id`, `db_id`, `similarity`, `aligned_res_num`.

### 8.3. Output

All three entity types are concatenated into `merged_df`, down-cast with `shrink_dataframe()`, and saved as Parquet (e.g. `recentpdb_low_homology_entity_homo.parquet`).

This file can be used to bucket cases by “homology to pre-cutoff PDB” during evaluation.

---

## Step 9 - Dataset analysis

**Script:** `benchmark/dataset_pipeline/step9_analysis_dataset.py`

**Goal:** Summarise the dataset into human-readable statistics and plots:

* How many PDB complexes survive each filter step.
* Token count distributions.
* Counts per evaluation type / subset.
* PDB ID lists per eval-type / subset.

### 9.1. Filter-step statistics

`stat_filtered_num()` re-applies the meta-level filters to a PDB meta info CSV (`pdb_meta_info_csv`):

* Date filter: `[after_date, before_date]`;
* drop NMR (same NMR method set);
* resolution threshold (default 4.5 Å);
* token threshold (default 2560);
* require standard polymers and limit max polymer chain copies;
* remove entries where all chains are `unk` or all chains have breaks;
* require that at least one chain has sufficient resolved residues.

It returns a dict of counts by step; `_log_filtered_num_stat()` turns this into a formatted text block.

### 9.2. Low-homology subset statistics

`stat_lowh_num()`:

1. Reads `recentpdb_low_homology_cluster_info.csv`, normalises `label_entity_id` to string.
2. Joins cluster IDs into the low-homology DataFrame (`add_cluster_id_to_df()`), using polymer cluster IDs for interfaces where one side is ligand or short chain.
3. Computes:

   * Number of complexes (`#unique entry_id`), chains, interfaces.
   * For each evaluation type (configured in `eval_type_config`):

     * number of rows,
     * number of unique clusters,
     * number of PDB IDs.
   * Token number lists (per complex and per chain/interface).
4. Explodes the `subset` column and counts occurrences per subset label, plus PDB ID and cluster counts per subset.

`_log_lowh_num()` formats these into human-readable text.

### 9.3. Plots and PDB ID lists

`_draw_token_num_plot()`:

* Saves histograms of token counts for:

  * Complex-level token counts (`num_tokens` field from meta info).
  * Chain/interface token counts per eval type.

All plots are written as PNG files under `output_dir/figs`.

`run_data_analysis()` writes:

* `stat.txt`:

  * `<DATA FILTERING PIPELINE STATISTICS>` section (step-wise counts).
  * `<LOW-HOMOLOGY SUBSET STATISTICS>` section (per eval type / subset).
* PDB ID lists under `output_dir/pdb_ids`:

  * `all.txt`, `lowh_polymer_only.txt`, plus one file per eval type and per subset label.
