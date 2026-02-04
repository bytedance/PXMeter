
# Dataset Output Files Reference

This document provides a detailed description of all files produced by the dataset construction pipeline.

For the full end-to-end dataset construction process, see
[Dataset Pipeline Overview](datapipeline.md).

Outputs are organized into two categories:

1. **SRC_DATA** - Intermediate and source data files used during dataset construction.
2. **SUPPORTED_DATA** - Final curated files required for evaluation.

Each file’s format, key fields, and intended use are described below.

## Table of Contents
- [1. SRC_DATA](#1-src_data)
  - [1.1. PDB Meta Info](#11-pdb-meta-info)
  - [1.2. PDB Sequences](#12-pdb-sequences)
  - [1.3. RecentPDB Chain / Interface Index](#13-recentpdb-chain--interface-index)
  - [1.4. Entity-Type Counts (Low Homology Subset)](#14-entity-type-counts-low-homology-subset)
  - [1.5. Train-Test High-Similarity Matches](#15-train-test-high-similarity-matches)
- [2. SUPPORTED_DATA](#2-supported_data)
  - [2.1. Low Homology Chain / Interface Index](#21-low-homology-chain--interface-index)
  - [2.2. Low Homology Cluster File](#22-low-homology-cluster-file)
  - [2.3. Low-Homology → Train Homology](#23-low-homology--train-homology-parquet-03-threshold)
  - [2.4. Ligand Information File](#24-ligand-information-file)
  - [2.5. Dataset Statistics](#25-dataset-statistics)
    - [a. Statistic Report](#a-statistic-report)
    - [b. PDB ID Lists](#b-pdb-id-lists)
    - [c. Token Distribution Figures](#c-token-distribution-figures)

---

# 1. SRC_DATA

(Intermediate products / files generated during dataset filtering and preparation)

`src_data/`

---

## 1.1. PDB Meta Info

**File:** `src_data/pdb_meta_info.csv`
Contains metadata for **all mmCIF entries**, including structure quality, polymer composition, and filter status.

### Columns

| Column              | Meaning                                                     | Type       | Notes                                  |
| ------------------- | ----------------------------------------------------------- | ---------- | -------------------------------------- |
| entry_id            | PDB ID                                                      | str        | —                                      |
| exptl_methods       | Experimental methods                                        | str        | Multiple values separated by `";"`     |
| classification      | Classification field from the RCSB PDB record               | str        | -                                      |
| release_date        | Public release date                                         | YYYY-MM-DD | —                                      |
| resolution          | Resolution                                                  | int        | Missing: `-1`                          |
| num_tokens          | Token count (AF3-style tokenization)                        | int        | Filtered-out values set to `-1`        |
| no_standard_polymer | No Protein(L), DNA, RNA in the entry                        | bool       | Filtered-out values default to `False` |
| max_chain_copies    | Max number of chains within any entity                      | int        | Filtered-out values: `-1`              |
| lacking_resolved    | No chain meets the resolved-residue requirement (>4 & >30%) | bool       | Filtered-out values default to `False` |
| all_chains_unk      | All polymer chains contain only unknown residues            | bool       | Filtered-out values default to `False` |
| all_chains_break    | All polymer chains contain Cα-Cα breaks                     | bool       | Filtered-out values default to `False` |
| pass_filter         | Whether the entry passes all Step 1 filters                 | bool       | Determines candidate RecentPDB set     |

---

## 1.2. PDB Sequences

**File:** `src_data/pdb_seqs.csv`
Contains **all entity sequences** extracted from mmCIF files.

### Columns

`entry_id, entity_id, release_date, entity_type, seq, seq_length`

Includes all polymer and non-polymer entity sequences.

---

## 1.3. RecentPDB Chain / Interface Index

**File:** `src_data/RecentPDB_chain_interface.csv`
Lists **all chains and interfaces** in the evaluation date window after Step 1 filtering.

### Columns

| Column                | Meaning                                    | Type | Notes                                                            |
| --------------------- | ------------------------------------------ | ---- | ---------------------------------------------------------------- |
| id                    | Unified ID for complex / chain / interface | str  | `pdbid`/ `pdbid_chain1`/ `pdbid_chain1_chain2` (sorted chain IDs)|
| type                  | `"chain"` or `"interface"`                 | str  | —                                                                |
| entry_id              | PDB ID                                     | str  | —                                                                |
| entity_id_1           | Entity ID of chain 1                       | str  | —                                                                |
| entity_type_1         | Entity type of chain 1                     | str  | Protein / DNA / RNA / Ligand / etc.                              |
| entity_type_2         | Entity type of chain 2                     | str  | Only for interfaces                                              |
| chain_id_1            | label_asym_id of chain 1                   | str  | Non-Asym Unit chains may contain `"."`, e.g., `"A.1"`            |
| chain_id_2            | label_asym_id of chain 2                   | str  | Same note as above                                               |
| auth_chain_id_1       | auth_asym_id of chain 1                    | str  | —                                                                |
| auth_chain_id_2       | auth_asym_id of chain 2                    | str  | —                                                                |
| seq_length_1          | Length of chain 1 sequence                 | int  | —                                                                |
| seq_length_2          | Length of chain 2 sequence                 | int  | —                                                                |
| resolved_seq_length_1 | Number of resolved residues on chain 1     | int  | —                                                                |
| resolved_seq_length_2 | Number of resolved residues on chain 2     | int  | —                                                                |

---

## 1.4. Entity-Type Counts (Low Homology Subset)

**File:**
`src_data/RecentPDB_low_homology_entity_types_count.csv`

Contains **entity-type composition** for entries in the Low Homology subset.

### Columns

`entry_id, polypeptide(L), ligand, polyribonucleotide, polydeoxyribonucleotide, polydeoxyribonucleotide/polyribonucleotide hybrid, polypeptide(D), is_protein_monomer, is_rna_monomer`

Used to determine monomer / homomer labels.

---

## 1.5. Train-Test High-Similarity Matches

**File:** `src_data/test_to_train_entity_homo.parquet`

This Parquet file stores the **similarity relationships between test-set entities** (those released between `after_date` and `before_date`) **and training-set entities** (those released on or before `after_date`).

The file is written in columnar **Parquet** format using `pyarrow` with `zstd` compression.

### Columns

| Column              | Meaning                    | Type      | Notes                                                           |
| ------------------- | -------------------------- | --------- | --------------------------------------------------------------- |
| **query_id**        | Test-set entity ID         | string    | Format: `"<test_entry_id>_<entity_id>"`                         |
| **db_id**           | Training-set entity ID     | string    | Format: `"<train_entry_id>_<entity_id>"`                        |
| **similarity**      | Similarity score           | float     | Protein/DNA/RNA: sequence identity; ligand: Tanimoto similarity |
| **aligned_res_num** | Number of aligned residues | float/int | Protein/DNA/RNA: aligned residue count; ligand: NaN             |

Notes:

* For **proteins and nucleic acids**, similarity and alignment come from MMseqs2 (or exact-identity matching for very short sequences).
* For **ligands**, similarity comes from Morgan fingerprint Tanimoto values; `aligned_res_num` is not applicable.
* This Parquet file is used in Step 3 for low-homology filtering.

---

# 2. SUPPORTED_DATA

(Final curated dataset files for evaluation)

`supported_data/`

---

## 2.1. Low Homology Chain / Interface Index

**File:** `supported_data/RecentPDB_low_homology.csv`

Final chain and interface records after all filters (including homology, ligand QC, short-chain rules, symmetry checks).

### Columns

`id, type, entry_id, entity_id_1, entity_id_2, entity_type_1, entity_type_2, chain_id_1, chain_id_2, auth_chain_id_1, auth_chain_id_2, seq_length_1, seq_length_2, resolved_seq_length_1, resolved_seq_length_2, subset`

### Subset Labels

A chain or interface can have **multiple labels**, separated by `";"`.

Example:

```
[antibody-protein];[antibody_HL-protein]
```

### Supported Labels

#### **Chain-level**

* `[antibody]`

  * `[antibody_HL]`
  * `[antibody_scFv]`
  * `[antibody_H]`
  * `[antibody_L]`
* `[protein-monomer]`

#### **Interface-level**

* `[antibody-antibody]`
* `[antibody-protein]`

  * `[antibody_HL-protein]`
  * `[antibody_scFv-protein]`
  * `[antibody_H-protein]`
  * `[antibody_L-protein]`
* `[peptide-interface]`

  * `[peptide-peptide]`
  * `[peptide-protein]` (polymer length ≥ 25)
  * `[peptide-dna]`
  * `[peptide-rna]`

**Antibody annotation source:**
SAbDab summary TSV

* Hchain = NaN → `[antibody_L]`
* Lchain = NaN → `[antibody_H]`
* Both present → `[antibody_HL]`

---

## 2.2. Low Homology Cluster File

**File:** `supported_data/RecentPDB_low_homology_cluster_info.csv`

Lists cluster IDs for all polymer and ligand entities in the low-homology dataset.

### Columns

`entry_id, label_entity_id, cluster_id, entity_type`

* Polymer clusters come from MMseqs2 clustering
* Ligand clusters use CCD code directly

---

## 2.3. Low-Homology → Train Homology (Parquet, 0.3 Threshold)

**File:** `supported_data/RecentPDB_low_homology_entity_homo.parquet`

This Parquet file stores **sequence homology relationships** between **entities in the Low Homology subset** and all **training-set entities** (release date < `after_date`), using a relaxed sequence identity threshold of **0.3**.
It is primarily used for **grouping evaluation samples by homology**.

### Columns

| Column              | Meaning                    | Type      | Notes                                                     |
| ------------------- | -------------------------- | --------- | --------------------------------------------------------- |
| **query_id**        | Low-homology entity ID     | string    | Format: `"<lowh_entry_id>_<entity_id>"`                   |
| **db_id**           | Training-set entity ID     | string    | Format: `"<train_entry_id>_<entity_id>"`                  |
| **similarity**      | Sequence identity          | float     | Smith-Waterman or MMseqs-based identity (≥ 0.3)           |
| **aligned_res_num** | Number of aligned residues | float/int | Number of query residues aligned to the training sequence |

Notes:

* Only **polymer entities** (protein, DNA, RNA) are included; **ligands are excluded**.
* Multiple rows per `query_id` are possible because a test entity may align to multiple training entities.

---

## 2.4. Ligand Information File

**File:** `supported_data/RecentPDB_low_homology_lig_info.csv`
This file lists the ligand chains associated with structures in the low-homology dataset.

**Columns:**

* `entry_id` - The PDB entry identifier.
* `label_asym_id` - The chain identifier of the ligand within the entry.

In the evaluation pipeline, this file is also used to identify ligand entities that require
**pocket-aligned RMSD computation** and **PoseBusters ligand validity checks**.
Only ligands listed in this file will be included in these ligand-specific quality assessments during benchmarking.

---

## 2.5. Dataset Statistics

Directory: `supported_data/stat_data/`

Contains all statistical summaries and plots.

### a. Statistic Report

**File:** `stat.txt`

Includes:

* Step 1 filtering counts
* Complex / Chain / Interface counts
* Per-eval-type and per-subset statistics
* Cluster counts

### b. PDB ID Lists

Directory: `supported_data/stat_data/pdb_ids/`

Contains:

* `all` - all low-homology PDB IDs
* `lowh_polymer_only` - IDs containing only polymer low-homology data
* Per-subset and per-eval-type lists for controlled evaluation sampling

### c. Token Distribution Figures

Directory: `supported_data/stat_data/figs/`

Contains token-count histograms for:

* Complex-level tokens
* Chain/interface-level tokens
* Per-eval-type token distributions
