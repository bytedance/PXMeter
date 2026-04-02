# Available datasets

With the exception of PXM-Legacy, each dataset provides both `src_data` and `supported_data`.
For evaluation purposes, downloading `supported_data` is sufficient.
If you wish to perform additional filtering, please use the files in `src_data`.
Refer to [datapipeline_output.md](datapipeline_output.md) for detailed file descriptions.

## Table of Contents

- [Annual Evaluation Benchmark](#annual-evaluation-benchmark)
  - [PXM-2025 dataset (2025-01-01 to 2025-12-31)](#pxm-2025-dataset-2025-01-01-to-2025-12-31)
  - [PXM-2024 dataset (2024-01-01 to 2024-12-31)](#pxm-2024-dataset-2024-01-01-to-2024-12-31)
- [Supplementary Evaluation Sets](#supplementary-evaluation-sets)
  - [PXM-24to25 dataset (2024-01-01 to 2025-12-31)](#pxm-24to25-dataset-2024-01-01-to-2025-12-31)
  - [PXM-22to25 dataset (2022-01-01 to 2025-12-31)](#pxm-22to25-dataset-2022-01-01-to-2025-12-31)
  - [PXM-2025-H2 dataset (2025-07-01 to 2025-12-31)](#pxm-2025-h2-dataset-2025-07-01-to-2025-12-31)
  - [PXM-Legacy dataset (Initial PXMeter Release Dataset)](#pxm-legacy-dataset-initial-pxmeter-release-dataset)

## Annual Evaluation Benchmark
### **PXM-2025 dataset (2025-01-01 to 2025-12-31)**

`supported_data` (603.58 MB): https://pxmeter.tos-cn-beijing.volces.com/PXM-2025-sup-only.tar.gz

`src_data` + `supported_data` (1.27 GB): https://pxmeter.tos-cn-beijing.volces.com/PXM-2025.tar.gz

```
# Complex: 2359
# Chain: 3722
# Interface: 5638
------------------
# Intra-Protein: 3005 (# PDB: 1539, # Cluster: 848)
# Intra-RNA: 246 (# PDB: 195, # Cluster: 95)
# Intra-DNA: 353 (# PDB: 182, # Cluster: 144)
# Intra-Ligand: 118 (# PDB: 117, # Cluster: 113)
# Protein-Protein: 3743 (# PDB: 1348, # Cluster: 1095)
# DNA-DNA: 206 (# PDB: 159, # Cluster: 102)
# RNA-RNA: 64 (# PDB: 56, # Cluster: 28)
# Protein-Ligand: 178 (# PDB: 155, # Cluster: 79)
# RNA-Protein: 347 (# PDB: 161, # Cluster: 121)
# DNA-Protein: 1018 (# PDB: 244, # Cluster: 290)
# DNA-RNA: 80 (# PDB: 68, # Cluster: 50)
# DNA-Ligand: 2 (# PDB: 2, # Cluster: 1)
------------------
# [protein_homomer]: 1295 (# PDB: 172, # Cluster: 261)
# [protein_monomer]: 474 (# PDB: 474, # Cluster: 242)
# [antibody-protein]: 314 (# PDB: 131, # Cluster: 95)
# [antibody_HL-protein]: 201 (# PDB: 69, # Cluster: 63)
# [antibody_H-protein]: 107 (# PDB: 56, # Cluster: 31)
# [peptide-interface]: 70 (# PDB: 22, # Cluster: 21)
# [peptide-protein]: 70 (# PDB: 22, # Cluster: 21)
# [rna_monomer]: 51 (# PDB: 51, # Cluster: 17)
# [antibody_scFv-protein]: 6 (# PDB: 6, # Cluster: 1)
# [cyclic_peptide-interface]: 4 (# PDB: 3, # Cluster: 3)
# [cyclic_peptide-protein]: 4 (# PDB: 3, # Cluster: 3)
```

### **PXM-2024 dataset (2024-01-01 to 2024-12-31)**

`supported_data` (545.99 MB): https://pxmeter.tos-cn-beijing.volces.com/PXM-2024-sup-only.tar.gz

`src_data` + `supported_data` (1.15 GB): https://pxmeter.tos-cn-beijing.volces.com/PXM-2024.tar.gz

```
# Complex: 2051
# Chain: 3150
# Interface: 4745
------------------
# Intra-Protein: 2651 (# PDB: 1371, # Cluster: 814)
# Intra-RNA: 176 (# PDB: 140, # Cluster: 71)
# Intra-DNA: 176 (# PDB: 110, # Cluster: 94)
# Intra-Ligand: 147 (# PDB: 147, # Cluster: 145)
# Protein-Protein: 3363 (# PDB: 1228, # Cluster: 960)
# DNA-DNA: 112 (# PDB: 78, # Cluster: 69)
# RNA-RNA: 48 (# PDB: 38, # Cluster: 23)
# Protein-Ligand: 198 (# PDB: 187, # Cluster: 74)
# RNA-Protein: 392 (# PDB: 157, # Cluster: 118)
# DNA-Protein: 553 (# PDB: 168, # Cluster: 211)
# DNA-RNA: 78 (# PDB: 53, # Cluster: 42)
# RNA-Ligand: 1 (# PDB: 1, # Cluster: 1)
------------------
# [protein_homomer]: 1156 (# PDB: 158, # Cluster: 256)
# [protein_monomer]: 424 (# PDB: 424, # Cluster: 280)
# [antibody-protein]: 220 (# PDB: 94, # Cluster: 82)
# [antibody_H-protein]: 116 (# PDB: 49, # Cluster: 27)
# [antibody_HL-protein]: 103 (# PDB: 44, # Cluster: 54)
# [peptide-interface]: 34 (# PDB: 17, # Cluster: 15)
# [peptide-protein]: 34 (# PDB: 17, # Cluster: 15)
# [rna_monomer]: 24 (# PDB: 24, # Cluster: 11)
# [antibody_scFv-protein]: 1 (# PDB: 1, # Cluster: 1)
```


## Supplementary Evaluation Sets

### **PXM-24to25 dataset (2024-01-01 to 2025-12-31)**

`supported_data` (2.39 GB): https://pxmeter.tos-cn-beijing.volces.com/PXM-24to25-sup-only.tar.gz

`src_data` + `supported_data` (4.28 GB): https://pxmeter.tos-cn-beijing.volces.com/PXM-24to25.tar.gz

```
<DATA FILTERING PIPELINE STATISTICS>
# Total: 246905
# FilteredByDate: 32881 (-214024)
# ExcludeNMR: 32356 (-525)
# FilteredByResolution: 31627 (-729)
# FilteredByTokenCount: 25320 (-6307)
# RequireStandardPolymer: 25318 (-2)
# LimitPolymerChainCopies: 25284 (-34)
# ExcludeAllChainsUnknown: 25284 (0)
# RequireResolvedStructure: 25239 (-45)


<LOW-HOMOLOGY SUBSET STATISTICS>
# Complex: 4691
# Chain: 7356
# Interface: 11095
------------------
# Intra-Protein: 6089 (# PDB: 3115, # Cluster: 1706)
# Intra-RNA: 456 (# PDB: 367, # Cluster: 167)
# Intra-DNA: 544 (# PDB: 306, # Cluster: 242)
# Intra-Ligand: 267 (# PDB: 267, # Cluster: 260)
# Protein-Protein: 7647 (# PDB: 2780, # Cluster: 2130)
# DNA-DNA: 322 (# PDB: 241, # Cluster: 176)
# RNA-RNA: 116 (# PDB: 98, # Cluster: 53)
# Protein-Ligand: 387 (# PDB: 351, # Cluster: 145)
# RNA-Protein: 804 (# PDB: 351, # Cluster: 256)
# DNA-Protein: 1645 (# PDB: 438, # Cluster: 531)
# DNA-RNA: 171 (# PDB: 134, # Cluster: 97)
# DNA-Ligand: 2 (# PDB: 2, # Cluster: 1)
# RNA-Ligand: 1 (# PDB: 1, # Cluster: 1)
------------------
# [protein_homomer]: 2570 (# PDB: 354, # Cluster: 541)
# [protein_monomer]: 951 (# PDB: 951, # Cluster: 534)
# [antibody-protein]: 583 (# PDB: 244, # Cluster: 179)
# [antibody_HL-protein]: 340 (# PDB: 126, # Cluster: 120)
# [antibody_H-protein]: 236 (# PDB: 111, # Cluster: 57)
# [peptide-interface]: 136 (# PDB: 41, # Cluster: 40)
# [peptide-protein]: 136 (# PDB: 41, # Cluster: 40)
# [rna_monomer]: 77 (# PDB: 77, # Cluster: 28)
# [antibody_scFv-protein]: 7 (# PDB: 7, # Cluster: 2)
# [cyclic_peptide-interface]: 4 (# PDB: 3, # Cluster: 3)
# [cyclic_peptide-protein]: 4 (# PDB: 3, # Cluster: 3)
```

### **PXM-22to25 dataset (2022-01-01 to 2025-12-31)**

`supported_data` (2.39 GB): https://pxmeter.tos-cn-beijing.volces.com/PXM-22to25-sup-only.tar.gz

`src_data` + `supported_data` (4.28 GB): https://pxmeter.tos-cn-beijing.volces.com/PXM-22to25.tar.gz

```
# Chain: 15070
# Interface: 21828
------------------
# Intra-Protein: 12890 (# PDB: 6647, # Cluster: 3454)
# Intra-RNA: 771 (# PDB: 620, # Cluster: 303)
# Intra-DNA: 944 (# PDB: 544, # Cluster: 444)
# Intra-Ligand: 465 (# PDB: 465, # Cluster: 451)
# Protein-Protein: 15606 (# PDB: 5795, # Cluster: 4289)
# DNA-DNA: 541 (# PDB: 424, # Cluster: 314)
# RNA-RNA: 214 (# PDB: 186, # Cluster: 111)
# Protein-Ligand: 697 (# PDB: 629, # Cluster: 259)
# RNA-Protein: 1473 (# PDB: 616, # Cluster: 518)
# DNA-Protein: 3017 (# PDB: 778, # Cluster: 975)
# DNA-RNA: 277 (# PDB: 219, # Cluster: 170)
# DNA-Ligand: 2 (# PDB: 2, # Cluster: 1)
# RNA-Ligand: 1 (# PDB: 1, # Cluster: 1)
------------------
# [protein_homomer]: 4822 (# PDB: 684, # Cluster: 1007)
# [protein_monomer]: 2092 (# PDB: 2092, # Cluster: 1148)
# [antibody-protein]: 1209 (# PDB: 528, # Cluster: 383)
# [antibody_HL-protein]: 755 (# PDB: 278, # Cluster: 258)
# [antibody_H-protein]: 440 (# PDB: 240, # Cluster: 119)
# [peptide-interface]: 253 (# PDB: 122, # Cluster: 112)
# [peptide-protein]: 253 (# PDB: 122, # Cluster: 112)
# [rna_monomer]: 100 (# PDB: 100, # Cluster: 41)
# [antibody_scFv-protein]: 14 (# PDB: 11, # Cluster: 6)
# [cyclic_peptide-interface]: 14 (# PDB: 13, # Cluster: 9)
# [cyclic_peptide-protein]: 14 (# PDB: 13, # Cluster: 9)
# [antibody]: 1 (# PDB: 1, # Cluster: 1)
# [antibody_HL]: 1 (# PDB: 1, # Cluster: 1)
```

### **PXM-Legacy dataset (Initial PXMeter Release Dataset)**

`supported_data` (~ 515 MB): https://pxmeter.tos-cn-beijing.volces.com/PXM-Legacy.tar.gz

This dataset contains the subsets included in the initial PXMeter release:
RecentPDB, PoseBusters, AF3-AB, RNA-Protein, and dsDNA-Protein.
The RecentPDB portion uses the time window `2022-05-01` to `2023-01-12`.

This dataset was not generated by the current data pipeline,
but its format has been updated to remain compatible with the current evaluation code.
The preserved information from the original version can be found
in the legacy documentation: **[Legacy Dataset Reference](legacy_dataset_reference.md)**
