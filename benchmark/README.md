# Benchmark for Biomolecular Structure Prediction Models

<div align="left" style="margin: 20px 0;">
<span style="margin: 0 10px;">📄 <a href="URL">From Dataset Diagnostics to Evaluation: Revisiting
Structure Prediction Benchmarks with PXMeter</a></span>
</div>


This repository provides evaluation codes for assessing models using our curated evaluation sets:
| Dataset       | Description                                                                                                  | Metrics                             |
|---------------|--------------------------------------------------------------------------------------------------------------|-------------------------------------|
| RecentPDB     | Evaluates RecentPDB low homology protein subset. Antibody-antigen and monomer subsets reported separately.   | DockQ success rates (>0.23), LDDT   |
| AF3-AB        | Analyses antibody-antigen subset of AlphaFold3.                                                              | DockQ success rates (>0.23), LDDT   |
| RecentPDB-NA  | Focuses on low homology nucleic acids subset of RecentPDB, aggregating LDDT for intra-DNA chains, intra-RNA chains, DNA-protein, and RNA-protein interfaces. | LDDT                                |
| dsDNA-Protein | Focuses on intra-DNA chains and DNA-protein interfaces, aggregating LDDT metrics.                            | LDDT                                |
| RNA-Protein   | Evaluates intra-DNA chains and RNA-protein interfaces with LDDT aggregation.                                 | LDDT                                |
| PoseBusters   | Assesses pocket-aligned RMSD of small molecules (PoseBusters V2).                                            | RMSD success rates (< 2 Å)          |

## 💡 Usage

### 0. Download the dataset

The benchmark data is licensed under CC0.

Before running the code for the first time, download the necessary dataset files:
```bash
wget https://pxmeter.tos-cn-beijing.volces.com/evaluation_supported_data.tar.gz
tar xzvf evaluation_supported_data.tar.gz -C [your_path]
```

To download the full dataset used in our article, including model input files, inference output structures, evaluation output JSON files, and summaries:
```bash
wget https://pxmeter.tos-cn-beijing.volces.com/evaluation_full_data.tar.gz
tar xzvf evaluation_full_data.tar.gz -C [your_path]
```

You can place the extracted `evaluation` folder in the directory from which you run the benchmark code, or specify its location via an environment variable:
```bash
export PXM_EVAL_DATA_ROOT_PATH="your/path/to/evaluation"
```

### 1. Evaluate inference results

Run the evaluation script as follows:
```bash
python benchmark/run_eval.py -i [infer_dir] -o [output_dir] -d [dataset] -c [chunk_str] -m [model] -n [num_cpu]
```

- `infer_dir`: Directory containing inference results (CIF files and confidence files).
- `output_dir`: Directory where evaluation results will be saved.
- `dataset`: Dataset to evaluate (options: "RecentPDB", "RecentPDB-NA", "PoseBusters", "dsDNA-Protein", "RNA-Protein", "AF3-AB").
- `chunk_str`: Chunk string for distributed evaluation (e.g., '1of5'), used when running evaluations across multiple machines. The default is None.
- `model`: Model name should be one of the supported options: "protenix", "boltz", "chai" or "af2m", as defined in "benchmark.run_eval.run_batch_eval", which calls corresponding Evaluators.
- `num_cpu`: Number of CPU cores to use. Default is 1.


### 2. Create JSON describing evaluation results paths

Example JSON structure:
```json
{
  "name": {
    "model": "model name (protenix, af2m, chai, boltz)",
    "seeds": [101, 102, "..."],
    "dataset_path": {
      "RecentPDB": "path/to/eval_results/RecentPDB",
      "RecentPDB-NA": "path/to/eval_results/RecentPDB_NA",
      "PoseBusters": "path/to/eval_results/PoseBusters",
      "AF3-AB": "path/to/eval_results/AF3_AB"
    }
  }
}
```

Allowed keys for `dataset_path` are:
`"RecentPDB"`, `"RecentPDB-NA"`, `"PoseBusters"`, `"dsDNA-Protein"`, `"RNA-Protein"`, and `"AF3-AB"`.
Each dataset can include only a subset of these keys.


### 3. Aggregate and display evaluation results

Run the aggregation script:
python benchmark/show_intersection_results.py -i [input_json] -o [output_path] -d [dataset_names] -n [num_cpu] -c [subset_csv] --overwrite_agg

Parameters:
- `input_json`: Path to JSON file with evaluation results (created in step 2).
- `output_path` (optional): Directory for final aggregated CSV output. Defaults to "./pxm_results".
- `dataset_names` (optional): Comma-separated dataset names to compare using intersections of "chain" and "interface". Defaults to all names in JSON.
- `num_cpu` (optional): Number of CPU cores for aggregation. Defaults to all available cores.
- `overwrite_agg` (optional): Overwrite existing aggregated CSV files. Defaults to False.
- `subset_csv` (optional): CSV file with columns ["type", "entry_id", "chain_id_1", "chain_id_2"]. Used to subset results. "type" can be "chain" or "interface".


#### Output structure

The script outputs summary CSV files in the directories specified by `dataset_path` in the JSON and creates a consolidated CSV with aggregated metrics in `output_path`:

```bash
pxm_results
├── DockQ_details.csv
├── DockQ_results.csv
├── LDDT_details.csv
├── LDDT_results.csv
├── RMSD_details.csv
├── RMSD_results.csv
├── Summary_table.csv
└── Summary_table.txt
```

- `Summary_table.csv` and `Summary_table.txt` provide a concise overview of key metrics such as DockQ success rate, PoseBusters success rate, and LDDT (with protein-protein interfaces indicated as prot_prot).
- `*_results.csv` files provide a full view of aggregated evaluation metrics.
- `*_details.csv` files allow exploration of sample-specific metrics selected by each ranker.


## 🔄 Reproduction of RecentPDB Low Homology Dataset Construction

This section outlines steps to construct the RecentPDB Low Homology dataset. File paths required for scripts are specified in `benchmark.configs.data_config`. It’s recommended to specify new output paths in scripts to prevent overwriting default files.

### 1. Filter to RecentPDB

Run this command to filter RecentPDB entries:
```bash
python benchmark/scripts/filter_for_recentpdb.py -c [mmcif_dir] -o [chain_interface_csv] -m [meta_csv] -n [num_cpu]
```

- `mmcif_dir`: Directory with MMCIF files downloaded from RCSB PDB.
- `chain_interface_csv`: Output CSV file recording filtered chain and interface information for RecentPDB dataset.
- `meta_csv`: Output CSV file containing meta information of filtered entries.
- `num_cpu`: Number of CPU cores for processing. Recommended CPU-to-memory (GB) ratio is 1:4.


### 2. Filter to Low Homology Subset

Run this command to filter to the Low Homology subset:
```bash
python benchmark/scripts/filter_to_lowh.py -c [chain_interface_csv] -t [test_to_train_json] -o [output_protein_lowh_csv] -n [output_nuc_lowh_csv]
```

- `chain_interface_csv`: Input CSV with filtered chain and interface info from RecentPDB.
- `test_to_train_json`: A JSON file containing a dictionary where keys are entities from the test set in the format '{PDB ID}_{Entity ID}', and values are lists of similar entities (high homology) from the train set in the same format.
- `output_protein_lowh_csv`: Output CSV with filtered chain and interface info for RecentPDB Low Homology Protein subset.
- `output_nuc_lowh_csv`: Output CSV with filtered chain and interface info for RecentPDB Low Homology Nucleic Acids subset.
