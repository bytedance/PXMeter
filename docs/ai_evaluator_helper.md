### **💡 Tips for Non-Developers: Generate Evaluator with AI**

If you are not familiar with Python, you can use LLMs to generate the `evaluator.py` for you.

#### **Step 1: Get your directory structure**
Run the following command in your terminal to get the structure of **one representative PDB folder**. This helps the AI understand the file hierarchy:
```bash
# Replace /path/to/your/predictions with your actual path
# This command picks the first subdirectory and shows its structure
tree /path/to/your/predictions/$(ls -F /path/to/your/predictions | grep / | head -n 1) -L 3
```

#### **Step 2: Use this Prompt Template**
Copy the following template, fill in your directory structure, and send it to an LLM:

```text
I am using a protein folding benchmark tool - PXMeter. I need to implement a Python class `MyModelEvaluator` that inherits from `BaseEvaluator` to parse my model's output directory.

My prediction directory contains many subfolders (one for each PDB). Here is the structure of **one representative PDB folder**:

[PASTE YOUR TREE OUTPUT HERE]

Please help me implement the `_get_info_from_each_pdb_dir(self, pdb_dir: Path) -> list` method.

Requirements:
1. `pdb_dir` is a direct subdirectory of the root prediction directory (e.g., the folder named `7rss`).
   - Crucial: Please check the tree output above carefully. If the files are inside a nested folder (like `7rss/7rss/model.cif`), your code MUST handle this extra level.
2. The method should return a list of tuples, where each tuple is: `(name, pdb_id, seed, sample, pred_cif_path, confidence_json_path, None)`.
3. `name` is usually the folder name. `pdb_id` is the PDB ID (e.g., "7rss").
4. It should correctly identify the `seed` and `sample` index from the file or folder names.
5. Only include samples where both the `.cif` and `.json` files exist.

Here is a reference implementation for a different directory structure (do not copy it exactly, but use it as a logic guide):
---
from pathlib import Path
from benchmark.evaluators.base import BaseEvaluator

class MyModelEvaluator(BaseEvaluator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_info_from_each_pdb_dir(self, pdb_dir: Path) -> list:
        name = pdb_dir.name
        pdb_id = name
        sub_data = []
        for seed_dir in pdb_dir.iterdir():
            if not seed_dir.is_dir(): continue
            seed = seed_dir.name
            for cif_path in seed_dir.glob("*.cif"):
                sample = cif_path.stem.split("_")[-1]
                conf_path = seed_dir / f"sample_{sample}.json"
                if conf_path.exists():
                    sub_data.append((name, pdb_id, seed, sample, cif_path, conf_path, None))
        return sub_data
---

Please provide the complete `evaluator.py` code based on MY directory structure provided above, including all necessary imports and the full class definition.
```
