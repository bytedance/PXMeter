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
import os
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from scipy import stats

from benchmark.configs.eval_type_config import EVAL_TYPE_TO_ENTITIY_TYPES
from pxmeter.constants import POLYMER


def divide_list_into_chunks(lst: list, n: int) -> list[list]:
    """
    Divide a Sequence into n approximately equal-sized chunks.

    Args:
        lst (list[Any]): The list to be divided.
        n (int): The number of chunks to create.

    Returns:
        list[list[Any]]: A list of n chunks, where each chunk is a sublist of lst.
    """
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def int_to_letters(n: int) -> str:
    """
    Convert int to letters.
    Useful for converting chain index to label_asym_id.

    Args:
        n (int): int number
    Returns:
        str: letters. e.g. 1 -> A, 2 -> B, 27 -> AA, 28 -> AB
    """
    result = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        result = chr(65 + remainder) + result
    return result


def nested_dict_to_sorted_list(data: dict | Any) -> list | Any:
    """
    Convert a nested dictionary into a sorted list.

    This function takes a nested dictionary and converts it into a sorted list.
    If the input is a dictionary, it sorts the keys and recursively processes the values.
    If the input is not a dictionary, it returns the value directly.

    Args:
        data (dict or Any): The input data, which can be a dictionary or any other type.

    Returns:
        list or Any: The sorted list or the original value if the input is not a dictionary.
    """
    if isinstance(data, dict):
        # If the input is a dictionary, sort the keys and recursively process the values
        try:
            for i in data.keys():
                int(i)
            key_type = int
        except ValueError:
            key_type = str
        return [
            nested_dict_to_sorted_list(data[key])
            for key in sorted(data.keys(), key=key_type)
        ]
    else:
        # If the input is not a dictionary, return the value directly
        return data


def get_infer_cif_path(
    infer_output_dir: Path, model: str, entry_id: str, seed: str, sample: str
) -> Path:
    """
    Get the path to the inferred CIF file based on the model name.

    Args:
        infer_output_dir (Path): The directory where inference outputs are stored.
        model (str): The name of the model used for inference.
        entry_id (str): The identifier for the entry.
        seed (str): The seed value used in the inference process.
        sample (str): The sample identifier.

    Returns:
        Path: The path to the inferred CIF file.

    Raises:
        NotImplementedError: If the provided model name is not recognized.
    """
    if model == "af3":
        if (infer_output_dir / entry_id / entry_id).exists():
            cif_path = (
                infer_output_dir
                / entry_id
                / entry_id
                / f"seed-{seed}_sample-{sample}"
                / f"{entry_id}_seed-{seed}_sample-{sample}_model.cif"
            )
        else:
            cif_path = (
                infer_output_dir
                / entry_id
                / f"seed-{seed}_sample-{sample}"
                / f"{entry_id}_seed-{seed}_sample-{sample}_model.cif"
            )
    elif model == "protenix":
        cif_path = (
            infer_output_dir
            / entry_id
            / entry_id
            / f"seed_{seed}"
            / "predictions"
            / f"{entry_id}_sample_{sample}.cif"
        )
    elif model == "chai":
        cif_path = infer_output_dir / entry_id / seed / f"pred.model_idx_{sample}.cif"
    elif model == "boltz":
        cif_path = (
            infer_output_dir
            / entry_id
            / f"seed_{seed}"
            / f"boltz_results_{entry_id}"
            / "predictions"
            / entry_id
            / f"{entry_id}_model_{sample}.cif"
        )
    else:
        raise NotImplementedError(f"Unknown model: {model}")
    return cif_path


def get_eval_result_json_path(
    eval_result_dir: Path, entry_id: str, seed: str, sample: str
) -> Path:
    """
    Get the path to the evaluation result JSON file.

    This function constructs the path to the JSON file that contains the evaluation results
    based on the provided evaluation result directory, entry ID, seed, and sample identifier.

    Args:
        eval_result_dir (Path): The directory where evaluation results are stored.
        entry_id (str): The identifier for the entry.
        seed (str): The seed value used in the evaluation process.
        sample (str): The sample identifier.

    Returns:
        Path: The path to the evaluation result JSON file.
    """
    return eval_result_dir / entry_id / str(seed) / f"sample_{sample}_metrics.json"


def build_case_study_samples(
    details_df: pd.DataFrame,
    infer_output_dir: Path,
    eval_result_dir: Path,
    output_dir: Path,
    true_cif_dir: Path,
    model=str,
    seed_col: str = "seed",
    sample_col: str = "sample",
):
    """
    Create per-entry study case directories with symlinked inference, ground-truth, and evaluation files.

    Iterates over rows of an input DataFrame and, for each row, constructs a subdirectory
    under output_dir for the corresponding entry (and chain pair). It then creates
    symbolic links to the model inference mmCIF, the ground-truth mmCIF, and the
    evaluation JSON file inside that subdirectory. Designed to gather files needed
    for manual inspection or downstream case-by-case analysis.

    Args:
        df (pd.DataFrame): DataFrame containing study-case metadata. Each row must
                        include at least the columns:
                        - "entry_id" (str): PDB entry identifier (used as a directory name).
                        - "chain_id_1" (str): First chain identifier.
                        - "chain_id_2" (str | NaN): Second chain identifier or NaN if absent.
                        - seed_col (str, optional): Column name for the random seed (default "seed").
                        - sample_col (str, optional): Column name for the sample index (default "sample").
        infer_output_dir (Path): Directory root where inference mmCIF files are stored.
                                get_infer_cif_path(infer_output_dir, model, entry_id, seed, sample) is
                                used to resolve the inference file path.
        eval_result_dir (Path): Directory root where per-sample evaluation JSON files are stored.
                        get_eval_result_json_path(eval_result_dir, entry_id, seed, sample) is used to
                        resolve the evaluation file path.
        output_dir (Path): Root directory under which per-entry study case subdirectories
                            will be created. Subdirectories take the form:
                            output_dir / entry_id / "{chain_id_1}" or
                            output_dir / entry_id / "{chain_id_1}_{chain_id_2}".
        true_cif_dir (Path): Directory containing ground-truth mmCIF files named {entry_id}.cif.
        model (str): Model identifier passed to get_infer_cif_path to locate inference results.
        seed_col (str, optional): Name of the DataFrame column holding the seed value.
                                Defaults to "seed".
        sample_col (str, optional): Name of the DataFrame column holding the sample index.
                                Defaults to "sample".
    """
    for _, row in details_df.iterrows():
        entry_id = row["entry_id"]
        chain_id_1 = row["chain_id_1"]
        chain_id_2 = row["chain_id_2"]
        seed = row[seed_col]
        sample = row[sample_col]
        infer_cif_path = get_infer_cif_path(
            infer_output_dir, model, entry_id, seed, sample
        )
        eval_result_json_path = get_eval_result_json_path(
            eval_result_dir, entry_id, seed, sample
        )
        true_cif_path = true_cif_dir / f"{entry_id}.cif"

        if pd.isna(chain_id_2):
            sub_output_dir = output_dir / entry_id / f"{chain_id_1}"
        else:
            sub_output_dir = output_dir / entry_id / f"{chain_id_1}_{chain_id_2}"

        sub_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            os.symlink(true_cif_path, output_dir / entry_id / true_cif_path.name)
        except FileExistsError:
            pass
        os.symlink(infer_cif_path, sub_output_dir / f"seed{seed}_{infer_cif_path.name}")
        os.symlink(
            eval_result_json_path,
            sub_output_dir / f"seed{seed}_{eval_result_json_path.name}",
        )


def select_df_by_eval_types(df: pd.DataFrame, eval_types: list[str]) -> pd.DataFrame:
    """
    Selects a subset of the DataFrame based on the specified evaluation types.
    The DataFrame must have following columns:
    - "type": The type of the evaluation, either "chain" or "interface".
    - "entity_type_1": The entity type of the first entity.
    - "entity_type_2": The entity type of the second entity.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        eval_types (list[str]): A list of evaluation types to consider.

    Returns:
        pd.DataFrame: A DataFrame containing the subset of metrics
                      data that matches the specified evaluation types.
    """
    mask = np.zeros(len(df), dtype=bool)
    for eval_type in eval_types:
        entity_type = EVAL_TYPE_TO_ENTITIY_TYPES[eval_type]
        if eval_type.startswith("Intra-"):
            # chain
            entity_type_mask = df.apply(
                lambda row, e_type=entity_type: str(row["entity_type_1"]) == e_type[0]
                and row["type"] == "chain",
                axis=1,
            )
        else:
            # interface
            entity_type = sorted(entity_type)
            entity_type_mask = df.apply(
                lambda row, e_type=entity_type: sorted(
                    [
                        str(row["entity_type_1"]),
                        str(row["entity_type_2"]),
                    ]
                )
                == e_type
                and row["type"] == "interface",
                axis=1,
            )
        mask |= entity_type_mask

    subset_df = df[mask].copy()
    return subset_df


def query_subset_labels(subset_series: pd.Series, query_label: str) -> pd.Series:
    """
    Query the labels of a subset series.

    For example, if the subset series is
    ["[antibody_HL];[antibody]", "[antibody_H];[antibody]", "[antibody_L];[antibody]"],

    and the query label is "[antibody_HL]", then the returned mask series is
    [True, False, False].

    Args:
        subset_series (pd.Series): The series containing the subset labels.
        query_label (str): The label to query.

    Returns:
        pd.Series: The mask series indicating whether each label is in the queried label.
    """
    mask = subset_series.str.contains(query_label, regex=False) | (
        subset_series == query_label
    )
    return mask


def add_comp_chain_iface_id(
    df: pd.DataFrame,
    entry_col: str = "entry_id",
    chain1_col: str = "chain_id_1",
    chain2_col: str = "chain_id_2",
    id_col: str = "id",
) -> pd.DataFrame:
    """
    Add an ID column based on whether chain_id_1 / chain_id_2 are present.

    Rules:
        - If both chain_id_1 and chain_id_2 are "empty" -> id = entry_id
        - If chain_id_1 is non-empty and chain_id_2 is empty -> id = entry_id + "_" + chain_id_1
        - If both chain_id_1 and chain_id_2 are non-empty (interface) ->
              sort(chain1, chain2) and use:
              id = entry_id + "_" + chain_min + "_" + chain_max

    "Empty" includes NaN, empty string "", and string forms like "nan", "NaN", "None".
    """
    out = df.copy()

    for col in (entry_col, chain1_col):
        if col not in out.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

    c1 = out[chain1_col].astype(str)

    if chain2_col not in out.columns:
        c2 = pd.Series("None", index=out.index)
    else:
        c2 = out[chain2_col].astype(str)

    def normalize_chain(s: pd.Series) -> pd.Series:
        s = s.fillna("")  # NaN -> ""
        s = s.str.strip()
        s = s.replace({"nan": "", "NaN": "", "None": "", "<NA>": ""})
        return s

    c1_norm = normalize_chain(c1)
    c2_norm = normalize_chain(c2)

    c1_empty = c1_norm == ""
    c2_empty = c2_norm == ""

    mask_complex = c1_empty & c2_empty
    mask_chain = ~c1_empty & c2_empty
    mask_iface = ~c1_empty & ~c2_empty

    if id_col in out.columns:
        # Drop the existing id_col if it exists
        out = out.drop(columns=[id_col])

    out.insert(0, id_col, "")
    entry_str = out[entry_col].astype(str)

    # complex: entry_id
    out.loc[mask_complex, id_col] = entry_str[mask_complex]

    # chain: entry_id_chain1
    out.loc[mask_chain, id_col] = entry_str[mask_chain] + "_" + c1_norm[mask_chain]

    # interface: entry_id_min(chain1, chain2)_max(chain1, chain2)
    if mask_iface.any():
        c1_iface = c1_norm[mask_iface]
        c2_iface = c2_norm[mask_iface]

        c1_le_c2 = c1_iface <= c2_iface
        first = pd.Series(
            np.where(c1_le_c2, c1_iface, c2_iface),
            index=c1_iface.index,
        )
        second = pd.Series(
            np.where(c1_le_c2, c2_iface, c1_iface),
            index=c1_iface.index,
        )

        out.loc[mask_iface, id_col] = entry_str[mask_iface] + "_" + first + "_" + second

    return out


def get_bootstrap_ci(
    data: list[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n: int = 10000,
) -> tuple[float, float]:
    """
    Bootstrap confidence interval for the mean of a distribution.

    Args:
        data (list[float]): The data to bootstrap.
        statistic (Callable[[np.ndarray], float], optional): The statistic to calculate. Defaults to np.mean.
        n (int, optional): The number of bootstrap samples to generate. Defaults to 10000.

    Returns:
        tuple[float, float]: The lower and upper bounds of the confidence interval.
    """
    if len(data) == 0:
        logging.debug(
            "Data is empty, cannot calculate confidence \
                interval for bootstrap. return (0, 0)"
        )
        ci_lower, ci_upper = 0.0, 0.0
    elif len(data) == 1:
        logging.debug(
            "Data has only one element, cannot calculate confidence \
                interval for bootstrap. return (data[0], data[0])"
        )
        ci_lower, ci_upper = data[0], data[0]

    elif np.nanstd(data) == 0:
        logging.debug(
            "Data has 0 std, cannot calculate confidence \
                interval for bootstrap. return (data[0], data[0])"
        )
        ci_lower, ci_upper = data[0], data[0]

    else:
        data = (data,)
        bootstrap_result = stats.bootstrap(data, statistic, n_resamples=n)

        ci_lower, ci_upper = bootstrap_result.confidence_interval
    return round(float(ci_lower), 4), round(float(ci_upper), 4)


def get_binomial_ci(total_num: int, success_num: int) -> tuple[float, float]:
    """
    Calculate the Clopper-Pearson interval (exact binomial confidence interval)
    for a binomial distribution.

    Args:
        total_num (int): The total number of trials.
        success_num (int): The number of successful trials.

    Returns:
        tuple[float, float]: The lower and upper bounds of the confidence interval.
    """
    binomtest_result = stats.binomtest(success_num, total_num).proportion_ci(0.95)
    ci_lower, ci_upper = binomtest_result
    return round(float(ci_lower), 4), round(float(ci_upper), 4)


def add_cluster_id_to_df(
    cluster_df: pd.DataFrame,
    df: pd.DataFrame,
    interface_only_use_polymer_cluster: bool = False,
) -> pd.DataFrame:
    """
    Adds cluster IDs to the DataFrame based on the cluster information in the provided CSV file.

    Args:
        cluster_df (pd.DataFrame): The DataFrame containing cluster information.
        df (pd.DataFrame): The DataFrame containing the data to add cluster IDs to.
        interface_only_use_polymer_cluster (bool, optional): Whether to only use polymer
                                           cluster for interface evaluation. Defaults to False.

    Returns:
        pd.DataFrame: The updated DataFrame with cluster IDs added.
    """
    out = df.copy()

    # Drop rows with NaN values in the "cluster_id" column
    cdf = cluster_df.dropna(subset=["cluster_id"]).copy()

    key = cdf["entry_id"].astype(str) + "_" + cdf["label_entity_id"].astype(str)
    entry_entity_to_cluster = dict(zip(key, cdf["cluster_id"].astype(str)))

    key1 = out["entry_id"].astype(str) + "_" + out["entity_id_1"].astype(str)
    cluster_id_1 = key1.map(entry_entity_to_cluster).astype(object)

    key2 = out["entry_id"].astype(str) + "_" + out["entity_id_2"].astype(str)
    cluster_id_2 = key2.map(entry_entity_to_cluster).astype(object)

    out["cluster_id_1"] = cluster_id_1
    out["cluster_id_2"] = cluster_id_2

    has_c1 = out["cluster_id_1"].notna() & (out["cluster_id_1"] != "")
    has_c2 = out["cluster_id_2"].notna() & (out["cluster_id_2"] != "")
    both = has_c1 & has_c2

    c1s = out["cluster_id_1"].fillna("")
    c2s = out["cluster_id_2"].fillna("")

    c1_le_c2 = c1s <= c2s
    pair_joined = np.where(
        c1_le_c2,
        (c1s + ":" + c2s),
        (c2s + ":" + c1s),
    )
    pair_joined = pd.Series(pair_joined, index=out.index).where(both, None)

    types = out.get("type", pd.Series([""] * len(out), index=out.index)).astype(str)

    # chain -> cluster_id_1
    mask_chain = types == "chain"
    # interface -> pair_joined
    mask_interface = types == "interface"

    final_cluster = pd.Series([None] * len(out), index=out.index, dtype=object)
    final_cluster = final_cluster.where(~mask_chain, out["cluster_id_1"])

    # interface:
    if interface_only_use_polymer_cluster:
        is_poly1 = out["entity_type_1"].isin(POLYMER)
        is_poly2 = out["entity_type_2"].isin(POLYMER)

        # case both polymer or both non-polymer -> use pair_joined if both exist
        both_poly_or_both_non = (is_poly1 & is_poly2) | (~is_poly1 & ~is_poly2)
        mask_use_pair = mask_interface & both_poly_or_both_non & both
        final_cluster = final_cluster.where(~mask_use_pair, pair_joined)

        # case only polymer_1 -> use cluster_id_1
        mask_use_c1 = mask_interface & is_poly1 & ~is_poly2
        final_cluster = final_cluster.where(~mask_use_c1, out["cluster_id_1"])

        # case only polymer_2 -> use cluster_id_2
        mask_use_c2 = mask_interface & is_poly2 & ~is_poly1
        final_cluster = final_cluster.where(~mask_use_c2, out["cluster_id_2"])

        if "ref_pocket_entity" in out.columns:
            # pocket-aligned rmsd: use ref_pocket_cluster_id if exist
            key_pocket = (
                out["entry_id"].astype(str) + "_" + out["ref_pocket_entity"].astype(str)
            )
            cluster_id_pocket = key_pocket.map(entry_entity_to_cluster).astype(object)
            out["ref_pocket_cluster_id"] = cluster_id_pocket
            has_pocket = out["ref_pocket_entity"].notna()
            final_cluster = final_cluster.where(
                ~has_pocket, out["ref_pocket_cluster_id"]
            )

        # other interface cases leave None (already None)
    else:
        # not restricting by polymer: if both cluster ids exist, use joined, else None
        mask_use_pair = mask_interface & both
        final_cluster = final_cluster.where(~mask_use_pair, pair_joined)

    # complex remain None (already default)
    # assign final_cluster
    out["cluster_id"] = final_cluster
    return out


def paired_test_auto(
    A: list[float],
    B: list[float],
    alpha: float = 0.05,
    alternative: str = "two-sided",
    normality_test: bool = True,
    min_n_for_t: int = 10,
    shapiro_max_n: int = 5000,
) -> dict:
    """
    Automatically choose between paired t-test (for approximately normal differences)
    and Wilcoxon signed-rank test (for non-normal differences).

    Always uses d = A - B as the difference direction, so effect_size > 0 means A > B.

    Args:
        A, B : array-like
            Two paired samples of equal length.
        alpha : float, optional
            Significance level, by default 0.05.
        alternative : {"two-sided", "greater", "less"}, optional
            Alternative hypothesis.
            - "greater": test mean(A-B) > 0  (A > B)
            - "less":    test mean(A-B) < 0  (B > A)
            - "two-sided": test mean(A-B) != 0
        normality_test : bool, optional
            Whether to run Shapiro-Wilk test on differences to check normality.
        min_n_for_t : int, optional
            Minimum sample size to allow t-test. For very small n, Wilcoxon is safer.
        shapiro_max_n : int, optional
            Maximum n to run Shapiro test. If n is larger, skip Shapiro and default to t-test.

    Returns
        dict: A dictionary containing:
            - n : sample size
            - alternative : alternative hypothesis used
            - mean_diff : mean of A - B
            - sd_diff : standard deviation of A - B
            - method : "paired t-test" or "wilcoxon"
            - stat : test statistic
            - p : p-value
            - effect_size : Cohen's d (paired) or rank-biserial correlation
            - effect_name : name of the effect size metric
            - decision : "A > B", "B > A", or "no significant difference"
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    assert (
        A.shape == B.shape and A.ndim == 1
    ), "A and B must be 1D arrays of the same length."

    # Remove NaNs
    mask = ~(np.isnan(A) | np.isnan(B))
    A, B = A[mask], B[mask]
    n = len(A)

    d = A - B
    mean_d = float(np.mean(d))
    sd_d = float(np.std(d, ddof=1)) if n > 1 else 0.0

    result = {
        "n": n,
        "alternative": alternative,
        "mean_diff": mean_d,
        "sd_diff": sd_d,
        "method": None,
        "stat": None,
        "p": None,
        "effect_size": None,
        "effect_name": None,  # "cohen_d_paired" or "rank_biserial"
        "decision": "no significant difference",  # "A > B" / "B > A" / "no significant difference"
    }

    if n < 3:
        result["decision"] = "Sample size too small (n < 3)"
        return result

    # -------- Normality check --------
    use_ttest = True
    if normality_test and (3 <= n <= shapiro_max_n):
        try:
            _shapiro_stat, shapiro_p = stats.shapiro(d)
            # If Shapiro test not rejected, and n is large enough, use t-test
            use_ttest = (shapiro_p >= alpha) and (n >= min_n_for_t)
        except Exception:
            # If Shapiro fails (e.g. constant vector), fall back to heuristic
            use_ttest = n >= min_n_for_t
    else:
        use_ttest = n >= min_n_for_t

    # -------- Run significance test + effect size --------
    if use_ttest and sd_d > 0:
        # Paired t-test
        t_stat, p = stats.ttest_rel(A, B, alternative=alternative)
        # Cohen's d for paired samples
        cohen_d = abs(mean_d / sd_d)
        result.update(
            {
                "method": "paired t-test",
                "stat": float(t_stat),
                "p": float(p),
                "effect_size": float(cohen_d),
                "effect_name": "cohen_d_paired",
            }
        )
    else:
        # Wilcoxon signed-rank test
        if np.allclose(d, 0):
            # All differences are zero -> no effect
            result.update(
                {
                    "method": "wilcoxon",
                    "stat": 0.0,
                    "p": 1.0,
                    "effect_size": 0.0,
                    "effect_name": "rank_biserial",
                    "decision": "no significant difference",
                }
            )
            return result

        w_stat, p = stats.wilcoxon(
            d, zero_method="wilcox", alternative=alternative, correction=False
        )

        # Rank-biserial correlation
        nonzero = d[d != 0]
        ranks = stats.rankdata(np.abs(nonzero))
        pos = nonzero > 0
        w_pos = ranks[pos].sum()
        w_neg = ranks[~pos].sum()
        denom = len(nonzero) * (len(nonzero) + 1) / 2.0
        r_rb = abs((w_pos - w_neg) / denom) if denom > 0 else 0.0

        result.update(
            {
                "method": "wilcoxon",
                "stat": float(w_stat),
                "p": float(p),
                "effect_size": float(r_rb),
                "effect_name": "rank_biserial",
            }
        )

    # -------- Decision based on p-value and direction --------
    alt = alternative
    p = result["p"]
    if alt == "two-sided":
        if p < alpha:
            result["decision"] = "A > B" if mean_d > 0 else "B > A"
        else:
            result["decision"] = "no significant difference"
    elif alt == "greater":
        # H1: mean(A-B) > 0
        result["decision"] = "A > B" if p < alpha else "no significant difference"
    elif alt == "less":
        # H1: mean(A-B) < 0
        result["decision"] = "B > A" if p < alpha else "no significant difference"
    else:
        raise ValueError('alternative must be "two-sided", "greater", or "less".')

    return result


def fmt_bytes(n: int) -> str:
    """Format a byte count into a human-readable string.

    Args:
        n: Number of bytes.

    Returns:
        A string formatted with a unit suffix (B, KB, MB, GB, TB, PB).
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def shrink_dataframe(
    df: pd.DataFrame,
    *,
    cat_threshold: int = 256,
    cat_ratio: float = 0.5,
    object_to_string: bool = True,
    downcast_float: bool = True,
    downcast_int: bool = True,
    use_nullable_int: bool = True,
    bool_cast: bool = True,
    exclude: Iterable[str] = (),
    report_topk: int = 30,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Downcast and recode columns to minimize memory and file size.

    This function applies a series of safe transformations to reduce memory
    footprint without changing column semantics:
    - Booleans: convert 0/1 or True/False to bool/boolean dtypes.
    - Floats: downcast float64 to float32 (optional).
    - Integers: downcast int64 to the smallest fitting integer dtype.
    - Object columns with integer-like values: convert to nullable Int* dtypes.
    - Low-cardinality text: convert to ``category``.
    - Other object text: convert to pandas ``string`` (Arrow/Parquet friendly).

    Args:
        df: Input DataFrame.
        cat_threshold: If the number of unique non-null values is ≤ this value,
            convert to ``category``.
        cat_ratio: Alternatively, if ``nunique / len(df)`` ≤ this ratio,
            convert to ``category``.
        object_to_string: Convert remaining ``object`` text columns to
            pandas ``string`` dtype.
        downcast_float: If True, downcast ``float64`` to ``float32``.
        downcast_int: If True, downcast ``int64`` to the smallest fitting
            signed integer dtype.
        use_nullable_int: If True, convert integer-like ``object`` columns
            (possibly with missing values) to nullable ``Int8/16/32/64``.
        bool_cast: If True, convert 0/1 (non-float) or boolean-like columns to
            ``bool`` / nullable ``boolean``.
        exclude: Column names to skip from any transformation.
        report_topk: Number of columns to include in the per-column memory
            savings summary.

    Returns:
        A 2-tuple ``(df_out, report)`` where:
        - ``df_out``: The transformed DataFrame with reduced memory usage.
        - ``report``: A dict containing summary metrics:
            - ``mem_before_bytes`` / ``mem_after_bytes`` / ``mem_saved_bytes``
            - human-readable counterparts (``*_readable``)
            - ``shrink_ratio`` (before/after)
            - ``changed_cols``: mapping of column -> (old_dtype, new_dtype)
            - ``top_saving_cols``: memory saved by column (top-K)

    Notes:
        - Conversions are conservative and try to preserve semantics.
        - For columns critical to numeric precision (e.g., scores), add them to
          ``exclude`` to prevent downcasting.
        - The function does not modify the input ``df`` in place.

    Examples:
        >>> df_small, rpt = shrink_dataframe(df, exclude=["critical_score"])
        >>> rpt["mem_before_readable"], rpt["mem_after_readable"]
        ('1.20 GB', '420.00 MB')
    """
    src_mem = df.memory_usage(deep=True).sum()
    out = df.copy()

    excl = set(exclude)
    changes: dict[str, tuple[str, str]] = {}

    # Iterate over columns and apply dtype reductions.
    for col in out.columns:
        if col in excl:
            continue

        s = out[col]
        old_dtype = s.dtype

        # 1) Boolean casting: recognize 0/1 (non-float) or existing bools.
        if bool_cast:
            if pd.api.types.is_bool_dtype(s):
                pass  # already boolean
            elif set(s.dropna()).issubset({0, 1}) and not pd.api.types.is_float_dtype(
                s
            ):
                out[col] = s.astype("boolean") if s.isna().any() else s.astype(bool)
                changes[col] = (old_dtype, out[col].dtype)
                continue

        # 2) Numeric downcasting.
        if pd.api.types.is_float_dtype(s):
            if downcast_float and s.dtype == "float64":
                out[col] = pd.to_numeric(s, downcast="float")
                changes[col] = (old_dtype, out[col].dtype)
                continue

        elif pd.api.types.is_integer_dtype(s):
            if downcast_int and s.dtype == "int64":
                out[col] = pd.to_numeric(s, downcast="integer")
                changes[col] = (old_dtype, out[col].dtype)
                continue

        # 3) Nullable integers for integer-like object columns.
        elif use_nullable_int and s.dtype == "object":
            sample = s.sample(min(len(s), 5000), random_state=0)
            try_parse = pd.to_numeric(sample, errors="coerce", downcast="integer")
            if (
                try_parse.notna().equals(sample.notna())
                and (try_parse.dropna() % 1 == 0).all()
            ):
                parsed = pd.to_numeric(s, errors="coerce", downcast="integer")
                if parsed.isna().any():
                    # Choose the smallest nullable integer dtype that can hold the range.
                    iinfo = pd.Series(parsed.dropna().astype("int64"))
                    minv, maxv = iinfo.min(), iinfo.max()
                    if -128 <= minv and maxv <= 127:
                        out[col] = parsed.astype("Int8")
                    elif -32768 <= minv and maxv <= 32767:
                        out[col] = parsed.astype("Int16")
                    elif -2147483648 <= minv and maxv <= 2147483647:
                        out[col] = parsed.astype("Int32")
                    else:
                        out[col] = parsed.astype("Int64")
                else:
                    out[col] = pd.to_numeric(parsed, downcast="integer")
                changes[col] = (old_dtype, out[col].dtype)
                continue

        # 4) Text handling: low-cardinality -> category; otherwise -> string.
        if s.dtype == "object":
            nunq = s.nunique(dropna=True)
            if nunq <= cat_threshold or nunq <= len(s) * cat_ratio:
                out[col] = s.astype("category")
            elif object_to_string:
                out[col] = s.astype("string")
            if out[col].dtype != old_dtype:
                changes[col] = (old_dtype, out[col].dtype)

    # Build memory report.
    dst_mem = out.memory_usage(deep=True).sum()
    delta = src_mem - dst_mem
    ratio = (src_mem / max(dst_mem, 1)) if dst_mem else np.inf

    mem_before = df.memory_usage(deep=True)
    mem_after = out.memory_usage(deep=True)
    diff = (mem_before - mem_after).sort_values(ascending=False)

    report: dict[str, Any] = {
        "mem_before_bytes": int(src_mem),
        "mem_after_bytes": int(dst_mem),
        "mem_saved_bytes": int(delta),
        "mem_before_readable": fmt_bytes(src_mem),
        "mem_after_readable": fmt_bytes(dst_mem),
        "mem_saved_readable": fmt_bytes(delta),
        "shrink_ratio": float(ratio),
        "changed_cols": {k: (str(v0), str(v1)) for k, (v0, v1) in changes.items()},
        "top_saving_cols": diff.head(report_topk).to_dict(),
    }
    return out, report
