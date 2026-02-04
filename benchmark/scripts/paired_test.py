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

import argparse
import itertools
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control
from tqdm import tqdm

from benchmark.utils import paired_test_auto

STD_RANKER_NAMES_MAPPING = {
    "iptm+ptm": "ranking_score",
    "chains_ptm": "chain_ptm",
    "confidence_score": "ranking_score",
    "complex_plddt": "plddt",
    "complex_pde": "pde",
    "complex_ipde": "ipde",
    "pair_chains_iptm": "chain_pair_iptm",
    "per_chain_ptm": "chain_ptm",
    "aggregate_score": "ranking_score",
    "per_chain_pair_iptm": "chain_pair_iptm",
}

RANKERS = (
    "best",
    "median",
    "best.ranking_score",
)


@dataclass
class StrictGraph:
    """Directed graph for strict preferences (A > B)."""

    edges: dict[str, set[str]]  # u -> set of v (strict edges u>v)
    indeg: dict[str, int]  # in-degree for Kahn's algorithm
    effect: dict[tuple[str, str], float]  # (u,v) -> effect size for strict edges


class UnionFind:
    """
    Disjoint Set Union (Union-Find) with path compression and union by rank.
    Used only for within-layer tie grouping (A ≈ B).
    """

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        """
        Find the representative (root) of the set containing the given element.

        This method applies path compression, which flattens the tree structure
        to make future queries faster.

        Args:
            x (hashable): The element to look up.

        Returns:
            hashable: The representative (root) of the set containing `x`.
        """
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        """
        Merge the sets containing two elements.

        This method uses union by rank to keep the tree depth minimal,
        which helps maintain nearly constant-time performance for future operations.

        Args:
            a (hashable): The first element.
            b (hashable): The second element.
        """
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def build_items(df: pd.DataFrame, a_col: str, b_col: str) -> list[str]:
    """Collect and sort all unique items that appear in columns A and B."""
    return sorted(set(df[a_col]).union(df[b_col]))


def build_strict_graph(
    g: pd.DataFrame,
    a_col: str,
    b_col: str,
    dec_col: str,
    effect_size_col: str,
    items: list[str],
) -> StrictGraph:
    """
    Build a directed graph using only strict relations (A > B).
    Also returns an indegree map for Kahn's topological sort and an edge->effect dict.
    """
    edges: dict[str, set[str]] = defaultdict(set)
    indeg: dict[str, int] = defaultdict(int)
    effect: dict[tuple[str, str], float] = {}

    # Add nodes
    for x in items:
        indeg.setdefault(x, 0)

    # Add strict edges and attach effect sizes
    strict_rows = g[g[dec_col] == "A > B"]
    for _, r in strict_rows.iterrows():
        a, b = r[a_col], r[b_col]
        if b not in edges[a]:
            edges[a].add(b)
            indeg[b] += 1
        # Keep the last effect size seen for (a,b)
        # you can also average duplicates if needed.
        effect[(a, b)] = float(r[effect_size_col])

    return StrictGraph(edges=edges, indeg=indeg, effect=effect)


def collect_ties(
    g: pd.DataFrame, a_col: str, b_col: str, dec_col: str
) -> dict[str, set[str]]:
    """
    Collect tie relations (A ≈ B) as an undirected adjacency list.
    """
    ties: dict[str, set[str]] = defaultdict(set)
    tie_rows = g[g[dec_col] == "show no difference"]
    for _, r in tie_rows.iterrows():
        a, b = r[a_col], r[b_col]
        ties[a].add(b)
        ties[b].add(a)
    return ties


def topo_layers_by_kahn(strict: StrictGraph) -> tuple[list[list[str]], dict[str, int]]:
    """
    Kahn's algorithm for topological layering using only strict edges.
    Returns:
        layers: list of layers, each is a list of nodes
        node_layer: node -> layer index
    If the graph has no edges (or nodes only), all nodes will end in layer 0.
    """
    indeg_mut = dict(strict.indeg)  # copy
    q = deque([n for n, d in indeg_mut.items() if d == 0])
    layers: list[list[str]] = []
    visited = 0

    while q:
        layer = list(q)
        layers.append(layer)
        q = deque()
        for u in layer:
            visited += 1
            for v in strict.edges.get(u, []):
                indeg_mut[v] -= 1
                if indeg_mut[v] == 0:
                    q.append(v)

    # If some nodes are unvisited due to cycles, we still produce partial layering.
    # However, in our “strict-first” scheme, cycles indicate conflicting strict statements.
    # We do NOT fail; we place any remaining nodes into the last layer as a fallback.
    if visited < len(strict.indeg):
        remaining = [n for n, d in indeg_mut.items() if d > 0]
        if remaining:
            layers.append(sorted(remaining))

    # Build node -> layer index
    node_layer: dict[str, int] = {}
    for i, layer in enumerate(layers):
        for n in layer:
            node_layer[n] = i

    # If no layers (no nodes), return empty structures
    if not layers and strict.indeg:
        # Should not happen, but keep safe
        nodes = list(strict.indeg.keys())
        layers = [nodes]
        node_layer = {x: 0 for x in nodes}

    return layers, node_layer


def adjust_layers_with_ties_downward(
    items: list[str],
    node_layer: dict[str, int],
    ties: dict[str, set[str]],
) -> dict[str, int]:
    """
    Tie-based downward adjustment:
    Iteratively move each node down to the minimum (i.e., numerically larger index)
    layer among its ≈ neighbors and itself until no changes occur.
    This preserves strict ordering while letting ties settle to lower (worse) layers
    if any tie neighbor is strictly below.
    """
    changed = True
    items_list = list(items)
    while changed:
        changed = False
        for x in items_list:
            if not ties.get(x):
                continue
            # Target is the max (numerically) layer among x and its tie neighbors.
            target = max(
                [node_layer.get(y, node_layer.get(x, 0)) for y in ties[x]]
                + [node_layer.get(x, 0)]
            )
            if target > node_layer.get(x, 0):
                node_layer[x] = target
                changed = True
    return node_layer


def rebuild_layers_from_assignment(
    node_layer: dict[str, int], mean_map: dict[str, float] | None = None
) -> list[list[str]]:
    """
    Rebuild a dense list of layers from the node -> layer index mapping
    and remove empty layers. Nodes in each layer are sorted by mean value
    (descending) if mean_map is provided, otherwise alphabetically.
    """
    if not node_layer:
        return []
    maxL = max(node_layer.values())
    buckets: list[list[str]] = [[] for _ in range(maxL + 1)]
    for x, L in node_layer.items():
        buckets[L].append(x)

    def sort_key(x):
        if mean_map and x in mean_map:
            return (-mean_map[x], x)  # primary: mean desc, secondary: name
        return (float("inf"), x)

    return [sorted(ly, key=sort_key) for ly in buckets if ly]


def group_ties_within_layer(
    layer: list[str],
    ties: dict[str, set[str]],
    mean_map: dict[str, float] | None = None,
) -> list[list[str]]:
    """
    Within a single layer, merge tie-connected components into groups
    using Union-Find (A ≈ B). Each group is sorted by mean value descending,
    and groups themselves are ordered by their max-mean descending.
    """
    uf = UnionFind()
    for x in layer:
        uf.find(x)
    for x in layer:
        for y in ties.get(x, set()):
            if y in layer:
                uf.union(x, y)

    root2members: dict[str, list[str]] = defaultdict(list)
    for x in layer:
        root2members[uf.find(x)].append(x)

    def inner_key(x: str):
        if mean_map and x in mean_map:
            return (-mean_map[x], x)
        return (float("inf"), x)

    groups = [sorted(v, key=inner_key) for v in root2members.values()]

    # --- order groups by their best (max) mean
    def group_key(g: list[str]):
        if mean_map:
            vals = [mean_map.get(x) for x in g if mean_map.get(x) is not None]
            if vals:
                return (-max(vals), g[0])
        return (float("inf"), g[0] if g else "")

    groups.sort(key=group_key)

    return groups


def format_layer_block(groups: list[list[str]]) -> str:
    """
    Format a single layer as 'A ≈ B ≈ C' with groups already sorted.
    """
    return " ≈ ".join(" ≈ ".join(g) for g in groups)


def compute_between_layer_effect(
    upper: list[str],
    lower: list[str],
    effect_map: dict[tuple[str, str], float],
) -> str | None:
    """
    Compute the max effect size between two adjacent layers
    using available strict edges only. If not all cross pairs have evidence,
    append '?' to indicate incomplete evidence.

    Returns:
        '(0.420?|1/2)' or '(0.420)' or None if no strict evidence at all.
    """
    eff_vals: list[float] = []
    complete_pairs = True
    for u in upper:
        for v in lower:
            if (u, v) in effect_map:
                eff_vals.append(effect_map[(u, v)])
            else:
                complete_pairs = False

    if not eff_vals:
        return None

    median_eff = np.median(eff_vals)
    s = f"({median_eff:.3f}"
    if not complete_pairs:
        num_pairs = len(upper) * len(lower)
        valid_eff_str = f"{len(eff_vals)}/{num_pairs}"
        s += f"?|{valid_eff_str}"
    s += ")"
    return s


def degrade_layers_by_zero_coverage(
    layers: list[list[str]],
    strict_edges: dict[str, set[str]],
    mean_map: dict[str, float] | None = None,
) -> list[list[str]]:
    """
    Degrade nodes with zero coverage against the immediate lower layer.

    If mean_map is provided, each layer is finally sorted by mean (desc).
    Otherwise, original relative order is preserved (no alpha sort).
    """
    # Work on a mutable copy
    layers = [list(layer) for layer in layers]

    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(layers) - 1:
            upper, lower = layers[i], layers[i + 1]
            to_move = []
            for u in list(upper):
                has_any = any(v in strict_edges.get(u, set()) for v in lower)
                if not has_any:
                    to_move.append(u)
            if to_move:
                for u in to_move:
                    upper.remove(u)
                    lower.append(u)
                changed = True
                # re-check this boundary
            else:
                i += 1
        layers = [ly for ly in layers if ly]

    # --- dedupe but preserve order
    def dedupe_keep_order(seq: list[str]) -> list[str]:
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    layers = [dedupe_keep_order(ly) for ly in layers]

    # --- final per-layer ordering
    if mean_map is not None:

        def key(x: str):
            m = mean_map.get(x, None)
            return (-m, x) if m is not None else (float("inf"), x)

        layers = [sorted(ly, key=key) for ly in layers]

    return layers


def find_strict_order_violations(
    node_layer: dict[str, int],
    strict: StrictGraph,
) -> tuple[set[tuple[str, str]], set[str]]:
    """
    Find all strict edges (u -> v) that are violated,
    i.e. where layer[u] >= layer[v].

    Returns:
        viol_edges (set[tuple[str, str]]): The set of violated edges {(u, v), ...}.
        viol_nodes (set[str]): The set of nodes involved in any violation {u, v, ...}.
    """
    viol_edges: set[tuple[str, str]] = set()
    viol_nodes: set[str] = set()
    for u, outs in strict.edges.items():
        Lu = node_layer.get(u, 0)
        for v in outs:
            Lv = node_layer.get(v, 0)
            if Lu >= Lv:  # Strict requirement should be Lu < Lv
                viol_edges.add((u, v))
                viol_nodes.update([u, v])
    return viol_edges, viol_nodes


def rank_with_ties(
    df: pd.DataFrame,
    a_col: str = "A",
    b_col: str = "B",
    mean_a_col: str = "mean_A",
    mean_b_col: str = "mean_B",
    dec_col: str = "bh_adj_decision",
    effect_size_col: str = "effect_size",
    n_sample_col: str = "n_sample",
    groupby_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Produce ranking strings per group with the following policy:
    1) Build a strict-only DAG (A > B), run Kahn to get layers (top is better).
    2) Apply tie-based downward adjustment (A ≈ B): move nodes down to the worst layer among their tie neighbors.
    3) Rebuild layers; within each layer, union tie-connected nodes and format as 'X ≈ Y'.
    4) Between adjacent layers, compute mean effect from available strict edges; add '?' if evidence is incomplete.

    Output example:
        'A (0.420?|1/2) > B ≈ C'
    """
    if groupby_cols is None:
        groupby_cols = ["eval_dataset", "subset", "eval_type", "ranker"]

    outputs = []
    for et, g in df.groupby(by=groupby_cols, dropna=False):
        # Unpack group keys safely (pad with Nones if fewer keys)
        keys = list(et) if isinstance(et, tuple) else [et]
        while len(keys) < 4:
            keys.append(None)
        eval_dataset, subset, eval_type, ranker = keys[:4]

        # Build basic output row
        output = {
            "eval_dataset": eval_dataset,
            "subset": subset,
            "eval_type": eval_type,
            "ranker": ranker,
            n_sample_col: (
                g[n_sample_col].iloc[0]
                if n_sample_col in g.columns and not g[n_sample_col].isna().all()
                else None
            ),
        }

        # Build mean_map from mean_A / mean_B
        mean_map: dict[str, float] = {}
        for _, row in g.iterrows():
            a, b = row[a_col], row[b_col]
            if not pd.isna(row[mean_a_col]):
                mean_map[a] = float(row[mean_a_col])
            if not pd.isna(row[mean_b_col]):
                mean_map[b] = float(row[mean_b_col])

        # 0) Collect items
        items = build_items(g, a_col, b_col)
        if not items:
            output["ranking"] = ""
            outputs.append(output)
            continue

        # Strict-only graph + Kahn layers
        strict = build_strict_graph(g, a_col, b_col, dec_col, effect_size_col, items)
        layers, node_layer = topo_layers_by_kahn(strict)

        # If no strict edges at all, put all items into a single layer 0
        if not layers:
            node_layer = {x: 0 for x in items}
            layers = [sorted(items)]

        # Collect ties and do downward adjustment
        ties = collect_ties(g, a_col, b_col, dec_col)
        node_layer = adjust_layers_with_ties_downward(items, node_layer, ties)

        # Rebuild layers and format within-layer groups
        layers = rebuild_layers_from_assignment(node_layer, mean_map)

        # Check for strict order violations
        viol_edges, viol_nodes = find_strict_order_violations(node_layer, strict)
        if viol_edges:
            example = next(iter(viol_edges))
            u, v = example
            conflict_info = (
                f"CONFLICT: strict edges contradict ties; "
                f"e.g., {u} > {v} violated by layering. "
                f"Involved: {', '.join(sorted(viol_nodes))}"
            )
            logging.warning(conflict_info)
            output["ranking"] = conflict_info
            outputs.append(output)
            continue

        # zero-coverage degradation
        layers = degrade_layers_by_zero_coverage(layers, strict.edges, mean_map)

        # per-layer tie grouping + effect formatting
        parts = []
        for i, layer in enumerate(layers):
            groups = group_ties_within_layer(layer, ties, mean_map)
            parts.append(format_layer_block(groups))
            if i < len(layers) - 1:
                eff_str = compute_between_layer_effect(
                    layers[i], layers[i + 1], strict.effect
                )
                if eff_str:
                    parts.append(eff_str)

        # Join with ' > ' and avoid patterns like '> ('
        ranking_str = " > ".join(parts).replace("> (", "(")
        output["ranking"] = ranking_str
        outputs.append(output)

    return pd.DataFrame(outputs)


def run_pairwise_test(
    details_csv: Path,
    metric_col: str,
    output_test_csv: Path | None = None,
    output_rank_csv: Path | None = None,
    cluster_pair_test: bool = True,
):
    """
    Run pairwise statistical tests and generate model rankings for each eval_type.

    Args:
        details_csv (Path): Path to the input CSV file containing evaluation details.
        metric_col (str): Name of the column containing the metric values to be compared.
        output_test_csv (Path | None, optional): Path to the CSV file where raw pairwise
            test results will be saved. Defaults to "<stem>_paired_test.csv" in the same
            directory as `details_csv`.
        output_rank_csv (Path | None, optional): Path to the CSV file where aggregated
            ranking results will be saved. Defaults to "<stem>_paired_test_rank.csv"
            in the same directory as `details_csv`.
        cluster_pair_test (bool, optional): Whether to perform paired tests on
            cluster-level aggregated scores (True) or on raw sample-level scores (False).
            Defaults to True.

    Notes:
        - Pairwise tests are conducted for each combination of (eval_dataset, subset,
          eval_type, ranker).
        - P-values are adjusted using the Benjamini-Hochberg (BH) procedure to control
          the false discovery rate (FDR).
        - Effect sizes, test statistics, and adjusted decisions are derived from
          `paired_test_auto`.
        - Rankings are generated with `rank_with_ties`, where models with no significant
          difference are grouped together.
    """
    if output_test_csv is None:
        output_test_csv = details_csv.parent / f"{details_csv.stem}_paired_test.csv"
    if output_rank_csv is None:
        output_rank_csv = (
            details_csv.parent / f"{details_csv.stem}_paired_test_rank.csv"
        )

    details_df = pd.read_csv(
        details_csv,
        dtype={
            "name": str,
            "entry_id": str,
            "chain_id_1": str,
            "chain_id_2": str,
            "cluster_id": str,
            "entity_id_1": str,
            "entity_id_2": str,
        },
    )

    if "subset" not in details_df.columns:
        details_df["subset"] = "All"

    assert (
        metric_col in details_df.columns
    ), f'The column "{metric_col}" not in {details_csv}'

    def mapping_ranker(row):
        ranker = row["ranker"]
        if "best." in ranker:
            ori_ranker = ranker.split(".")[-1]
            new_ranker = "best." + STD_RANKER_NAMES_MAPPING.get(ori_ranker, ori_ranker)
            return new_ranker
        else:
            return ranker

    details_df["ranker"] = details_df.apply(mapping_ranker, axis=1)

    combos = set(
        details_df[["eval_dataset", "subset", "eval_type"]].itertuples(
            index=False, name=None
        )
    )

    results = []
    ori_ps = []
    for (
        eval_dataset_i,
        subset_i,
        eval_type_i,
    ) in tqdm(combos, total=len(combos), desc="pairwise test"):
        for ranker_i in RANKERS:
            sub_df = details_df[
                (details_df["eval_dataset"] == eval_dataset_i)
                & (details_df["subset"] == subset_i)
                & (details_df["eval_type"] == eval_type_i)
                & (details_df["ranker"] == ranker_i)
            ]
            if len(sub_df) == 0:
                continue
            names = sub_df["name"].unique()
            pairwise_names = itertools.combinations(names, 2)

            for name1, name2 in pairwise_names:
                name1_mask = sub_df["name"] == name1
                name2_mask = sub_df["name"] == name2
                if cluster_pair_test:
                    name1_metric = []
                    name2_metric = []
                    for _cluster_id, group_df in sub_df.groupby("cluster_id"):
                        name1_metric.append(group_df[metric_col][name1_mask].mean())
                        name2_metric.append(group_df[metric_col][name2_mask].mean())
                else:
                    name1_metric = sub_df[metric_col][name1_mask]
                    name2_metric = sub_df[metric_col][name2_mask]

                mean_1 = np.mean(name1_metric)
                mean_2 = np.mean(name2_metric)

                _test_result = paired_test_auto(
                    name1_metric,
                    name2_metric,
                )

                if _test_result["decision"] == "Sample size too small (n < 3)":
                    logging.warning(
                        "Sample size too small (n < 3) for %s vs %s in %s %s %s %s",
                        name1,
                        name2,
                        eval_dataset_i,
                        subset_i,
                        eval_type_i,
                        ranker_i,
                    )
                    continue

                paired_test_result = {
                    "A": name1,
                    "B": name2,
                    "mean_A": mean_1,
                    "mean_B": mean_2,
                    "eval_dataset": eval_dataset_i,
                    "eval_type": eval_type_i,
                    "ranker": ranker_i,
                    "subset": subset_i,
                    "cluster_pair_test": cluster_pair_test,
                    "n_sample": len(name1_metric),
                    "n_cluster": sub_df["cluster_id"].nunique(),
                }

                for i in [
                    "decision",
                    "p",
                    "stat",
                    "effect_size",
                    "effect_name",
                    "method",
                ]:
                    paired_test_result[i] = _test_result[i]

                if paired_test_result["decision"] == "B > A":
                    paired_test_result["A"] = name2
                    paired_test_result["B"] = name1
                    paired_test_result["mean_A"] = mean_2
                    paired_test_result["mean_B"] = mean_1
                    paired_test_result["decision"] = "A > B"

                ori_ps.append(paired_test_result["p"])
                results.append(paired_test_result)

    result_df = pd.DataFrame(results)

    # Adjust p-values to control the FDR by Benjamini-Hochberg method
    result_df["bh_adj_p"] = false_discovery_control(ori_ps, method="bh")
    result_df["bh_adj_decision"] = result_df.apply(
        lambda x: (
            "no significant difference" if x["bh_adj_p"] >= 0.05 else x["decision"]
        ),
        axis=1,
    )

    result_df = result_df.round(6)
    result_df.to_csv(output_test_csv, index=False)

    if cluster_pair_test:
        n_sample_col = "n_cluster"
    else:
        n_sample_col = "n_sample"

    rank_df = rank_with_ties(result_df, n_sample_col=n_sample_col)
    rank_df.sort_values(
        by=["ranker", "eval_dataset", "subset", "eval_type"], inplace=True
    )
    rank_df.to_csv(output_rank_csv, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--details_csv", type=Path, required=True)
    parser.add_argument("-m", "--metric_col", type=str, required=True)
    parser.add_argument("-t", "--output_test_csv", type=Path, default=None)
    parser.add_argument("-r", "--output_rank_csv", type=Path, default=None)
    parser.add_argument("-c", "--cluster_pair_test", action="store_true")

    args = parser.parse_args()

    run_pairwise_test(
        details_csv=args.details_csv,
        output_test_csv=args.output_test_csv,
        output_rank_csv=args.output_rank_csv,
        metric_col=args.metric_col,
        cluster_pair_test=args.cluster_pair_test,
    )
