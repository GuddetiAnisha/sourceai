"""Test portfolio optimization for redundant experiment reduction.

This module supports software-only analysis of a portfolio of physical or
software tests. It detects overlap in requirement coverage and numeric outcomes,
then builds a smaller candidate plan using a transparent greedy coverage
heuristic. It does not claim optimality and does not use Volvo data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import math
import pandas as pd


@dataclass(frozen=True)
class TestPlanResult:
    selected_ids: tuple[str, ...]
    original_count: int
    selected_count: int
    removed_count: int
    requirement_coverage: float
    original_cost: float
    selected_cost: float
    original_duration: float
    selected_duration: float


def parse_requirements(value) -> set[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return set()
    if isinstance(value, str):
        return {x.strip() for x in value.replace(",", "|").split("|") if x.strip()}
    if isinstance(value, Iterable):
        return {str(x).strip() for x in value if str(x).strip()}
    return {str(value).strip()}


def coverage_matrix(tests: pd.DataFrame, requirement_col: str = "requirements") -> pd.DataFrame:
    req_sets = tests[requirement_col].map(parse_requirements)
    universe = sorted(set().union(*req_sets.tolist()) if len(req_sets) else set())
    matrix = pd.DataFrame(
        [{req: int(req in reqs) for req in universe} for reqs in req_sets],
        index=tests["test_id"].astype(str),
    )
    matrix.index.name = "test_id"
    return matrix


def pairwise_requirement_overlap(tests: pd.DataFrame, requirement_col: str = "requirements") -> pd.DataFrame:
    rows = []
    reqs = [parse_requirements(x) for x in tests[requirement_col]]
    ids = tests["test_id"].astype(str).tolist()

    for i in range(len(tests)):
        for j in range(i + 1, len(tests)):
            union = reqs[i] | reqs[j]
            similarity = len(reqs[i] & reqs[j]) / len(union) if union else 1.0
            rows.append(
                {
                    "test_a": ids[i],
                    "test_b": ids[j],
                    "requirement_jaccard": round(float(similarity), 4),
                }
            )
    return pd.DataFrame(rows)


def numeric_similarity(
    tests: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    rows = []
    ids = tests["test_id"].astype(str).tolist()

    for i in range(len(tests)):
        for j in range(i + 1, len(tests)):
            scores = []
            for col in columns:
                a = float(tests.iloc[i][col])
                b = float(tests.iloc[j][col])
                scale = max(abs(a), abs(b), 1.0)
                scores.append(max(0.0, 1.0 - abs(a - b) / scale))
            rows.append(
                {
                    "test_a": ids[i],
                    "test_b": ids[j],
                    "numeric_similarity": round(
                        float(sum(scores) / len(scores)) if scores else 1.0, 4
                    ),
                }
            )
    return pd.DataFrame(rows)


def redundant_candidates(
    tests: pd.DataFrame,
    outcome_columns: Sequence[str],
    requirement_threshold: float = 0.75,
    numeric_threshold: float = 0.90,
) -> pd.DataFrame:
    overlap = pairwise_requirement_overlap(tests)
    numeric = numeric_similarity(tests, outcome_columns)
    merged = overlap.merge(numeric, on=["test_a", "test_b"], how="inner")
    return merged[
        (merged["requirement_jaccard"] >= requirement_threshold)
        & (merged["numeric_similarity"] >= numeric_threshold)
    ].reset_index(drop=True)


def optimize_test_plan(
    tests: pd.DataFrame,
    requirement_col: str = "requirements",
    cost_col: str = "cost",
    duration_col: str = "duration",
    cost_weight: float = 0.5,
    duration_weight: float = 0.5,
) -> TestPlanResult:
    required = {"test_id", requirement_col, cost_col, duration_col}
    missing = required - set(tests.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if len(tests) == 0:
        return TestPlanResult((), 0, 0, 0, 1.0, 0.0, 0.0, 0.0, 0.0)

    data = tests.copy()
    data["_reqs"] = data[requirement_col].map(parse_requirements)
    all_requirements = set().union(*data["_reqs"].tolist())
    uncovered = set(all_requirements)
    selected: list[str] = []

    max_cost = max(float(data[cost_col].max()), 1.0)
    max_duration = max(float(data[duration_col].max()), 1.0)

    while uncovered:
        best_idx = None
        best_key = None

        for idx, row in data.iterrows():
            test_id = str(row["test_id"])
            if test_id in selected:
                continue

            gain = len(row["_reqs"] & uncovered)
            if gain == 0:
                continue

            burden = (
                cost_weight * float(row[cost_col]) / max_cost
                + duration_weight * float(row[duration_col]) / max_duration
            )
            score = gain / max(burden, 1e-9)
            key = (score, gain, -float(row[cost_col]), -float(row[duration_col]), test_id)

            if best_key is None or key > best_key:
                best_key = key
                best_idx = idx

        if best_idx is None:
            break

        row = data.loc[best_idx]
        selected.append(str(row["test_id"]))
        uncovered -= row["_reqs"]

    selected_df = data[data["test_id"].astype(str).isin(selected)]
    selected_requirements = set().union(*selected_df["_reqs"].tolist()) if len(selected_df) else set()
    coverage = len(selected_requirements) / len(all_requirements) if all_requirements else 1.0

    return TestPlanResult(
        selected_ids=tuple(selected),
        original_count=len(data),
        selected_count=len(selected_df),
        removed_count=len(data) - len(selected_df),
        requirement_coverage=float(coverage),
        original_cost=float(data[cost_col].sum()),
        selected_cost=float(selected_df[cost_col].sum()),
        original_duration=float(data[duration_col].sum()),
        selected_duration=float(selected_df[duration_col].sum()),
    )
