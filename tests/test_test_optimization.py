import pandas as pd

from test_optimization import (
    coverage_matrix,
    redundant_candidates,
    optimize_test_plan,
)


def sample_tests():
    return pd.DataFrame(
        [
            {"test_id": "T1", "requirements": "R1|R2", "cost": 100, "duration": 30, "signal_a": 10.0},
            {"test_id": "T2", "requirements": "R1|R2", "cost": 130, "duration": 38, "signal_a": 10.1},
            {"test_id": "T3", "requirements": "R2|R3", "cost": 90, "duration": 28, "signal_a": 20.0},
            {"test_id": "T4", "requirements": "R3|R4", "cost": 80, "duration": 24, "signal_a": 30.0},
        ]
    )


def test_coverage_matrix_contains_requirements():
    matrix = coverage_matrix(sample_tests())
    assert set(matrix.columns) == {"R1", "R2", "R3", "R4"}
    assert matrix.loc["T1", "R1"] == 1


def test_redundant_candidate_detected():
    redundant = redundant_candidates(sample_tests(), ["signal_a"])
    assert ((redundant["test_a"] == "T1") & (redundant["test_b"] == "T2")).any()


def test_optimizer_preserves_requirement_coverage():
    result = optimize_test_plan(sample_tests())
    assert result.requirement_coverage == 1.0
    assert result.selected_count <= result.original_count
    assert result.selected_cost <= result.original_cost
    assert result.selected_duration <= result.original_duration
