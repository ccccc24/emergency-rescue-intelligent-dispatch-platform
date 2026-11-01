# -*- coding: utf-8 -*-
from __future__ import annotations

import pytest

from app.common.data_loader import (
    build_plan_summary,
    compute_risk_reduction_pct,
    get_default_selection,
    load_ablation,
    load_network_layout,
    load_scenario,
    load_solution,
    load_solution_index,
    load_village_names,
)


def test_default_scenario_and_solution_load() -> None:
    index = load_solution_index()
    scenario_date, group = get_default_selection(index)
    assert scenario_date == "2025-08-29"
    assert group == "C"

    scenario = load_scenario(scenario_date, index)
    solution = load_solution(scenario_date, group)
    assert scenario["weather"] == "晴"
    assert len(scenario["villages"]) == 6
    assert len(scenario["roads"]) == 30
    assert len(scenario["priority_villages"]) == 3
    assert solution["solver_status"] == "success"
    assert len(solution["vehicle_trajectories"]) == 3


def test_all_six_historical_solutions_load() -> None:
    index = load_solution_index()
    statuses = {}
    for scenario_date, metadata in index["scenarios"].items():
        for group in metadata["modes"]:
            solution = load_solution(scenario_date, group)
            statuses[(scenario_date, group)] = solution["solver_status"]
    assert statuses[("2025-08-01", "A")] == "timeout"
    assert statuses[("2025-08-01", "B")] == "success"
    assert statuses[("2025-08-01", "C")] == "success"
    assert statuses[("2025-08-29", "A")] == "success"


def test_layout_names_and_summary() -> None:
    index = load_solution_index()
    scenario_date, group = get_default_selection(index)
    scenario = load_scenario(scenario_date, index)
    solution = load_solution(scenario_date, group)
    layout = load_network_layout()
    names = load_village_names()
    assert len(layout) == 15
    assert names[1] == "青石村"
    summary = build_plan_summary(scenario, solution, names, group)
    assert summary["scenario_date"] == "2025-08-29"
    assert len(summary["priority_villages"]) == 3
    assert len(summary["top_risk_roads"]) == 5


@pytest.mark.parametrize(
    ("kind", "expected"),
    (("calm", 23.52664850485099), ("storm", 28.41229656309134)),
)
def test_risk_reduction_is_computed_from_csv(kind: str, expected: float) -> None:
    frame = load_ablation(kind)
    assert compute_risk_reduction_pct(frame) == pytest.approx(expected, rel=1e-9)
