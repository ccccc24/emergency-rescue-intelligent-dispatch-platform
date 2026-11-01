# -*- coding: utf-8 -*-
from __future__ import annotations

import pytest

from app.common.metrics import compute_delivery_metrics, format_satisfaction_rate


def _partial_solution() -> dict:
    return {
        "solver_status": "timeout",
        "metrics": {
            "slowest_vehicle_time": 2.0,
            "total_exposed_risk": 10.0,
            "priority_villages_satisfied": 1,
            "total_demand_satisfied_rate": 1.0,
        },
        "village_delivery_summary": {
            "1": {"demand": 100.0, "delivered": 100.0},
            "2": {"demand": 50.0, "delivered": 25.0},
        },
    }


def test_partial_solution_uses_current_plan_demand_denominator() -> None:
    scenario = {"demands": {"1": 999.0, "2": 999.0}, "priority_villages": [1, 2]}
    metrics = compute_delivery_metrics(_partial_solution(), scenario)
    assert metrics.total_demand == pytest.approx(150.0)
    assert metrics.delivered == pytest.approx(125.0)
    assert metrics.satisfaction_rate == pytest.approx(125.0 / 150.0)
    assert metrics.satisfaction_rate < 1.0
    assert format_satisfaction_rate(metrics) == "83.33%"



def test_zero_demand_is_not_displayed_as_100_percent() -> None:
    solution = {
        "solver_status": "success",
        "metrics": {},
        "village_delivery_summary": {
            "1": {"demand": 0.0, "delivered": 0.0},
        },
    }
    metrics = compute_delivery_metrics(solution)
    assert metrics.total_demand == 0.0
    assert metrics.satisfaction_rate is None
    assert format_satisfaction_rate(metrics) == "无可配送需求"
    assert "100%" not in format_satisfaction_rate(metrics)


def test_unserved_villages_missing_from_summary_are_still_in_denominator() -> None:
    solution = {
        "solver_status": "timeout",
        "metrics": {},
        "village_delivery_summary": {
            "1": {"demand": 100.0, "delivered": 100.0},
        },
    }
    scenario = {"demands": {"1": 100.0, "2": 50.0}, "priority_villages": [1]}
    metrics = compute_delivery_metrics(solution, scenario)
    assert metrics.total_demand == pytest.approx(150.0)
    assert metrics.delivered == pytest.approx(100.0)
    assert metrics.satisfaction_rate == pytest.approx(2.0 / 3.0)

def test_full_satisfaction_formats_as_exactly_two_decimals() -> None:
    from app.common.metrics import DeliveryMetrics

    metrics = DeliveryMetrics(
        total_demand=100.0,
        delivered=100.0,
        satisfaction_rate=1.0,
        unserved_demand=0.0,
        source="village_delivery_summary",
    )
    text = format_satisfaction_rate(metrics)
    assert text == "100.00%"
    assert "..." not in text
    assert "…" not in text

