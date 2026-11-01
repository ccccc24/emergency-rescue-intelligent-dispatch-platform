# -*- coding: utf-8 -*-
from __future__ import annotations

import pytest

from src.ml.feature_builder import (
    DEMAND_FEATURE_ORDER,
    FeatureValidationError,
    RISK_FEATURE_ORDER,
    build_demand_feature_frame,
    build_risk_feature_frame,
    demand_row_features,
    road_row_features,
)


def _village(**overrides):
    row = {
        "name": "东村",
        "x": 0,
        "y": 10,
        "priority": 1,
        "population": 3000,
        "disaster_level": 2,
        "road_accessibility": 0.8,
        "hist_demand_mean": 100.0,
    }
    row.update(overrides)
    return row


def _road(**overrides):
    row = {
        "from": 0,
        "to": 1,
        "distance": 12,
        "base_speed": 50.0,
        "geology_risk": 0.2,
        "hist_break_count": 1,
    }
    row.update(overrides)
    return row


def test_feature_order_and_derived_values() -> None:
    village_features, errors = demand_row_features(_village(), "雨")
    road_features, road_errors = road_row_features(_road(), "雨")
    assert errors == road_errors == []
    assert tuple(village_features) == DEMAND_FEATURE_ORDER
    assert tuple(road_features) == RISK_FEATURE_ORDER
    assert village_features["distance_to_center"] == 10.0
    assert village_features["weather_code"] == 1
    assert village_features["is_priority"] == 1
    assert road_features["length"] == 12.0
    assert road_features["weather_code"] == 1


def test_missing_hist_demand_mean_is_rejected_without_default() -> None:
    features, errors = demand_row_features(_village(hist_demand_mean=None), "晴")
    assert features is None
    assert any("hist_demand_mean 缺失" in error for error in errors)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("population", 0, "population"),
        ("disaster_level", 6, "disaster_level"),
        ("road_accessibility", 1.2, "road_accessibility"),
        ("hist_demand_mean", -1, "hist_demand_mean"),
    ),
)
def test_out_of_range_village_feature_is_rejected(field: str, value: float, message: str) -> None:
    features, errors = demand_row_features(_village(**{field: value}), "晴")
    assert features is None
    assert any(message in error for error in errors)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("base_speed", 0, "base_speed"),
        ("geology_risk", 1.5, "geology_risk"),
        ("hist_break_count", -1, "hist_break_count"),
        ("distance", 0, "distance"),
    ),
)
def test_out_of_range_road_feature_is_rejected(field: str, value: float, message: str) -> None:
    features, errors = road_row_features(_road(**{field: value}), "晴")
    assert features is None
    assert any(message in error for error in errors)


def test_invalid_weather_is_rejected() -> None:
    with pytest.raises(FeatureValidationError) as exc_info:
        build_demand_feature_frame([_village()], "雪")
    assert "weather" in str(exc_info.value)


def test_feature_frames_match_model_order() -> None:
    demand_frame = build_demand_feature_frame([_village()], "晴")
    risk_frame = build_risk_feature_frame([_road()], "暴风")
    assert tuple(demand_frame.columns) == DEMAND_FEATURE_ORDER
    assert tuple(risk_frame.columns) == RISK_FEATURE_ORDER
