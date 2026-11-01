# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
from pathlib import Path

from app.common.scenario_builder import build_user_active_scenario, default_scheduler_config
from src.ml.user_prediction import (
    DEMAND_MODEL_PATH,
    RISK_MODEL_PATH,
    load_model_bundle,
    predict_user_inputs,
)


def _villages():
    return [
        {
            "name": "东村",
            "x": 0,
            "y": 10,
            "priority": 1,
            "population": 3000,
            "disaster_level": 2,
            "road_accessibility": 0.8,
            "hist_demand_mean": 100.0,
        },
        {
            "name": "西村",
            "x": -12,
            "y": 0,
            "priority": 0,
            "population": 2400,
            "disaster_level": 1,
            "road_accessibility": 0.9,
            "hist_demand_mean": 80.0,
        },
    ]


def _roads():
    return [
        {"from": 0, "to": 1, "distance": 12, "base_speed": 50, "geology_risk": 0.2, "hist_break_count": 1},
        {"from": 0, "to": 2, "distance": 14, "base_speed": 50, "geology_risk": 0.3, "hist_break_count": 0},
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_missing_final_values_are_predicted() -> None:
    result = predict_user_inputs(_villages(), _roads(), source="manual", weather="晴")
    metadata = result["prediction_metadata"]
    assert all(row["demand"] >= 0 for row in result["villages"])
    assert all(0 <= row["risk"] <= 1 for row in result["roads"])
    assert metadata["rows_predicted"] == 4
    assert metadata["rows_missing_features"] == 0
    assert metadata["source"] == "manual"
    assert metadata["demand_model_sha256"] == _sha256(DEMAND_MODEL_PATH)
    assert metadata["risk_model_sha256"] == _sha256(RISK_MODEL_PATH)


def test_default_mode_preserves_user_values() -> None:
    villages = _villages()
    roads = _roads()
    villages[0]["demand"] = 222.0
    roads[0]["risk"] = 0.91
    result = predict_user_inputs(villages, roads, source="excel", weather="晴", overwrite_existing=False)
    assert result["villages"][0]["demand"] == 222.0
    assert result["roads"][0]["risk"] == 0.91
    assert result["prediction_metadata"]["rows_overridden_by_user"] == 2
    assert result["prediction_metadata"]["rows_predicted"] == 2


def test_overwrite_mode_replaces_all_user_values() -> None:
    villages = _villages()
    roads = _roads()
    villages[0]["demand"] = 222.0
    roads[0]["risk"] = 0.91
    result = predict_user_inputs(villages, roads, source="manual", weather="晴", overwrite_existing=True)
    assert result["villages"][0]["demand"] != 222.0
    assert result["roads"][0]["risk"] != 0.91
    assert result["prediction_metadata"]["rows_predicted"] == 4
    assert result["prediction_metadata"]["rows_overridden_by_user"] == 0


def test_missing_feature_isolated_to_single_row() -> None:
    villages = _villages()
    villages[0]["hist_demand_mean"] = None
    result = predict_user_inputs(villages, _roads(), source="excel", weather="晴")
    assert "demand" not in result["villages"][0]
    assert result["villages"][1]["demand"] >= 0
    assert result["prediction_metadata"]["rows_missing_features"] == 1
    assert result["prediction_metadata"]["rows_predicted"] == 3
    assert result["village_status"][0]["status"] == "error"


def test_model_contract_hashes_and_feature_order() -> None:
    bundle = load_model_bundle()
    assert bundle.demand_model_sha256 == _sha256(DEMAND_MODEL_PATH)
    assert bundle.risk_model_sha256 == _sha256(RISK_MODEL_PATH)
    assert tuple(bundle.demand_model.feature_names_in_) == tuple(bundle.demand_metadata["feature_order"])
    assert tuple(bundle.risk_model.feature_names_in_) == tuple(bundle.risk_metadata["feature_order"])


def test_prediction_changes_hash_and_repeat_prediction_is_stable() -> None:
    villages = _villages()
    roads = _roads()
    for row in villages:
        row["demand"] = 1.0
    for row in roads:
        row["risk"] = 0.01
    config = default_scheduler_config()
    before = build_user_active_scenario(
        villages,
        roads,
        config,
        input_mode="full",
        weather="晴",
    )
    first = predict_user_inputs(villages, roads, source="manual", weather="晴", overwrite_existing=True)
    first_active = build_user_active_scenario(
        first["villages"],
        first["roads"],
        config,
        input_mode="full",
        weather="晴",
        prediction=first["prediction_metadata"],
    )
    second = predict_user_inputs(villages, roads, source="manual", weather="晴", overwrite_existing=True)
    second_active = build_user_active_scenario(
        second["villages"],
        second["roads"],
        config,
        input_mode="full",
        weather="晴",
        prediction=second["prediction_metadata"],
    )
    assert first_active["scenario_hash"] != before["scenario_hash"]
    assert second_active["scenario_hash"] == first_active["scenario_hash"]
    assert first["prediction_metadata"]["predicted_at"]
    assert second["prediction_metadata"]["predicted_at"]
