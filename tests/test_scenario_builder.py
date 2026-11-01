# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
import io
import json

import pytest
from openpyxl import load_workbook

from app.common.scenario_builder import (
    CONFIG_SCHEMA_VERSION,
    ROAD_COLUMNS,
    ROAD_TEMPLATE_COLUMNS,
    STARTER_ROADS,
    STARTER_VILLAGES,
    TEMPLATE_PATH,
    VILLAGE_COLUMNS,
    VILLAGE_TEMPLATE_COLUMNS,
    ScenarioValidationError,
    build_excel_active_scenario,
    build_scenario,
    build_user_active_scenario,
    compute_scenario_hash,
    default_scheduler_config,
    excel_template_workbook,
    make_active_scenario,
    normalize_scheduler_config,
    read_excel_scenario,
    save_excel_template,
    scheduler_config_from_mapping,
)
from src.ml.user_prediction import predict_user_inputs


def _villages() -> list[dict]:
    return [dict(row) for row in STARTER_VILLAGES]


def _roads() -> list[dict]:
    return [dict(row) for row in STARTER_ROADS]


def test_valid_manual_scenario_builds() -> None:
    scenario = build_scenario(_villages(), _roads(), default_scheduler_config(), source="manual")
    assert len(scenario["villages"]) == 3
    assert len(scenario["roads"]) == 4
    assert scenario["demands"] == {1: 100.0, 2: 80.0, 3: 60.0}
    assert scenario["priority_villages"] == [1]
    assert scenario["scenario_hash"]
    assert scenario["config_schema_version"] == CONFIG_SCHEMA_VERSION
    active = make_active_scenario(scenario, source="manual", config=default_scheduler_config())
    assert active["source"] == "manual"
    assert active["scenario_hash"] == scenario["scenario_hash"]
    assert active["config_schema_version"] == CONFIG_SCHEMA_VERSION
    assert active["scenario"]["metadata"]["scheduler_config"] == default_scheduler_config()
    assert active["created_at"]


@pytest.mark.parametrize(
    ("villages", "roads", "message"),
    (
        ([dict(STARTER_VILLAGES[0])], _roads(), "村庄数量必须至少为 2"),
        (_villages(), [{"from": 0, "to": 9, "distance": 10, "risk": 0.2}], "不存在"),
        (_villages(), [{"from": 0, "to": 1, "distance": 0, "risk": 0.2}], "distance 必须大于 0"),
        ([{**_villages()[0], "demand": -1}, *_villages()[1:]], _roads(), "demand 必须 ≥ 0"),
        (
            [{**_villages()[0], "name": _villages()[1]["name"]}, *_villages()[1:]],
            _roads(),
            "村庄 name 必须唯一",
        ),
        (
            _villages(),
            [{**_roads()[0], "risk": 1.5}, *_roads()[1:]],
            "risk 必须在 [0, 1] 内",
        ),
        (
            _villages(),
            [
                {"from": 0, "to": 1, "distance": 10, "risk": 0.2},
                {"from": 0, "to": 2, "distance": 10, "risk": 0.2},
            ],
            "路网不连通",
        ),
    ),
)
def test_invalid_inputs_are_rejected(villages: list[dict], roads: list[dict], message: str) -> None:
    with pytest.raises(ScenarioValidationError) as exc_info:
        build_scenario(villages, roads, default_scheduler_config(), source="manual")
    assert message in str(exc_info.value)


def test_excel_template_round_trip(tmp_path) -> None:
    template = tmp_path / "scenario_template.xlsx"
    save_excel_template(template)
    assert template.is_file()
    workbook = load_workbook(template, data_only=True)
    assert workbook.sheetnames == ["villages", "roads", "config"]
    assert tuple(cell.value for cell in workbook["villages"][1]) == VILLAGE_TEMPLATE_COLUMNS
    assert tuple(cell.value for cell in workbook["roads"][1]) == ROAD_TEMPLATE_COLUMNS
    workbook.close()

    parsed = read_excel_scenario(template.read_bytes())
    assert parsed["detected_input_mode"] == "full"
    assert len(parsed["villages"]) == 3
    assert len(parsed["roads"]) == 4
    assert parsed["config"]["num_vehicles"] == 3
    assert parsed["config"]["risk_weight"] == default_scheduler_config()["risk_cost_alpha"]
    assert parsed["config"]["search_time_limit"] == default_scheduler_config()["time_limit"]
    result = predict_user_inputs(
        parsed["villages"],
        parsed["roads"],
        source="excel",
        weather=parsed["weather"],
    )
    active = build_user_active_scenario(
        result["villages"],
        result["roads"],
        parsed["scheduler_config"],
        source="excel",
        label="模板预测场景",
        input_mode=parsed["detected_input_mode"],
        weather=parsed["weather"],
        prediction=result["prediction_metadata"],
    )
    assert active["source"] == "excel"
    assert active["scenario_hash"]
    assert active["config_schema_version"] == CONFIG_SCHEMA_VERSION
    assert active["config"]["risk_cost_alpha"] == default_scheduler_config()["risk_cost_alpha"]
    assert active["config"]["time_limit"] == default_scheduler_config()["time_limit"]
    assert active["scenario"]["metadata"]["prediction"]["source"] == "excel"
    assert active["scenario"]["metadata"]["prediction"]["rows_predicted"] > 0
    assert len(active["scenario"]["villages"]) == 3
    assert len(active["scenario"]["roads"]) == 4


def test_static_template_exists_and_matches_schema() -> None:
    assert TEMPLATE_PATH.is_file()
    parsed = read_excel_scenario(TEMPLATE_PATH.read_bytes())
    assert parsed["villages"]
    assert parsed["roads"]
    assert parsed["config"]


def test_hash_stable_and_changes_with_any_input() -> None:
    config = default_scheduler_config()
    first = build_scenario(_villages(), _roads(), config, source="manual")
    second = build_scenario(_villages(), _roads(), config, source="manual")
    assert first["scenario_hash"] == second["scenario_hash"]

    changed_villages = copy.deepcopy(_villages())
    changed_villages[0]["demand"] += 1.0
    changed = build_scenario(changed_villages, _roads(), config, source="manual")
    assert changed["scenario_hash"] != first["scenario_hash"]

    changed_roads = copy.deepcopy(_roads())
    changed_roads[0]["risk"] = 0.9
    road_changed = build_scenario(_villages(), changed_roads, config, source="manual")
    assert road_changed["scenario_hash"] != first["scenario_hash"]

    changed_config = dict(config)
    changed_config["vehicle_capacity"] = float(config["vehicle_capacity"]) + 10.0
    config_changed = build_scenario(_villages(), _roads(), changed_config, source="manual")
    assert config_changed["scenario_hash"] != first["scenario_hash"]


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("risk_cost_alpha", 0.4),
        ("speed_penalty", 0.3),
        ("time_limit", 15.0),
        ("max_states", 12000),
    ),
)
def test_each_scheduler_parameter_changes_hash(key: str, value: int | float) -> None:
    config = default_scheduler_config()
    baseline = build_scenario(_villages(), _roads(), config, source="manual")
    changed = build_scenario(
        _villages(),
        _roads(),
        {**config, key: value},
        source="manual",
    )
    assert changed["scenario_hash"] != baseline["scenario_hash"]


def test_config_aliases_produce_same_hash() -> None:
    config = default_scheduler_config()
    explicit = {
        **config,
        "risk_cost_alpha": 0.35,
        "time_limit": 18.0,
    }
    alias = dict(config)
    alias.pop("risk_cost_alpha")
    alias.pop("time_limit")
    alias.update({"risk_weight": 0.35, "search_time_limit": 18.0})
    explicit_scenario = build_scenario(_villages(), _roads(), explicit, source="manual")
    alias_scenario = build_scenario(_villages(), _roads(), alias, source="manual")
    assert explicit_scenario["scenario_hash"] == alias_scenario["scenario_hash"]


@pytest.mark.parametrize(
    ("key", "value", "message"),
    (
        ("risk_cost_alpha", -0.1, "risk_cost_alpha"),
        ("speed_penalty", 0.6, "speed_penalty"),
        ("time_limit", 4.9, "time_limit"),
        ("max_states", 999, "max_states"),
    ),
)
def test_scheduler_parameter_ranges_are_rejected(key: str, value: float | int, message: str) -> None:
    with pytest.raises(ScenarioValidationError) as exc_info:
        normalize_scheduler_config({**default_scheduler_config(), key: value})
    assert message in str(exc_info.value)


def test_excel_config_aliases_are_normalized() -> None:
    workbook = excel_template_workbook()
    config_sheet = workbook["config"]
    for row in config_sheet.iter_rows(min_row=2):
        if row[0].value == "risk_weight":
            row[1].value = 0.35
        elif row[0].value == "search_time_limit":
            row[1].value = 18
    village_sheet = workbook["villages"]
    village_headers = {cell.value: cell.column for cell in village_sheet[1]}
    for row_index in range(2, village_sheet.max_row + 1):
        village_sheet.cell(row_index, village_headers["demand"], 100)
    road_sheet = workbook["roads"]
    road_headers = {cell.value: cell.column for cell in road_sheet[1]}
    for row_index in range(2, road_sheet.max_row + 1):
        road_sheet.cell(row_index, road_headers["risk"], 0.2)
    buffer = io.BytesIO()
    workbook.save(buffer)
    workbook.close()

    active = build_excel_active_scenario(buffer.getvalue())
    assert active["config"]["risk_cost_alpha"] == 0.35
    assert active["config"]["time_limit"] == 18.0
    assert "risk_weight" not in active["config"]
    assert "search_time_limit" not in active["config"]
    assert scheduler_config_from_mapping({"risk_weight": 0.35})["risk_cost_alpha"] == 0.35


def test_schema_v1_persisted_scenario_migrates_and_preserves_label(
    tmp_path,
    monkeypatch,
) -> None:
    from app.common import scenario_builder as builder
    from app.common.data_loader import load_scenario, load_solution_index

    index = load_solution_index()
    date = index["default_scenario"]
    label = index["scenarios"][date]["label"]
    scenario = load_scenario(date, index)
    active = make_active_scenario(
        scenario,
        source="example",
        config=default_scheduler_config(),
        label=label,
        example_date=date,
    )
    legacy_config = {
        key: active["config"][key]
        for key in ("num_vehicles", "vehicle_capacity", "vehicle_time_limit")
    }
    active.pop("config_schema_version")
    active["scenario"].pop("config_schema_version")
    active["scenario"]["metadata"].pop("config_schema_version")
    active["scenario"]["metadata"].pop("scheduler_config", None)
    active["config"] = legacy_config
    legacy_hash = builder._compute_legacy_scenario_hash(active["scenario"], legacy_config)
    active["scenario_hash"] = legacy_hash
    active["scenario"]["scenario_hash"] = legacy_hash
    active["scenario"]["metadata"]["scenario_hash"] = legacy_hash

    active_path = tmp_path / "active_scenario.json"
    active_path.write_text(json.dumps(active, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(builder, "LIVE_DIR", tmp_path)
    monkeypatch.setattr(builder, "ACTIVE_SCENARIO_PATH", active_path)

    migrated = builder.load_persisted_active_scenario()
    assert migrated is not None
    assert migrated["label"] == label
    assert migrated["config_schema_version"] == CONFIG_SCHEMA_VERSION
    assert migrated["config"] == default_scheduler_config()
    assert migrated["scenario"]["metadata"]["config_schema_version"] == CONFIG_SCHEMA_VERSION
    assert migrated["scenario_hash"] != legacy_hash

    persisted = json.loads(active_path.read_text(encoding="utf-8"))
    assert persisted["label"] == label
    assert persisted["config_schema_version"] == CONFIG_SCHEMA_VERSION
    assert persisted["config"]["risk_cost_alpha"] == default_scheduler_config()["risk_cost_alpha"]


def test_hash_ignores_volatile_created_at() -> None:
    config = default_scheduler_config()
    first = build_scenario(_villages(), _roads(), config, source="manual")
    second = copy.deepcopy(first)
    second["created_at"] = "2099-01-01T00:00:00+08:00"
    second["metadata"]["created_at"] = second["created_at"]
    assert compute_scenario_hash(first, config) == compute_scenario_hash(second, config)

def test_invalid_persisted_label_uses_scenario_fallback() -> None:
    from app.common.data_loader import load_scenario, load_solution_index
    from app.common.scenario_builder import scenario_summary

    index = load_solution_index()
    scenario_date = index["default_scenario"]
    scenario = load_scenario(scenario_date, index)
    active = make_active_scenario(
        scenario,
        source="example",
        config=default_scheduler_config(),
        label="??",
        example_date=scenario_date,
    )
    active["label"] = "??"
    summary = scenario_summary(active)
    assert summary["label"] != "??"
    assert summary["label"] == index["scenarios"][scenario_date]["label"]

def _full_rows() -> tuple[list[dict], list[dict]]:
    villages = [
        {
            **row,
            "population": 3000,
            "disaster_level": 2,
            "road_accessibility": 0.8,
            "hist_demand_mean": 100.0,
        }
        for row in _villages()
    ]
    roads = [
        {
            **row,
            "base_speed": 50.0,
            "geology_risk": row["risk"],
            "hist_break_count": 1,
        }
        for row in _roads()
    ]
    return villages, roads


def test_input_mode_not_in_hash_and_mode_metadata_is_explicit() -> None:
    villages, roads = _full_rows()
    config = default_scheduler_config()
    full = build_user_active_scenario(
        villages,
        roads,
        config,
        input_mode="full",
        weather="雨",
        prediction={"source": "manual", "rows_predicted": 2},
    )
    minimal = build_user_active_scenario(
        villages,
        roads,
        config,
        input_mode="minimal",
        weather="雨",
        ml_unused=True,
    )
    assert full["scenario_hash"] == minimal["scenario_hash"]
    assert full["scenario"]["metadata"]["input_mode"] == "full"
    assert full["scenario"]["metadata"]["ml_inputs"]["weather"] == "雨"
    assert full["scenario"]["metadata"]["prediction"]["source"] == "manual"
    assert minimal["scenario"]["metadata"]["input_mode"] == "minimal"
    assert minimal["scenario"]["metadata"]["ml_unused"] is True
    assert "ml_inputs" not in minimal["scenario"]["metadata"]
    assert "prediction" not in minimal["scenario"]["metadata"]


def test_full_manual_can_apply_without_prediction_when_final_values_exist() -> None:
    villages, roads = _full_rows()
    for row in villages:
        row["population"] = None
        row["hist_demand_mean"] = None
    active = build_user_active_scenario(
        villages,
        roads,
        default_scheduler_config(),
        input_mode="full",
        weather="晴",
    )
    assert active["scenario"]["demands"]
    assert active["scenario"]["risk_scores"]
    assert "prediction" not in active["scenario"]["metadata"]


def test_old_basic_excel_template_uses_minimal_mode() -> None:
    from openpyxl import Workbook

    workbook = Workbook()
    village_sheet = workbook.active
    village_sheet.title = "villages"
    village_sheet.append(list(VILLAGE_COLUMNS))
    for row in _villages():
        village_sheet.append([row["name"], row["x"], row["y"], row["demand"], row["priority"]])
    road_sheet = workbook.create_sheet("roads")
    road_sheet.append(list(ROAD_COLUMNS))
    for row in _roads():
        road_sheet.append([row["from"], row["to"], row["distance"], row["risk"]])
    config_sheet = workbook.create_sheet("config")
    config_sheet.append(["key", "value"])
    config_sheet.append(["num_vehicles", 3])
    buffer = io.BytesIO()
    workbook.save(buffer)
    workbook.close()

    parsed = read_excel_scenario(buffer.getvalue())
    assert parsed["detected_input_mode"] == "minimal"
    active = build_excel_active_scenario(buffer.getvalue())
    assert active["scenario"]["metadata"]["input_mode"] == "minimal"
    assert active["config"]["num_vehicles"] == 3


def test_base_speed_fallback_hash_and_metadata() -> None:
    villages, roads = _full_rows()
    roads_without_speed = [
        {key: value for key, value in road.items() if key != "base_speed"}
        for road in roads
    ]
    roads_with_50 = [{**road, "base_speed": 50.0} for road in roads_without_speed]
    roads_with_80 = [{**road, "base_speed": 80.0} for road in roads_without_speed]
    config = default_scheduler_config()

    fallback = build_user_active_scenario(
        villages,
        roads_without_speed,
        config,
        input_mode="full",
        weather="晴",
    )
    explicit_50 = build_user_active_scenario(
        villages,
        roads_with_50,
        config,
        input_mode="full",
        weather="晴",
    )
    explicit_80 = build_user_active_scenario(
        villages,
        roads_with_80,
        config,
        input_mode="full",
        weather="晴",
    )

    assert fallback["scenario_hash"] != explicit_50["scenario_hash"]
    assert explicit_50["scenario_hash"] != explicit_80["scenario_hash"]
    assert fallback["scenario_hash"] != explicit_80["scenario_hash"]
    fallback_records = fallback["scenario"]["metadata"]["fallback_fields"]["roads"]
    assert len(fallback_records) == len(roads_without_speed)
    assert all(record["fields"] == ["base_speed"] for record in fallback_records)
    assert explicit_50["scenario"]["metadata"]["fallback_fields"]["roads"] == []
    assert explicit_80["scenario"]["metadata"]["fallback_fields"]["roads"] == []
    assert fallback["scenario"]["roads"][1]["base_speed_is_fallback"] is True
    assert explicit_50["scenario"]["roads"][1]["base_speed_is_fallback"] is False
    assert explicit_80["scenario"]["roads"][1]["base_speed_is_fallback"] is False


def test_schema_v2_persisted_scenario_migrates_without_losing_label(tmp_path, monkeypatch) -> None:
    from app.common import scenario_builder as builder
    from app.common.data_loader import load_scenario, load_solution_index

    index = load_solution_index()
    date = index["default_scenario"]
    label = index["scenarios"][date]["label"]
    scenario = load_scenario(date, index)
    active = make_active_scenario(
        scenario,
        source="example",
        config=default_scheduler_config(),
        label=label,
        example_date=date,
    )
    active["config_schema_version"] = 2
    active["scenario"]["config_schema_version"] = 2
    active["scenario"]["metadata"]["config_schema_version"] = 2
    active["scenario"]["metadata"].pop("fallback_fields", None)
    for road in active["scenario"]["roads"].values():
        road.pop("base_speed_is_fallback", None)
    legacy_hash = builder._compute_schema_v2_scenario_hash(
        active["scenario"],
        active["config"],
    )
    active["scenario_hash"] = legacy_hash
    active["scenario"]["scenario_hash"] = legacy_hash

    active_path = tmp_path / "active_scenario.json"
    active_path.write_text(json.dumps(active, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(builder, "LIVE_DIR", tmp_path)
    monkeypatch.setattr(builder, "ACTIVE_SCENARIO_PATH", active_path)

    migrated = builder.load_persisted_active_scenario()
    assert migrated is not None
    assert migrated["label"] == label
    assert migrated["config_schema_version"] == CONFIG_SCHEMA_VERSION
    assert migrated["scenario_hash"] != legacy_hash
    assert migrated["scenario"]["metadata"]["fallback_fields"] == {
        "roads": [],
        "villages": [],
    }
