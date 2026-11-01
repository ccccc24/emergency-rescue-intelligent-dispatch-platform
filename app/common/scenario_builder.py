# -*- coding: utf-8 -*-
"""用户场景构建、校验、指纹和会话绑定。"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

from app.common.data_loader import PROJECT_ROOT, load_network_layout, load_scenario, load_solution, load_solution_index

SOURCE_LABELS = {
    "example": "示例数据",
    "manual": "手动场景",
    "excel": "Excel 场景",
}

def _valid_label(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text) and "\ufffd" not in text and not re.search(r"[?？]{2,}", text)


CONFIG_SCHEMA_VERSION = 3
SUPPORTED_CONFIG_KEYS = (
    "num_vehicles",
    "vehicle_capacity",
    "vehicle_time_limit",
    "risk_cost_alpha",
    "speed_penalty",
    "time_limit",
    "max_states",
)
CONFIG_ALIASES = {
    "risk_weight": "risk_cost_alpha",
    "search_time_limit": "time_limit",
}
DEFAULT_BASE_SPEED = 50.0
DEFAULT_SPEED_PENALTY = 0.1
TEMPLATE_PATH = PROJECT_ROOT / "app" / "assets" / "scenario_template.xlsx"
LIVE_DIR = PROJECT_ROOT / "outputs" / "live"
ACTIVE_SCENARIO_PATH = LIVE_DIR / "active_scenario.json"
VILLAGE_COLUMNS = ("name", "x", "y", "demand", "priority")
ROAD_COLUMNS = ("from", "to", "distance", "risk")
VILLAGE_FEATURE_COLUMNS = ("population", "disaster_level", "road_accessibility", "hist_demand_mean")
ROAD_FEATURE_COLUMNS = ("base_speed", "geology_risk", "hist_break_count")
VILLAGE_TEMPLATE_COLUMNS = VILLAGE_COLUMNS + VILLAGE_FEATURE_COLUMNS
ROAD_TEMPLATE_COLUMNS = ROAD_COLUMNS + ROAD_FEATURE_COLUMNS
CONFIG_COLUMNS = ("key", "value")
INPUT_MODES = ("minimal", "full")
INPUT_CONFIG_KEYS = ("weather", "input_mode")
SUPPORTED_WEATHER = ("晴", "雨", "暴风")
WEATHER_CODES = {"晴": 0, "雨": 1, "暴风": 2}
STARTER_VILLAGES = (
    {"name": "东村", "x": 0.0, "y": 10.0, "demand": 100.0, "priority": 1},
    {"name": "西村", "x": -12.0, "y": 0.0, "demand": 80.0, "priority": 0},
    {"name": "南村", "x": 8.0, "y": -8.0, "demand": 60.0, "priority": 0},
)
STARTER_ROADS = (
    {"from": 0, "to": 1, "distance": 12.0, "risk": 0.20},
    {"from": 0, "to": 2, "distance": 14.0, "risk": 0.30},
    {"from": 1, "to": 2, "distance": 10.0, "risk": 0.25},
    {"from": 0, "to": 3, "distance": 15.0, "risk": 0.15},
)


class ScenarioValidationError(ValueError):
    """用户场景输入不满足调度前置条件。"""

    def __init__(self, errors: list[str] | str) -> None:
        self.errors = [errors] if isinstance(errors, str) else list(errors)
        super().__init__("；".join(self.errors))


def default_scheduler_config() -> dict[str, int | float]:
    from src.ml.data_gen import load_config

    config = load_config(PROJECT_ROOT / "configs" / "live_demo.yaml")
    optimization = config["optimization"]
    risk = config.get("risk", {})
    return {
        "num_vehicles": int(optimization["num_vehicles"]),
        "vehicle_capacity": float(optimization["vehicle_capacity"]),
        "vehicle_time_limit": float(optimization["vehicle_time_limit"]),
        "risk_cost_alpha": float(optimization.get("risk_cost_alpha", 0.2)),
        "speed_penalty": float(risk.get("speed_penalty", 0.1)),
        "time_limit": float(optimization["time_limit"]),
        "max_states": int(optimization["max_states"]),
    }


def _finite_float(value: Any, field: str, errors: list[str]) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        errors.append(f"{field} 必须是数字")
        return 0.0
    if not math.isfinite(result):
        errors.append(f"{field} 必须是有限数字")
    return result


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip() == ""


def _optional_finite_float(value: Any, field: str, errors: list[str]) -> float | None:
    if _is_blank(value):
        return None
    return _finite_float(value, field, errors)


def normalize_input_mode(value: Any) -> str:
    mode = str(value or "minimal").strip().lower()
    if mode not in INPUT_MODES:
        raise ScenarioValidationError("input_mode 必须是 minimal 或 full")
    return mode


def normalize_weather(value: Any) -> str:
    weather = str(value or "晴").strip()
    if weather not in SUPPORTED_WEATHER:
        raise ScenarioValidationError("weather 必须是 晴 / 雨 / 暴风")
    return weather


def scheduler_config_from_mapping(config: dict[str, Any] | None) -> dict[str, int | float]:
    scheduler_values: dict[str, Any] = {}
    if isinstance(config, dict):
        for raw_key, value in config.items():
            raw_name = str(raw_key).strip().lower()
            key = CONFIG_ALIASES.get(raw_name, raw_name)
            if key in SUPPORTED_CONFIG_KEYS:
                scheduler_values[key] = value
            elif key in INPUT_CONFIG_KEYS or not key or key == "key":
                continue
            else:
                raise ScenarioValidationError(f"不支持的 Excel config 参数：{raw_key}")
    return normalize_scheduler_config(scheduler_values)


def _priority_value(value: Any) -> int:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "是", "优先"}:
            return 1
        if normalized in {"0", "false", "no", "n", "否", ""}:
            return 0
    try:
        return 1 if int(float(value)) != 0 else 0
    except (TypeError, ValueError):
        return 0


def normalize_scheduler_config(config: dict[str, Any] | None = None) -> dict[str, int | float]:
    defaults = default_scheduler_config()
    values = dict(defaults)
    if config:
        for raw_key, value in config.items():
            key = str(raw_key).strip().lower()
            if not key or key == "key":
                continue
            key = CONFIG_ALIASES.get(key, key)
            if key not in SUPPORTED_CONFIG_KEYS:
                raise ScenarioValidationError(
                    f"不支持的调度参数：{raw_key}；当前仅支持 {', '.join(SUPPORTED_CONFIG_KEYS)}"
                )
            values[key] = value
    errors: list[str] = []
    try:
        num_vehicles = int(values["num_vehicles"])
    except (TypeError, ValueError):
        num_vehicles = 0
    if not 1 <= num_vehicles <= 20:
        errors.append("num_vehicles 必须在 1–20 之间")
    capacity = _finite_float(values["vehicle_capacity"], "vehicle_capacity", errors)
    if capacity <= 0:
        errors.append("vehicle_capacity 必须大于 0")
    vehicle_time_limit = _finite_float(values["vehicle_time_limit"], "vehicle_time_limit", errors)
    if vehicle_time_limit <= 0:
        errors.append("vehicle_time_limit 必须大于 0")
    risk_alpha = _finite_float(values["risk_cost_alpha"], "risk_cost_alpha", errors)
    if not 0.0 <= risk_alpha <= 0.5:
        errors.append("risk_cost_alpha 必须在 [0, 0.5] 范围内")
    speed_penalty = _finite_float(values["speed_penalty"], "speed_penalty", errors)
    if not 0.0 <= speed_penalty <= 0.5:
        errors.append("speed_penalty 必须在 [0, 0.5] 范围内")
    search_time_limit = _finite_float(values["time_limit"], "time_limit", errors)
    if not 5.0 <= search_time_limit <= 35.0:
        errors.append("time_limit 必须在 [5, 35] 秒范围内")
    try:
        max_states = int(values["max_states"])
    except (TypeError, ValueError):
        max_states = 0
    if not 1000 <= max_states <= 50000:
        errors.append("max_states 必须在 [1000, 50000] 范围内")
    if errors:
        raise ScenarioValidationError(errors)
    return {
        "num_vehicles": num_vehicles,
        "vehicle_capacity": round(capacity, 6),
        "vehicle_time_limit": round(vehicle_time_limit, 6),
        "risk_cost_alpha": round(risk_alpha, 6),
        "speed_penalty": round(speed_penalty, 6),
        "time_limit": round(search_time_limit, 6),
        "max_states": max_states,
    }


def normalize_village_rows(
    rows: list[dict[str, Any]] | pd.DataFrame,
    *,
    require_final_values: bool = True,
) -> list[dict[str, Any]]:
    if isinstance(rows, pd.DataFrame):
        records = rows.to_dict("records")
    else:
        records = list(rows)
    errors: list[str] = []
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(records, start=1):
        name = str(row.get("name", "") or "").strip()
        if not name:
            errors.append(f"第 {index} 个村庄 name 不能为空")
        x = _finite_float(row.get("x", 0.0), f"村庄 {index} 的 x", errors)
        y = _finite_float(row.get("y", 0.0), f"村庄 {index} 的 y", errors)
        if _is_blank(row.get("demand")):
            demand = None
            if require_final_values:
                errors.append(f"村庄 {index} 的 demand 缺失")
        else:
            demand = _finite_float(row.get("demand"), f"村庄 {index} 的 demand", errors)
            if demand < 0:
                errors.append(f"村庄 {index} 的 demand 必须 ≥ 0")
        normalized.append(
            {
                "id": index,
                "name": name,
                "x": round(x, 6),
                "y": round(y, 6),
                "demand": None if demand is None else round(demand, 6),
                "priority": _priority_value(row.get("priority", 0)),
                "population": _optional_finite_float(row.get("population"), f"村庄 {index} 的 population", errors),
                "disaster_level": _optional_finite_float(row.get("disaster_level"), f"村庄 {index} 的 disaster_level", errors),
                "road_accessibility": _optional_finite_float(row.get("road_accessibility"), f"村庄 {index} 的 road_accessibility", errors),
                "hist_demand_mean": _optional_finite_float(row.get("hist_demand_mean"), f"村庄 {index} 的 hist_demand_mean", errors),
            }
        )
    if errors:
        raise ScenarioValidationError(errors)
    return normalized



def normalize_road_rows(
    rows: list[dict[str, Any]] | pd.DataFrame,
    *,
    require_final_values: bool = True,
) -> list[dict[str, Any]]:
    if isinstance(rows, pd.DataFrame):
        records = rows.to_dict("records")
    else:
        records = list(rows)
    errors: list[str] = []
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(records, start=1):
        from_raw = row.get("from", row.get("起点"))
        to_raw = row.get("to", row.get("终点"))
        try:
            from_node = int(from_raw)
            to_node = int(to_raw)
        except (TypeError, ValueError):
            errors.append(f"第 {index} 条道路的 from/to 必须是整数节点编号")
            continue
        if float(from_raw) != from_node or float(to_raw) != to_node:
            errors.append(f"第 {index} 条道路的 from/to 必须是整数")
        distance = _finite_float(row.get("distance", 0.0), f"道路 {index} 的 distance", errors)
        if _is_blank(row.get("risk")):
            risk = None
            if require_final_values:
                errors.append(f"道路 {index} 的 risk 缺失")
        else:
            risk = _finite_float(row.get("risk"), f"道路 {index} 的 risk", errors)
            if not 0.0 <= risk <= 1.0:
                errors.append(f"道路 {index} 的 risk 必须在 [0, 1] 内")
        if distance <= 0:
            errors.append(f"道路 {index} 的 distance 必须大于 0")
        normalized.append(
            {
                "id": index,
                "road_id": f"R{index:02d}",
                "from": from_node,
                "to": to_node,
                "distance": round(distance, 6),
                "risk": None if risk is None else round(risk, 6),
                "base_speed": _optional_finite_float(row.get("base_speed"), f"道路 {index} 的 base_speed", errors),
                "geology_risk": _optional_finite_float(row.get("geology_risk"), f"道路 {index} 的 geology_risk", errors),
                "hist_break_count": _optional_finite_float(row.get("hist_break_count"), f"道路 {index} 的 hist_break_count", errors),
            }
        )
    if errors:
        raise ScenarioValidationError(errors)
    return normalized



def validate_scenario_inputs(
    villages: list[dict[str, Any]],
    roads: list[dict[str, Any]],
    config: dict[str, int | float],
    *,
    require_final_values: bool = True,
) -> None:
    errors: list[str] = []
    if len(villages) < 2:
        errors.append("村庄数量必须至少为 2")
    if require_final_values:
        for village in villages:
            if village.get("demand") is None:
                errors.append(f"村庄 {village.get('name', village.get('id'))} 的 demand 缺失")
        for index, road in enumerate(roads, start=1):
            if road.get("risk") is None:
                errors.append(f"道路 {index} 的 risk 缺失")
    names = [str(village["name"]).strip() for village in villages]
    if len(set(names)) != len(names):
        errors.append("村庄 name 必须唯一")
    if not roads:
        errors.append("至少需要 1 条道路")

    village_ids = {int(village["id"]) for village in villages}
    valid_nodes = village_ids | {0}
    seen_edges: set[tuple[int, int]] = set()
    adjacency: dict[int, set[int]] = {node: set() for node in valid_nodes}
    for index, road in enumerate(roads, start=1):
        from_node = int(road["from"])
        to_node = int(road["to"])
        if from_node not in valid_nodes:
            errors.append(f"道路 {index} 的起点 {from_node} 不存在")
        if to_node not in valid_nodes:
            errors.append(f"道路 {index} 的终点 {to_node} 不存在")
        if from_node == to_node:
            errors.append(f"道路 {index} 的起点和终点不能相同")
        edge = (min(from_node, to_node), max(from_node, to_node))
        if edge in seen_edges:
            errors.append(f"道路 {index} 重复连接节点 {edge[0]}–{edge[1]}")
        seen_edges.add(edge)
        if from_node in valid_nodes and to_node in valid_nodes and from_node != to_node:
            adjacency[from_node].add(to_node)
            adjacency[to_node].add(from_node)

    visited = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for neighbor in adjacency.get(node, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    disconnected = sorted(village_ids - visited)
    if disconnected:
        errors.append(f"路网不连通，无法从指挥中心 0 到达村庄：{disconnected}")
    if len(config) != len(SUPPORTED_CONFIG_KEYS):
        errors.append("调度参数不完整")
    if errors:
        raise ScenarioValidationError(errors)


def compute_scenario_hash(
    scenario: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> str:
    """只对调度相关规范化字段计算稳定 SHA-256。"""

    normalized_config = normalize_scheduler_config(config)
    raw_villages = scenario.get("villages", {})
    demands = scenario.get("demands", {})
    priority_villages = {int(value) for value in scenario.get("priority_villages", [])}
    villages = []
    for raw_id, village in raw_villages.items():
        village_id = int(village.get("id", raw_id))
        villages.append(
            {
                "id": village_id,
                "name": str(village.get("name", f"村庄{village_id}")),
                "x": round(float(village.get("x", 0.0)), 9),
                "y": round(float(village.get("y", 0.0)), 9),
                "demand": round(float(demands.get(str(raw_id), demands.get(raw_id, village.get("demand", 0.0)))), 9),
                "priority": 1 if village_id in priority_villages else 0,
            }
        )
    roads = []
    for raw_id, road in scenario.get("roads", {}).items():
        roads.append(
            {
                "from": int(road["from"]),
                "to": int(road["to"]),
                "distance": round(float(road.get("length", road.get("distance", 0.0))), 9),
                "risk": round(float(road.get("risk_score", scenario.get("risk_scores", {}).get(str(raw_id), scenario.get("risk_scores", {}).get(raw_id, 0.0)))), 9),
                "base_speed": round(float(road.get("base_speed", DEFAULT_BASE_SPEED)), 9),
                "base_speed_is_fallback": bool(road.get("base_speed_is_fallback", False)),
            }
        )
    payload = {
        "config_schema_version": CONFIG_SCHEMA_VERSION,
        "villages": sorted(villages, key=lambda item: item["id"]),
        "roads": sorted(
            roads,
            key=lambda item: (
                item["from"],
                item["to"],
                item["distance"],
                item["risk"],
                item["base_speed"],
                item["base_speed_is_fallback"],
            ),
        ),
        "config": normalized_config,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _scenario_from_normalized(
    villages: list[dict[str, Any]],
    roads: list[dict[str, Any]],
    config: dict[str, int | float],
    source: str,
    label: str,
    created_at: str,
    *,
    input_mode: str,
    weather: str,
    prediction: dict[str, Any] | None,
    ml_unused: bool,
) -> dict[str, Any]:
    weather_code = WEATHER_CODES[weather]
    village_payload: dict[str, dict[str, Any]] = {}
    demands: dict[str, float] = {}
    priority_villages: list[int] = []
    ml_villages: list[dict[str, Any]] = []
    for village in villages:
        village_id = int(village["id"])
        demand = float(village["demand"])
        distance_to_center = math.hypot(float(village["x"]), float(village["y"]))
        population = village.get("population")
        disaster_level = village.get("disaster_level")
        road_accessibility = village.get("road_accessibility")
        hist_demand_mean = village.get("hist_demand_mean")
        village_payload[village_id] = {
            "id": village_id,
            "name": str(village["name"]),
            "x": float(village["x"]),
            "y": float(village["y"]),
            "population": int(population) if population is not None else (0 if input_mode == "minimal" else None),
            "disaster_level": int(disaster_level) if disaster_level is not None else (1 if input_mode == "minimal" else None),
            "is_priority": int(village["priority"]),
            "distance_to_center": round(distance_to_center, 6),
            "road_accessibility": float(road_accessibility) if road_accessibility is not None else (1.0 if input_mode == "minimal" else None),
            "weather_code": weather_code,
            "hist_demand_mean": float(hist_demand_mean) if hist_demand_mean is not None else (demand if input_mode == "minimal" else None),
            "demand": demand,
        }
        demands[village_id] = demand
        if village["priority"]:
            priority_villages.append(village_id)
        ml_villages.append(
            {
                "row": village_id,
                "name": str(village["name"]),
                "population": population,
                "disaster_level": disaster_level,
                "road_accessibility": road_accessibility,
                "hist_demand_mean": hist_demand_mean,
            }
        )

    road_payload: dict[str, dict[str, Any]] = {}
    risk_scores: dict[str, float] = {}
    adjusted_speed: dict[str, float] = {}
    ml_roads: list[dict[str, Any]] = []
    fallback_roads: list[dict[str, Any]] = []
    for road in roads:
        road_id = int(road["id"])
        risk = float(road["risk"])
        base_speed_provided = road.get("base_speed") is not None
        base_speed = float(road["base_speed"]) if base_speed_provided else DEFAULT_BASE_SPEED
        geology_risk = road.get("geology_risk")
        hist_break_count = road.get("hist_break_count")
        road_payload[road_id] = {
            "id": road_id,
            "road_id": str(road["road_id"]),
            "from": int(road["from"]),
            "to": int(road["to"]),
            "length": float(road["distance"]),
            "base_speed": base_speed,
            "base_speed_is_fallback": not base_speed_provided,
            "terrain": "用户输入",
            "weather": weather,
            "weather_code": weather_code,
            "geology_risk": float(geology_risk) if geology_risk is not None else (risk if input_mode == "minimal" else None),
            "weather_sensitivity": 0.0,
            "hist_break_count": int(hist_break_count) if hist_break_count is not None else (0 if input_mode == "minimal" else None),
            "risk_score": risk,
        }
        risk_scores[road_id] = risk
        adjusted_speed[road_id] = round(base_speed * (1.0 - risk * float(config["speed_penalty"])), 6)
        if not base_speed_provided:
            fallback_roads.append(
                {
                    "from": int(road["from"]),
                    "to": int(road["to"]),
                    "fields": ["base_speed"],
                }
            )
        ml_roads.append(
            {
                "row": road_id,
                "from": int(road["from"]),
                "to": int(road["to"]),
                "distance": float(road["distance"]),
                "base_speed": road.get("base_speed"),
                "geology_risk": geology_risk,
                "hist_break_count": hist_break_count,
            }
        )

    has_ml_inputs = any(
        value is not None
        for village in ml_villages
        for value in (village["population"], village["disaster_level"], village["road_accessibility"], village["hist_demand_mean"])
    ) or any(
        value is not None
        for road in ml_roads
        for value in (road["base_speed"], road["geology_risk"], road["hist_break_count"])
    )
    metadata: dict[str, Any] = {
        "source": source,
        "generator": "scenario_builder",
        "created_at": created_at,
        "config_schema_version": CONFIG_SCHEMA_VERSION,
        "scheduler_config": dict(config),
        "input_mode": input_mode,
        "weather": weather,
        "ml_schema_version": 1,
        "fallback_fields": {"roads": fallback_roads, "villages": []},
        "input_schema": {
            "villages": list(VILLAGE_TEMPLATE_COLUMNS),
            "roads": list(ROAD_TEMPLATE_COLUMNS),
            "config": list(SUPPORTED_CONFIG_KEYS) + ["weather", "input_mode"],
        },
    }
    if input_mode == "full":
        metadata["ml_inputs"] = {"weather": weather, "villages": ml_villages, "roads": ml_roads}
        metadata["ml_used"] = bool(prediction and int(prediction.get("rows_predicted", 0)) > 0)
        if prediction:
            metadata["prediction"] = {**copy.deepcopy(prediction), "source": source}
    else:
        metadata["ml_used"] = False
        if ml_unused or has_ml_inputs:
            metadata["ml_unused"] = True

    scenario: dict[str, Any] = {
        "date": "custom",
        "label": label,
        "weather": weather,
        "source": source,
        "input_mode": input_mode,
        "config_schema_version": CONFIG_SCHEMA_VERSION,
        "created_at": created_at,
        "villages": village_payload,
        "roads": road_payload,
        "demands": demands,
        "risk_scores": risk_scores,
        "priority_villages": priority_villages,
        "adjusted_speed": adjusted_speed,
        "isolated_villages": [],
        "metadata": metadata,
    }
    scenario_hash = compute_scenario_hash(scenario, config)
    scenario["scenario_hash"] = scenario_hash
    scenario["metadata"]["scenario_hash"] = scenario_hash
    return scenario



def build_scenario(
    villages: list[dict[str, Any]] | pd.DataFrame,
    roads: list[dict[str, Any]] | pd.DataFrame,
    config: dict[str, Any] | None = None,
    *,
    source: Literal["manual", "excel"] = "manual",
    label: str | None = None,
    input_mode: Literal["minimal", "full"] = "minimal",
    weather: str = "晴",
    prediction: dict[str, Any] | None = None,
    ml_unused: bool = False,
) -> dict[str, Any]:
    if source not in {"manual", "excel"}:
        raise ScenarioValidationError("用户场景 source 必须是 manual 或 excel")
    normalized_mode = normalize_input_mode(input_mode)
    normalized_weather = normalize_weather(weather)
    normalized_villages = normalize_village_rows(villages, require_final_values=True)
    normalized_roads = normalize_road_rows(roads, require_final_values=True)
    normalized_config = normalize_scheduler_config(config)
    validate_scenario_inputs(normalized_villages, normalized_roads, normalized_config)
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    return _scenario_from_normalized(
        normalized_villages,
        normalized_roads,
        normalized_config,
        source,
        label or ("手动编辑场景" if source == "manual" else "Excel 上传场景"),
        created_at,
        input_mode=normalized_mode,
        weather=normalized_weather,
        prediction=prediction,
        ml_unused=ml_unused,
    )


def validate_user_scenario_inputs(
    villages: list[dict[str, Any]] | pd.DataFrame,
    roads: list[dict[str, Any]] | pd.DataFrame,
    config: dict[str, Any] | None = None,
    *,
    weather: str = "晴",
) -> None:
    """校验预览阶段的基础字段和图结构，允许 demand/risk 暂缺。"""

    normalize_weather(weather)
    normalized_config = normalize_scheduler_config(config)
    normalized_villages = normalize_village_rows(villages, require_final_values=False)
    normalized_roads = normalize_road_rows(roads, require_final_values=False)
    validate_scenario_inputs(
        normalized_villages,
        normalized_roads,
        normalized_config,
        require_final_values=False,
    )



def make_active_scenario(
    scenario: dict[str, Any],
    *,
    source: Literal["example", "manual", "excel"],
    config: dict[str, Any] | None = None,
    label: str | None = None,
    example_date: str | None = None,
) -> dict[str, Any]:
    scenario_copy = copy.deepcopy(scenario)
    normalized_config = normalize_scheduler_config(config)
    scenario_hash = compute_scenario_hash(scenario_copy, normalized_config)
    created_at = str(scenario_copy.get("created_at") or datetime.now().astimezone().isoformat(timespec="seconds"))
    scenario_copy["source"] = source
    scenario_copy["scenario_hash"] = scenario_hash
    scenario_copy["config_schema_version"] = CONFIG_SCHEMA_VERSION
    scenario_copy["created_at"] = created_at
    metadata = scenario_copy.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        scenario_copy["metadata"] = metadata
    metadata["config_schema_version"] = CONFIG_SCHEMA_VERSION
    metadata["scheduler_config"] = dict(normalized_config)
    scenario_fallback = str(scenario_copy.get("label") or "").strip()
    fallback_label = scenario_fallback if source == "example" and _valid_label(scenario_fallback) else SOURCE_LABELS[source]
    safe_label = str(label).strip() if _valid_label(label) else fallback_label
    scenario_copy["label"] = safe_label
    return {
        "scenario": scenario_copy,
        "source": source,
        "label": safe_label,
        "scenario_hash": scenario_hash,
        "config_schema_version": CONFIG_SCHEMA_VERSION,
        "created_at": created_at,
        "example_date": example_date,
        "config": normalized_config,
    }


def build_user_active_scenario(
    villages: list[dict[str, Any]] | pd.DataFrame,
    roads: list[dict[str, Any]] | pd.DataFrame,
    config: dict[str, Any] | None = None,
    *,
    source: Literal["manual", "excel"] = "manual",
    label: str | None = None,
    input_mode: Literal["minimal", "full"] = "minimal",
    weather: str = "晴",
    prediction: dict[str, Any] | None = None,
    ml_unused: bool = False,
) -> dict[str, Any]:
    scenario = build_scenario(
        villages,
        roads,
        config,
        source=source,
        label=label,
        input_mode=input_mode,
        weather=weather,
        prediction=prediction,
        ml_unused=ml_unused,
    )
    return make_active_scenario(
        scenario,
        source=source,
        config=normalize_scheduler_config(config),
        label=label,
    )



def editable_rows_for_active(active: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if active.get("source") != "example":
        scenario = active["scenario"]
        villages = [
            {
                "id": int(village.get("id", raw_id)),
                "name": str(village.get("name", f"村庄{raw_id}")),
                "x": float(village.get("x", 0.0)),
                "y": float(village.get("y", 0.0)),
                "demand": float(scenario["demands"].get(str(raw_id), scenario["demands"].get(raw_id, village.get("demand", 0.0)))),
                "priority": 1 if int(village.get("id", raw_id)) in scenario.get("priority_villages", []) else 0,
                "population": village.get("population"),
                "disaster_level": village.get("disaster_level"),
                "road_accessibility": village.get("road_accessibility"),
                "hist_demand_mean": village.get("hist_demand_mean"),
            }
            for raw_id, village in scenario.get("villages", {}).items()
        ]
        roads = [
            {
                "from": int(road["from"]),
                "to": int(road["to"]),
                "distance": float(road.get("length", 0.0)),
                "risk": float(road.get("risk_score", scenario.get("risk_scores", {}).get(str(raw_id), scenario.get("risk_scores", {}).get(raw_id, 0.0)))),
                "base_speed": road.get("base_speed"),
                "geology_risk": road.get("geology_risk"),
                "hist_break_count": road.get("hist_break_count"),
            }
            for raw_id, road in scenario.get("roads", {}).items()
        ]
        return villages, roads, normalize_scheduler_config(active.get("config"))
    return [dict(row) for row in STARTER_VILLAGES], [dict(row) for row in STARTER_ROADS], default_scheduler_config()




def _normalize_persisted_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    villages = {
        int(raw_id): dict(village)
        for raw_id, village in scenario.get("villages", {}).items()
    }
    roads = {
        int(raw_id): dict(road)
        for raw_id, road in scenario.get("roads", {}).items()
    }
    demands = {
        int(village_id): float(value)
        for village_id, value in scenario.get("demands", {}).items()
    }
    risk_scores = {
        int(road_id): float(value)
        for road_id, value in scenario.get("risk_scores", {}).items()
    }
    adjusted_speed = {
        int(road_id): float(value)
        for road_id, value in scenario.get("adjusted_speed", {}).items()
    }
    for village_id, village in villages.items():
        village["id"] = int(village.get("id", village_id))
    for road_id, road in roads.items():
        road["id"] = int(road.get("id", road_id))
        road["from"] = int(road["from"])
        road["to"] = int(road["to"])
        road["risk_score"] = float(risk_scores.get(road_id, road.get("risk_score", 0.0)))
        road["adjusted_speed"] = float(adjusted_speed.get(road_id, road.get("adjusted_speed", road.get("base_speed", 0.0))))
    return {
        **scenario,
        "villages": villages,
        "roads": roads,
        "demands": demands,
        "risk_scores": risk_scores,
        "adjusted_speed": adjusted_speed,
        "priority_villages": [int(value) for value in scenario.get("priority_villages", [])],
        "isolated_villages": [int(value) for value in scenario.get("isolated_villages", [])],
    }


def _legacy_config(config: dict[str, Any] | None) -> dict[str, int | float]:
    """还原 schema v1 的三个调度参数，用于校验旧持久化文件。"""

    defaults = default_scheduler_config()
    values = dict(defaults)
    if isinstance(config, dict):
        for key in ("num_vehicles", "vehicle_capacity", "vehicle_time_limit"):
            if key in config:
                values[key] = config[key]
    return {
        "num_vehicles": int(values["num_vehicles"]),
        "vehicle_capacity": round(float(values["vehicle_capacity"]), 6),
        "vehicle_time_limit": round(float(values["vehicle_time_limit"]), 6),
    }


def _compute_legacy_scenario_hash(scenario: dict[str, Any], config: dict[str, Any] | None) -> str:
    """计算 schema v1 指纹，仅供旧持久化场景迁移校验。"""

    normalized_config = _legacy_config(config)
    raw_villages = scenario.get("villages", {})
    demands = scenario.get("demands", {})
    priority_villages = {int(value) for value in scenario.get("priority_villages", [])}
    villages = []
    for raw_id, village in raw_villages.items():
        village_id = int(village.get("id", raw_id))
        villages.append(
            {
                "id": village_id,
                "name": str(village.get("name", f"村庄{village_id}")),
                "x": round(float(village.get("x", 0.0)), 9),
                "y": round(float(village.get("y", 0.0)), 9),
                "demand": round(
                    float(
                        demands.get(
                            str(raw_id),
                            demands.get(raw_id, village.get("demand", 0.0)),
                        )
                    ),
                    9,
                ),
                "priority": 1 if village_id in priority_villages else 0,
            }
        )
    roads = []
    for raw_id, road in scenario.get("roads", {}).items():
        roads.append(
            {
                "from": int(road["from"]),
                "to": int(road["to"]),
                "distance": round(
                    float(road.get("length", road.get("distance", 0.0))),
                    9,
                ),
                "risk": round(
                    float(
                        road.get(
                            "risk_score",
                            scenario.get("risk_scores", {}).get(
                                str(raw_id),
                                scenario.get("risk_scores", {}).get(raw_id, 0.0),
                            ),
                        )
                    ),
                    9,
                ),
            }
        )
    payload = {
        "villages": sorted(villages, key=lambda item: item["id"]),
        "roads": sorted(
            roads,
            key=lambda item: (
                item["from"],
                item["to"],
                item["distance"],
                item["risk"],
            ),
        ),
        "config": normalized_config,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _compute_schema_v2_scenario_hash(
    scenario: dict[str, Any],
    config: dict[str, Any] | None,
) -> str:
    """计算 schema v2 指纹，仅供旧 active_scenario 迁移校验。"""

    normalized_config = normalize_scheduler_config(config)
    raw_villages = scenario.get("villages", {})
    demands = scenario.get("demands", {})
    priority_villages = {int(value) for value in scenario.get("priority_villages", [])}
    villages = []
    for raw_id, village in raw_villages.items():
        village_id = int(village.get("id", raw_id))
        villages.append(
            {
                "id": village_id,
                "name": str(village.get("name", f"村庄{village_id}")),
                "x": round(float(village.get("x", 0.0)), 9),
                "y": round(float(village.get("y", 0.0)), 9),
                "demand": round(
                    float(
                        demands.get(
                            str(raw_id),
                            demands.get(raw_id, village.get("demand", 0.0)),
                        )
                    ),
                    9,
                ),
                "priority": 1 if village_id in priority_villages else 0,
            }
        )
    roads = []
    for raw_id, road in scenario.get("roads", {}).items():
        roads.append(
            {
                "from": int(road["from"]),
                "to": int(road["to"]),
                "distance": round(
                    float(road.get("length", road.get("distance", 0.0))),
                    9,
                ),
                "risk": round(
                    float(
                        road.get(
                            "risk_score",
                            scenario.get("risk_scores", {}).get(
                                str(raw_id),
                                scenario.get("risk_scores", {}).get(raw_id, 0.0),
                            ),
                        )
                    ),
                    9,
                ),
            }
        )
    payload = {
        "config_schema_version": 2,
        "villages": sorted(villages, key=lambda item: item["id"]),
        "roads": sorted(
            roads,
            key=lambda item: (
                item["from"],
                item["to"],
                item["distance"],
                item["risk"],
            ),
        ),
        "config": normalized_config,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _annotate_base_speed_fallbacks(scenario: dict[str, Any]) -> None:
    """从 v2 场景推断并写入 base_speed 兜底元数据。"""

    metadata = scenario.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        scenario["metadata"] = metadata
    source = str(scenario.get("source") or metadata.get("source") or "")
    input_mode = str(metadata.get("input_mode") or "minimal")
    ml_inputs = metadata.get("ml_inputs", {})
    raw_speeds: dict[int, Any] = {}
    if isinstance(ml_inputs, dict):
        for item in ml_inputs.get("roads", []):
            if isinstance(item, dict) and item.get("row") is not None:
                raw_speeds[int(item["row"])] = item.get("base_speed")

    fallback_roads: list[dict[str, Any]] = []
    for raw_id, road in scenario.get("roads", {}).items():
        road_id = int(road.get("id", raw_id))
        marker = road.get("base_speed_is_fallback")
        if marker is None:
            if source == "example":
                is_fallback = False
            elif input_mode == "full" and road_id in raw_speeds:
                is_fallback = raw_speeds[road_id] is None
            else:
                is_fallback = source in {"manual", "excel"}
        else:
            is_fallback = bool(marker)
        road["base_speed_is_fallback"] = is_fallback
        if is_fallback:
            fallback_roads.append(
                {
                    "from": int(road["from"]),
                    "to": int(road["to"]),
                    "fields": ["base_speed"],
                }
            )
    metadata["fallback_fields"] = {"roads": fallback_roads, "villages": []}



def _upgrade_active_scenario(active: dict[str, Any]) -> dict[str, Any] | None:
    """校验并升级 active wrapper；内容不一致时返回 None。"""

    if not isinstance(active, dict) or not isinstance(active.get("scenario"), dict):
        return None
    source = str(active.get("source") or active["scenario"].get("source") or "")
    if source not in SOURCE_LABELS:
        return None
    declared_hash = str(active.get("scenario_hash") or "")
    if not declared_hash:
        return None

    scenario = _normalize_persisted_scenario(active["scenario"])
    metadata = scenario.get("metadata", {})
    metadata_version = (
        metadata.get("config_schema_version")
        if isinstance(metadata, dict)
        else None
    )
    schema_version = int(
        active.get("config_schema_version")
        or scenario.get("config_schema_version")
        or metadata_version
        or 1
    )
    config = normalize_scheduler_config(active.get("config"))

    if schema_version <= 1:
        if _compute_legacy_scenario_hash(scenario, active.get("config")) != declared_hash:
            return None
    elif schema_version == 2:
        if _compute_schema_v2_scenario_hash(scenario, active.get("config")) != declared_hash:
            return None
    elif schema_version == CONFIG_SCHEMA_VERSION:
        if compute_scenario_hash(scenario, config) != declared_hash:
            return None
    else:
        return None

    if schema_version < CONFIG_SCHEMA_VERSION:
        _annotate_base_speed_fallbacks(scenario)

    label = _resolved_scenario_label(active)
    scenario["source"] = source
    scenario["label"] = label
    scenario["config_schema_version"] = CONFIG_SCHEMA_VERSION
    scenario["metadata"] = metadata if isinstance(metadata, dict) else {}
    scenario["metadata"]["config_schema_version"] = CONFIG_SCHEMA_VERSION
    scenario["metadata"]["scheduler_config"] = dict(config)
    scenario_hash = compute_scenario_hash(scenario, config)
    scenario["scenario_hash"] = scenario_hash
    scenario["metadata"]["scenario_hash"] = scenario_hash
    return {
        **active,
        "scenario": scenario,
        "source": source,
        "label": label,
        "scenario_hash": scenario_hash,
        "config_schema_version": CONFIG_SCHEMA_VERSION,
        "config": config,
    }


def _persist_active_scenario(active: dict[str, Any]) -> None:
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVE_SCENARIO_PATH.write_text(json.dumps(active, ensure_ascii=False, indent=2), encoding="utf-8")


def load_persisted_active_scenario() -> dict[str, Any] | None:
    if not ACTIVE_SCENARIO_PATH.is_file():
        return None
    try:
        raw_active = json.loads(ACTIVE_SCENARIO_PATH.read_text(encoding="utf-8"))
        active = _upgrade_active_scenario(raw_active)
        if active is None:
            return None
        if raw_active != active:
            _persist_active_scenario(active)
        return active
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None




def layout_for_active_scenario(active: dict[str, Any]) -> dict[int, dict[str, float]]:
    scenario = active["scenario"]
    villages = scenario.get("villages", {})
    if villages and all("x" in village and "y" in village for village in villages.values()):
        layout = {0: {"x": 0.0, "y": 0.0}}
        for raw_id, village in villages.items():
            layout[int(village.get("id", raw_id))] = {
                "x": float(village["x"]),
                "y": float(village["y"]),
            }
        return layout
    return load_network_layout()


def _resolved_scenario_label(active: dict[str, Any]) -> str:
    source = str(active.get("source", ""))
    raw_label = active.get("label")
    if _valid_label(raw_label):
        return str(raw_label).strip()
    scenario_label = active.get("scenario", {}).get("label")
    if source == "example":
        try:
            index = load_solution_index()
            date = str(active.get("example_date") or index.get("default_scenario", ""))
            indexed_label = index.get("scenarios", {}).get(date, {}).get("label")
            if _valid_label(indexed_label):
                return str(indexed_label).strip()
        except Exception:
            pass
        if _valid_label(scenario_label):
            return str(scenario_label).strip()
    return SOURCE_LABELS.get(source, "当前场景")


def scenario_summary(active: dict[str, Any]) -> dict[str, Any]:
    scenario = active["scenario"]
    return {
        "source": active["source"],
        "source_label": SOURCE_LABELS.get(active["source"], active["source"]),
        "label": _resolved_scenario_label(active),
        "scenario_hash": active["scenario_hash"],
        "created_at": active.get("created_at", ""),
        "village_count": len(scenario.get("villages", {})),
        "road_count": len(scenario.get("roads", {})),
        "total_demand": sum(float(value) for value in scenario.get("demands", {}).values()),
        "priority_count": len(scenario.get("priority_villages", [])),
        "config": dict(active.get("config", {})),
    }









def excel_template_workbook() -> Workbook:
    workbook = Workbook()
    workbook.remove(workbook.active)
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    thin = Side(style="thin", color="D9E2F3")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    village_sheet = workbook.create_sheet("villages")
    village_sheet.append(list(VILLAGE_TEMPLATE_COLUMNS))
    village_features = ((3000, 2, 0.8, 100.0), (2400, 1, 0.9, 80.0), (1800, 2, 0.75, 60.0))
    for row, features in zip(STARTER_VILLAGES, village_features, strict=True):
        village_sheet.append([row["name"], row["x"], row["y"], None, row["priority"], *features])

    road_sheet = workbook.create_sheet("roads")
    road_sheet.append(list(ROAD_TEMPLATE_COLUMNS))
    road_features = ((50.0, 0.20, 1), (50.0, 0.30, 1), (45.0, 0.25, 0), (50.0, 0.15, 0))
    for row, features in zip(STARTER_ROADS, road_features, strict=True):
        road_sheet.append([row["from"], row["to"], row["distance"], None, *features])

    config_sheet = workbook.create_sheet("config")
    config_sheet.append(list(CONFIG_COLUMNS))
    defaults = default_scheduler_config()
    for key, value in (
        ("num_vehicles", defaults["num_vehicles"]),
        ("vehicle_capacity", defaults["vehicle_capacity"]),
        ("vehicle_time_limit", defaults["vehicle_time_limit"]),
        ("risk_weight", defaults["risk_cost_alpha"]),
        ("speed_penalty", defaults["speed_penalty"]),
        ("search_time_limit", defaults["time_limit"]),
        ("max_states", defaults["max_states"]),
        ("weather", "晴"),
        ("input_mode", "full"),
    ):
        config_sheet.append([key, value])

    widths = {
        "villages": {"A": 18, "B": 12, "C": 12, "D": 14, "E": 10, "F": 14, "G": 16, "H": 20, "I": 20},
        "roads": {"A": 10, "B": 10, "C": 14, "D": 12, "E": 14, "F": 16, "G": 20},
        "config": {"A": 24, "B": 16},
    }
    for sheet in (village_sheet, road_sheet, config_sheet):
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        sheet.sheet_view.showGridLines = False
        for cell in sheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = border
        for row in sheet.iter_rows(min_row=1, max_row=sheet.max_row, min_col=1, max_col=sheet.max_column):
            for cell in row:
                cell.border = border
        for column, width in widths[sheet.title].items():
            sheet.column_dimensions[column].width = width
        sheet.row_dimensions[1].height = 24

    for cell in village_sheet["E"][1:]:
        cell.number_format = "0"
    for cell in road_sheet["C"][1:]:
        cell.number_format = "0.00"
    for cell in road_sheet["D"][1:]:
        cell.number_format = "0.000"
    priority_validation = DataValidation(type="list", formula1='"0,1"', allow_blank=False)
    village_sheet.add_data_validation(priority_validation)
    priority_validation.add("E2:E1000")
    mode_validation = DataValidation(type="list", formula1='"minimal,full"', allow_blank=False)
    config_sheet.add_data_validation(mode_validation)
    mode_validation.add("B2:B1000")
    return workbook


def save_excel_template(path: str | Path = TEMPLATE_PATH) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    excel_template_workbook().save(output)
    return output


def _read_header(sheet: Any, required: tuple[str, ...]) -> dict[str, int]:
    headers = {
        str(cell.value).strip().lower(): index
        for index, cell in enumerate(sheet[1], start=1)
        if cell.value is not None
    }
    missing = [column for column in required if column not in headers]
    if missing:
        raise ScenarioValidationError(f"{sheet.title} 缺少列：{', '.join(missing)}")
    return headers


def read_excel_scenario(data: bytes) -> dict[str, Any]:
    try:
        workbook = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    except Exception as error:
        raise ScenarioValidationError(f"无法读取 Excel：{error}") from error
    try:
        for sheet_name in ("villages", "roads"):
            if sheet_name not in workbook.sheetnames:
                raise ScenarioValidationError(f"Excel 缺少工作表：{sheet_name}")
        village_sheet = workbook["villages"]
        road_sheet = workbook["roads"]
        village_headers = _read_header(village_sheet, VILLAGE_COLUMNS)
        road_headers = _read_header(road_sheet, ROAD_COLUMNS)

        villages = []
        for row in village_sheet.iter_rows(min_row=2, values_only=True):
            if all(_is_blank(value) for value in row):
                continue
            record = {
                column: row[village_headers[column] - 1] if column in village_headers else None
                for column in VILLAGE_TEMPLATE_COLUMNS
            }
            villages.append(record)
        roads = []
        for row in road_sheet.iter_rows(min_row=2, values_only=True):
            if all(_is_blank(value) for value in row):
                continue
            record = {
                column: row[road_headers[column] - 1] if column in road_headers else None
                for column in ROAD_TEMPLATE_COLUMNS
            }
            roads.append(record)

        config: dict[str, Any] = {}
        if "config" in workbook.sheetnames:
            config_sheet = workbook["config"]
            config_headers = _read_header(config_sheet, CONFIG_COLUMNS)
            for row in config_sheet.iter_rows(min_row=2, values_only=True):
                key = row[config_headers["key"] - 1]
                value = row[config_headers["value"] - 1]
                if _is_blank(key):
                    continue
                config[str(key).strip().lower()] = value

        village_header_set = set(village_headers)
        road_header_set = set(road_headers)
        configured_mode = config.get("input_mode")
        if not _is_blank(configured_mode):
            detected_mode = normalize_input_mode(configured_mode)
        else:
            has_feature_columns = bool(set(VILLAGE_FEATURE_COLUMNS) & village_header_set) or bool(
                set(ROAD_FEATURE_COLUMNS) & road_header_set
            )
            detected_mode = "full" if has_feature_columns else "minimal"
        weather = normalize_weather(config.get("weather", "晴"))
        return {
            "villages": villages,
            "roads": roads,
            "config": config,
            "scheduler_config": scheduler_config_from_mapping(config),
            "detected_input_mode": detected_mode,
            "weather": weather,
        }
    finally:
        workbook.close()


def build_excel_active_scenario(
    data: bytes,
    *,
    label: str = "Excel 上传场景",
    prediction: dict[str, Any] | None = None,
    input_mode: str | None = None,
    weather: str | None = None,
) -> dict[str, Any]:
    parsed = read_excel_scenario(data)
    mode = normalize_input_mode(input_mode or parsed["detected_input_mode"])
    normalized_weather = normalize_weather(weather or parsed["weather"])
    scenario = build_scenario(
        parsed["villages"],
        parsed["roads"],
        parsed["scheduler_config"],
        source="excel",
        label=label,
        input_mode=mode,
        weather=normalized_weather,
        prediction=prediction,
    )
    return make_active_scenario(
        scenario,
        source="excel",
        config=parsed["scheduler_config"],
        label=label,
    )
