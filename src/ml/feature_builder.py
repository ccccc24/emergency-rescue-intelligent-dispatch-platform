# -*- coding: utf-8 -*-
"""把用户场景特征转换为需求/风险模型输入，不修改 bridge.py。"""

from __future__ import annotations

import math
from typing import Any, Iterable

import pandas as pd

WEATHER_CODES = {"晴": 0, "雨": 1, "暴风": 2}
DEMAND_FEATURE_ORDER = (
    "population",
    "disaster_level",
    "distance_to_center",
    "road_accessibility",
    "weather_code",
    "is_priority",
    "hist_demand_mean",
)
RISK_FEATURE_ORDER = (
    "length",
    "base_speed",
    "weather_code",
    "geology_risk",
    "hist_break_count",
)
VILLAGE_FEATURE_INPUTS = (
    "population",
    "disaster_level",
    "road_accessibility",
    "hist_demand_mean",
)
ROAD_FEATURE_INPUTS = ("base_speed", "geology_risk", "hist_break_count")


class FeatureValidationError(ValueError):
    """用户特征不满足预测前置条件。"""

    def __init__(self, errors: list[str] | str) -> None:
        self.errors = [errors] if isinstance(errors, str) else list(errors)
        super().__init__("；".join(self.errors))


def is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip() == ""


def as_records(rows: Iterable[dict[str, Any]] | pd.DataFrame) -> list[dict[str, Any]]:
    if isinstance(rows, pd.DataFrame):
        return rows.to_dict("records")
    return [dict(row) for row in rows]


def _number(
    row: dict[str, Any],
    field: str,
    errors: list[str],
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    integer: bool = False,
) -> float | int | None:
    value = row.get(field)
    if is_blank(value):
        errors.append(f"{field} 缺失")
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(f"{field} 必须是数字")
        return None
    if not math.isfinite(number):
        errors.append(f"{field} 必须是有限数字")
        return None
    if integer and not number.is_integer():
        errors.append(f"{field} 必须是整数")
        return None
    if minimum is not None and number < minimum:
        errors.append(f"{field} 必须 ≥ {minimum:g}")
    if maximum is not None and number > maximum:
        errors.append(f"{field} 必须 ≤ {maximum:g}")
    return int(number) if integer else number


def _weather_code(weather: str, errors: list[str]) -> int | None:
    code = WEATHER_CODES.get(str(weather).strip())
    if code is None:
        errors.append("weather 必须是 晴 / 雨 / 暴风")
    return code


def demand_row_features(
    row: dict[str, Any],
    weather: str,
) -> tuple[dict[str, float] | None, list[str]]:
    """构造单村庄需求特征；任何缺失或越界均拒绝该行。"""

    errors: list[str] = []
    population = _number(row, "population", errors, minimum=1, maximum=1_000_000, integer=True)
    disaster_level = _number(row, "disaster_level", errors, minimum=1, maximum=5, integer=True)
    road_accessibility = _number(row, "road_accessibility", errors, minimum=0, maximum=1)
    hist_demand_mean = _number(row, "hist_demand_mean", errors, minimum=0, maximum=10000)
    x = _number(row, "x", errors, minimum=-100000, maximum=100000)
    y = _number(row, "y", errors, minimum=-100000, maximum=100000)
    priority_raw = row.get("priority")
    if is_blank(priority_raw):
        is_priority = 0
    elif isinstance(priority_raw, bool):
        is_priority = int(priority_raw)
    else:
        try:
            priority_number = float(priority_raw)
        except (TypeError, ValueError):
            errors.append("priority 必须是 0 或 1")
            priority_number = -1
        if priority_number not in {0.0, 1.0}:
            errors.append("priority 必须是 0 或 1")
        is_priority = int(priority_number) if priority_number in {0.0, 1.0} else None
    weather_code = _weather_code(weather, errors)
    if errors:
        return None, errors
    distance_to_center = math.hypot(float(x), float(y))
    return {
        "population": int(population),
        "disaster_level": int(disaster_level),
        "distance_to_center": round(distance_to_center, 6),
        "road_accessibility": float(road_accessibility),
        "weather_code": int(weather_code),
        "is_priority": int(is_priority),
        "hist_demand_mean": float(hist_demand_mean),
    }, []


def road_row_features(
    row: dict[str, Any],
    weather: str,
) -> tuple[dict[str, float] | None, list[str]]:
    """构造单道路风险特征；任何缺失或越界均拒绝该行。"""

    errors: list[str] = []
    distance_value = row.get("distance", row.get("length"))
    if is_blank(distance_value):
        errors.append("distance 缺失")
        distance = None
    else:
        try:
            distance = float(distance_value)
        except (TypeError, ValueError):
            errors.append("distance 必须是数字")
            distance = None
        else:
            if not math.isfinite(distance):
                errors.append("distance 必须是有限数字")
            elif not 0 < distance <= 1000:
                errors.append("distance 必须在 (0, 1000] 内")
    base_speed = _number(row, "base_speed", errors, minimum=1, maximum=200)
    geology_risk = _number(row, "geology_risk", errors, minimum=0, maximum=1)
    hist_break_count = _number(row, "hist_break_count", errors, minimum=0, maximum=100, integer=True)
    weather_code = _weather_code(weather, errors)
    if errors:
        return None, errors
    return {
        "length": round(float(distance), 6),
        "base_speed": float(base_speed),
        "weather_code": int(weather_code),
        "geology_risk": float(geology_risk),
        "hist_break_count": int(hist_break_count),
    }, []


def build_demand_feature_frame(
    villages: Iterable[dict[str, Any]] | pd.DataFrame,
    weather: str,
) -> pd.DataFrame:
    rows = as_records(villages)
    features: list[dict[str, float]] = []
    errors: list[str] = []
    for index, row in enumerate(rows, start=1):
        feature_row, row_errors = demand_row_features(row, weather)
        if row_errors:
            label = str(row.get("name") or f"村庄 {index}")
            errors.extend(f"{label}：{message}" for message in row_errors)
        elif feature_row is not None:
            features.append(feature_row)
    if errors:
        raise FeatureValidationError(errors)
    return pd.DataFrame(features, columns=list(DEMAND_FEATURE_ORDER), dtype=float)


def build_risk_feature_frame(
    roads: Iterable[dict[str, Any]] | pd.DataFrame,
    weather: str,
) -> pd.DataFrame:
    rows = as_records(roads)
    features: list[dict[str, float]] = []
    errors: list[str] = []
    for index, row in enumerate(rows, start=1):
        feature_row, row_errors = road_row_features(row, weather)
        if row_errors:
            errors.extend(f"道路 {index}：{message}" for message in row_errors)
        elif feature_row is not None:
            features.append(feature_row)
    if errors:
        raise FeatureValidationError(errors)
    return pd.DataFrame(features, columns=list(RISK_FEATURE_ORDER), dtype=float)
