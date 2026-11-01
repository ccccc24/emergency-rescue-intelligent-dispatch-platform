# -*- coding: utf-8 -*-
"""只读加载需求/风险模型，为用户输入提供可覆盖的预测结果。"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd

from src.ml.feature_builder import (
    DEMAND_FEATURE_ORDER,
    RISK_FEATURE_ORDER,
    as_records,
    demand_row_features,
    is_blank,
    road_row_features,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMAND_MODEL_PATH = PROJECT_ROOT / "models" / "demand_model.pkl"
DEMAND_META_PATH = PROJECT_ROOT / "models" / "demand_model_meta.json"
RISK_MODEL_PATH = PROJECT_ROOT / "models" / "risk_model.pkl"
RISK_META_PATH = PROJECT_ROOT / "models" / "risk_model_meta.json"


class UserPredictionError(ValueError):
    """模型契约或预测结果不满足页面展示要求。"""


@dataclass(frozen=True)
class ModelBundle:
    demand_model: Any
    risk_model: Any
    demand_metadata: dict[str, Any]
    risk_metadata: dict[str, Any]
    demand_model_sha256: str
    risk_model_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise UserPredictionError(f"模型元数据格式无效：{path.name}")
    return payload


def _validate_feature_contract(model: Any, metadata: dict[str, Any], expected: tuple[str, ...], name: str) -> None:
    feature_order = tuple(metadata.get("feature_order", []))
    if feature_order != expected:
        raise UserPredictionError(f"{name} metadata.feature_order 与代码定义不一致：{feature_order}")
    model_order = tuple(str(value) for value in getattr(model, "feature_names_in_", []))
    if model_order != feature_order:
        raise UserPredictionError(f"{name} pkl feature_names_in_ 与 metadata 不一致：{model_order}")


@lru_cache(maxsize=1)
def load_model_bundle() -> ModelBundle:
    for path in (DEMAND_MODEL_PATH, DEMAND_META_PATH, RISK_MODEL_PATH, RISK_META_PATH):
        if not path.is_file():
            raise UserPredictionError(f"模型文件不存在：{path.name}")
    demand_metadata = _load_json(DEMAND_META_PATH)
    risk_metadata = _load_json(RISK_META_PATH)
    try:
        demand_model = joblib.load(DEMAND_MODEL_PATH)
        risk_model = joblib.load(RISK_MODEL_PATH)
    except Exception as error:
        raise UserPredictionError(f"模型加载失败：{error}") from error
    _validate_feature_contract(demand_model, demand_metadata, DEMAND_FEATURE_ORDER, "需求模型")
    _validate_feature_contract(risk_model, risk_metadata, RISK_FEATURE_ORDER, "风险模型")
    return ModelBundle(
        demand_model=demand_model,
        risk_model=risk_model,
        demand_metadata=demand_metadata,
        risk_metadata=risk_metadata,
        demand_model_sha256=_sha256(DEMAND_MODEL_PATH),
        risk_model_sha256=_sha256(RISK_MODEL_PATH),
    )


def _existing_value(value: Any) -> float | None:
    if is_blank(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number


def _ordered_frame(feature_rows: list[dict[str, float]], columns: tuple[str, ...]) -> pd.DataFrame:
    if not feature_rows:
        return pd.DataFrame(columns=list(columns), dtype=float)
    return pd.DataFrame(feature_rows, columns=list(columns), dtype=float)


def _positive_probability(model: Any, features: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    classes = list(getattr(model, "classes_", []))
    if probabilities.ndim != 2 or 1 not in classes:
        raise UserPredictionError("风险模型未返回可用的二分类概率")
    return probabilities[:, classes.index(1)]


def predict_user_inputs(
    villages: list[dict[str, Any]] | pd.DataFrame,
    roads: list[dict[str, Any]] | pd.DataFrame,
    *,
    source: Literal["manual", "excel"],
    weather: str,
    overwrite_existing: bool = False,
) -> dict[str, Any]:
    """预测缺失的 demand/risk，并按行返回保留、预测或错误状态。"""

    if source not in {"manual", "excel"}:
        raise UserPredictionError("source 必须是 manual 或 excel")
    village_rows = as_records(villages)
    road_rows = as_records(roads)
    merged_villages = copy.deepcopy(village_rows)
    merged_roads = copy.deepcopy(road_rows)

    demand_features: list[dict[str, float]] = []
    demand_positions: list[int] = []
    demand_status: list[dict[str, Any]] = []
    village_user_kept = 0
    village_errors = 0
    for position, row in enumerate(village_rows):
        label = str(row.get("name") or f"村庄 {position + 1}")
        existing = _existing_value(row.get("demand"))
        if existing is not None and not overwrite_existing:
            if not math.isfinite(existing) or existing < 0:
                status = "error"
                village_errors += 1
                errors = ["demand 必须是非负有限数字"]
            else:
                status = "kept_user"
                village_user_kept += 1
                errors = []
            demand_status.append(
                {"row": position + 1, "name": label, "field": "demand", "status": status, "value": existing, "errors": errors}
            )
            continue
        features, errors = demand_row_features(row, weather)
        if errors:
            village_errors += 1
            demand_status.append(
                {"row": position + 1, "name": label, "field": "demand", "status": "error", "value": None, "errors": errors}
            )
            continue
        demand_features.append(features)
        demand_positions.append(position)
        demand_status.append(
            {"row": position + 1, "name": label, "field": "demand", "status": "pending", "value": None, "errors": []}
        )

    road_features: list[dict[str, float]] = []
    road_positions: list[int] = []
    road_status: list[dict[str, Any]] = []
    road_user_kept = 0
    road_errors = 0
    for position, row in enumerate(road_rows):
        label = f"道路 {position + 1}"
        existing = _existing_value(row.get("risk"))
        if existing is not None and not overwrite_existing:
            if not math.isfinite(existing) or not 0 <= existing <= 1:
                status = "error"
                road_errors += 1
                errors = ["risk 必须在 [0, 1] 内"]
            else:
                status = "kept_user"
                road_user_kept += 1
                errors = []
            road_status.append(
                {"row": position + 1, "name": label, "field": "risk", "status": status, "value": existing, "errors": errors}
            )
            continue
        features, errors = road_row_features(row, weather)
        if errors:
            road_errors += 1
            road_status.append(
                {"row": position + 1, "name": label, "field": "risk", "status": "error", "value": None, "errors": errors}
            )
            continue
        road_features.append(features)
        road_positions.append(position)
        road_status.append(
            {"row": position + 1, "name": label, "field": "risk", "status": "pending", "value": None, "errors": []}
        )

    bundle = load_model_bundle()
    predicted_count = 0
    if demand_positions:
        demand_frame = _ordered_frame(demand_features, DEMAND_FEATURE_ORDER)
        predictions = np.asarray(bundle.demand_model.predict(demand_frame), dtype=float)
        if predictions.shape[0] != len(demand_positions):
            raise UserPredictionError("需求模型输出数量与待预测行数不一致")
        for position, prediction in zip(demand_positions, predictions, strict=True):
            if not math.isfinite(float(prediction)) or float(prediction) < 0:
                status_index = next(i for i, item in enumerate(demand_status) if item["row"] == position + 1)
                demand_status[status_index].update({"status": "error", "errors": ["模型输出 demand 非法"]})
                village_errors += 1
                continue
            merged_villages[position]["demand"] = round(float(prediction), 6)
            status_index = next(i for i, item in enumerate(demand_status) if item["row"] == position + 1)
            demand_status[status_index].update({"status": "predicted", "value": round(float(prediction), 6)})
            predicted_count += 1
    if road_positions:
        risk_frame = _ordered_frame(road_features, RISK_FEATURE_ORDER)
        probabilities = _positive_probability(bundle.risk_model, risk_frame)
        if probabilities.shape[0] != len(road_positions):
            raise UserPredictionError("风险模型输出数量与待预测行数不一致")
        for position, probability in zip(road_positions, probabilities, strict=True):
            if not math.isfinite(float(probability)) or not 0 <= float(probability) <= 1:
                status_index = next(i for i, item in enumerate(road_status) if item["row"] == position + 1)
                road_status[status_index].update({"status": "error", "errors": ["模型输出 risk 非法"]})
                road_errors += 1
                continue
            merged_roads[position]["risk"] = round(float(probability), 6)
            status_index = next(i for i, item in enumerate(road_status) if item["row"] == position + 1)
            road_status[status_index].update({"status": "predicted", "value": round(float(probability), 6)})
            predicted_count += 1

    prediction_metadata = {
        "source": source,
        "weather": weather,
        "demand_model_sha256": bundle.demand_model_sha256,
        "risk_model_sha256": bundle.risk_model_sha256,
        "predicted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "rows_predicted": predicted_count,
        "rows_overridden_by_user": village_user_kept + road_user_kept,
        "rows_missing_features": village_errors + road_errors,
        "overwrite_existing": bool(overwrite_existing),
        "village_rows_predicted": sum(item["status"] == "predicted" for item in demand_status),
        "road_rows_predicted": sum(item["status"] == "predicted" for item in road_status),
    }
    return {
        "villages": merged_villages,
        "roads": merged_roads,
        "village_status": demand_status,
        "road_status": road_status,
        "prediction_metadata": prediction_metadata,
    }
