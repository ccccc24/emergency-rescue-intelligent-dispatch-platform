# -*- coding: utf-8 -*-
"""读取 bridge 生成的 scenario JSON，并转换为调度器输入。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from src.ml.data_gen import load_config, resolve_project_path


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH: Final[Path] = PROJECT_ROOT / "configs" / "default.yaml"


@dataclass(frozen=True)
class ScenarioInput:
    """调度器所需的场景子集。"""

    villages: dict[int, dict[str, Any]]
    roads: dict[int, dict[str, Any]]
    demands: dict[int, float]
    priority_villages: set[int]
    isolated_villages: set[int]
    date: str = ""
    depot_id: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def _read_village_names(config: dict[str, Any]) -> dict[int, str]:
    """从村庄名称配置读取整数 ID 到名称的映射。"""

    names_path = resolve_project_path(config["paths"]["village_names"])
    if not names_path.is_file():
        return {}
    names_config = load_config(names_path)
    return {
        int(village_id): str(name)
        for village_id, name in names_config.get("villages", {}).items()
    }


def _derive_isolated_villages(
    villages: dict[int, dict[str, Any]],
    roads: dict[int, dict[str, Any]],
    threshold: float,
) -> set[int]:
    """按当前边风险复算孤岛村，且与 JSON 中已有标记取并集。"""

    incident: dict[int, list[float]] = {village_id: [] for village_id in villages}
    for road in roads.values():
        risk_score = float(road["risk_score"])
        for node_id in (int(road["from"]), int(road["to"])):
            if node_id in incident:
                incident[node_id].append(risk_score)
    return {
        village_id
        for village_id, risks in incident.items()
        if not risks or all(risk > threshold for risk in risks)
    }


def load_scenario(
    scenario_json_path: str | Path,
    config_path: str | Path | None = None,
) -> ScenarioInput:
    """读取 ``outputs/scenario_*.json`` 并校验跨字段一致性。"""

    scenario_path = resolve_project_path(scenario_json_path)
    if not scenario_path.is_file():
        raise FileNotFoundError(f"场景文件不存在：{scenario_path}")
    payload = json.loads(scenario_path.read_text(encoding="utf-8"))

    config_file = (
        resolve_project_path(config_path)
        if config_path is not None
        else DEFAULT_CONFIG_PATH
    )
    config = load_config(config_file)
    village_names = _read_village_names(config)

    villages: dict[int, dict[str, Any]] = {}
    for raw_id, raw_village in payload["villages"].items():
        village_id = int(raw_id)
        villages[village_id] = {
            "id": village_id,
            "name": village_names.get(village_id, str(raw_village.get("name", f"村庄{village_id}"))),
            "population": int(raw_village.get("population", 0)),
            "is_priority": int(raw_village.get("is_priority", 0)),
            "disaster_level": int(raw_village.get("disaster_level", 0)),
        }

    roads: dict[int, dict[str, Any]] = {}
    for raw_id, raw_road in payload["roads"].items():
        road_id = int(raw_id)
        risk_score = float(payload["risk_scores"][raw_id])
        adjusted_speed = float(payload["adjusted_speed"][raw_id])
        if adjusted_speed <= 0:
            raise ValueError(f"道路 {raw_id} 的调整速度必须为正数")
        roads[road_id] = {
            "id": road_id,
            "code": str(raw_road.get("road_id", f"R{road_id:02d}")),
            "from": int(raw_road["from"]),
            "to": int(raw_road["to"]),
            "length": float(raw_road["length"]),
            "base_speed": float(raw_road.get("base_speed", adjusted_speed)),
            "adjusted_speed": adjusted_speed,
            "risk_score": risk_score,
            "terrain": str(raw_road.get("terrain", "未知")),
        }

    demands = {
        int(village_id): float(demand)
        for village_id, demand in payload["demands"].items()
    }
    priority_villages = {int(value) for value in payload["priority_villages"]}
    isolated_from_json = {int(value) for value in payload.get("isolated_villages", [])}

    if not set(demands) <= set(villages):
        raise ValueError("demands 中存在未定义的村庄")
    if not priority_villages <= set(villages):
        raise ValueError("priority_villages 中存在未定义的村庄")
    if not isolated_from_json <= set(villages):
        raise ValueError("isolated_villages 中存在未定义的村庄")

    isolation_threshold = float(
        config.get("risk", {}).get("isolated_road_threshold", 0.75)
    )
    isolated_villages = isolated_from_json | _derive_isolated_villages(
        villages,
        roads,
        isolation_threshold,
    )

    return ScenarioInput(
        villages=villages,
        roads=roads,
        demands=demands,
        priority_villages=priority_villages,
        isolated_villages=isolated_villages,
        date=str(payload.get("date", "")),
        depot_id=0,
        metadata=dict(payload.get("metadata", {})),
    )
