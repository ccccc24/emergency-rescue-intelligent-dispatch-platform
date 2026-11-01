# -*- coding: utf-8 -*-
"""统一读取示例方案、当前方案、布局和消融数据。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from app.common.metrics import compute_delivery_metrics, format_satisfaction_rate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CONFIG_DIR = PROJECT_ROOT / "configs"
LAYOUT_PATH = PROJECT_ROOT / "data" / "geo" / "network_layout.json"

GROUP_LABELS = {
    "A": "A 历史均值需求",
    "B": "B 仅需求预测",
    "C": "C 完整 ML",
}
ABLATION_FILES = {
    "calm": OUTPUTS_DIR / "ablation_calm.csv",
    "storm": OUTPUTS_DIR / "ablation_storm.csv",
}
ABLATION_NUMERIC_COLUMNS = (
    "slowest_vehicle_time",
    "total_distance",
    "priority_villages_satisfied",
    "total_exposed_risk",
    "solve_time",
    "priority_demand_satisfied_rate",
    "total_demand_satisfied_rate",
    "high_risk_edge_count",
    "high_risk_edge_visits",
    "unserved_demand_total",
)


class DataContractError(ValueError):
    """数据字段或格式不符合前端契约。"""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"数据文件不存在：{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DataContractError(f"JSON 顶层必须是对象：{path}")
    return payload


def _resolve_output_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def get_default_selection(index: dict[str, Any] | None = None) -> tuple[str, str]:
    """返回默认场景日期和组别，缺失时按稳定规则兜底。"""

    index = index or load_solution_index()
    scenarios = index.get("scenarios", {})
    if not scenarios:
        raise DataContractError("solution_index.json 缺少 scenarios")
    scenario_date = str(index.get("default_scenario") or sorted(scenarios)[0])
    scenario_meta = scenarios.get(scenario_date)
    if not scenario_meta:
        scenario_date = sorted(scenarios)[0]
        scenario_meta = scenarios[scenario_date]
    modes = [str(mode).upper() for mode in scenario_meta.get("modes", ["A", "B", "C"])]
    default_mode = str(index.get("default_mode") or "C").upper()
    if default_mode not in modes:
        default_mode = "C" if "C" in modes else modes[0]
    return scenario_date, default_mode


def load_solution_index() -> dict[str, Any]:
    """读取场景与 A/B/C 文件索引。"""

    index = _read_json(OUTPUTS_DIR / "solution_index.json")
    scenarios = index.get("scenarios")
    if not isinstance(scenarios, dict) or not scenarios:
        raise DataContractError("solution_index.json 的 scenarios 无效")
    for scenario_date, metadata in scenarios.items():
        files = metadata.get("files") if isinstance(metadata, dict) else None
        if not isinstance(files, dict) or not files:
            raise DataContractError(f"场景 {scenario_date} 缺少 files 映射")
    default_date, default_mode = get_default_selection(index)
    index["default_scenario"] = default_date
    index["default_mode"] = default_mode
    return index


def load_solution(scenario_date: str, group: str) -> dict[str, Any]:
    """读取指定场景和组别的方案。"""

    scenario_date = str(scenario_date)
    group = str(group).upper()
    index = load_solution_index()
    scenario_meta = index["scenarios"].get(scenario_date)
    if scenario_meta is None:
        raise KeyError(f"未知场景：{scenario_date}")
    raw_path = scenario_meta.get("files", {}).get(group)
    if not raw_path:
        raise KeyError(f"场景 {scenario_date} 没有组别 {group}")
    solution = _read_json(_resolve_output_path(raw_path))
    for field in ("vehicle_trajectories", "village_delivery_summary", "metrics", "solver_status"):
        if field not in solution:
            raise DataContractError(f"solution 缺少字段 {field}：{raw_path}")
    if not isinstance(solution["vehicle_trajectories"], dict):
        raise DataContractError(f"vehicle_trajectories 不是对象：{raw_path}")
    if not isinstance(solution["metrics"], dict):
        raise DataContractError(f"metrics 不是对象：{raw_path}")
    return solution


def load_scenario(scenario_date: str, index: dict[str, Any] | None = None) -> dict[str, Any]:
    """读取场景并做前端友好的整数键归一化。"""

    scenario_date = str(scenario_date)
    index = index or load_solution_index()
    scenario_meta = index.get("scenarios", {}).get(scenario_date, {})
    payload = _read_json(OUTPUTS_DIR / f"scenario_{scenario_date}.json")

    for field in ("villages", "roads", "demands", "risk_scores", "adjusted_speed"):
        if field not in payload:
            raise DataContractError(f"scenario 缺少字段 {field}：{scenario_date}")

    villages = {
        int(raw_id): dict(village)
        for raw_id, village in payload["villages"].items()
    }
    roads = {int(raw_id): dict(road) for raw_id, road in payload["roads"].items()}
    demands = {
        int(village_id): float(value)
        for village_id, value in payload["demands"].items()
    }
    risk_scores = {
        int(road_id): float(value)
        for road_id, value in payload["risk_scores"].items()
    }
    adjusted_speed = {
        int(road_id): float(value)
        for road_id, value in payload["adjusted_speed"].items()
    }
    for village_id, village in villages.items():
        village["id"] = int(village.get("id", village_id))
        village["demand"] = float(demands.get(village_id, 0.0))
    for road_id, road in roads.items():
        road["id"] = int(road.get("id", road_id))
        road["from"] = int(road["from"])
        road["to"] = int(road["to"])
        road["risk_score"] = float(risk_scores.get(road_id, 0.0))
        road["adjusted_speed"] = float(adjusted_speed.get(road_id, road.get("base_speed", 0.0)))

    return {
        **payload,
        "date": str(payload.get("date", scenario_date)),
        "weather": str(payload.get("weather") or scenario_meta.get("weather") or ""),
        "label": str(scenario_meta.get("label") or scenario_date),
        "villages": villages,
        "roads": roads,
        "demands": demands,
        "risk_scores": risk_scores,
        "adjusted_speed": adjusted_speed,
        "priority_villages": [int(value) for value in payload.get("priority_villages", [])],
        "isolated_villages": [int(value) for value in payload.get("isolated_villages", [])],
    }


def load_network_layout() -> dict[int, dict[str, float]]:
    payload = _read_json(LAYOUT_PATH)
    layout = {
        int(node_id): {"x": float(value["x"]), "y": float(value["y"])}
        for node_id, value in payload.items()
    }
    if not layout:
        raise DataContractError("network_layout.json 为空")
    return layout


def load_village_names() -> dict[int, str]:
    """读取简单 YAML 名称映射，避免引入额外 YAML 依赖。"""

    path = CONFIG_DIR / "village_names.yaml"
    if not path.is_file():
        return {}
    names: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s*(\d+)\s*:\s*(.+?)\s*$", line)
        if match:
            value = match.group(2).strip().strip('"\'')
            names[int(match.group(1))] = value
    return names


def load_ablation(kind: Literal["calm", "storm"]) -> pd.DataFrame:
    """读取消融 CSV，并转换为数值列。"""

    key = str(kind).lower()
    if key not in ABLATION_FILES:
        raise KeyError(f"未知消融场景：{kind}")
    path = ABLATION_FILES[key]
    if not path.is_file():
        raise FileNotFoundError(f"消融文件不存在：{path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    for column in ABLATION_NUMERIC_COLUMNS:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["group"] = frame["group"].astype(str).str.upper()
    if set(frame["group"]) != {"A", "B", "C"}:
        raise DataContractError(f"消融文件组别不完整：{path}")
    return frame


def compute_risk_reduction_pct(frame: pd.DataFrame) -> float:
    """使用 B、C 两组风险暴露动态计算 C 的降幅。"""

    indexed = frame.set_index("group")
    if "B" not in indexed.index or "C" not in indexed.index:
        raise DataContractError("风险降幅计算需要 B、C 两组")
    baseline = float(indexed.loc["B", "total_exposed_risk"])
    full = float(indexed.loc["C", "total_exposed_risk"])
    if baseline == 0:
        raise DataContractError("B 组总暴露风险为 0，无法计算降幅")
    return (baseline - full) / baseline * 100.0


def scenario_option_label(scenario_date: str, index: dict[str, Any] | None = None) -> str:
    index = index or load_solution_index()
    meta = index.get("scenarios", {}).get(str(scenario_date), {})
    label = meta.get("label", scenario_date)
    weather = meta.get("weather", "")
    return f"{label}（{scenario_date}，{weather}）" if weather else f"{label}（{scenario_date}）"


def build_plan_summary(
    scenario: dict[str, Any],
    solution: dict[str, Any],
    village_names: dict[int, str],
    group: str,
) -> dict[str, Any]:
    """构造 LLM 与页面共用的结构化方案摘要。"""

    group = str(group).upper()
    delivery = solution.get("village_delivery_summary", {})
    delivery_metrics = compute_delivery_metrics(solution, scenario)
    priority_rows = []
    for village_id in scenario.get("priority_villages", []):
        village = scenario["villages"].get(int(village_id), {})
        delivery_row = delivery.get(str(village_id), {})
        delivered = float(delivery_row.get("delivered", 0.0))
        priority_rows.append(
            {
                "id": int(village_id),
                "name": village_names.get(int(village_id), f"村庄{village_id}"),
                "demand": float(
                    delivery_row.get("demand", scenario["demands"].get(int(village_id), 0.0))
                ),
                "delivered": delivered,
            }
        )

    top_risk_roads = []
    for road_id, road in scenario["roads"].items():
        top_risk_roads.append(
            {
                "road_id": str(road.get("road_id") or f"R{int(road_id):02d}"),
                "from": int(road["from"]),
                "to": int(road["to"]),
                "risk_score": float(road.get("risk_score", 0.0)),
            }
        )
    top_risk_roads.sort(key=lambda item: item["risk_score"], reverse=True)

    village_rows = []
    for village_id, village in scenario["villages"].items():
        village_id = int(village.get("id", village_id))
        delivery_row = delivery.get(str(village_id), {})
        demand = float(delivery_row.get("demand", scenario["demands"].get(village_id, 0.0)))
        delivered = float(delivery_row.get("delivered", 0.0))
        village_rows.append(
            {
                "id": village_id,
                "name": village_names.get(village_id, f"村庄{village_id}"),
                "population": int(village.get("population", 0)),
                "disaster_level": int(village.get("disaster_level", 0)),
                "demand": round(demand, 6),
                "delivered": round(delivered, 6),
                "satisfaction_rate": (
                    round(min(max(delivered, 0.0), demand) / demand, 6)
                    if demand > 0
                    else None
                ),
                "is_priority": village_id in scenario.get("priority_villages", []),
            }
        )

    road_rows = [
        {
            "road_id": str(road.get("road_id") or f"R{int(road_id):02d}"),
            "from": int(road["from"]),
            "to": int(road["to"]),
            "distance": round(float(road.get("length", 0.0)), 6),
            "risk_score": round(float(road.get("risk_score", 0.0)), 6),
            "adjusted_speed": round(float(road.get("adjusted_speed", 0.0)), 6),
            "terrain": str(road.get("terrain", "")),
        }
        for road_id, road in scenario["roads"].items()
    ]

    vehicle_rows = []
    for vehicle_id, frames in solution.get("vehicle_trajectories", {}).items():
        if not isinstance(frames, list) or not frames:
            continue
        last_frame = frames[-1]
        vehicle_rows.append(
            {
                "vehicle_id": str(vehicle_id),
                "route_locations": [int(frame.get("location", 0)) for frame in frames],
                "start_time": float(frames[0].get("time", 0.0)),
                "end_time": float(last_frame.get("time", 0.0)),
                "final_load": float(last_frame.get("load", 0.0)),
                "final_delivered": dict(last_frame.get("delivered", {})),
                "last_action": str(last_frame.get("action", "")),
            }
        )

    summary_metrics = dict(solution.get("metrics", {}))
    summary_metrics["total_demand"] = delivery_metrics.total_demand
    summary_metrics["delivered_demand"] = delivery_metrics.delivered
    summary_metrics["total_demand_satisfied_rate"] = delivery_metrics.satisfaction_rate
    summary_metrics["unserved_demand_total"] = delivery_metrics.unserved_demand
    summary_metrics["satisfaction_text"] = format_satisfaction_rate(delivery_metrics)

    return {
        "scenario_date": str(solution.get("scenario_date", scenario.get("date", ""))),
        "scenario_label": str(scenario.get("label", "")),
        "weather": str(scenario.get("weather", "")),
        "group": group,
        "group_label": GROUP_LABELS.get(group, group),
        "solver_status": str(solution.get("solver_status", "")),
        "solve_time_actual": float(solution.get("solve_time_actual", 0.0)),
        "metrics": summary_metrics,
        "priority_villages": priority_rows,
        "top_risk_roads": top_risk_roads[:5],
        "isolated_villages": list(scenario.get("isolated_villages", [])),
        "village_count": len(village_rows),
        "road_count": len(road_rows),
        "vehicle_count": len(vehicle_rows),
        "villages": village_rows,
        "roads": road_rows,
        "vehicles": vehicle_rows,
    }




