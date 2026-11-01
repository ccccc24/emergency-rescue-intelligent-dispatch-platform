# -*- coding: utf-8 -*-
"""把消融场景的调度结果导出为前端可直接播放的 solution JSON。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from src.ml.data_gen import load_config, resolve_project_path

from .run import DEFAULT_CONFIG_PATH, AblationMode, Solution, solve_with_ablation


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
SCENARIO_DATES: Final[dict[str, dict[str, str]]] = {
    "2025-08-29": {"label": "常规天气日", "weather": "晴"},
    "2025-08-01": {"label": "恶劣天气日", "weather": "暴风"},
}
MODE_CODES: Final[dict[str, AblationMode]] = {
    "A": "hist_demand",
    "B": "no_risk_adjust",
    "C": "full",
}
METRIC_KEYS: Final[tuple[str, ...]] = (
    "slowest_vehicle_time",
    "total_distance",
    "priority_villages_satisfied",
    "total_exposed_risk",
    "high_risk_edge_visits",
    "total_demand_satisfied_rate",
    "unserved_demand_total",
)


def _read_scenario(scenario_json_path: str | Path) -> tuple[Path, dict[str, Any]]:
    """读取场景 JSON，并返回解析后的路径和内容。"""

    path = resolve_project_path(scenario_json_path)
    if not path.is_file():
        raise FileNotFoundError(f"场景文件不存在：{path}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def _load_hist_demands(
    scenario_payload: dict[str, Any],
    config: dict[str, Any],
) -> dict[int, float]:
    """按场景日期读取 demand.csv 的 ``hist_demand_mean``。"""

    date = str(scenario_payload.get("date", ""))
    data_path = resolve_project_path(config["paths"]["data_synthetic"]) / "demand.csv"
    if not data_path.is_file():
        raise FileNotFoundError(f"需求数据不存在：{data_path}")

    demands: dict[int, float] = {}
    with data_path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if row["date"] != date:
                continue
            village_id = int(row["village_id"])
            if str(village_id) in scenario_payload.get("villages", {}):
                demands[village_id] = float(row["hist_demand_mean"])
    missing = {
        int(village_id)
        for village_id in scenario_payload.get("villages", {})
    } - set(demands)
    if missing:
        raise ValueError(f"场景 {date} 缺少历史均值需求：{sorted(missing)}")
    return demands


def _resolve_demands(
    scenario_payload: dict[str, Any],
    mode: AblationMode,
    config: dict[str, Any],
) -> dict[int, float]:
    """按消融模式还原实际送入调度器的村庄需求。"""

    if mode == "hist_demand":
        return _load_hist_demands(scenario_payload, config)
    if mode == "fixed_demand":
        return {
            int(village_id): 100.0
            for village_id in scenario_payload.get("villages", {})
        }
    return {
        int(village_id): float(value)
        for village_id, value in scenario_payload.get("demands", {}).items()
    }


def _serialize_trajectories(
    raw_trajectories: dict[str, list[dict[str, int | float]]],
) -> dict[str, list[dict[str, Any]]]:
    """给逐车轨迹补充累计配送量和动作语义。

    现有轨迹点包含 ``time/location/load``；导出层按相邻点载荷差还原村庄配送量，
    并生成前端播放需要的 ``delivered`` 与 ``action`` 字段。
    """

    serialized: dict[str, list[dict[str, Any]]] = {}
    for vehicle_key, points in raw_trajectories.items():
        previous_load: float | None = None
        cumulative: dict[str, float] = {}
        vehicle_points: list[dict[str, Any]] = []
        for index, point in enumerate(points):
            location = int(point["location"])
            load = float(point["load"])
            delivered_now = 0.0
            if (
                previous_load is not None
                and 1 <= location <= 6
                and load < previous_load - 1e-9
            ):
                delivered_now = previous_load - load
                village_key = str(location)
                cumulative[village_key] = (
                    cumulative.get(village_key, 0.0) + delivered_now
                )

            if index == 0:
                action = "depart"
            elif location == 0 and previous_load is not None and load > previous_load + 1e-9:
                action = "reload"
            elif delivered_now > 1e-9:
                action = "deliver"
            else:
                action = "move"

            vehicle_points.append(
                {
                    "time": round(float(point["time"]), 6),
                    "location": location,
                    "load": round(load, 6),
                    "delivered": {
                        key: round(value, 6)
                        for key, value in cumulative.items()
                    },
                    "action": action,
                }
            )
            previous_load = load
        serialized[vehicle_key] = vehicle_points
    return serialized


def _delivery_summary(
    demands: dict[int, float],
    trajectories: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, float]]:
    """汇总每村需求、累计配送量和满足率。"""

    delivered: dict[int, float] = {village_id: 0.0 for village_id in demands}
    for points in trajectories.values():
        if not points:
            continue
        for village_id, amount in points[-1]["delivered"].items():
            delivered[int(village_id)] = delivered.get(int(village_id), 0.0) + float(amount)

    summary: dict[str, dict[str, float]] = {}
    for village_id, demand in sorted(demands.items()):
        delivered_value = min(max(delivered.get(village_id, 0.0), 0.0), demand)
        summary[str(village_id)] = {
            "demand": round(float(demand), 6),
            "delivered": round(float(delivered_value), 6),
            "satisfied_pct": (
                round(delivered_value / demand * 100.0, 2)
                if demand > 0
                else 100.0
            ),
        }
    return summary


def _config_snapshot(config: dict[str, Any]) -> dict[str, int | float | str]:
    """保存本次导出实际使用的调度参数。"""

    optimization = config["optimization"]
    risk = config.get("risk", {})
    return {
        "time_limit": float(optimization["time_limit"]),
        "max_states": int(optimization["max_states"]),
        "risk_cost_alpha": float(optimization.get("risk_cost_alpha", 0.0)),
        "speed_penalty": float(risk.get("speed_penalty", 0.5)),
        "num_vehicles": int(optimization["num_vehicles"]),
        "vehicle_capacity": float(optimization["vehicle_capacity"]),
        "heuristic": str(optimization["heuristic_type"]),
    }


def export_solution(
    scenario_json_path: str | Path,
    output_path: str | Path,
    ablation_mode: AblationMode = "full",
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """求解一个场景并导出前端可播放的 solution JSON。"""

    config_file = resolve_project_path(config_path or DEFAULT_CONFIG_PATH)
    config = load_config(config_file)
    scenario_path, scenario_payload = _read_scenario(scenario_json_path)
    solution: Solution = solve_with_ablation(
        scenario_path,
        ablation_mode,
        config_path=config_file,
    )

    demands = _resolve_demands(scenario_payload, ablation_mode, config)
    trajectories = _serialize_trajectories(solution.vehicle_trajectories)
    metrics = {
        key: solution.metrics[key]
        for key in METRIC_KEYS
    }
    payload: dict[str, Any] = {
        "scenario_date": str(scenario_payload.get("date", "")),
        "ablation_mode": ablation_mode,
        "config_snapshot": _config_snapshot(config),
        "vehicle_trajectories": trajectories,
        "village_delivery_summary": _delivery_summary(demands, trajectories),
        "metrics": metrics,
        "solver_status": str(solution.metrics.get("status", "")),
        "solve_time_actual": round(float(solution.metrics.get("solve_time", 0.0)), 6),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scenario_json_md5": hashlib.md5(scenario_path.read_bytes()).hexdigest(),
    }

    output = resolve_project_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def export_all_solutions(config_path: str | Path | None = None) -> dict[str, Any]:
    """导出 3 组 × 2 场景共 6 个 solution JSON，并写索引文件。"""

    config_file = resolve_project_path(config_path or DEFAULT_CONFIG_PATH)
    config = load_config(config_file)
    outputs_dir = resolve_project_path(config["paths"]["outputs"])
    index: dict[str, Any] = {"scenarios": {}}

    for date, scenario_meta in SCENARIO_DATES.items():
        scenario_path = outputs_dir / f"scenario_{date}.json"
        if not scenario_path.is_file():
            raise FileNotFoundError(f"场景文件不存在：{scenario_path}")
        index["scenarios"][date] = {
            "label": scenario_meta["label"],
            "weather": scenario_meta["weather"],
            "modes": ["A", "B", "C"],
            "files": {},
        }
        for code, mode in MODE_CODES.items():
            output_path = outputs_dir / f"solution_{date}_{code}.json"
            export_solution(
                scenario_path,
                output_path,
                ablation_mode=mode,
                config_path=config_file,
            )
            index["scenarios"][date]["files"][code] = str(
                output_path.relative_to(PROJECT_ROOT)
            ).replace("\\", "/")

    index_path = outputs_dir / "solution_index.json"
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return index


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出前端可播放的 solution JSON")
    parser.add_argument("--scenario", type=Path, help="scenario JSON 路径")
    parser.add_argument(
        "--mode",
        choices=tuple(MODE_CODES.values()),
        default="full",
        help="消融模式",
    )
    parser.add_argument("--out", type=Path, help="输出 JSON 路径")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="配置文件路径",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="导出 6 个 solution JSON 和索引文件",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.all:
        index = export_all_solutions(arguments.config)
        print(json.dumps(index, ensure_ascii=False, indent=2))
        return
    if arguments.scenario is None or arguments.out is None:
        raise SystemExit("非 --all 模式必须同时提供 --scenario 和 --out")
    payload = export_solution(
        arguments.scenario,
        arguments.out,
        ablation_mode=arguments.mode,
        config_path=arguments.config,
    )
    print(
        f"已导出 {arguments.out}："
        f"status={payload['solver_status']}，"
        f"solve_time={payload['solve_time_actual']:.3f}s"
    )


if __name__ == "__main__":
    main()
