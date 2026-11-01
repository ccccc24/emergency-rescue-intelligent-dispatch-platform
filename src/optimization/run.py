# -*- coding: utf-8 -*-
"""场景级调度入口：加载 bridge 输出并求解多车辆配送方案。"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Final, Literal

from src.ml.data_gen import load_config, resolve_project_path

from .planner import MultiVehiclePlanner
from .scenario_loader import ScenarioInput, load_scenario
from .trajectory import state_sequence_to_trajectory


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH: Final[Path] = PROJECT_ROOT / "configs" / "default.yaml"
DEFAULT_SCENARIO_PATH: Final[Path] = PROJECT_ROOT / "outputs" / "scenario_latest.json"
AblationMode = Literal["fixed_demand", "hist_demand", "no_risk_adjust", "full"]


@dataclass(frozen=True)
class Solution:
    """一次场景求解的结构化结果。"""

    vehicle_trajectories: dict[str, list[dict[str, int | float]]]
    served_villages: list[int]
    isolated_villages: list[int]
    metrics: dict[str, int | float | str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _SolveOutcome:
    """内部求解产物，额外携带消融指标所需的配送明细。"""

    solution: Solution
    final_delivered: dict[int, float]
    high_risk_edge_count: int
    high_risk_edge_visits: int


def _build_graph(
    scenario: ScenarioInput,
) -> tuple[
    dict[int, list[tuple[int, float, float]]],
    dict[tuple[int, int], tuple[float, float, int]],
]:
    """构建双向邻接表，并保留无向边的长度与风险用于暴露风险统计。"""

    graph: dict[int, list[tuple[int, float, float]]] = {}
    edge_attributes: dict[tuple[int, int], tuple[float, float, int]] = {}
    for road in scenario.roads.values():
        from_node = int(road["from"])
        to_node = int(road["to"])
        if (
            from_node in scenario.isolated_villages
            or to_node in scenario.isolated_villages
        ):
            continue
        length = float(road["length"])
        speed = float(road["adjusted_speed"])
        if speed <= 0:
            continue
        graph.setdefault(from_node, []).append((to_node, length, speed))
        graph.setdefault(to_node, []).append((from_node, length, speed))
        edge_key = (min(from_node, to_node), max(from_node, to_node))
        edge_attributes[edge_key] = (
            length,
            float(road["risk_score"]),
            int(road["id"]),
        )

    for edges in graph.values():
        edges.sort(key=lambda edge: (edge[0], edge[1]))
    return graph, edge_attributes


def _build_heuristic_nodes(
    graph: dict[int, list[tuple[int, float, float]]],
    depot_id: int,
) -> dict[int, tuple[float, float]]:
    """按到指挥中心的最短道路距离构造一维坐标。

    场景 JSON 没有节点经纬度。使用 ``(到指挥中心最短距离, 0)`` 作为坐标，
    欧氏距离仍然是有意义的路网距离下界，避免凭空编造地图坐标。
    """

    distances: dict[int, float] = {node_id: float("inf") for node_id in graph}
    if depot_id not in distances:
        raise ValueError(f"指挥中心节点 {depot_id} 不在场景路网中")
    distances[depot_id] = 0.0
    queue: list[tuple[float, int]] = [(0.0, depot_id)]
    while queue:
        current_distance, current_node = heapq.heappop(queue)
        if current_distance != distances[current_node]:
            continue
        for neighbor_id, length, _ in graph[current_node]:
            next_distance = current_distance + length
            if next_distance < distances[neighbor_id]:
                distances[neighbor_id] = next_distance
                heapq.heappush(queue, (next_distance, neighbor_id))
    return {
        node_id: (distance, 0.0)
        for node_id, distance in distances.items()
        if distance < float("inf")
    }


def _resolve_speed_penalty(
    config: dict[str, Any],
    speed_penalty: float | None,
) -> float:
    """读取并校验速度惩罚参数。"""

    value = (
        float(config.get("risk", {}).get("speed_penalty", 0.5))
        if speed_penalty is None
        else float(speed_penalty)
    )
    if not 0.0 <= value <= 1.0:
        raise ValueError("speed_penalty 必须在 [0, 1] 范围内")
    return value


def _apply_speed_penalty(
    scenario: ScenarioInput,
    speed_penalty: float,
) -> ScenarioInput:
    """按 base_speed 和风险分数重算 adjusted_speed。"""

    roads = {}
    for road_id, road in scenario.roads.items():
        adjusted_speed = float(road["base_speed"]) * (
            1.0 - float(road["risk_score"]) * speed_penalty
        )
        if adjusted_speed <= 0:
            raise ValueError(f"道路 {road_id} 的速度惩罚后速度非正")
        roads[road_id] = {**road, "adjusted_speed": adjusted_speed}
    return replace(scenario, roads=roads)


def _traversed_edges(states: list[Any]) -> list[tuple[int, int]]:
    """按先后状态差分还原实际经过的无向边序列。"""

    traversed: list[tuple[int, int]] = []
    for previous_state, current_state in zip(states, states[1:]):
        previous_locations = {
            vehicle.id: vehicle.location for vehicle in previous_state.vehicles
        }
        for vehicle in current_state.vehicles:
            previous_location = previous_locations[vehicle.id]
            if vehicle.location == previous_location:
                continue
            traversed.append(
                (min(previous_location, vehicle.location), max(previous_location, vehicle.location))
            )
    return traversed


def _edge_metrics(
    states: list[Any],
    edge_attributes: dict[tuple[int, int], tuple[float, float, int]],
    high_risk_threshold: float,
) -> tuple[float, int, int]:
    """返回暴露风险、高风险不同道路数和高风险边经过次数。"""

    total_risk = 0.0
    high_risk_road_ids: set[int] = set()
    high_risk_edge_visits = 0
    for edge_key in _traversed_edges(states):
        attributes = edge_attributes.get(edge_key)
        if attributes is None:
            raise ValueError(f"轨迹经过未登记道路：{edge_key}")
        length, risk_score, road_id = attributes
        total_risk += length * risk_score
        if risk_score > high_risk_threshold:
            high_risk_road_ids.add(road_id)
            high_risk_edge_visits += 1
    return total_risk, len(high_risk_road_ids), high_risk_edge_visits


def _solve_loaded_scenario(
    scenario: ScenarioInput,
    config: dict[str, Any],
    risk_alpha: float | None = None,
) -> _SolveOutcome:
    """对已加载场景执行规划，供正常入口和消融实验复用。"""

    optimization_config = config["optimization"]
    resolved_risk_alpha = (
        float(optimization_config.get("risk_cost_alpha", 0.0))
        if risk_alpha is None
        else float(risk_alpha)
    )
    if resolved_risk_alpha < 0:
        raise ValueError("risk_alpha 不能为负数")

    graph, edge_attributes = _build_graph(scenario)
    edge_risk = {
        edge_key: attributes[1]
        for edge_key, attributes in edge_attributes.items()
    }
    heuristic_nodes = _build_heuristic_nodes(graph, scenario.depot_id)

    active_demands = {
        village_id: demand
        for village_id, demand in scenario.demands.items()
        if village_id not in scenario.isolated_villages
    }
    missing_nodes = set(active_demands) - set(heuristic_nodes)
    if missing_nodes:
        raise ValueError(f"非孤岛村庄与指挥中心不连通：{sorted(missing_nodes)}")

    active_priority_villages = (
        scenario.priority_villages - scenario.isolated_villages
    )
    vehicle_capacity = float(optimization_config["vehicle_capacity"])
    vehicle_time_limit = float(optimization_config["vehicle_time_limit"])
    num_vehicles = int(optimization_config["num_vehicles"])
    vehicle_properties = [
        (vehicle_capacity, vehicle_time_limit)
        for _ in range(num_vehicles)
    ]

    planner = MultiVehiclePlanner(
        nodes=heuristic_nodes,
        graph=graph,
        demands=active_demands,
        vehicle_props=vehicle_properties,
        priority_villages=active_priority_villages,
        heuristic_type=str(optimization_config["heuristic_type"]),
        depot_id=scenario.depot_id,
        max_states=int(optimization_config["max_states"]),
        time_limit=float(optimization_config["time_limit"]),
        edge_risk=edge_risk,
        risk_cost_alpha=resolved_risk_alpha,
    )

    solve_start = time.perf_counter()
    state_path, slowest_vehicle_time, states_explored = planner.a_star()
    solve_time = time.perf_counter() - solve_start
    if not state_path:
        raise RuntimeError("A* 未返回任何状态路径")

    final_state = state_path[-1]
    tolerance = 1e-6
    served_villages = sorted(
        village_id
        for village_id, demand in active_demands.items()
        if final_state.delivered.get(village_id, 0) >= demand - tolerance
    )
    priority_satisfied = sum(
        1
        for village_id in scenario.priority_villages
        if final_state.delivered.get(village_id, 0)
        >= active_demands.get(village_id, float("inf")) - tolerance
    )

    trajectories = state_sequence_to_trajectory(state_path)
    exposed_risk, high_risk_edge_count, high_risk_edge_visits = _edge_metrics(
        state_path,
        edge_attributes,
        float(config.get("risk", {}).get("high_risk_edge_threshold", 0.6)),
    )
    metrics: dict[str, int | float | str] = {
        "slowest_vehicle_time": float(slowest_vehicle_time),
        "total_distance": float(final_state.total_system_distance),
        "priority_villages_satisfied": int(priority_satisfied),
        "total_exposed_risk": float(exposed_risk),
        "high_risk_edge_visits": int(high_risk_edge_visits),
        "risk_cost_alpha": float(resolved_risk_alpha),
        "solve_time": float(solve_time),
        "states_explored": int(states_explored),
        "status": str(final_state.status),
    }
    solution = Solution(
        vehicle_trajectories=trajectories,
        served_villages=served_villages,
        isolated_villages=sorted(scenario.isolated_villages),
        metrics=metrics,
    )
    return _SolveOutcome(
        solution=solution,
        final_delivered=dict(final_state.delivered),
        high_risk_edge_count=high_risk_edge_count,
        high_risk_edge_visits=high_risk_edge_visits,
    )


def solve_scenario(
    scenario_json_path: str | Path,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    risk_alpha: float | None = None,
    speed_penalty: float | None = None,
) -> Solution:
    """加载场景、调用 A* 风格搜索并返回轨迹与指标。"""

    config_file = resolve_project_path(config_path)
    config = load_config(config_file)
    scenario = load_scenario(scenario_json_path, config_file)
    scenario = _apply_speed_penalty(
        scenario,
        _resolve_speed_penalty(config, speed_penalty),
    )
    return _solve_loaded_scenario(scenario, config, risk_alpha=risk_alpha).solution


def _load_hist_demands(
    scenario: ScenarioInput,
    config: dict[str, Any],
) -> dict[int, float]:
    """从 demand.csv 读取场景日期的 ``hist_demand_mean`` 作为朴素基线。"""

    data_path = resolve_project_path(config["paths"]["data_synthetic"]) / "demand.csv"
    if not data_path.is_file():
        raise FileNotFoundError(f"需求数据不存在：{data_path}")
    if not scenario.date:
        raise ValueError("场景缺少 date，无法读取历史均值需求")

    hist_demands: dict[int, float] = {}
    with data_path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if row["date"] != scenario.date:
                continue
            village_id = int(row["village_id"])
            if village_id in scenario.villages:
                hist_demands[village_id] = float(row["hist_demand_mean"])

    missing = set(scenario.villages) - set(hist_demands)
    if missing:
        raise ValueError(
            f"日期 {scenario.date} 缺少村庄历史均值需求：{sorted(missing)}"
        )
    return hist_demands


def _apply_ablation_mode(
    scenario: ScenarioInput,
    mode: AblationMode,
    hist_demands: dict[int, float] | None = None,
) -> ScenarioInput:
    """在不修改原始场景文件的前提下生成消融输入。"""

    if mode == "full":
        return scenario
    if mode == "fixed_demand":
        fixed_demands = {village_id: 100.0 for village_id in scenario.villages}
        return replace(scenario, demands=fixed_demands)
    if mode == "hist_demand":
        if hist_demands is None:
            raise ValueError("hist_demand 模式需要历史均值需求")
        return replace(scenario, demands=dict(hist_demands))
    if mode == "no_risk_adjust":
        roads = {
            road_id: {**road, "adjusted_speed": float(road["base_speed"])}
            for road_id, road in scenario.roads.items()
        }
        return replace(scenario, roads=roads)
    raise ValueError(f"未知消融模式：{mode}")


def solve_with_ablation(
    scenario_json_path: str | Path,
    mode: AblationMode,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    risk_alpha: float | None = None,
    speed_penalty: float | None = None,
) -> Solution:
    """按消融模式求解，并返回报告所需的指标。

    ``hist_demand`` 使用 demand.csv 的历史均值需求；``fixed_demand`` 把每村
    需求固定为 100 箱；``no_risk_adjust`` 用 ``base_speed`` 替换风险调整速度；
    ``full`` 使用完整 ML 输入。``risk_alpha`` 可覆盖配置中的风险代价权重。
    """

    config_file = resolve_project_path(config_path)
    config = load_config(config_file)
    scenario = load_scenario(scenario_json_path, config_file)
    scenario = _apply_speed_penalty(
        scenario,
        _resolve_speed_penalty(config, speed_penalty),
    )
    hist_demands = (
        _load_hist_demands(scenario, config) if mode == "hist_demand" else None
    )
    ablated_scenario = _apply_ablation_mode(scenario, mode, hist_demands)
    effective_risk_alpha = 0.0 if mode == "no_risk_adjust" else risk_alpha
    outcome = _solve_loaded_scenario(
        ablated_scenario,
        config,
        risk_alpha=effective_risk_alpha,
    )
    solution = outcome.solution

    total_demand = sum(ablated_scenario.demands.values())
    total_satisfied = sum(
        min(max(outcome.final_delivered.get(village_id, 0.0), 0.0), demand)
        for village_id, demand in ablated_scenario.demands.items()
    )
    unserved_demand_total = sum(
        max(demand - outcome.final_delivered.get(village_id, 0.0), 0.0)
        for village_id, demand in ablated_scenario.demands.items()
    )
    priority_demand = sum(
        ablated_scenario.demands.get(village_id, 0.0)
        for village_id in ablated_scenario.priority_villages
    )
    priority_satisfied_boxes = sum(
        min(
            max(outcome.final_delivered.get(village_id, 0.0), 0.0),
            ablated_scenario.demands.get(village_id, 0.0),
        )
        for village_id in ablated_scenario.priority_villages
    )

    metrics = {
        **solution.metrics,
        "priority_demand_satisfied_rate": (
            float(priority_satisfied_boxes / priority_demand) if priority_demand > 0 else 0.0
        ),
        "total_demand_satisfied_rate": (
            float(total_satisfied / total_demand) if total_demand > 0 else 0.0
        ),
        "unserved_demand_total": float(unserved_demand_total),
        "high_risk_edge_count": int(outcome.high_risk_edge_count),
        "high_risk_edge_visits": int(outcome.high_risk_edge_visits),
    }
    return Solution(
        vehicle_trajectories=solution.vehicle_trajectories,
        served_villages=solution.served_villages,
        isolated_villages=solution.isolated_villages,
        metrics=metrics,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="求解 bridge 场景的多车辆调度问题")
    parser.add_argument(
        "--scenario",
        type=Path,
        default=DEFAULT_SCENARIO_PATH,
        help="scenario JSON 路径",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="配置文件路径",
    )
    parser.add_argument(
        "--risk-alpha",
        type=float,
        default=None,
        help="风险代价权重；不传时使用 configs/default.yaml 中的值",
    )
    parser.add_argument(
        "--speed-penalty",
        type=float,
        default=None,
        help="速度惩罚系数；不传时使用 configs/default.yaml 中的值",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    solution = solve_scenario(
        arguments.scenario,
        arguments.config,
        risk_alpha=arguments.risk_alpha,
        speed_penalty=arguments.speed_penalty,
    )
    print(json.dumps(solution.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
