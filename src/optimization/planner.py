# -*- coding: utf-8 -*-
"""多车辆调度状态空间、状态转移和启发式调度器。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Final

from src.ml.data_gen import load_config

from .heuristics import (
    HEURISTIC_FUNCTIONS,
    basic,
    distance_based,
    priority_based,
)
from .models import State, Vehicle


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH: Final[Path] = PROJECT_ROOT / "configs" / "default.yaml"


class MultiVehiclePlanner:
    """多车辆配送规划器。

    该类迁移自 ``团队作业.py``，保留原有的状态转移、目标判定和优先村比较
    语义；同时把指挥中心、状态数上限、墙钟时间上限和风险代价权重参数化。
    风险代价按每次经过的 ``alpha * risk * length`` 累加到状态上，供 A* 的
    g 值使用；``alpha=0`` 时完全退化为原版最慢车辆时间目标。
    """

    def __init__(
        self,
        nodes: dict[int, tuple[float, float]],
        graph: dict[int, list[tuple[int, float, float]]],
        demands: dict[int, float],
        vehicle_props: list[tuple[float, float]],
        priority_villages: set[int] | None = None,
        heuristic_type: str = "basic",
        depot_id: int | None = None,
        max_states: int | None = None,
        time_limit: float | None = None,
        edge_risk: dict[tuple[int, int], float] | None = None,
        risk_cost_alpha: float = 0.0,
        config_path: str | Path | None = None,
    ) -> None:
        self.nodes = nodes
        self.graph = graph
        self.demands = demands
        self.vehicle_properties = vehicle_props
        self.priority_villages = set(priority_villages or set())
        self.heuristic_type = heuristic_type
        self.depot_id = depot_id if depot_id is not None else (0 if 0 in nodes else 1)
        self.last_status = "success"
        self.edge_risk = {
            (min(int(from_node), int(to_node)), max(int(from_node), int(to_node))): float(risk)
            for (from_node, to_node), risk in (edge_risk or {}).items()
        }
        self.risk_cost_alpha = float(risk_cost_alpha)
        if self.risk_cost_alpha < 0:
            raise ValueError("risk_cost_alpha 不能为负数")

        if max_states is None or time_limit is None:
            resolved_config_path = (
                Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
            )
            config = load_config(resolved_config_path)
            optimization_config = config["optimization"]
            if max_states is None:
                max_states = int(optimization_config["max_states"])
            if time_limit is None:
                time_limit = float(optimization_config["time_limit"])
        self.max_states = int(max_states)
        self.time_limit = float(time_limit)
        if self.max_states <= 0:
            raise ValueError("max_states 必须为正整数")
        if self.time_limit <= 0:
            raise ValueError("time_limit 必须为正数")
        if self.depot_id not in self.graph:
            raise ValueError(f"指挥中心节点 {self.depot_id} 不在图中")

        if self.graph:
            all_speeds = [
                speed
                for edges in self.graph.values()
                for _, _, speed in edges
                if speed > 0
            ]
            self.min_speed = min(all_speeds) if all_speeds else 1.0
        else:
            self.min_speed = 1.0

    def initial_state(self) -> State:
        initial_vehicles = []
        for vehicle_id, (capacity, max_time_limit) in enumerate(
            self.vehicle_properties, start=1
        ):
            initial_vehicles.append(
                Vehicle(
                    id=vehicle_id,
                    location=self.depot_id,
                    capacity=capacity,
                    initial_max_individual_time=max_time_limit,
                    time=0,
                    distance_traveled=0,
                )
            )

        initial_delivered = {village_id: 0 for village_id in self.demands}
        return State(
            initial_vehicles,
            initial_delivered,
            0,
            self.priority_villages,
        )

    def is_goal(self, current_state: State) -> bool:
        for village_id, required_demand in self.demands.items():
            if current_state.delivered.get(village_id, 0) < required_demand:
                return False
        return True

    def successors(self, current_state: State) -> list[State]:
        possible_next_states: list[State] = []
        for vehicle_index, current_vehicle in enumerate(current_state.vehicles):
            if current_vehicle.location not in self.graph:
                continue

            for neighbor_id, distance, speed in self.graph[current_vehicle.location]:
                if speed <= 0:
                    continue

                new_vehicles_list = [
                    Vehicle(
                        id=vehicle.id,
                        location=vehicle.location,
                        capacity=vehicle.capacity,
                        initial_max_individual_time=vehicle.max_individual_time_limit,
                        time=vehicle.time,
                        distance_traveled=vehicle.total_distance_traveled,
                    )
                    for vehicle in current_state.vehicles
                ]

                vehicle_to_move = new_vehicles_list[vehicle_index]
                travel_time = distance / speed
                if (
                    vehicle_to_move.time + travel_time
                    > vehicle_to_move.max_individual_time_limit
                ):
                    continue

                vehicle_to_move.location = neighbor_id
                vehicle_to_move.time += travel_time
                vehicle_to_move.total_distance_traveled += distance

                edge_key = (
                    min(current_vehicle.location, neighbor_id),
                    max(current_vehicle.location, neighbor_id),
                )
                edge_risk = self.edge_risk.get(edge_key, 0.0)
                new_risk_cost = (
                    current_state.risk_cost
                    + self.risk_cost_alpha * edge_risk * distance
                )

                new_delivered_dict = current_state.delivered.copy()
                if neighbor_id == self.depot_id:
                    original_capacity, _ = self.vehicle_properties[
                        vehicle_to_move.id - 1
                    ]
                    vehicle_to_move.capacity = float(original_capacity)
                elif neighbor_id in self.demands and vehicle_to_move.capacity > 0:
                    demand_at_target = self.demands[neighbor_id]
                    delivered_so_far = new_delivered_dict.get(neighbor_id, 0)
                    remaining_demand = demand_at_target - delivered_so_far
                    if remaining_demand > 0:
                        can_deliver = min(
                            vehicle_to_move.capacity, remaining_demand
                        )
                        vehicle_to_move.capacity -= can_deliver
                        new_delivered_dict[neighbor_id] += can_deliver

                new_max_overall_time = max(
                    vehicle.time for vehicle in new_vehicles_list
                )
                possible_next_states.append(
                    State(
                        new_vehicles_list,
                        new_delivered_dict,
                        new_max_overall_time,
                        self.priority_villages,
                        risk_cost=new_risk_cost,
                    )
                )

        return possible_next_states

    def heuristic(self, current_state: State) -> float:
        heuristic_function = HEURISTIC_FUNCTIONS.get(self.heuristic_type, basic)
        return heuristic_function(self, current_state)

    def basic_heuristic(self, current_state: State) -> float:
        """兼容原方法接口的基础启发式。"""

        return basic(self, current_state)

    def distance_based_heuristic(self, current_state: State) -> float:
        """兼容原方法接口的距离启发式。"""

        return distance_based(self, current_state)

    def priority_based_heuristic(self, current_state: State) -> float:
        """兼容原方法接口的优先村启发式。"""

        return priority_based(self, current_state)

    def a_star(self) -> tuple[list[State] | None, float, int]:
        """调用独立 A* 模块，保持原三返回值接口。"""

        from .astar import a_star

        return a_star(self)
