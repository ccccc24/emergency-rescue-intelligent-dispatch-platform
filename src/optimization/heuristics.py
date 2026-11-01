# -*- coding: utf-8 -*-
"""多车辆 A* 风格搜索使用的三个启发式函数。"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, Protocol

if TYPE_CHECKING:
    from .models import State
    from .planner import MultiVehiclePlanner


class HeuristicContext(Protocol):
    """启发式函数需要读取的规划器字段。"""

    nodes: dict[int, tuple[float, float]]
    graph: dict[int, list[tuple[int, float, float]]]
    demands: dict[int, float]
    vehicle_properties: list[tuple[float, float]]
    priority_villages: set[int]
    min_speed: float
    depot_id: int


def euclidean_distance(
    p1_coords: tuple[float, float] | None,
    p2_coords: tuple[float, float] | None,
) -> float:
    """计算两点之间的欧氏距离；任一坐标缺失时返回正无穷。"""

    if p1_coords is None or p2_coords is None:
        return float("inf")
    return math.sqrt(
        (p1_coords[0] - p2_coords[0]) ** 2
        + (p1_coords[1] - p2_coords[1]) ** 2
    )


def basic(planner: HeuristicContext, current_state: State) -> float:
    """基础启发式：按最大单载容量估算剩余满载行程数量。

    该函数只使用剩余需求、车辆容量和到指挥中心的最远距离，原实现保持不变。
    """

    total_remaining_demand = 0.0
    needy_village_ids: list[int] = []
    for village_id, required_demand in planner.demands.items():
        remaining = required_demand - current_state.delivered.get(village_id, 0)
        if remaining > 0:
            total_remaining_demand += remaining
            needy_village_ids.append(village_id)
    if total_remaining_demand == 0:
        return 0.0

    max_one_vehicle_capacity = 1.0
    if planner.vehicle_properties:
        max_one_vehicle_capacity = max(
            capacity for capacity, _ in planner.vehicle_properties
        )
    if max_one_vehicle_capacity <= 0:
        max_one_vehicle_capacity = 1.0

    num_full_loads_needed = math.ceil(
        total_remaining_demand / max_one_vehicle_capacity
    )

    max_dist_to_needy_village = 0.0
    depot_coords = planner.nodes.get(planner.depot_id)
    if depot_coords and needy_village_ids:
        current_max_dist = 0.0
        for village_id in needy_village_ids:
            village_coords = planner.nodes.get(village_id)
            distance = euclidean_distance(depot_coords, village_coords)
            if distance != float("inf") and distance > current_max_dist:
                current_max_dist = distance
        max_dist_to_needy_village = current_max_dist
    elif not needy_village_ids:
        return 0.0
    else:
        all_distances_in_graph = [
            distance
            for edges in planner.graph.values()
            for _, distance, _ in edges
            if distance > 0
        ]
        max_dist_to_needy_village = (
            max(all_distances_in_graph) if all_distances_in_graph else 10.0
        )

    if max_dist_to_needy_village == 0 and total_remaining_demand > 0:
        return total_remaining_demand / (
            max_one_vehicle_capacity * planner.min_speed + 1e-6
        )

    time_for_one_furthest_trip = max_dist_to_needy_village / planner.min_speed
    h_value = num_full_loads_needed * time_for_one_furthest_trip
    if planner.vehicle_properties:
        h_value /= len(planner.vehicle_properties)
    return h_value


def distance_based(planner: HeuristicContext, current_state: State) -> float:
    """距离启发式：在基础启发式上加入剩余需求所需的最少往返时间。

    该版本比 ``basic`` 更充分利用需求总量和车队总容量，原实现保持不变。
    """

    base_heuristic = basic(planner, current_state)

    total_remaining_demand = 0.0
    for village_id, required_demand in planner.demands.items():
        remaining = required_demand - current_state.delivered.get(village_id, 0)
        if remaining > 0:
            total_remaining_demand += remaining

    if total_remaining_demand == 0:
        return 0.0

    total_max_capacity = sum(
        capacity for capacity, _ in planner.vehicle_properties
    )
    if total_max_capacity > 0:
        min_trips_needed = math.ceil(total_remaining_demand / total_max_capacity)
    else:
        min_trips_needed = float("inf")

    needy_villages_coords: dict[int, tuple[float, float] | None] = {}
    for village_id, required_demand in planner.demands.items():
        remaining = required_demand - current_state.delivered.get(village_id, 0)
        if remaining > 0:
            needy_villages_coords[village_id] = planner.nodes.get(village_id)

    max_dist_to_village = 0.0
    resupply_coord = planner.nodes.get(planner.depot_id)
    if resupply_coord:
        for village_coord in needy_villages_coords.values():
            distance = euclidean_distance(resupply_coord, village_coord)
            if distance > max_dist_to_village:
                max_dist_to_village = distance

    avg_trip_time_lower_bound = (2 * max_dist_to_village) / planner.min_speed
    return base_heuristic + min_trips_needed * avg_trip_time_lower_bound


def priority_based(planner: HeuristicContext, current_state: State) -> float:
    """优先村启发式：在距离启发式上增加未满足优先需求的惩罚。

    该启发式会放大优先村权重，因此不代表可采纳下界；它属于 A* 风格的
    多车启发式搜索，不保证全局最优。
    """

    base_heuristic = distance_based(planner, current_state)
    priority_penalty = 0.0

    has_needy_priority_villages = any(
        (planner.demands.get(village_id, 0) - current_state.delivered.get(village_id, 0))
        > 0
        and village_id in planner.priority_villages
        for village_id in planner.demands
    )
    if has_needy_priority_villages:
        total_priority_demand = sum(
            planner.demands.get(village_id, 0)
            - current_state.delivered.get(village_id, 0)
            for village_id in planner.priority_villages
        )
        priority_penalty = total_priority_demand * 0.1

    return base_heuristic + priority_penalty


HeuristicFunction = Callable[[HeuristicContext, "State"], float]
HEURISTIC_FUNCTIONS: dict[str, HeuristicFunction] = {
    "basic": basic,
    "distance_based": distance_based,
    "priority_based": priority_based,
}
