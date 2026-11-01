# -*- coding: utf-8 -*-
"""把 A* 状态序列转换为前端可播放的逐车轨迹。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .models import State


def state_sequence_to_trajectory(
    states: Sequence[State],
) -> dict[str, list[dict[str, int | float]]]:
    """把状态序列转换为 ``{vehicle_1: [{time, location, load, remaining_capacity}]}``。

    ``load`` 是车辆当前随车物资量，``remaining_capacity`` 是相对初始满载容量的
    剩余装载空间。初始状态会为每辆车写入 ``time=0`` 的起点记录。
    """

    if not states:
        return {}

    initial_state = states[0]
    initial_capacity = {
        vehicle.id: float(vehicle.capacity) for vehicle in initial_state.vehicles
    }
    trajectories: dict[str, list[dict[str, int | float]]] = {}

    def append_point(vehicle_id: int, location: int, time_value: float, load: float) -> None:
        key = f"vehicle_{vehicle_id}"
        trajectories.setdefault(key, []).append(
            {
                "time": round(float(time_value), 6),
                "location": int(location),
                "load": round(float(load), 6),
                "remaining_capacity": round(
                    float(initial_capacity[vehicle_id] - load), 6
                ),
            }
        )

    for vehicle in initial_state.vehicles:
        append_point(vehicle.id, vehicle.location, vehicle.time, vehicle.capacity)

    previous_locations = {
        vehicle.id: vehicle.location for vehicle in initial_state.vehicles
    }
    for current_state in states[1:]:
        for vehicle in current_state.vehicles:
            if vehicle.location != previous_locations.get(vehicle.id):
                append_point(vehicle.id, vehicle.location, vehicle.time, vehicle.capacity)
                previous_locations[vehicle.id] = vehicle.location

    return trajectories
