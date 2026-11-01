# -*- coding: utf-8 -*-
"""多车辆调度中的基础数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(init=False)
class Vehicle:
    """单车状态。

    ``capacity`` 表示车辆当前随车携带的物资量；车辆回到指挥中心后会恢复为
    初始容量。字段名称和构造参数保持与原 ``团队作业.py`` 兼容。
    """

    id: int
    location: int
    capacity: float
    max_individual_time_limit: float
    time: float
    total_distance_traveled: float

    def __init__(
        self,
        id: int,
        location: int,
        capacity: float,
        initial_max_individual_time: float,
        time: float = 0,
        distance_traveled: float = 0,
    ) -> None:
        self.id = int(id)
        self.location = int(location)
        self.capacity = float(capacity)
        self.max_individual_time_limit = float(initial_max_individual_time)
        self.time = float(time)
        self.total_distance_traveled = float(distance_traveled)

    def __repr__(self) -> str:
        return (
            f"Vehicle(id={self.id}, loc={self.location}, cap={self.capacity}, "
            f"time={self.time:.2f}, dist_travel={self.total_distance_traveled:.2f}, "
            f"time_limit={self.max_individual_time_limit:.2f})"
        )


@dataclass(eq=False)
class State:
    """多车辆搜索状态。

    ``status`` 是本次重构新增字段，用于标记最终状态是正常完成、超时还是达到
    状态上限；它不参与状态相等性和哈希，避免同一路径因标记不同而重复扩展。
    ``risk_cost`` 是已走路径累计的风险附加代价，只作为 g 值的一部分，不参与
    状态相等性与哈希；同一车辆/配送状态下保留代价更低的路径。
    """

    vehicles: list[Vehicle]
    delivered: dict[int, float]
    max_time: float
    priority_villages: set[int] = field(default_factory=set)
    risk_cost: float = 0.0
    total_system_distance: float = field(init=False)
    status: str = field(default="success", compare=False)

    def __post_init__(self) -> None:
        self.vehicles = sorted(self.vehicles, key=lambda vehicle: vehicle.id)
        self.priority_villages = set(self.priority_villages)
        self.total_system_distance = sum(
            vehicle.total_distance_traveled for vehicle in self.vehicles
        )

    def __hash__(self) -> int:
        vehicle_details = tuple(
            (
                vehicle.location,
                vehicle.capacity,
                round(vehicle.time, 5),
                round(vehicle.total_distance_traveled, 5),
            )
            for vehicle in self.vehicles
        )
        delivered_details = tuple(sorted(self.delivered.items()))
        return hash(
            (
                vehicle_details,
                delivered_details,
                round(self.max_time, 5),
                tuple(sorted(self.priority_villages)),
            )
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, State):
            return False
        if len(self.vehicles) != len(other.vehicles):
            return False

        vehicles_equal = all(
            self_vehicle.id == other_vehicle.id
            and self_vehicle.location == other_vehicle.location
            and self_vehicle.capacity == other_vehicle.capacity
            and round(self_vehicle.time, 5) == round(other_vehicle.time, 5)
            and round(self_vehicle.total_distance_traveled, 5)
            == round(other_vehicle.total_distance_traveled, 5)
            for self_vehicle, other_vehicle in zip(self.vehicles, other.vehicles)
        )
        return (
            vehicles_equal
            and self.delivered == other.delivered
            and round(self.max_time, 5) == round(other.max_time, 5)
            and self.priority_villages == other.priority_villages
        )

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, State):
            return NotImplemented

        self_priority_delivered = sum(
            self.delivered.get(village_id, 0) for village_id in self.priority_villages
        )
        other_priority_delivered = sum(
            other.delivered.get(village_id, 0) for village_id in other.priority_villages
        )
        if self_priority_delivered != other_priority_delivered:
            return self_priority_delivered > other_priority_delivered

        if round(self.max_time, 5) != round(other.max_time, 5):
            return round(self.max_time, 5) < round(other.max_time, 5)

        if round(self.risk_cost, 5) != round(other.risk_cost, 5):
            return round(self.risk_cost, 5) < round(other.risk_cost, 5)

        if round(self.total_system_distance, 5) != round(
            other.total_system_distance, 5
        ):
            return round(self.total_system_distance, 5) < round(
                other.total_system_distance, 5
            )

        self_total_delivered = sum(self.delivered.values())
        other_total_delivered = sum(other.delivered.values())
        if self_total_delivered != other_total_delivered:
            return self_total_delivered > other_total_delivered
        return False

    def __repr__(self) -> str:
        return (
            f"State(max_time={self.max_time:.2f}, "
            f"risk_cost={self.risk_cost:.2f}, "
            f"total_dist={self.total_system_distance:.2f}, "
            f"delivered={self.delivered}, priority={self.priority_villages}, "
            f"vehicles={self.vehicles}, status={self.status})"
        )
