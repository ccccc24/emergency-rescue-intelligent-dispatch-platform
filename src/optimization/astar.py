# -*- coding: utf-8 -*-
"""带状态数和墙钟上限的 A* 风格多车启发式搜索。"""

from __future__ import annotations

import heapq
import time
from typing import TYPE_CHECKING, Final

from .models import State

if TYPE_CHECKING:
    from .planner import MultiVehiclePlanner


SUCCESS: Final[str] = "success"
TIMEOUT: Final[str] = "timeout"
LIMIT_REACHED: Final[str] = "limit_reached"


class PriorityQueue:
    """基于 ``heapq`` 的最小优先队列。"""

    def __init__(self) -> None:
        self.items: list[tuple[float, State]] = []

    def enqueue(self, item: State, priority: float) -> None:
        heapq.heappush(self.items, (priority, item))

    def dequeue(self) -> State:
        if self.is_empty():
            raise IndexError("无法从空优先队列出队")
        return heapq.heappop(self.items)[1]

    def is_empty(self) -> bool:
        return len(self.items) == 0


def _reconstruct_path(
    state: State,
    came_from: dict[State, State | None],
) -> list[State]:
    """从父指针恢复从初始状态到目标状态的序列。"""

    path: list[State] = []
    current: State | None = state
    while current is not None:
        path.append(current)
        current = came_from[current]
    return path[::-1]


def _is_better_partial(candidate: State, current_best: State) -> bool:
    """比较部分解：优先已配送总量，其次最慢车辆时间。"""

    candidate_delivered = sum(candidate.delivered.values())
    best_delivered = sum(current_best.delivered.values())
    if candidate_delivered != best_delivered:
        return candidate_delivered > best_delivered
    return candidate.max_time < current_best.max_time


def a_star(planner: MultiVehiclePlanner) -> tuple[list[State] | None, float, int]:
    """执行 A* 风格搜索。

    返回值保持原接口 ``(path, total_time, explored_count)``。达到状态数或时间
    上限时返回当前已配送量最高的部分路径，并分别在最终 ``State.status`` 和
    ``planner.last_status`` 中写入 ``limit_reached`` 或 ``timeout``。

    g 值采用 ``最慢车辆时间 + 累计风险代价``。累计风险代价由规划器按
    ``alpha * risk * length`` 写入 ``State.risk_cost``；``alpha=0`` 时该项为 0，
    搜索行为与纯时间目标完全一致。
    """

    queue = PriorityQueue()
    start_state = planner.initial_state()
    start_priority = (
        start_state.max_time + start_state.risk_cost + planner.heuristic(start_state)
    )
    queue.enqueue(start_state, start_priority)

    came_from: dict[State, State | None] = {start_state: None}
    g_score: dict[State, float] = {start_state: 0.0}
    explored_count = 0
    best_state = start_state
    start_clock = time.perf_counter()

    while not queue.is_empty():
        if explored_count >= planner.max_states:
            best_state.status = LIMIT_REACHED
            planner.last_status = LIMIT_REACHED
            return _reconstruct_path(best_state, came_from), best_state.max_time, explored_count

        if time.perf_counter() - start_clock >= planner.time_limit:
            best_state.status = TIMEOUT
            planner.last_status = TIMEOUT
            return _reconstruct_path(best_state, came_from), best_state.max_time, explored_count

        current_state = queue.dequeue()
        explored_count += 1

        if (
            current_state.max_time + current_state.risk_cost
            > g_score.get(current_state, float("inf"))
        ):
            continue

        if _is_better_partial(current_state, best_state):
            best_state = current_state

        if planner.is_goal(current_state):
            current_state.status = SUCCESS
            planner.last_status = SUCCESS
            return _reconstruct_path(current_state, came_from), current_state.max_time, explored_count

        for next_state in planner.successors(current_state):
            tentative_g = next_state.max_time + next_state.risk_cost
            if tentative_g < g_score.get(next_state, float("inf")):
                g_score[next_state] = tentative_g
                came_from[next_state] = current_state
                priority = tentative_g + planner.heuristic(next_state)
                queue.enqueue(next_state, priority)
                if _is_better_partial(next_state, best_state):
                    best_state = next_state

    best_state.status = LIMIT_REACHED
    planner.last_status = LIMIT_REACHED
    return _reconstruct_path(best_state, came_from), best_state.max_time, explored_count
