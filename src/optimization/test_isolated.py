# -*- coding: utf-8 -*-
"""孤岛村处理的最小闭环测试。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .run import solve_scenario


def build_isolated_scenario() -> dict[str, Any]:
    """构造北崖村两条连边均为高风险、其他村庄正常的最小场景。"""

    villages: dict[str, dict[str, Any]] = {}
    roads: dict[str, dict[str, Any]] = {}
    risk_scores: dict[str, float] = {}
    adjusted_speed: dict[str, float] = {}
    demands: dict[str, float] = {}

    demands_by_village = {1: 5, 2: 6, 3: 5, 4: 6, 5: 5, 6: 9}
    for village_id in range(1, 7):
        villages[str(village_id)] = {
            "id": village_id,
            "name": f"村庄{village_id}",
            "population": 1000,
            "disaster_level": 3,
            "is_priority": 0,
        }
        demands[str(village_id)] = demands_by_village[village_id]

    road_specs = [
        (1, 0, 1, 10.0, 40.0, 0.10),
        (2, 0, 2, 10.0, 40.0, 0.10),
        (3, 0, 3, 10.0, 40.0, 0.10),
        (4, 0, 4, 10.0, 40.0, 0.10),
        (5, 0, 5, 10.0, 40.0, 0.10),
        (6, 0, 6, 10.0, 40.0, 0.80),
        (7, 4, 6, 10.0, 40.0, 0.80),
    ]
    for road_id, from_node, to_node, length, base_speed, risk_score in road_specs:
        roads[str(road_id)] = {
            "id": road_id,
            "road_id": f"R{road_id:02d}",
            "from": from_node,
            "to": to_node,
            "length": length,
            "base_speed": base_speed,
            "terrain": "山区" if to_node == 6 else "平原",
        }
        risk_scores[str(road_id)] = risk_score
        adjusted_speed[str(road_id)] = base_speed * (1 - risk_score * 0.5)

    return {
        "date": "2099-01-01",
        "villages": villages,
        "roads": roads,
        "demands": demands,
        "risk_scores": risk_scores,
        "priority_villages": [],
        "adjusted_speed": adjusted_speed,
        # 故意留空，验证加载器会根据 0.75 阈值自动识别北崖村为孤岛。
        "isolated_villages": [],
        "metadata": {},
    }


def test_isolated_village_is_excluded() -> None:
    """高风险孤岛村不进入需求图，其余 5 村仍能完成配送。"""

    scenario = build_isolated_scenario()
    output_root = Path(__file__).resolve().parents[2] / "outputs"
    scenario_path = output_root / "scenario_test_isolated.json"
    try:
        scenario_path.write_text(
            json.dumps(scenario, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        solution = solve_scenario(scenario_path, "configs/default.yaml")
    finally:
        scenario_path.unlink(missing_ok=True)

    assert 6 in solution.isolated_villages, solution.isolated_villages
    trajectory_locations = {
        int(point["location"])
        for points in solution.vehicle_trajectories.values()
        for point in points
    }
    assert 6 not in trajectory_locations, trajectory_locations
    assert set(solution.served_villages) == {1, 2, 3, 4, 5}, solution.served_villages
    assert solution.metrics["status"] == "success", solution.metrics
    print("孤岛村测试通过：isolated=", solution.isolated_villages)
    print("实际配送村庄：", solution.served_villages)
    print("轨迹出现节点：", sorted(trajectory_locations))


def main() -> None:
    test_isolated_village_is_excluded()


if __name__ == "__main__":
    main()
