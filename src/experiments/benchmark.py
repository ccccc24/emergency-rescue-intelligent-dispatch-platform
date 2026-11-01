# -*- coding: utf-8 -*-
"""调度算法基准案例。

本文件迁移自根目录 ``团队作业.py`` 中的 ``run_test_case``、
``generate_large_scale_problem`` 和主程序测试逻辑，只调整 import 路径。
"""

from __future__ import annotations

import argparse
import time
from typing import Final

from src.optimization.heuristics import euclidean_distance
from src.optimization.planner import MultiVehiclePlanner


HEURISTIC_TYPES: Final[tuple[str, ...]] = (
    "basic",
    "distance_based",
    "priority_based",
)


def run_test_case(
    name: str,
    nodes: dict[int, tuple[float, float]],
    graph: dict[int, list[tuple[int, float, float]]],
    demands: dict[int, float],
    vehicle_props: list[tuple[float, float]],
    priority_villages: set[int] | None = None,
    heuristic_type: str = "distance_based",
) -> tuple[float, int, float]:
    """运行一个基准算例，返回耗时、探索状态数和解的最慢车辆时间。"""

    print(f"\n--- 测试案例: {name} ---")
    print(f"使用启发式函数: {heuristic_type}")
    if priority_villages:
        print(f"优先村庄: {priority_villages}")

    planner = MultiVehiclePlanner(
        nodes,
        graph,
        demands,
        vehicle_props,
        priority_villages,
        heuristic_type,
    )

    start_time = time.perf_counter()
    final_path, total_time, num_explored = planner.a_star()
    execution_time = time.perf_counter() - start_time

    if final_path:
        final_state = final_path[-1]
        total_dist_all_vehicles = sum(
            vehicle.total_distance_traveled for vehicle in final_state.vehicles
        )
        print("  🎉 解决方案找到!" if final_state.status == "success" else "  ⚠️ 返回部分解")
        print(f"    状态: {final_state.status}")
        print(f"    最慢车辆用时: {total_time:.2f}")
        print(f"    所有车辆总行驶距离: {total_dist_all_vehicles:.2f}")
        print(f"    共探索了 {num_explored} 个状态。")
        print(f"    路径长度 (状态数): {len(final_path)}")
        print(f"    运行时间: {execution_time:.4f} 秒")

        if priority_villages:
            priority_delivered = sum(
                final_state.delivered.get(village_id, 0)
                for village_id in priority_villages
            )
            priority_total = sum(
                demands.get(village_id, 0) for village_id in priority_villages
            )
            if priority_total > 0:
                print(
                    "    优先村庄配送情况: "
                    f"{priority_delivered}/{priority_total} "
                    f"({priority_delivered / priority_total * 100:.1f}%)"
                )
    else:
        print(f"  😥 未能找到解决方案。共探索了 {num_explored} 个状态。")
        print(f"    运行时间: {execution_time:.4f} 秒")

    return execution_time, num_explored, total_time if final_path else float("inf")


def generate_large_scale_problem() -> tuple[
    dict[int, tuple[float, float]],
    dict[int, list[tuple[int, float, float]]],
    dict[int, float],
    list[tuple[float, float]],
    set[int],
]:
    """返回原文件中的 35 节点大规模算例。"""

    nodes = {
        1: (0, 0), 2: (5, 149), 3: (118, 103), 4: (140, 132), 5: (102, 108),
        6: (90, 63), 7: (110, 140), 8: (18, 137), 9: (109, 130), 10: (100, 137),
        11: (142, 145), 12: (61, 12), 13: (40, 118), 14: (111, 10), 15: (147, 36),
        16: (123, 129), 17: (128, 81), 18: (13, 132), 19: (10, 80), 20: (70, 140),
        21: (28, 50), 22: (11, 138), 23: (111, 113), 24: (148, 108), 25: (93, 14),
        26: (149, 78), 27: (1, 104), 28: (102, 3), 29: (12, 13), 30: (81, 118),
        31: (110, 11), 32: (122, 117), 33: (136, 101), 34: (13, 101), 35: (141, 14),
    }

    graph = {
        1: [(2, 149.1, 6), (35, 141.0, 3), (29, 17.7, 7), (25, 94.0, 3), (21, 57.3, 7), (12, 62.2, 6)],
        2: [(1, 149.1, 6), (3, 120.0, 7), (8, 13.9, 7), (20, 65.2, 7), (22, 6.1, 6), (27, 45.1, 3)],
        3: [(2, 120.0, 7), (4, 30.4, 3), (5, 17.0, 5), (10, 37.6, 7), (16, 10.0, 6), (23, 7.1, 4)],
        4: [(3, 30.4, 3), (5, 41.8, 8), (7, 31.6, 5), (11, 13.3, 7), (16, 17.7, 5), (24, 24.7, 6)],
        5: [(3, 17.0, 5), (4, 41.8, 8), (6, 47.4, 8), (9, 23.0, 6), (17, 30.0, 7), (23, 10.2, 7)],
        6: [(5, 47.4, 8), (7, 54.1, 5), (12, 59.1, 7), (17, 38.0, 3), (19, 82.0, 4), (25, 49.2, 7)],
        7: [(4, 31.6, 5), (6, 54.1, 5), (8, 92.0, 5), (11, 32.0, 5), (16, 18.4, 6), (20, 40.0, 7)],
        8: [(2, 13.9, 7), (7, 92.0, 5), (9, 91.0, 5), (13, 28.6, 6), (18, 5.8, 5), (22, 7.1, 6)],
        9: [(5, 23.0, 6), (8, 91.0, 5), (10, 11.4, 6), (16, 13.0, 7), (23, 19.2, 5), (30, 30.5, 6)],
        10: [(3, 37.6, 7), (9, 11.4, 6), (11, 42.8, 7), (16, 26.4, 4), (20, 30.2, 3), (30, 24.2, 7)],
        11: [(4, 13.3, 7), (7, 32.0, 5), (10, 42.8, 7), (16, 19.9, 5), (20, 32.8, 7), (32, 33.9, 6)],
        12: [(1, 62.2, 6), (6, 59.1, 7), (14, 50.0, 6), (19, 53.1, 7), (21, 53.1, 6), (29, 49.0, 8)],
        13: [(8, 28.6, 6), (18, 27.0, 8), (19, 46.1, 5), (27, 40.8, 6), (30, 7.1, 6), (34, 27.9, 4)],
        14: [(12, 50.0, 6), (15, 43.2, 6), (17, 72.8, 5), (25, 18.4, 6), (28, 12.0, 7), (31, 0.0, 5)],
        15: [(14, 43.2, 6), (17, 50.1, 7), (26, 42.2, 4), (28, 51.1, 4), (31, 41.1, 8), (35, 23.3, 7)],
        16: [(3, 10.0, 6), (4, 17.7, 5), (7, 18.4, 6), (9, 13.0, 7), (10, 26.4, 4), (11, 19.9, 5)],
        17: [(5, 30.0, 7), (6, 38.0, 3), (14, 72.8, 5), (15, 50.1, 7), (23, 34.0, 3), (26, 21.2, 7)],
        18: [(8, 5.8, 5), (13, 27.0, 8), (19, 52.0, 7), (22, 7.8, 6), (27, 13.3, 6), (34, 31.0, 3)],
        19: [(6, 82.0, 4), (12, 53.1, 7), (13, 46.1, 5), (18, 52.0, 7), (21, 34.5, 5), (27, 26.0, 7)],
        20: [(2, 65.2, 7), (7, 40.0, 7), (10, 30.2, 3), (11, 32.8, 7), (22, 59.0, 7), (30, 30.5, 3)],
        21: [(1, 57.3, 7), (12, 53.1, 6), (19, 34.5, 5), (29, 39.8, 7), (34, 53.2, 8)],
        22: [(2, 6.1, 6), (8, 7.1, 6), (18, 7.8, 6), (20, 59.0, 7), (27, 10.0, 6)],
        23: [(3, 7.1, 4), (5, 10.2, 7), (9, 19.2, 5), (17, 34.0, 3), (30, 31.0, 8), (32, 11.0, 5)],
        24: [(4, 24.7, 6), (26, 43.2, 3), (32, 29.7, 7), (33, 14.6, 6), (35, 94.6, 5)],
        25: [(1, 94.0, 3), (6, 49.2, 7), (14, 18.4, 6), (28, 9.9, 3), (31, 18.0, 7)],
        26: [(15, 42.2, 4), (17, 21.2, 7), (24, 43.2, 3), (28, 76.2, 7), (33, 26.0, 7), (35, 20.0, 6)],
        27: [(2, 45.1, 3), (13, 40.8, 6), (18, 13.3, 6), (19, 26.0, 7), (22, 10.0, 6), (34, 12.0, 8)],
        28: [(1, 102.0, 6), (14, 12.0, 7), (15, 51.1, 4), (25, 9.9, 3), (26, 76.2, 7), (31, 7.1, 6)],
        29: [(1, 17.7, 7), (12, 49.0, 8), (21, 39.8, 7), (34, 88.1, 5)],
        30: [(9, 30.5, 6), (10, 24.2, 7), (13, 7.1, 6), (20, 30.5, 3), (23, 31.0, 8), (32, 42.0, 6)],
        31: [(14, 0.0, 5), (15, 41.1, 8), (25, 18.0, 7), (28, 7.1, 6), (35, 6.1, 6)],
        32: [(11, 33.9, 6), (23, 11.0, 5), (24, 29.7, 7), (30, 42.0, 6), (33, 18.4, 8)],
        33: [(24, 14.6, 6), (26, 26.0, 7), (32, 18.4, 8), (35, 87.4, 5)],
        34: [(13, 27.9, 4), (18, 31.0, 3), (21, 53.2, 8), (27, 12.0, 8), (29, 88.1, 5)],
        35: [(1, 141.0, 3), (15, 23.3, 7), (24, 94.6, 5), (26, 20.0, 6), (31, 6.1, 6), (33, 87.4, 5)],
    }
    demands = {18: 58, 20: 38, 22: 36, 34: 22, 27: 52, 8: 35, 21: 41}
    priority_villages = {20, 8, 18}
    vehicle_props = [(78, 153.4), (86, 139.2), (52, 229.5), (70, 169.8)]
    return nodes, graph, demands, vehicle_props, priority_villages


def build_small_problem() -> tuple[
    dict[int, tuple[float, float]],
    dict[int, list[tuple[int, float, float]]],
    dict[int, float],
    list[tuple[float, float]],
    set[int],
]:
    """构造原文件中的 5 节点完全图小规模算例。"""

    nodes = {
        1: (0, 0),
        2: (10, 10),
        3: (20, 5),
        4: (5, 15),
        5: (15, 15),
    }
    graph: dict[int, list[tuple[int, float, float]]] = {}
    for from_node in nodes:
        graph[from_node] = []
        for to_node in nodes:
            if from_node != to_node:
                distance = euclidean_distance(nodes[from_node], nodes[to_node])
                graph[from_node].append((to_node, distance, 5.0))
    return nodes, graph, {2: 8, 3: 10, 4: 6, 5: 12}, [(20, 100.0), (15, 100.0)], {2, 4}


def run_benchmark(case: str = "all") -> None:
    """运行小规模和/或大规模算例的三种启发式对比。"""

    cases = []
    if case in {"all", "small"}:
        cases.append(("小规模问题", build_small_problem()))
    if case in {"all", "large"}:
        cases.append(("大规模问题", generate_large_scale_problem()))

    for case_name, problem in cases:
        nodes, graph, demands, vehicle_props, priority_villages = problem
        results: dict[str, dict[str, float]] = {}
        print(f"\n=== {case_name} ===")
        for heuristic_type in HEURISTIC_TYPES:
            execution_time, states_explored, solution_quality = run_test_case(
                f"{case_name} - {heuristic_type}",
                nodes,
                graph,
                demands,
                vehicle_props,
                priority_villages,
                heuristic_type,
            )
            results[heuristic_type] = {
                "执行时间": execution_time,
                "探索状态数": float(states_explored),
                "解决方案质量": solution_quality,
            }

        print("\n=== 性能比较 ===")
        print(f"{'启发式函数':<20} {'执行时间(秒)':<15} {'探索状态数':<15} {'解决方案质量':<15}")
        print("-" * 65)
        for heuristic_type, metrics in results.items():
            quality = metrics["解决方案质量"]
            quality_text = f"{quality:.2f}" if quality != float("inf") else "无解"
            print(
                f"{heuristic_type:<20} {metrics['执行时间']:<15.4f} "
                f"{int(metrics['探索状态数']):<15d} {quality_text:<15}"
            )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行调度算法基准案例")
    parser.add_argument(
        "--case",
        choices=("all", "small", "large"),
        default="all",
        help="选择小规模、大规模或全部算例",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    run_benchmark(arguments.case)


if __name__ == "__main__":
    main()
