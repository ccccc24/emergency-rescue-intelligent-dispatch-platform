import heapq
import math
import time
import random
import requests
import json
import os
from typing import List, Dict, Tuple, Set, Any

# 从原始文件导入必要的类和函数
from disaster_relief_enhanced import (
    PriorityQueue, euclidean_distance, Vehicle, State,
    MultiVehiclePlanner, run_test_case, generate_large_scale_problem
)


# -----------------------------------------------------------------------------
# LLM 评估状态相关函数
# -----------------------------------------------------------------------------

class LLMStateEvaluator:
    """使用LLM评估状态的类"""

    def __init__(self, api_key: str, model_name: str = "qwen-max"):
        """
        初始化LLM状态评估器
        
        参数:
            api_key: 阿里云百炼API密钥
            model_name: 使用的模型名称，默认为qwen-max
        """
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def format_state_for_prompt(self, state: State, problem_context: Dict) -> str:
        """
        将状态格式化为提示词
        
        参数:
            state: 需要评估的状态
            problem_context: 问题上下文信息，包括节点、图、需求等
        
        返回:
            格式化后的提示词
        """
        nodes = problem_context.get("nodes", {})
        demands = problem_context.get("demands", {})
        priority_villages = problem_context.get("priority_villages", set())

        # 构建状态描述
        state_desc = []

        # 1. 总体状态信息
        state_desc.append("## 当前状态总览")
        state_desc.append(f"- 最慢车辆用时: {state.max_time:.2f}")
        state_desc.append(f"- 总行驶距离: {state.total_system_distance:.2f}")

        # 2. 配送情况
        state_desc.append("\n## 配送情况")
        total_demand = sum(demands.values())
        total_delivered = sum(state.delivered.values())
        state_desc.append(
            f"- 总配送进度: {total_delivered}/{total_demand} ({total_delivered / total_demand * 100:.1f}%)")

        # 优先村庄配送情况
        if priority_villages:
            priority_total = sum(demands.get(v, 0) for v in priority_villages)
            priority_delivered = sum(state.delivered.get(v, 0) for v in priority_villages)
            state_desc.append(
                f"- 优先村庄配送进度: {priority_delivered}/{priority_total} ({priority_delivered / priority_total * 100:.1f}%)")

        # 详细村庄配送情况
        state_desc.append("\n### 各村庄配送详情:")
        for village_id, required in demands.items():
            delivered = state.delivered.get(village_id, 0)
            priority_mark = "🌟" if village_id in priority_villages else ""
            state_desc.append(
                f"- 村庄{village_id}{priority_mark}: {delivered}/{required} ({delivered / required * 100:.1f}%)")

        # 3. 车辆情况
        state_desc.append("\n## 车辆情况")
        for vehicle in state.vehicles:
            location_type = "指挥中心" if vehicle.location == 1 else f"村庄{vehicle.location}"
            state_desc.append(f"- 车辆{vehicle.id}: 位置={location_type}, 剩余容量={vehicle.capacity}, " +
                              f"已行驶时间={vehicle.time:.2f}/{vehicle.max_individual_time_limit:.2f}, " +
                              f"已行驶距离={vehicle.total_distance_traveled:.2f}")

        return "\n".join(state_desc)

    def evaluate_state(self, state: State, problem_context: Dict) -> float:
        """
        使用LLM评估状态，返回评分
        
        参数:
            state: 需要评估的状态
            problem_context: 问题上下文信息
            
        返回:
            状态评分 (0-100)
        """
        # 格式化状态为提示词
        state_description = self.format_state_for_prompt(state, problem_context)

        # 构建完整提示词
        prompt = f"""你是一个灾难救援路径规划专家，请评估以下救援状态的质量。

问题背景：
- 有多辆车从指挥中心出发，为多个村庄配送救援物资
- 每个村庄有特定的物资需求量
- 部分村庄被标记为优先村庄(🌟)，需要优先考虑
- 车辆有容量限制和时间限制
- 目标是在满足所有村庄需求的前提下，最小化最慢车辆的用时

请根据以下状态信息，从0-100给出一个评分，100分表示最优状态：

{state_description}

评分时请综合考虑以下因素：
1. 优先村庄的配送完成度（最重要）
2. 整体配送完成度
3. 最慢车辆用时（越短越好）
4. 总行驶距离（越短越好）
5. 车辆资源利用率

请直接返回一个0-100之间的数字作为评分，不要有任何其他文字。"""

        # 调用API
        payload = {
            "model": self.model_name,
            "input": {
                "messages": [
                    {"role": "system", "content": "你是一个灾难救援路径规划专家，擅长评估救援状态的质量。"},
                    {"role": "user", "content": prompt}
                ]
            },
            "parameters": {}
        }

        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()

            # 解析响应
            result = response.json()
            score_text = result["output"]["text"].strip()

            # 提取数字
            try:
                score = float(score_text)
                # 确保分数在0-100之间
                score = max(0, min(100, score))
                return score
            except ValueError:
                print(f"无法解析LLM返回的评分: {score_text}")
                return 50.0  # 默认中等分数

        except Exception as e:
            print(f"调用LLM API时出错: {str(e)}")
            return 50.0  # 默认中等分数


class LLMMultiVehiclePlanner(MultiVehiclePlanner):
    """使用LLM评估状态的多车辆规划器"""

    def __init__(self, nodes, graph, demands, vehicle_props, priority_villages=None,
                 heuristic_type="basic", llm_evaluator=None, use_llm_for_ranking=False):
        """
        初始化使用LLM的多车辆规划器
        
        参数:
            nodes: 节点坐标字典
            graph: 图的邻接表表示
            demands: 各村庄的需求量
            vehicle_props: 车辆属性列表，每个元素是 (容量, 时间限制) 元组
            priority_villages: 优先村庄集合
            heuristic_type: 启发式函数类型
            llm_evaluator: LLM评估器实例
            use_llm_for_ranking: 是否使用LLM进行状态排序
        """
        super().__init__(nodes, graph, demands, vehicle_props, priority_villages, heuristic_type)
        self.llm_evaluator = llm_evaluator
        self.use_llm_for_ranking = use_llm_for_ranking
        self.state_scores_cache = {}  # 缓存状态评分，避免重复评估

        # 构建问题上下文
        self.problem_context = {
            "nodes": self.nodes,
            "graph": self.graph,
            "demands": self.demands,
            "priority_villages": self.priority_villages
        }

    def get_state_score(self, state):
        """获取状态的LLM评分，使用缓存避免重复计算"""
        state_hash = hash(state)
        if state_hash not in self.state_scores_cache:
            score = self.llm_evaluator.evaluate_state(state, self.problem_context)
            self.state_scores_cache[state_hash] = score
        return self.state_scores_cache[state_hash]

    def a_star_with_llm(self):
        """使用LLM评估的A*搜索算法"""
        pq = PriorityQueue()
        start_node_state = self.initial_state()

        # 初始状态的评分
        if self.use_llm_for_ranking and self.llm_evaluator:
            # 对于初始状态，使用100减去LLM评分作为优先级（评分越高，优先级越高）
            llm_score = self.get_state_score(start_node_state)
            start_priority = 100 - llm_score
        else:
            # 使用传统方式计算优先级
            g_start = start_node_state.max_time
            h_start = self.heuristic(start_node_state)
            start_priority = g_start + h_start

        pq.enqueue(start_node_state, start_priority)
        came_from = {start_node_state: None}
        g_score = {start_node_state: start_node_state.max_time}

        explored_count = 0
        max_explored_limit = 50000  # 探索上限

        while not pq.is_empty():
            current_s = pq.dequeue()
            explored_count += 1

            if current_s.max_time > g_score.get(current_s, float('inf')):
                continue  # 已找到更好的路径

            if explored_count > max_explored_limit:
                print(f"探索达到上限 {max_explored_limit}，提前终止。")
                return None, float('inf'), explored_count

            if self.is_goal(current_s):
                total_path = []
                temp_current = current_s
                while temp_current is not None:
                    total_path.append(temp_current)
                    temp_current = came_from[temp_current]
                return total_path[::-1], current_s.max_time, explored_count

            for next_s in self.successors(current_s):
                tentative_g_next = next_s.max_time

                if tentative_g_next < g_score.get(next_s, float('inf')):
                    g_score[next_s] = tentative_g_next
                    came_from[next_s] = current_s

                    if self.use_llm_for_ranking and self.llm_evaluator:
                        # 使用LLM评分作为优先级
                        llm_score = self.get_state_score(next_s)
                        # 评分越高，优先级越高（所以用100减去评分）
                        priority = 100 - llm_score
                    else:
                        # 使用传统启发式函数
                        h_next = self.heuristic(next_s)
                        priority = tentative_g_next + h_next

                    pq.enqueue(next_s, priority)

        return None, float('inf'), explored_count


def run_llm_test_case(name, nodes, graph, demands, vehicle_props, priority_villages=None,
                      heuristic_type="basic", use_llm=True, api_key=None):
    """
    运行使用LLM的测试案例
    
    参数:
        name: 测试案例名称
        nodes: 节点坐标字典
        graph: 图的邻接表表示
        demands: 各村庄的需求量
        vehicle_props: 车辆属性列表
        priority_villages: 优先村庄集合
        heuristic_type: 启发式函数类型
        use_llm: 是否使用LLM进行状态评估
        api_key: 阿里云百炼API密钥
    """
    print(f"\n--- 测试案例: {name} ---")
    print(f"使用启发式函数: {heuristic_type}")
    if priority_villages:
        print(f"优先村庄: {priority_villages}")
    print(f"使用LLM评估: {'是' if use_llm else '否'}")

    llm_evaluator = None
    if use_llm and api_key:
        llm_evaluator = LLMStateEvaluator(api_key=api_key)

    planner = LLMMultiVehiclePlanner(
        nodes, graph, demands, vehicle_props,
        priority_villages, heuristic_type,
        llm_evaluator=llm_evaluator,
        use_llm_for_ranking=use_llm
    )

    start_time = time.time()
    if use_llm and llm_evaluator:
        final_path, total_time, num_explored = planner.a_star_with_llm()
    else:
        final_path, total_time, num_explored = planner.a_star()
    end_time = time.time()
    execution_time = end_time - start_time

    if final_path:
        final_state = final_path[-1]
        total_dist_all_vehicles = sum(v.total_distance_traveled for v in final_state.vehicles)
        print(f"  🎉 解决方案找到!")
        print(f"    最慢车辆用时: {total_time:.2f}")
        print(f"    所有车辆总行驶距离: {total_dist_all_vehicles:.2f}")
        print(f"    共探索了 {num_explored} 个状态。")
        print(f"    路径长度 (状态数): {len(final_path)}")
        print(f"    运行时间: {execution_time:.4f} 秒")

        # 计算优先村庄的配送情况
        if priority_villages:
            priority_delivered = sum(final_state.delivered.get(v, 0) for v in priority_villages)
            priority_total = sum(demands.get(v, 0) for v in priority_villages)
            print(
                f"    优先村庄配送情况: {priority_delivered}/{priority_total} ({priority_delivered / priority_total * 100:.1f}%)")

        # 如果使用了LLM，显示最终状态的LLM评分
        if use_llm and llm_evaluator:
            final_score = llm_evaluator.evaluate_state(final_state, {
                "nodes": nodes,
                "graph": graph,
                "demands": demands,
                "priority_villages": priority_villages
            })
            print(f"    LLM评分: {final_score:.2f}/100")
    else:
        print(f"  😥 未能找到解决方案。共探索了 {num_explored} 个状态。")
        print(f"    运行时间: {execution_time:.4f} 秒")

    return execution_time, num_explored, total_time if final_path else float('inf')


# -----------------------------------------------------------------------------
# 主程序
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # 从环境变量或配置文件获取API密钥
    api_key = os.environ.get("ALIYUN_API_KEY")  # environment only

    # 测试案例
    nodes_ex1 = {1: (0, 0), 2: (10, 0), 3: (0, 10)}
    graph_ex1 = {
        1: [(2, 10, 5), (3, 10, 5)],
        2: [(1, 10, 5), (3, 14.14, 2)],
        3: [(1, 10, 5), (2, 14.14, 2)]
    }
    demands_ex1 = {2: 10, 3: 5}
    vehicle_props_ex1 = [(12, 100.0)]  # 1辆车, 容量12, 个体时间上限100
    priority_villages_ex1 = {2}  # 村庄2为优先村庄

    print("\n=== LLM评估测试 ===")

    # 使用传统启发式函数
    print("\n--- 使用传统启发式函数 ---")
    run_llm_test_case(
        "基础案例 - 传统启发式",
        nodes_ex1, graph_ex1, demands_ex1, vehicle_props_ex1,
        priority_villages_ex1, "priority_based", use_llm=False
    )

    # 使用LLM评估
    print("\n--- 使用LLM评估 ---")
    run_llm_test_case(
        "基础案例 - LLM评估",
        nodes_ex1, graph_ex1, demands_ex1, vehicle_props_ex1,
        priority_villages_ex1, "priority_based", use_llm=True, api_key=api_key
    )

    # 测试更复杂的案例
    test_nodes = {
        1: (0, 0),  # 指挥中心
        2: (10, 10),  # 村庄1
        3: (20, 5),  # 村庄2
        4: (5, 15),  # 村庄3
        5: (15, 15)  # 村庄4
    }

    # 构建完全图
    test_graph = {}
    for i in test_nodes:
        test_graph[i] = []
        for j in test_nodes:
            if i != j:
                dist = euclidean_distance(test_nodes[i], test_nodes[j])
                speed = 5.0  # 统一速度使问题更可控
                test_graph[i].append((j, dist, speed))

    # 设置适当的需求
    test_demands = {
        2: 8,
        3: 10,
        4: 6,
        5: 12
    }

    # 设置充足的车辆属性
    test_vehicle_props = [
        (20, 100.0),  # 车辆1: 容量20，时间限制100
        (15, 100.0)  # 车辆2: 容量15，时间限制100
    ]

    # 设置优先村庄
    test_priority_villages = {2, 4}

    print("\n=== 复杂案例比较 ===")

    # 使用传统启发式函数
    print("\n--- 使用传统启发式函数 ---")
    traditional_exec_time, traditional_states, traditional_quality = run_llm_test_case(
        "复杂案例 - 传统启发式",
        test_nodes, test_graph, test_demands, test_vehicle_props,
        test_priority_villages, "priority_based", use_llm=False
    )

    # 使用LLM评估
    print("\n--- 使用LLM评估 ---")
    llm_exec_time, llm_states, llm_quality = run_llm_test_case(
        "复杂案例 - LLM评估",
        test_nodes, test_graph, test_demands, test_vehicle_props,
        test_priority_villages, "priority_based", use_llm=True, api_key=api_key
    )

    # 打印比较结果
    print("\n=== 性能比较 ===")
    print(f"{'方法':<20} {'执行时间(秒)':<15} {'探索状态数':<15} {'解决方案质量':<15}")
    print("-" * 65)

    traditional_quality_str = f"{traditional_quality:.2f}" if traditional_quality != float('inf') else "无解"
    llm_quality_str = f"{llm_quality:.2f}" if llm_quality != float('inf') else "无解"

    print(f"{'传统启发式':<20} {traditional_exec_time:<15.4f} {traditional_states:<15d} {traditional_quality_str:<15}")
    print(f"{'LLM评估':<20} {llm_exec_time:<15.4f} {llm_states:<15d} {llm_quality_str:<15}")
