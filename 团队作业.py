import heapq
import math


# -----------------------------------------------------------------------------
# 辅助类和函数
# -----------------------------------------------------------------------------

class PriorityQueue:
    def __init__(self):
        self.items = []  # 初始化一个空列表作为堆

    def enqueue(self, item, priority):
        heapq.heappush(self.items, (priority, item))  # 添加元素及其优先级

    def dequeue(self):
        if not self.is_empty():
            return heapq.heappop(self.items)[1]  # 返回元素（忽略优先级）
        raise IndexError("无法从空优先队列出队")

    def is_empty(self):
        return len(self.items) == 0


def euclidean_distance(p1_coords, p2_coords):
    """计算两点之间的欧氏距离。p1_coords 和 p2_coords 是 (x,y) 元组。"""
    if p1_coords is None or p2_coords is None:
        return float('inf')
    return math.sqrt((p1_coords[0] - p2_coords[0]) ** 2 + (p1_coords[1] - p2_coords[1]) ** 2)


# -----------------------------------------------------------------------------
# 核心类定义
# -----------------------------------------------------------------------------

class Vehicle:
    def __init__(self, id, location, capacity, initial_max_individual_time, time=0, distance_traveled=0):
        self.id = id
        self.location = location
        self.capacity = capacity
        self.time = time
        self.max_individual_time_limit = initial_max_individual_time
        self.total_distance_traveled = distance_traveled

    def __repr__(self):
        return (f"Vehicle(id={self.id}, loc={self.location}, cap={self.capacity}, "
                f"time={self.time:.2f}, dist_travel={self.total_distance_traveled:.2f}, "
                f"time_limit={self.max_individual_time_limit:.2f})")


class State:
    def __init__(self, vehicles_list, delivered_dict, current_max_time, priority_villages=None):
        self.vehicles = sorted(vehicles_list, key=lambda v: v.id)
        self.delivered = delivered_dict
        self.max_time = current_max_time
        self.total_system_distance = sum(v.total_distance_traveled for v in self.vehicles)
        # 新增：优先村庄集合
        self.priority_villages = priority_villages or set()

    def __hash__(self):
        vehicle_details = tuple(
            (v.location, v.capacity, round(v.time, 5), round(v.total_distance_traveled, 5)) for v in self.vehicles)
        delivered_details = tuple(sorted(self.delivered.items()))
        return hash(
            (vehicle_details, delivered_details, round(self.max_time, 5), tuple(sorted(self.priority_villages))))

    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        if len(self.vehicles) != len(other.vehicles):
            return False

        vehicles_equal = all(
            s_v.id == o_v.id and
            s_v.location == o_v.location and
            s_v.capacity == o_v.capacity and
            round(s_v.time, 5) == round(o_v.time, 5) and
            round(s_v.total_distance_traveled, 5) == round(o_v.total_distance_traveled, 5)
            for s_v, o_v in zip(self.vehicles, other.vehicles)
        )
        return (
                vehicles_equal and
                self.delivered == other.delivered and
                round(self.max_time, 5) == round(other.max_time, 5) and
                self.priority_villages == other.priority_villages
        )

    def __lt__(self, other):
        if not isinstance(other, State):
            return NotImplemented

        # 任务1：优先村庄比较
        # 计算优先村庄的满足程度
        self_priority_delivered = sum(self.delivered.get(v, 0) for v in self.priority_villages)
        other_priority_delivered = sum(other.delivered.get(v, 0) for v in other.priority_villages)

        # 如果优先村庄配送量不同，优先考虑优先村庄配送量更多的状态
        if self_priority_delivered != other_priority_delivered:
            return self_priority_delivered > other_priority_delivered

        # 主要比较标准: max_time (越小越好)
        if round(self.max_time, 5) != round(other.max_time, 5):
            return round(self.max_time, 5) < round(other.max_time, 5)

        # 次要比较标准: 总行驶距离
        if round(self.total_system_distance, 5) != round(other.total_system_distance, 5):
            return round(self.total_system_distance, 5) < round(other.total_system_distance, 5)

        # 第三比较标准: 总配送量 (越大越好)
        self_total_delivered = sum(self.delivered.values())
        other_total_delivered = sum(other.delivered.values())
        if self_total_delivered != other_total_delivered:
            return self_total_delivered > other_total_delivered

        return False  # 如果所有标准都相同

    def __repr__(self):
        return (f"State(max_time={self.max_time:.2f}, total_dist={self.total_system_distance:.2f}, "
                f"delivered={self.delivered}, priority={self.priority_villages}, vehicles={self.vehicles})")


class MultiVehiclePlanner:
    def __init__(self, nodes, graph, demands, vehicle_props, priority_villages=None, heuristic_type="basic"):
        """
        初始化多车辆规划器
        
        参数:
            nodes: 节点坐标字典
            graph: 图的邻接表表示
            demands: 各村庄的需求量
            vehicle_props: 车辆属性列表，每个元素是 (容量, 时间限制) 元组
            priority_villages: 优先村庄集合
            heuristic_type: 启发式函数类型，可选值: "basic", "distance_based", "priority_based"
        """
        self.nodes = nodes
        self.graph = graph
        self.demands = demands
        self.vehicle_properties = vehicle_props
        # 任务1：设置优先村庄
        self.priority_villages = priority_villages or set()
        # 任务2：设置启发式函数类型
        self.heuristic_type = heuristic_type

        if self.graph:
            all_speeds = [speed for edges in self.graph.values() for _, _, speed in edges if speed > 0]
            self.min_speed = min(all_speeds) if all_speeds else 1
        else:
            self.min_speed = 1

    def initial_state(self):
        initial_vehicles = []
        for i, (capacity, max_time_limit) in enumerate(self.vehicle_properties, start=1):
            vehicle = Vehicle(id=i, location=1, capacity=capacity, initial_max_individual_time=max_time_limit, time=0,
                              distance_traveled=0)
            initial_vehicles.append(vehicle)

        initial_delivered = {village_id: 0 for village_id in self.demands}
        return State(initial_vehicles, initial_delivered, 0, self.priority_villages)

    def is_goal(self, current_state):
        for village_id, required_demand in self.demands.items():
            if current_state.delivered.get(village_id, 0) < required_demand:
                return False
        return True

    def successors(self, current_state):
        possible_next_states = []
        for vehicle_idx_in_state_list, current_vehicle_from_state in enumerate(current_state.vehicles):
            if current_vehicle_from_state.location not in self.graph:
                continue

            for neighbor_id, distance, speed in self.graph[current_vehicle_from_state.location]:
                if speed <= 0: continue

                # --- 创建深拷贝 ---
                new_vehicles_list = []
                for v_orig in current_state.vehicles:
                    new_vehicles_list.append(
                        Vehicle(id=v_orig.id,
                                location=v_orig.location,
                                capacity=v_orig.capacity,
                                initial_max_individual_time=v_orig.max_individual_time_limit,
                                time=v_orig.time,
                                distance_traveled=v_orig.total_distance_traveled)
                    )

                vehicle_to_move = new_vehicles_list[vehicle_idx_in_state_list]
                travel_time = distance / speed

                # 检查移动是否会导致超出个体时间限制
                if vehicle_to_move.time + travel_time > vehicle_to_move.max_individual_time_limit:
                    continue

                # 更新车辆状态
                vehicle_to_move.location = neighbor_id
                vehicle_to_move.time += travel_time
                vehicle_to_move.total_distance_traveled += distance

                new_delivered_dict = current_state.delivered.copy()

                if neighbor_id == 1:  # 返回指挥中心补给
                    # 根据其原始属性找到车辆的初始容量
                    original_cap, _ = self.vehicle_properties[vehicle_to_move.id - 1]
                    vehicle_to_move.capacity = original_cap
                elif neighbor_id in self.demands and vehicle_to_move.capacity > 0:  # 到村庄配送
                    demand_at_target = self.demands[neighbor_id]
                    delivered_so_far = new_delivered_dict.get(neighbor_id, 0)
                    remaining_demand = demand_at_target - delivered_so_far
                    if remaining_demand > 0:
                        can_deliver = min(vehicle_to_move.capacity, remaining_demand)
                        vehicle_to_move.capacity -= can_deliver
                        new_delivered_dict[neighbor_id] += can_deliver

                new_max_overall_time = max(v.time for v in new_vehicles_list)
                possible_next_states.append(
                    State(new_vehicles_list, new_delivered_dict, new_max_overall_time, self.priority_villages))

        return possible_next_states

    def heuristic(self, current_state):
        """
        根据选择的启发式函数类型返回适当的启发式值
        """
        if self.heuristic_type == "distance_based":
            return self.distance_based_heuristic(current_state)
        elif self.heuristic_type == "priority_based":
            return self.priority_based_heuristic(current_state)
        else:
            return self.basic_heuristic(current_state)

    def basic_heuristic(self, current_state):
        """原始基础启发式函数"""
        total_remaining_demand = 0
        needy_village_ids = []
        for village_id, required_demand in self.demands.items():
            remaining = required_demand - current_state.delivered.get(village_id, 0)
            if remaining > 0:
                total_remaining_demand += remaining
                needy_village_ids.append(village_id)
        if total_remaining_demand == 0: return 0

        max_one_vehicle_capacity = 1
        if self.vehicle_properties:
            max_one_vehicle_capacity = max(p[0] for p in self.vehicle_properties) if self.vehicle_properties else 1
        if max_one_vehicle_capacity <= 0: max_one_vehicle_capacity = 1

        num_full_loads_needed = math.ceil(total_remaining_demand / max_one_vehicle_capacity)

        max_dist_to_needy_village = 0
        depot_coords = self.nodes.get(1)
        if depot_coords and needy_village_ids:
            current_max_dist = 0
            for v_id in needy_village_ids:
                village_coords = self.nodes.get(v_id)
                dist = euclidean_distance(depot_coords, village_coords)
                if dist != float('inf') and dist > current_max_dist:
                    current_max_dist = dist
            max_dist_to_needy_village = current_max_dist
        elif not needy_village_ids:
            return 0
        else:  # Fallback if coordinates are missing
            all_distances_in_graph = [d for edges_list in self.graph.values() for _, d, _ in edges_list if d > 0]
            max_dist_to_needy_village = max(all_distances_in_graph) if all_distances_in_graph else 10

        if max_dist_to_needy_village == 0 and total_remaining_demand > 0:
            return total_remaining_demand / (max_one_vehicle_capacity * self.min_speed + 1e-6)

        time_for_one_furthest_trip = max_dist_to_needy_village / self.min_speed
        h_value = (num_full_loads_needed * time_for_one_furthest_trip)
        if self.vehicle_properties:
            h_value /= len(self.vehicle_properties)
        return h_value

    def distance_based_heuristic(self, current_state):
        """
        在 basic_heuristic 的基础上，增加对总需求量的考虑。
        这个函数估算完成所有配送任务所需的最少行程次数。

        """
        base_heuristic = self.basic_heuristic(current_state)

        total_remaining_demand = 0
        for village_id, required_demand in self.demands.items():
            remaining = required_demand - current_state.delivered.get(village_id, 0)
            if remaining > 0:
                total_remaining_demand += remaining

        # 如果所有需求都已满足，启发式值为0
        if total_remaining_demand == 0:
            return 0.0

        total_max_capacity = sum(p[0] for p in self.vehicle_properties)

        # 估算总共需要的满载行程数（下界）
        if total_max_capacity > 0:
            min_trips_needed = math.ceil(total_remaining_demand / total_max_capacity)
        else:
            min_trips_needed = float('inf')

        # 估算每次往返的平均时间下界（可以根据指挥中心到最远村庄的往返时间作为下界）
        needy_villages_coords = {}
        for village_id, required_demand in self.demands.items():
            remaining = required_demand - current_state.delivered.get(village_id, 0)
            if remaining > 0:
                needy_villages_coords[village_id] = self.nodes.get(village_id)

        max_dist_to_village = 0
        resupply_coord = self.nodes.get(1)
        if resupply_coord:
            for village_coord in needy_villages_coords.values():
                dist = euclidean_distance(resupply_coord, village_coord)
                if dist > max_dist_to_village:
                    max_dist_to_village = dist

        avg_trip_time_lower_bound = (2 * max_dist_to_village) / self.min_speed

        # 将行程数成本加到基础启发式值上
        return base_heuristic + min_trips_needed * avg_trip_time_lower_bound

    def priority_based_heuristic(self, current_state):
        """
        在 `distance_based_heuristic` 的基础上，为优先村庄的配送增加惩罚权重。
        为了保持可接受性，我们不能减少启发式值，而是可以增加一个惩罚项。
        """
        base_heuristic = self.distance_based_heuristic(current_state)

        priority_penalty = 0.0

        # 检查是否有未满足需求的优先级村庄
        has_needy_priority_villages = any(
            (self.demands.get(vid, 0) - current_state.delivered.get(vid, 0)) > 0 and vid in self.priority_villages
            for vid in self.demands
        )

        if has_needy_priority_villages:
            # 增加一个与未满足的优先级需求量相关的惩罚
            total_priority_demand = sum(
                self.demands.get(vid, 0) - current_state.delivered.get(vid, 0) for vid in self.priority_villages)
            priority_penalty = total_priority_demand * 0.1

        return base_heuristic + priority_penalty

    def a_star(self):
        pq = PriorityQueue()
        start_node_state = self.initial_state()

        g_start = start_node_state.max_time
        h_start = self.heuristic(start_node_state)
        f_start = g_start + h_start

        pq.enqueue(start_node_state, f_start)
        came_from = {start_node_state: None}
        g_score = {start_node_state: g_start}

        explored_count = 0
        max_explored_limit = 1000000  # 增加探索上限

        while not pq.is_empty():
            current_s = pq.dequeue()
            explored_count += 1

            if current_s.max_time > g_score.get(current_s, float('inf')):
                continue  # Found a better path already

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
                    h_next = self.heuristic(next_s)
                    f_value_next = tentative_g_next + h_next
                    pq.enqueue(next_s, f_value_next)

        return None, float('inf'), explored_count


# -----------------------------------------------------------------------------
# 测试算例和主程序
# -----------------------------------------------------------------------------
def run_test_case(name, nodes, graph, demands, vehicle_props, priority_villages=None, heuristic_type="distance_based"):
    print(f"\n--- 测试案例: {name} ---")
    print(f"使用启发式函数: {heuristic_type}")
    if priority_villages:
        print(f"优先村庄: {priority_villages}")

    planner = MultiVehiclePlanner(nodes, graph, demands, vehicle_props, priority_villages, heuristic_type)

    start_time = time.time()
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
    else:
        print(f"  😥 未能找到解决方案。共探索了 {num_explored} 个状态。")
        print(f"    运行时间: {execution_time:.4f} 秒")

    return execution_time, num_explored, total_time if final_path else float('inf')


def generate_large_scale_problem():
    """Return the large-scale vehicle routing problem with the provided data"""
    # Use the provided nodes data
    nodes = {
        1: (0, 0), 2: (5, 149), 3: (118, 103), 4: (140, 132), 5: (102, 108),
        6: (90, 63), 7: (110, 140), 8: (18, 137), 9: (109, 130), 10: (100, 137),
        11: (142, 145), 12: (61, 12), 13: (40, 118), 14: (111, 10), 15: (147, 36),
        16: (123, 129), 17: (128, 81), 18: (13, 132), 19: (10, 80), 20: (70, 140),
        21: (28, 50), 22: (11, 138), 23: (111, 113), 24: (148, 108), 25: (93, 14),
        26: (149, 78), 27: (1, 104), 28: (102, 3), 29: (12, 13), 30: (81, 118),
        31: (110, 11), 32: (122, 117), 33: (136, 101), 34: (13, 101), 35: (141, 14)
    }

    # Use the provided graph data
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
        35: [(1, 141.0, 3), (15, 23.3, 7), (24, 94.6, 5), (26, 20.0, 6), (31, 6.1, 6), (33, 87.4, 5)]
    }
    # Use the provided demands data
    demands = {
        18: 58, 20: 38, 22: 36, 34: 22, 27: 52, 8: 35, 21: 41
    }
    # Use the provided priority villages
    priority_villages = [20, 8, 18]
    # Use the provided vehicle properties
    vehicle_props = [
        (78, 153.4), (86, 139.2), (52, 229.5), (70, 169.8)
    ]
    return nodes, graph, demands, vehicle_props, priority_villages


if __name__ == '__main__':
    import time

    # 小规模数据算例下，不同启发式函数性能对比
    print("\n=== 小规模数据算例下，不同启发式函数性能对比 ===")

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

    heuristic_types = ["basic", "distance_based", "priority_based"]
    results1 = {}

    for h_type in heuristic_types:
        print(f"\n测试启发式函数: {h_type}")
        exec_time, states_explored, solution_quality = run_test_case(
            f"小规模问题 - {h_type}",
            test_nodes, test_graph, test_demands, test_vehicle_props, test_priority_villages, h_type
        )
        results1[h_type] = {
            "执行时间": exec_time,
            "探索状态数": states_explored,
            "解决方案质量": solution_quality
        }

    # 打印性能比较表格
    print("\n=== 性能比较 ===")
    print(f"{'启发式函数':<20} {'执行时间(秒)':<15} {'探索状态数':<15} {'解决方案质量':<15}")
    print("-" * 65)
    for h_type, metrics in results1.items():
        quality_str = f"{metrics['解决方案质量']:.2f}" if metrics['解决方案质量'] != float('inf') else "无解"
        print(f"{h_type:<20} {metrics['执行时间']:<15.4f} {metrics['探索状态数']:<15d} {quality_str:<15}")

    # 大规模问题测试
    print("\n=== 大规模问题测试 ===")
    large_nodes, large_graph, large_demands, large_vehicle_props, large_priority_villages = generate_large_scale_problem()

    results2 = {}

    for h_type in heuristic_types:
        print(f"\n测试启发式函数: {h_type}")
        exec_time, states_explored, solution_quality = run_test_case(
            f"大规模问题 - {h_type}",
            large_nodes, large_graph, large_demands, large_vehicle_props, large_priority_villages, h_type
        )
        results2[h_type] = {
            "执行时间": exec_time,
            "探索状态数": states_explored,
            "解决方案质量": solution_quality
        }

    # 打印性能比较表格
    print("\n=== 性能比较 ===")
    print(f"{'启发式函数':<20} {'执行时间(秒)':<15} {'探索状态数':<15} {'解决方案质量':<15}")
    print("-" * 65)
    for h_type, metrics in results2.items():
        quality_str = f"{metrics['解决方案质量']:.2f}" if metrics['解决方案质量'] != float('inf') else "无解"
        print(f"{h_type:<20} {metrics['执行时间']:<15.4f} {metrics['探索状态数']:<15d} {quality_str:<15}")
