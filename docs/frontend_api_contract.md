# 前端 API 契约

> 本文档定义前端读取调度结果的数据契约。`outputs/` 下的 solution JSON 与 ablation CSV 已定稿；前端不得改写这些文件，也不得在页面内重新求解。

## 1. 文件清单

| 文件 | 用途 |
|---|---|
| `outputs/solution_index.json` | 场景/组别索引与默认选择 |
| `outputs/solution_{date}_{mode}.json` | 指定日期、指定组的完整调度结果 |
| `outputs/solution_latest_A.json` | 常规日 A 组默认别名 |
| `outputs/solution_latest_B.json` | 常规日 B 组默认别名 |
| `outputs/solution_latest_C.json` | 常规日 C 组默认别名 |
| `configs/village_names.yaml` | 村庄 ID 到中文名的映射 |
| `data/geo/network_layout.json` | 仅用于可视化的节点坐标 |

`mode` 取值：`A`、`B`、`C`。`date` 取值：`2025-08-29`、`2025-08-01`。

## 2. 场景切换

读取 `outputs/solution_index.json`：

```json
{
  "default_scenario": "2025-08-29",
  "default_mode": "C",
  "scenarios": {
    "2025-08-29": {
      "label": "常规天气日",
      "weather": "晴",
      "modes": ["A", "B", "C"],
      "files": {
        "A": "outputs/solution_2025-08-29_A.json",
        "B": "outputs/solution_2025-08-29_B.json",
        "C": "outputs/solution_2025-08-29_C.json"
      }
    }
  }
}
```

加载顺序：

1. 读取 `default_scenario` 与 `default_mode`；
2. 用 `scenarios[date].files[mode]` 找到 solution JSON；
3. 默认页面可直接读 `outputs/solution_latest_C.json`，无需写死日期；
4. 用户切换场景/组别后，重新读取对应文件。

## 3. solution 顶层字段

以下字段与当前 `outputs/solution_2025-08-29_C.json` 完全一致。

| 字段 | 类型 | 示例 | 说明 |
|---|---|---|---|
| `scenario_date` | string | `"2025-08-29"` | 场景日期 |
| `ablation_mode` | string | `"full"` | `hist_demand` / `no_risk_adjust` / `full` |
| `config_snapshot` | object | `{"time_limit":45.0,...}` | 导出时的调度参数快照 |
| `vehicle_trajectories` | object | `{"vehicle_1":[...]}` | 逐车轨迹，见第 4 节 |
| `village_delivery_summary` | object | `{"1":{"demand":28.93,...}}` | 逐村配送汇总 |
| `metrics` | object | `{"slowest_vehicle_time":7.449,...}` | 核心指标，见第 6 节 |
| `solver_status` | string | `"success"` | `success` / `timeout` / `limit_reached` |
| `solve_time_actual` | float | `30.506593` | 实际求解墙钟秒数 |
| `generated_at` | string | `"2026-10-23T17:39:36+08:00"` | 导出时间，ISO 8601 |
| `scenario_json_md5` | string | `"e782ff10a710f5a9869f80f9a0670027"` | 场景 JSON 的 MD5 |

`config_snapshot` 示例：

```json
{
  "time_limit": 45.0,
  "max_states": 50000,
  "risk_cost_alpha": 0.2,
  "speed_penalty": 0.1,
  "num_vehicles": 3,
  "vehicle_capacity": 100.0,
  "heuristic": "priority_based"
}
```

## 4. 轨迹字段与插值

### 4.1 结构

```json
{
  "vehicle_1": [
    {"time": 0.0, "location": 0, "load": 100.0, "delivered": {}, "action": "depart"},
    {"time": 0.882634, "location": 4, "load": 0.0, "delivered": {"4": 100.0}, "action": "deliver"},
    {"time": 1.101171, "location": 7, "load": 0.0, "delivered": {"4": 100.0}, "action": "move"}
  ]
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `time` | float | 车辆时间，单位小时；不是全局时钟，也不是秒 |
| `location` | int | 节点 ID：0 指挥中心，1–6 村庄，7–14 中间节点 |
| `load` | float | 到达该节点后的剩余载货量，单位箱 |
| `delivered` | object | 截至该节点的累计配送量；键为村庄 ID 字符串，值为箱数 |
| `action` | string | 原始动作，见 4.2 的映射 |

### 4.2 action 适配

原始 JSON 实际取值为 `depart`、`move`、`deliver`、`reload`。前端契约使用以下展示枚举：

| 原始值 | 前端枚举 | 含义 |
|---|---|---|
| `depart` | `depart` | 从指挥中心出发 |
| `move` | `travel` | 普通路段移动 |
| `deliver` | `deliver` | 到达村庄并卸货 |
| `reload` | `resupply` | 回到指挥中心补货 |
| 无原始值 | `arrive` | 播放到最后一帧时由前端补充的到达状态 |

前端适配伪代码：

```python
ACTION_MAP = {
    "depart": "depart",
    "move": "travel",
    "deliver": "deliver",
    "reload": "resupply",
}
ui_action = ACTION_MAP.get(frame["action"], "travel")
if frame is last_frame:
    ui_action = "arrive"
```

### 4.3 delivered 的新送达量

原始 `delivered` 是**累计值**。前端若需要“本帧新送达量”，用相邻帧差分：

```python
new_delivered = {
    village_id: amount - previous_cumulative.get(village_id, 0.0)
    for village_id, amount in frame["delivered"].items()
    if amount > previous_cumulative.get(village_id, 0.0)
}
```

### 4.4 插值规则

- 相邻两帧之间，节点位置用 `data/geo/network_layout.json` 的 `(x, y)` 线性插值；
- 时间使用两帧的真实时间差，不假设固定帧率；
- `load`、`delivered`、`action` 在到达帧生效，不做数值插值；
- `time` 单位是小时，若动画按秒播放，可乘以 3600 或按场景总时长归一化。

## 5. village_delivery_summary

```json
{
  "1": {"demand": 28.934324, "delivered": 28.934324, "satisfied_pct": 100.0}
}
```

| 字段 | 类型 | 单位 | 说明 |
|---|---|---|---|
| `demand` | float | 箱 | 该村总需求 |
| `delivered` | float | 箱 | 实际配送量，已按需求截断 |
| `satisfied_pct` | float | % | 满足率，0–100 |

键为村庄 ID 字符串，前端显示前用 `int()` 转换。

## 6. metrics：8 项核心 + 3 项诊断

### 6.1 核心 8 项

| 字段 | 类型 | 单位 | 典型范围 | 说明 |
|---|---|---|---:|---|
| `slowest_vehicle_time` | float | 小时 | 0–100 | 最慢车辆完成时间 |
| `total_distance` | float | 长度单位 | 0–1000 | 全部车辆总里程 |
| `priority_villages_satisfied` | int | 个 | 0–6 | 完整满足的优先村数 |
| `total_exposed_risk` | float | 风险×长度 | 0–1000 | 路径累计暴露风险 |
| `high_risk_edge_visits` | int | 次 | 0–30 | 经过高风险边的次数 |
| `total_demand_satisfied_rate` | float | 0–1 | 0–1 | 总需求满足率 |
| `unserved_demand_total` | float | 箱 | 0–1200 | 未送达总箱数 |
| `solve_time_actual` | float | 秒 | 0–45 | 实际求解墙钟时间（顶层字段） |

前 7 项位于 `metrics` 对象内；`solve_time_actual` 是顶层字段，但前端应把它与 `metrics` 一起展示。

### 6.2 诊断 3 项

| 字段 | 类型 | 说明 |
|---|---|---|
| `solver_status` | string | `success` / `timeout` / `limit_reached` |
| `scenario_json_md5` | string | 场景输入校验值 |
| `generated_at` | string | 结果生成时间，ISO 8601 |

## 7. 部分解提示

当 `solver_status == "timeout"` 或 `"limit_reached"` 时，页面顶部必须显示警示条：

```text
本方案为部分解：总满足率 XX.XX%，未送达 XX.XX 箱
```

数值来自：

- `metrics.total_demand_satisfied_rate * 100`
- `metrics.unserved_demand_total`

`solver_status == "success"` 时不显示警示条。当前恶劣天气日 A 组为 timeout，前端必须明确标注。

## 8. 村庄显示名

读取 `configs/village_names.yaml`：

```yaml
villages:
  1: 青石村
  2: 望江村
  3: 石门村
  4: 云岭村
  5: 松溪村
  6: 北崖村
```

显示规则：`name = villages.get(int(village_id), f"村庄{village_id}")`。若配置缺项，fallback 到“村庄N”。

## 9. 路网可视化坐标

读取 `data/geo/network_layout.json`：

```json
{
  "0": {"x": 0.0, "y": 0.0},
  "1": {"x": 14.0, "y": 0.0},
  "6": {"x": -12.0, "y": -34.0}
}
```

- 共 15 个节点：0–14；
- 0 为指挥中心，1–6 为村庄，7–14 为中间节点；
- `x` 向东为正，`y` 向北为正；
- 坐标只用于可视化，不参与调度计算；
- 画边时以 scenario JSON 的 `roads[].from/to` 为准，不要假设端点都是村庄。

## 10. 推荐加载流程

```text
1. 读 solution_index.json，取 default_scenario 和 default_mode
2. 若用户未切换，直接读 solution_latest_C.json
3. 若用户切换，读 scenarios[date].files[mode]
4. 读 village_names.yaml 和 network_layout.json
5. 按 vehicle_trajectories 插值播放，按 action 映射显示动作
6. 若 solver_status != success，显示部分解警示条
```

## 11. 定稿约束

- `outputs/` 下 solution JSON、索引和 ablation CSV 本轮后定稿；
- 前端只读这些文件，不写回、不重新求解；
- 任何字段或语义改动必须建立新版本契约，并同步更新 `docs/optimization_final_state.md`。


