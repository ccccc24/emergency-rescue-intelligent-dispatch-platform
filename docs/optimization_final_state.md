# 调度算法最终状态

> 本文档是调度算法定稿的交付物。自本轮起，`src/optimization/` 与 `configs/default.yaml` 中的调度参数不再改动；前端开发、报告撰写和答辩演示均以本文档为准。后续如需调整，必须建立新版本并同步更新本文件。

## 1. 定稿范围

- 定稿模块：`src/optimization/`：`models.py`、`heuristics.py`、`planner.py`、`astar.py`、`trajectory.py`、`scenario_loader.py`、`run.py`。
- 定稿参数：`configs/default.yaml` 中 `optimization` 与 `risk` 段。
- 定稿输入：`outputs/scenario_2025-08-29.json`、`outputs/scenario_2025-08-01.json`、`outputs/scenario_latest.json`。
- 前端只读取场景 JSON 和调度结果，不在页面内重新实现或修改调度逻辑。

## 2. 最终配置

| 参数 | 最终值 | 说明 |
|---|---:|---|
| `seed` | 42 | 全局随机种子 |
| `optimization.max_states` | 50000 | A* 状态数上限 |
| `optimization.time_limit` | 45 | 单次求解墙钟上限，单位秒 |
| `optimization.num_vehicles` | 3 | 车辆数 |
| `optimization.vehicle_capacity` | 100 | 每车每趟容量，单位箱 |
| `optimization.vehicle_time_limit` | 100.0 | 单车任务累计时间上限 |
| `optimization.heuristic_type` | `priority_based` | 优先村启发式 |
| `optimization.risk_cost_alpha` | 0.2 | 风险代价权重，量纲为“风险×长度对应的时间” |
| `risk.priority_threshold` | 0.7 | 风险预警阈值 |
| `risk.priority_village_threshold` | 0.6 | 村庄平均连边风险达到该值时纳入优先村 |
| `risk.isolated_road_threshold` | 0.75 | 所有连边风险均高于该值时视为孤岛村 |
| `risk.warning_threshold` | 0.7 | 前端预警展示阈值 |
| `risk.high_risk_edge_threshold` | 0.6 | 高风险边统计阈值 |
| `risk.speed_penalty` | 0.1 | `adjusted_speed = base_speed × (1 - risk × 0.1)` |

说明：`priority_based` 不满足可采纳性，因此本系统统一表述为“A* 风格的多车启发式搜索”，不宣称全局最优。

## 3. 核心消融结论

最终消融使用 3 次重复平均，场景为常规天气日 2025-08-29 和恶劣天气日 2025-08-01。下表指标采集于 `time_limit=30` 批次；最终配置为 `time_limit=45`，核心指标不变。

### 3.1 C 组相对 B 组

| 场景 | 暴露风险变化 | 高风险边经过次数 | 总里程变化 | 最慢车辆时间变化 |
|---|---:|---:|---:|---:|
| 常规天气日 | **-23.53%** | 6 → 4 | +0.21% | +10.11% |
| 恶劣天气日 | **-28.41%** | 15 → 6 | +4.55% | -4.15% |

恶劣天气日实现“风险下降、最慢车时间同时下降”的双赢；常规天气日风险下降明显，但时间有小幅代价，符合风险差异较小的场景特征。

### 3.2 A 组 timeout 发现

- 恶劣天气日 A 组（历史 7 日均值需求 + 风险调整速度）3 次运行全部 `timeout`，返回部分解：总满足率 99.32%，未送达 7.86 箱。
- 20 秒和 30 秒上限下均 timeout，`states_explored≈16291`，说明瓶颈不是单纯时限不足。
- A 组历史均值总需求约 1155.6 箱，C 组 ML 预测总需求约 1165.5 箱，总量接近；但历史均值把村间差异平均化后，北崖村 A 组约 409.1 箱，超过 4×100 箱，需要第 5 车次，而 C 组约 383.4 箱只需 4 车次。该装载边界差异会放大状态空间。
- 结论：ML 需求预测的价值不仅是精度，还可能降低求解复杂度；该发现作为定性说明，不作为核心定量证据。

### 3.3 三组最终指标

常规天气日 2025-08-29：

| 组 | 最慢用时 | 总里程 | 总暴露风险 | 求解时间 | 总满足率 | 高风险经过 | 状态 |
|---|---:|---:|---:|---:|---:|---:|---|
| A 历史均值需求 | 6.427 | 708.9 | 223.513 | 1.346 | 100.00% | 4 | success |
| B 无风险调整 | 6.765 | 760.8 | 317.360 | 4.477 | 100.00% | 6 | success |
| C 完整 ML | 7.449 | 762.4 | 242.696 | 28.318 | 100.00% | 4 | success |

恶劣天气日 2025-08-01：

| 组 | 最慢用时 | 总里程 | 总暴露风险 | 求解时间 | 总满足率 | 高风险经过 | 状态 |
|---|---:|---:|---:|---:|---:|---:|---|
| A 历史均值需求 | 6.842 | 814.0 | 350.744 | 30.383 | 99.32% | 8 | timeout |
| B 无风险调整 | 7.269 | 823.8 | 484.856 | 11.904 | 100.00% | 15 | success |
| C 完整 ML | 6.967 | 861.3 | 347.097 | 13.624 | 100.00% | 6 | success |

## 4. 已知局限

- 恶劣天气日 A 组 3/3 timeout，主表结果需标注为部分解，不能与完整解直接比较。
- 两个真实场景的 `isolated_villages` 均为空，孤岛村逻辑已有单元测试覆盖，但未在正式场景中触发。
- 常规天气日 A/C 总需求满足率均为 100%，需求预测的精度优势未在满足率上体现；只有运力紧张时才可能体现。
- 当前机器常规天气日 C 组：消融批次 28.32 秒 / 30 秒；最终 `time_limit=45` 配置下导出 30.51 秒（余量 32.2%），答辩机基准 31.`time_limit=45`（余量 30.1%）。同一 `states_explored=21949` 下，较快环境约 15.3 秒，耗时差异主要来自机器负载。
- alpha 和 speed_penalty 扫描在 `time_limit=10` 下完成，最终消融使用 30 秒批次，当前最终配置为 `time_limit=45`；扫描选优基于相对比较，时限变化不改变参数选择结论。
- alpha=0.2 的量纲经过校准，但仍是经验参数；部署到真实路网前应按真实时间单位重新标定。

## 5. 预计算方案

演示阶段不依赖实时求解。所有演示场景的完整轨迹和指标已预先计算并保存为 solution JSON，前端直接读取播放；研究实验场景仍保留 `time_limit=45` 作为兜底。

| 场景 | 组别 | 文件 |
|---|---|---|
| 常规天气日 2025-08-29 | A 历史均值需求 | `outputs/solution_2025-08-29_A.json` |
| 常规天气日 2025-08-29 | B 无风险调整 | `outputs/solution_2025-08-29_B.json` |
| 常规天气日 2025-08-29 | C 完整 ML | `outputs/solution_2025-08-29_C.json` |
| 恶劣天气日 2025-08-01 | A 历史均值需求 | `outputs/solution_2025-08-01_A.json` |
| 恶劣天气日 2025-08-01 | B 无风险调整 | `outputs/solution_2025-08-01_B.json` |
| 恶劣天气日 2025-08-01 | C 完整 ML | `outputs/solution_2025-08-01_C.json` |

索引文件：`outputs/solution_index.json`，包含两个场景的 `label`、`weather`、`modes` 和 `files` 映射。

每个 solution JSON 的顶层字段为：

- `scenario_date: str`
- `ablation_mode: str`
- `config_snapshot: dict`
- `vehicle_trajectories: dict[str, list[dict]]`
- `village_delivery_summary: dict[str, dict]`
- `metrics: dict`
- `solver_status: str`
- `solve_time_actual: float`
- `generated_at: str`
- `scenario_json_md5: str`

轨迹点字段为 `time`、`location`、`load`、`delivered`、`action`；当前文件约 9–15KB，原因是只记录位置变化点，未做逐秒插值。前端可按相邻点时间差插值播放，不需要额外轨迹数据。

`action` 枚举为 `depart`（出发）、`deliver`（村庄卸货）、`reload`（回指挥中心补货）、`move`（普通移动）。`delivered` 是该车截至当前点的累计配送量。

当前答辩机基准（`python -m src.experiments.benchmark_env`）：

| 场景 | CPU | 耗时 | states_explored | 占最终时限比例 | 余量 |
|---|---|---:|---:|---:|---:|
| 常规天气日 C 组 | Intel Core Ultra 5 225H，14 逻辑核，31.43GB | 31.450s | 21949 | 69.89% | 30.11% |
| 恶劣天气日 C 组 | 同上 | 16.985s | 11052 | 37.75% | 62.25% |

## 6. 场景 JSON Schema

文件位置：`outputs/scenario_{date}.json`，`outputs/scenario_latest.json` 为最新场景副本。

### 5.1 顶层字段

| 字段 | 类型 | 示例 | 说明 |
|---|---|---|---|
| `date` | string | `"2025-08-29"` | 场景日期，ISO `YYYY-MM-DD` |
| `villages` | object | `{"1": {...}, ...}` | 村庄字典，键为字符串 ID |
| `roads` | object | `{"1": {...}, ...}` | 道路字典，键为字符串 ID，共 30 条边 |
| `demands` | object | `{"1": 28.93, "6": 416.24}` | `{村庄ID: 预测需求箱数}` |
| `risk_scores` | object | `{"1": 0.2136, ...}` | `{道路ID: 中断概率}`，范围 `[0, 1]` |
| `priority_villages` | array[int] | `[4, 5, 6]` | 优先村 ID 列表 |
| `adjusted_speed` | object | `{"1": 53.825, ...}` | `{道路ID: 风险调整后速度}`，已含 `speed_penalty` |
| `isolated_villages` | array[int] | `[]` | 孤岛村 ID 列表，可能为空 |
| `metadata` | object | `{...}` | 模型版本、生成时间、数据 MD5 |

### 5.2 `villages` 元素

示例：村庄 1

| 字段 | 类型 | 示例 | 说明 |
|---|---|---|---|
| `id` | int | `1` | 村庄 ID，1–6 |
| `name` | string | `"村庄1"` | 场景文件中的通用名；前端展示建议改用 `configs/village_names.yaml` 的映射 |
| `population` | int | `1200` | 人口 |
| `disaster_level` | int | `1` | 灾情等级 1–5 |
| `is_priority` | int | `0` | 是否优先村，0/1 |
| `distance_to_center` | float | `10.0` | 到指挥中心距离 |
| `road_accessibility` | float | `0.92` | 道路可达性 0–1 |
| `weather_code` | int | `1` | 天气编码：晴=0，雨=1，暴风=2 |
| `hist_demand_mean` | float | `22.285714` | 过去 7 天需求均值，供基线和展示使用 |

显示名映射：`1=青石村`、`2=望江村`、`3=石门村`、`4=云岭村`、`5=松溪村`、`6=北崖村`。

### 5.3 `roads` 元素

示例：道路 1

| 字段 | 类型 | 示例 | 说明 |
|---|---|---|---|
| `id` | int | `1` | 道路 ID，1–30 |
| `road_id` | string | `"R01"` | 人类可读道路编号 |
| `from` | int | `0` | 起点节点，0 为指挥中心，1–6 为村庄，7–14 为中间节点 |
| `to` | int | `1` | 终点节点，编码同上 |
| `length` | float | `11.5` | 边长 |
| `base_speed` | float | `55.0` | 基础速度 |
| `terrain` | string | `"平原"` | 地形：平原/丘陵/山区 |
| `weather` | string | `"雨"` | 天气：晴/雨/暴风 |
| `weather_code` | int | `1` | 天气有序编码 |
| `geology_risk` | float | `0.064` | 地质风险 0–1 |
| `weather_sensitivity` | float | `0.813` | 该边对天气的敏感度 |
| `hist_break_count` | int | `1` | 截至当天前的历史中断次数，不包含当天 |

### 5.4 `metadata`

| 字段 | 类型 | 示例 |
|---|---|---|
| `demand_model.type` | string | `"RandomForestRegressor"` |
| `demand_model.trained_date` | string | `"2026-10-21T22:35:53+08:00"` |
| `demand_model.test_mae` | float | `18.8501` |
| `risk_model.type` | string | `"RandomForestClassifier"` |
| `risk_model.trained_date` | string | `"2026-10-22T16:13:49+08:00"` |
| `risk_model.test_auc` | float | `0.7909` |
| `risk_model.features_used` | array[string] | `["length", "base_speed", ...]` |
| `config_seed` | int | `42` |
| `generated_at` | string | ISO 时间戳 |
| `data_md5.demand` | string | 需求 CSV 的 MD5 |
| `data_md5.road` | string | 道路 CSV 的 MD5 |

## 7. 前端接入注意

1. JSON 中的字典键全部是字符串；前端转换村庄/道路 ID 时要显式 `int()`。
2. `adjusted_speed` 已经是最终速度，前端不要再次乘以 `(1 - risk × 0.1)`。
3. `risk_scores` 是概率，不是等级；颜色映射和阈值展示由 UI 决定，调度器不再二次判定封路。
4. `priority_villages` 和 `isolated_villages` 是整数数组，不是字典。
5. `roads` 的 `from`/`to` 可能引用 7–14 号中间节点，而 `villages` 只有 1–6；画路网时必须以 `roads` 的端点为准，不能假设所有端点都是村庄。
6. 场景文件不包含节点坐标；前端需要单独的布局/地图数据，或按节点 ID 自动布局。
7. `name` 字段是通用的“村庄N”；正式展示请读取 `configs/village_names.yaml`。
8. 页面只读取 `scenario_latest.json` 或指定日期文件，不要在页面中重新训练模型或改写调度参数。

## 8. 定稿验收清单

- [x] `time_limit=45`、`max_states=50000`、`risk_cost_alpha=0.2`、`speed_penalty=0.1` 已写入配置。
- [x] 两个场景 JSON 已重新生成，`risk_scores` 与 `demands` 保持稳定，`adjusted_speed` 已按 0.1 惩罚更新。
- [x] A/B/C 三组 × 两场景 × 3 次消融完成。
- [x] 核心结论保持：恶劣日风险 -28.41%，常规日风险 -23.53%。
- [x] A 组 timeout 已作为已知局限记录，未通过调参或降低问题规模掩盖。
- [x] 6 个 solution JSON 与 `solution_index.json` 已生成，metrics 与消融 CSV 一致。
- [x] `src/experiments/benchmark_env.py` 可在当前机器运行并输出时间余量。
- [x] 前端所需场景字段、类型和示例已记录。

后续任何算法或配置改动，必须新建版本说明并更新本文件，不得直接覆盖本定稿状态。

