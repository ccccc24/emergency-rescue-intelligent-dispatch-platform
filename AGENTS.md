# 航空应急救援调度平台

> 面向 AI 编码代理。项目根目录："C:\Users\xy179\Desktop\机器学习与数据分析课\大作业"

## 1. 项目定位

以多车/多机救援调度算法为核心，用机器学习预测灾区物资需求与道路风险，将结果输入 A\* 风格启发式调度器，Streamlit 前端展示调度过程，LLM 作为可选解释层。

课程主线：**数据分析 → 机器学习预测 → 调度优化 → 实验对比 → 可视化产品 → AI 辅助开发过程**。

一句话：`团队作业.py` 是核心调度引擎，ML 负责需求预测和风险评估，LLM 是可选的智能分析与交互层，Streamlit 负责产品化展示。

## 2. 三层架构
历史/模拟数据
├─→ 需求预测模型 ─→ 每村未来 24h 需求量
└─→ 道路风险评估模型 ─→ 风险等级 / 通行速度
│
▼
src/optimization/ A* 多车启发式调度（核心引擎，离线可跑）
│
▼
调度方案 + 每车轨迹
├─→ Streamlit 前端（地图动画 / KPI / 对比图）
└─→ LLM 可选层（方案解释 / what-if）

text

**边界**：`src/optimization/` 不 import LLM、不 import Streamlit、不联网；LLM 不得进入 A\* 内层搜索循环；ML 输出必须真正影响调度输入（`demands` / `priority_villages` / 边权）。

## 3. 环境基线（已确认，不要重复检查）

- 使用 **Anaconda base 环境**，**不创建虚拟环境**。
- Python 3.13.9 / pip 25.3。
- 已装：streamlit 1.51.0、plotly 6.3.0、pandas 2.3.3、numpy 2.3.5、scikit-learn 1.7.2。
- 未装（可选，需要时再 `pip install`）：xgboost、lightgbm、pydeck。
- 新增依赖必须同步写入 `requirements.txt`；禁止 `pip install` 后不记录。

## 4. 目录结构
航空应急救援/
├── AGENTS.md
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── configs/default.yaml
├── data/
│ ├── synthetic/ # data_gen.py 生成的历史数据
│ └── geo/ # 离线 GeoJSON
├── models/ # 训练好的模型（.pkl）
├── src/
│ ├── optimization/ # models / planner / heuristics / astar / trajectory
│ ├── ml/ # data_gen / demand_model / risk_model / bridge
│ ├── llm/ # explainer / whatif（可选，mock 模式）
│ └── experiments/ # benchmark / ablation
├── app/
│ ├── main.py
│ └── pages/
├── tests/
└── docs/
├── ai_logs/
├── prompt_iteration.md
├── design_decisions.md
└── failure_cases.md

## 5. 核心算法模块约定（`src/optimization/`）

以 `团队作业.py` 为主干重构，`团队作业LLM实现尝试.py` 的 LLM 部分拆到 `src/llm/`。

必须修复：
1. `import time` 移到文件顶部（现在只在 `__main__` 里导入，模块化后调用 `run_test_case()` 会报 `name 'time' is not defined`）。
2. 测试代码（`run_test_case` / `generate_large_scale_problem`）移到 `src/experiments/`。
3. A\* 增加 `max_states=20000`、`time_limit=10s`，超限返回当前最优解并标记 `status="timeout"`。
4. 输出每车轨迹：`{"vehicle_1": [{"time": 0, "location": 1}, ...], ...}`，供前端动画。
5. 表述规范：`priority_based` 不满足可采纳性，**不保证全局最优**，统一写"A\* 风格的多车启发式搜索"。
6. A\* 是优化算法，不冒充 ML。

## 6. 机器学习层约定（`src/ml/`）

- **数据生成**：用带业务规则的生成器（`data_gen.py`），公式可解释、随机种子固定（默认 42）、噪声 10%～20%。禁止直接手写 CSV。
- **需求预测**（回归）：特征含人口、灾情等级、距离、可达性、天气、历史需求、是否优先村；模型 RF / XGBoost / LightGBM 至少一个；基线历史均值 + 线性回归；指标 MAE / RMSE / R²；**按时间切分 train/test，禁止随机切分**；输出 `demands`。
- **风险评估**（分类/回归）：输入道路/航线/天气/地质/历史中断；指标 Accuracy / F1 / AUC 或 MAE；输出风险等级、`adjusted_speed = base_speed * (1 - risk_score * 0.5)`，并参与 `priority_villages` 判定（阈值 0.7）。
- 可选：KMeans/DBSCAN 村庄聚类为紧急/高/中/低。
- ML 输出必须经 `bridge.py` 转换为调度器输入，禁止只画图。

## 7. LLM 层约定（`src/llm/`）

允许：方案解释、对算法生成的 3～5 个候选方案做自然语言评价、what-if 解析成结构化参数再重算。
禁止：A\* 每扩展一状态调一次 LLM。
工程：结构化输出、缓存、规则兜底、Key 只从 `.env` 读、无 Key 自动 mock 模式。

## 8. 前端约定（`app/`）

Streamlit + Plotly/pydeck。5 个页面：态势总览、需求预测、智能调度（含时间轴动画）、方案对比、AI 决策解释。地图用离线 GeoJSON，答辩不依赖网络。

## 9. 文档与 Vibe Coding 证据

持续产出 `docs/ai_logs/`、`prompt_iteration.md`、`design_decisions.md`、`failure_cases.md`，保留 Git commit 与任务看板截图。

## 10. 开发里程碑（按顺序）

1. ~~base 环境检查~~ ✅ 已完成（见第 3 节）
2. **数据生成** `src/ml/data_gen.py` → 两份 CSV（当前任务，负责人：ML）
3. 需求预测模型 + 基线对比
4. 道路风险评估模型
5. `bridge.py` 打通 ML → A\* 闭环
6. 重构 `团队作业.py` → `src/optimization/`（负责人：调度）
7. Streamlit 页面 + 车辆动画
8. 可选 LLM 解释模块
9. 实验对比、报告、演示视频

MVP 优先：5～10 村、2～3 车/机；一个预测模型；A\* vs 改进启发式；地图动画；方案对比；完整文档。

## 11. 已知坑（禁止事项）

- ❌ 未记录依赖就先 `pip install`。
- ❌ 直接 import 原文件进 Streamlit（必须先模块化）。
- ❌ `time` 留在 `__main__`。
- ❌ A\* 无上限运行导致页面卡死。
- ❌ 宣称 priority_based "保证最优"。
- ❌ LLM 进内层搜索循环。
- ❌ API Key 硬编码 / 提交 `.env`。
- ❌ 依赖在线地图做 demo。
- ❌ 保留重复的 `disaster_relief_*` 旧版本（只留主版本）。
- ❌ 中文乱码（统一 UTF-8）。
- ❌ 随机切分训练/测试集（时序数据必须按时间切）。

## 12. 协作规范

- 所有文件 UTF-8；随机种子默认 42，从 `configs/default.yaml` 读。
- 关键 AI 对话记录到 `docs/ai_logs/`。
- 分工按真实工时写，报告附"模块—负责人—工时—产出"表。
- 跨模块接口（尤其 `bridge.py` 输出 JSON 格式）改动必须双方确认后同步到本文件。