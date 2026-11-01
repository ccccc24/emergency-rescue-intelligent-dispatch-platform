# 驰援：应急救援智能调度平台

## 1. 项目定位
支持示例场景与用户自定义数据的应急救援应用。前端采用 **Next.js + React**，以本地 SVG 路网展示风险航路与车辆轨迹；后端采用 **FastAPI**，复用桥接层、ML 预测和 A* 风格启发式调度器。LLM 仅做方案解释，不进入 A* 搜索循环，也不改变调度结果。

## 2. 环境与安装

当前主版本采用 **Next.js + React 前端 + FastAPI 后端**。Vercel 部署步骤见 [`docs/vercel_deployment.md`](docs/vercel_deployment.md)。

```powershell
pip install -r requirements-dev.txt
cd frontend
npm install
Copy-Item .env.local.example .env.local
```

LLM 可选：在后端环境变量中配置 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL`。未配置 Key 时自动使用规则 mock 模式。不要把 `.env` 提交到仓库。


## 3. 启动

后端：

```powershell
python -m uvicorn api.index:app --reload --host 127.0.0.1 --port 8000
```

前端（另开终端）：

```powershell
cd frontend
npm run dev
```

打开 `http://localhost:3000`。

## 4. 页面说明
- **态势总览**：当前方案 KPI、风险着色路网、优先村和计算入口。
- **数据输入**：示例、手动编辑、Excel 上传及两级输入模式。
- **智能调度**：轨迹示意动画，支持播放、暂停、时间轴和倍速。
- **方案对比**：用户单方案与示例 A/B/C 分区展示。
- **AI 决策解释**：方案摘要、建议问题和自由提问。

## 5. 两级输入模式
- **最小输入（直接调度）**：适合已有最终 demand/risk 的用户。村庄填写 `name,x,y,demand,priority`，道路填写 `from,to,distance,risk`，再设置车辆和搜索参数。
- **完整输入（ML 预测辅助）**：适合只有村庄/道路特征的用户。除基础字段外，村庄增加 `population,disaster_level,road_accessibility,hist_demand_mean`；道路增加 `base_speed,geology_risk,hist_break_count`。点击“预测需求与风险”后回填 demand/risk，用户仍可覆盖。

完整模式不点预测、但已手填全部 demand/risk 时也可直接应用。缺失特征不会用于预测；`hist_demand_mean` 是需求模型必填特征。

## 6. ML 预测
- 手动模式通过“预测需求与风险”触发。
- Excel 模式根据特征列或 `config.input_mode` 自动判断；完整模式显示预测按钮。
- 模型：`models/demand_model.pkl`、`models/risk_model.pkl`，只读加载，不训练。
- 预测严格按模型 metadata 的 `feature_order`；缺失、NaN、非法或越界特征只拒绝对应行，不填默认值。
- “预测并覆盖已填 demand/risk”默认不勾选；未勾选时保留用户最终值，勾选后全部重新预测。
- `metadata.prediction` 记录来源、天气、模型 SHA-256、预测时间、预测行数、保留用户值行数和特征错误行数。
- 道路缺少 `base_speed` 时，调度结构使用默认值 50.0，并写入 `metadata.fallback_fields`；页面显示提示。该兜底也参与 hash。

## 7. 使用路径
1. 态势总览查看示例方案。
2. 数据输入页选择最小或完整输入，编辑或上传 Excel。
3. 应用当前场景后，在态势总览点击“用当前数据计算方案”。
4. 查看智能调度、方案对比和 AI 解释。
5. 场景变化后 hash 改变，旧结果不会复用。

## 8. Excel 模板
模板：`app/assets/scenario_template.xlsx`，也可在数据输入页“下载模板.xlsx”。

- `villages`：`name,x,y,demand,priority,population,disaster_level,road_accessibility,hist_demand_mean`
- `roads`：`from,to,distance,risk,base_speed,geology_risk,hist_break_count`
- `config`：调度参数，以及 `weather`、`input_mode=minimal|full`

只保留基础列和 demand/risk 的旧模板仍可直接使用。校验包括村庄数、名称唯一、道路端点、距离、需求、风险和图连通性。

## 9. Hash 与调度参数
`scenario_hash` 覆盖场景几何、最终 demand/risk、道路实际 `base_speed`、base_speed 来源标记和调度参数。`input_mode` 只写入 metadata，不进入 hash。

| 页面/Excel 别名 | 真实字段 | 范围 |
|---|---|---|
| `risk_weight` | `risk_cost_alpha` | `[0, 0.5]` |
| `speed_penalty` | `speed_penalty` | `[0, 0.5]` |
| `search_time_limit` | `time_limit` | `[5, 35]` |
| `max_states` | `max_states` | `[1000, 50000]` |

Excel 同时接受别名和真实字段名。

## 10. 实时计算
计算成功结果绑定 `scenario_hash + group`。超时或失败时，有上一次有效结果则保留并提示；此前从未成功则提示本次未成功，不使用示例结果代替用户场景。

## 11. 数据与清理
- `outputs/`：示例场景与结果。
- `outputs/live/`：实时结果和 `active_scenario.json`。
- 清理命令：

```powershell
python scripts/clean_live_outputs.py --keep-last 20 --older-than 7
```

默认预览，加 `--yes` 才删除；不会删除 `active_scenario.json`。

## 12. 测试
```powershell
python -m compileall -q app src scripts tests
pytest -q
```

## 13. 常见问题
- **场景校验失败**：检查村庄数、名称、端点、距离、demand/risk 和图连通性。
- **预测失败**：检查完整模式特征，特别是 `hist_demand_mean`。
- **计算未成功**：确认当前场景已有完整最终值，再查看页面提示。
- **持久化异常**：应用会切回示例场景并显示中性提示。
- **无 Key**：AI 页显示“规则兜底模式”。

## 14. 已知限制
- 范围校验是业务保护，不是统计置信区间。
- 用户场景只计算单方案，不参与示例 A/B/C 对比。
- 单用户持久化，未做多用户隔离。
- 示意路网，不是真实 GIS。
- what-if 不自动重跑 A*。
- 外部 LLM 成功路径主要使用模拟响应验证。


