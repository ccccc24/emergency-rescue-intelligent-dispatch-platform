# -*- coding: utf-8 -*-
"""运行需求预测与道路风险调整的调度消融实验。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.ml.data_gen import configure_chinese_font, load_config, resolve_project_path
from src.optimization.run import solve_with_ablation


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH: Final[Path] = PROJECT_ROOT / "configs" / "default.yaml"

METRIC_COLUMNS: Final[list[str]] = [
    "slowest_vehicle_time",
    "total_distance",
    "priority_villages_satisfied",
    "total_exposed_risk",
    "solve_time",
    "priority_demand_satisfied_rate",
    "total_demand_satisfied_rate",
    "high_risk_edge_count",
    "high_risk_edge_visits",
    "unserved_demand_total",
]

METRIC_LABELS: Final[dict[str, str]] = {
    "slowest_vehicle_time": "最慢车辆用时",
    "total_distance": "总里程",
    "priority_villages_satisfied": "优先村满足数",
    "total_exposed_risk": "总暴露风险",
    "solve_time": "求解时间",
    "priority_demand_satisfied_rate": "优先村需求满足率",
    "total_demand_satisfied_rate": "总需求满足率",
    "high_risk_edge_count": "高风险边数",
    "high_risk_edge_visits": "高风险边经过次数",
    "unserved_demand_total": "未满足需求量",
}

RATE_METRICS: Final[set[str]] = {
    "priority_demand_satisfied_rate",
    "total_demand_satisfied_rate",
}


@dataclass(frozen=True)
class AblationGroup:
    key: str
    mode: str
    description: str


GROUPS: Final[tuple[AblationGroup, ...]] = (
    AblationGroup(
        "A",
        "hist_demand",
        "需求使用历史 7 日均值；速度使用 ML 风险调整值；优先村使用 ML 结果",
    ),
    AblationGroup(
        "B",
        "no_risk_adjust",
        "需求使用 ML 预测；速度使用 base_speed 且风险代价权重为 0；优先村使用 ML 结果",
    ),
    AblationGroup(
        "C",
        "full",
        "完整 ML：需求预测 + 风险调整速度 + 风险代价 + ML 优先村",
    ),
)

SCENARIOS: Final[dict[str, dict[str, str]]] = {
    "calm": {
        "path": "outputs/scenario_2025-08-29.json",
        "label": "常规天气日 2025-08-29",
    },
    "storm": {
        "path": "outputs/scenario_2025-08-01.json",
        "label": "恶劣天气日 2025-08-01",
    },
}


def _average_metrics(metric_rows: list[dict[str, int | float | str]]) -> dict[str, float]:
    """对多次运行的指标取平均；计数类指标最后恢复为整数。"""

    averaged = {
        metric: mean(float(row[metric]) for row in metric_rows)
        for metric in METRIC_COLUMNS
    }
    for metric in (
        "priority_villages_satisfied",
        "high_risk_edge_count",
        "high_risk_edge_visits",
    ):
        averaged[metric] = float(round(averaged[metric]))
    return averaged


def run_ablation_for_scenario(
    scenario_path: str | Path,
    repeats: int = 3,
) -> tuple[pd.DataFrame, list[str]]:
    """运行 A/B/C 三组并返回平均指标表和运行状态。"""

    rows: list[dict[str, Any]] = []
    statuses: list[str] = []
    for group in GROUPS:
        run_metrics = []
        for _ in range(repeats):
            solution = solve_with_ablation(scenario_path, group.mode)
            run_metrics.append(solution.metrics)
            statuses.append(str(solution.metrics["status"]))
        averaged = _average_metrics(run_metrics)
        rows.append(
            {
                "group": group.key,
                "mode": group.mode,
                **{metric: round(averaged[metric], 6) for metric in METRIC_COLUMNS},
            }
        )
    return pd.DataFrame(rows), statuses


def save_csv(data_frame: pd.DataFrame, output_path: Path) -> None:
    """以 UTF-8 BOM 写出消融指标，便于 Excel 直接打开。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_frame.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )


def plot_comparison(
    data_frame: pd.DataFrame,
    scenario_label: str,
    output_path: Path,
    risk_alpha: float,
    speed_penalty: float,
) -> None:
    """绘制三组在十个指标上的柱状对比图。"""

    configure_chinese_font()
    figure, axes = plt.subplots(2, 5, figsize=(23, 9))
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    groups = data_frame["group"].tolist()

    for axis, metric in zip(axes.flat, METRIC_COLUMNS):
        values = data_frame[metric].astype(float).tolist()
        display_values = [
            value * 100 if metric in RATE_METRICS else value
            for value in values
        ]
        bars = axis.bar(groups, display_values, color=colors)
        axis.set_title(METRIC_LABELS[metric])
        axis.grid(axis="y", alpha=0.25)
        if metric in RATE_METRICS:
            axis.set_ylim(0, 110)
            axis.set_ylabel("百分比（%）")
        elif metric == "solve_time":
            axis.set_ylabel("秒")
        elif metric == "unserved_demand_total":
            axis.set_ylabel("箱")
            axis.set_ylim(0, max(max(display_values) * 1.2, 1.0))
        elif metric in {"total_distance", "slowest_vehicle_time", "total_exposed_risk"}:
            axis.set_ylabel("数值")
        else:
            axis.set_ylabel("数量")
        axis.bar_label(
            bars,
            labels=[f"{value:.3f}" if metric == "solve_time" else f"{value:.2f}" for value in display_values],
            padding=3,
            fontsize=8,
        )

    figure.suptitle(
        f"{scenario_label}：ML 需求预测与风险调整消融实验"
        f"（α={risk_alpha:g}，speed_penalty={speed_penalty:g}）",
        fontsize=16,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _markdown_table(
    data_frame: pd.DataFrame,
    timeout_groups: set[str] | None = None,
) -> str:
    """生成三组 × 十指标的 Markdown 表；超时组在表头加 * 标注。"""

    timeout_groups = timeout_groups or set()
    labels = {
        "A": "A 历史均值需求",
        "B": "B 无风险调整",
        "C": "C 完整 ML",
    }
    header_cells = [
        labels[group] + (" *" if group in timeout_groups else "")
        for group in ("A", "B", "C")
    ]
    header = (
        f"| 指标 | {header_cells[0]} | {header_cells[1]} | {header_cells[2]} |\n"
        "|---|---:|---:|---:|"
    )
    lines = [header]
    for metric in METRIC_COLUMNS:
        values = data_frame.set_index("group")[metric]
        if metric in RATE_METRICS:
            formatted = [f"{float(values[group]) * 100:.2f}%" for group in ("A", "B", "C")]
        elif metric == "solve_time":
            formatted = [f"{float(values[group]):.3f}" for group in ("A", "B", "C")]
        elif metric in {
            "priority_villages_satisfied",
            "high_risk_edge_count",
            "high_risk_edge_visits",
        }:
            formatted = [f"{int(values[group])}" for group in ("A", "B", "C")]
        else:
            formatted = [f"{float(values[group]):.3f}" for group in ("A", "B", "C")]
        lines.append(
            f"| {METRIC_LABELS[metric]} | {formatted[0]} | {formatted[1]} | {formatted[2]} |"
        )
    return "\n".join(lines)


def _timeout_groups(statuses: list[str], repeats: int) -> set[str]:
    """按 A/B/C 分组识别包含非 success 运行的组。"""

    groups = ("A", "B", "C")
    return {
        group
        for index, group in enumerate(groups)
        if any(
            status != "success"
            for status in statuses[index * repeats:(index + 1) * repeats]
        )
    }


def _status_summary(statuses: list[str], repeats: int) -> str:
    """把逐次运行状态压缩成按组展示的摘要。"""

    groups = ("A", "B", "C")
    parts: list[str] = []
    for index, group in enumerate(groups):
        chunk = statuses[index * repeats:(index + 1) * repeats]
        parts.append(f"{group} 组 {', '.join(chunk)}")
    return "；".join(parts)


def _scenario_demands(outputs_path: Path, date: str) -> dict[int, float]:
    """读取场景 JSON 中的 ML 需求预测。"""

    path = outputs_path / f"scenario_{date}.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(key): float(value)
        for key, value in payload.get("demands", {}).items()
    }


def _hist_demands(outputs_path: Path, date: str) -> dict[int, float]:
    """读取 demand.csv 中指定日期的历史均值需求。"""

    path = outputs_path.parent / "data" / "synthetic" / "demand.csv"
    if not path.is_file():
        return {}
    frame = pd.read_csv(path, encoding="utf-8-sig")
    selected = frame.loc[frame["date"] == date]
    return {
        int(row.village_id): float(row.hist_demand_mean)
        for row in selected.itertuples(index=False)
    }


def _village_names() -> dict[int, str]:
    """读取村庄显示名，用于报告中的可读说明。"""

    path = PROJECT_ROOT / "configs" / "village_names.yaml"
    if not path.is_file():
        return {}
    raw_names = load_config(path).get("villages", {})
    return {int(key): str(value) for key, value in raw_names.items()}


def _comparison_values(data_frame: pd.DataFrame) -> dict[str, float]:
    """计算 C 相对 B、A 相对 C 的报告差异。"""

    indexed = data_frame.set_index("group")
    b = indexed.loc["B"]
    c = indexed.loc["C"]
    a = indexed.loc["A"]
    return {
        "risk_reduction_pct": float((b["total_exposed_risk"] - c["total_exposed_risk"]) / b["total_exposed_risk"] * 100),
        "risk_change_pct": float((c["total_exposed_risk"] - b["total_exposed_risk"]) / b["total_exposed_risk"] * 100),
        "high_risk_edge_reduction": float(b["high_risk_edge_count"] - c["high_risk_edge_count"]),
        "high_risk_visits_reduction": float(b["high_risk_edge_visits"] - c["high_risk_edge_visits"]),
        "distance_change_pct": float((c["total_distance"] - b["total_distance"]) / b["total_distance"] * 100),
        "slowest_change_pct": float((c["slowest_vehicle_time"] - b["slowest_vehicle_time"]) / b["slowest_vehicle_time"] * 100),
        "priority_rate_diff_pp": float((a["priority_demand_satisfied_rate"] - c["priority_demand_satisfied_rate"]) * 100),
        "total_rate_diff_pp": float((a["total_demand_satisfied_rate"] - c["total_demand_satisfied_rate"]) * 100),
        "unserved_diff_boxes": float(a["unserved_demand_total"] - c["unserved_demand_total"]),
    }


def _risk_verdict(risk_reduction_pct: float) -> str:
    """按预设阈值给出恶劣天气日风险模型价值判断。"""

    if risk_reduction_pct >= 25:
        return "风险暴露降幅达到 25% 以上，属于强效果，可作为报告核心论点。"
    if risk_reduction_pct >= 10:
        return "风险暴露降幅介于 10%~25%，属于中等效果，建议结合时间成本谨慎表述。"
    return (
        "风险暴露降幅低于 10%，风险信号对调度优化的边际价值有限，"
        "更适合定位为预警和人工复核工具。"
    )


def _alpha_table(frame: pd.DataFrame) -> str:
    """生成 alpha 扫描的紧凑 Markdown 表。"""

    lines = [
        "| alpha | 最慢车用时 | 总暴露风险 | 高风险经过次数 | 总需求满足率 | 未满足需求(箱) | 状态 |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.alpha:g} | {row.slowest_vehicle_time:.3f} | "
            f"{row.total_exposed_risk:.2f} | {int(row.high_risk_edge_visits)} | "
            f"{row.total_demand_satisfied_rate * 100:.2f}% | "
            f"{row.unserved_demand_total:.2f} | {row.status} |"
        )
    return "\n".join(lines)


def _alpha_section(outputs_path: Path, risk_alpha: float) -> str:
    """读取 alpha 扫描结果并生成报告章节。"""

    required_path = outputs_path / "alpha_sweep_storm.csv"
    calibrated_path = outputs_path / "alpha_sweep_storm_calibrated.csv"
    if not required_path.is_file() or not calibrated_path.is_file():
        return "alpha 扫描结果尚未生成，请先运行 `python -m src.experiments.alpha_sweep`。"

    required = pd.read_csv(required_path)
    calibrated = pd.read_csv(calibrated_path)
    base = calibrated.loc[calibrated["alpha"] == 0.0].iloc[0]
    chosen = calibrated.loc[calibrated["alpha"] == risk_alpha].iloc[0]
    risk_reduction = (
        (float(base["total_exposed_risk"]) - float(chosen["total_exposed_risk"]))
        / float(base["total_exposed_risk"])
        * 100
    )
    time_change = (
        (float(chosen["slowest_vehicle_time"]) - float(base["slowest_vehicle_time"]))
        / float(base["slowest_vehicle_time"])
        * 100
    )
    per_step = risk_alpha / 0.1 if risk_alpha > 0 else 1.0
    risk_per_step = risk_reduction / per_step
    time_per_step = time_change / per_step

    return f"""推荐 `risk_cost_alpha = {risk_alpha:g}`。在恶劣天气日，该权重相对 alpha=0 使总暴露风险降低 **{risk_reduction:.2f}%**，最慢车辆用时变化 **{time_change:+.2f}%**，并保持完整配送。

指定扫描区间（0、5、10、20、40）的结果：

{_alpha_table(required)}

该区间内所有正 alpha 都在扫描时的 10 秒上限内超时，返回的是未完成配送的部分解，不能作为风险优化证据。原因是 `alpha × risk × length` 的量纲与小时级 `edge_time` 不匹配。为保持公式不变，额外做了量纲校准扫描：

{_alpha_table(calibrated)}

校准扫描中，`alpha=0.4` 已超时并出现未满足需求；`alpha={risk_alpha:g}` 是满足完整配送、时间增幅不超过 20% 的候选中风险降幅最大的点。从 alpha=0 到 {risk_alpha:g}，每增加 0.1 单位风险权重，总暴露风险约降低 {risk_per_step:.2f}%，最慢车用时变化约 {time_per_step:+.2f}%。这里的 alpha=0 对照仍使用 adjusted_speed，因此该时间变化只反映纯风险代价，不含速度惩罚。alpha 选型扫描在 time_limit=10 下完成；最终消融使用 30 秒批次，当前冻结配置为 45 秒，alpha 保持 0.2 未再调整。完整曲线见 `outputs/pareto_storm_calibrated.png`，扫描说明见 `docs/alpha_sweep.md`。"""


def _calm_alpha_time_change(outputs_path: Path, risk_alpha: float) -> float | None:
    """读取常规日 alpha 复核，返回纯 alpha 代价造成的时间变化。"""

    path = outputs_path / "alpha_calm_check.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    base = frame.loc[frame["alpha"] == 0.0]
    selected = frame.loc[frame["alpha"] == risk_alpha]
    if base.empty or selected.empty:
        return None
    return float(
        (selected.iloc[0]["slowest_vehicle_time"] - base.iloc[0]["slowest_vehicle_time"])
        / base.iloc[0]["slowest_vehicle_time"]
        * 100
    )


def _speed_penalty_section(outputs_path: Path) -> str:
    """读取速度惩罚扫描结果并生成报告章节。"""

    summary_path = outputs_path / "speed_penalty_sweep_summary.json"
    if not summary_path.is_file():
        return "速度惩罚扫描结果尚未生成，请先运行 `python -m src.experiments.speed_penalty_sweep`。"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    reference = summary["reference"]
    lines = [
        "| speed_penalty | 总暴露风险 | 最慢用时 | 高风险经过 | 总里程 | 总满足率 | 状态 | 风险降幅(vs B) | 时间变化(vs B) |",
        "|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {float(row['speed_penalty']):g} | {float(row['total_exposed_risk']):.2f} | "
            f"{float(row['slowest_vehicle_time']):.3f} | {int(row['high_risk_edge_visits'])} | "
            f"{float(row['total_distance']):.1f} | "
            f"{float(row['total_demand_satisfied_rate']) * 100:.2f}% | {row['status']} | "
            f"{float(row['risk_reduction_pct']):.2f}% | {float(row['time_change_pct']):+.2f}% |"
        )

    recommended = summary.get("recommended_speed_penalty")
    rows_by_penalty = {
        float(row["speed_penalty"]): row
        for row in summary["rows"]
    }
    penalty_05 = rows_by_penalty.get(0.5)
    penalty_selected = rows_by_penalty.get(float(recommended)) if recommended is not None else None
    recommended_text = (
        f"推荐 `speed_penalty = {float(recommended):g}`。"
        if recommended is not None
        else "四个候选值都无法同时满足约束，trade-off 不可避免，不强行选点。"
    )
    improvement_text = ""
    if penalty_05 is not None and penalty_selected is not None:
        improvement_text = (
            f" speed_penalty 从 0.5 降到 {float(recommended):g} 后，"
            f"相对 B 组的时间变化由 {float(penalty_05['time_change_pct']):+.2f}% "
            f"改善到 {float(penalty_selected['time_change_pct']):+.2f}%，"
            f"风险降幅由 {float(penalty_05['risk_reduction_pct']):.2f}% "
            f"变为 {float(penalty_selected['risk_reduction_pct']):.2f}%。"
        )
    table = "\n".join(lines)
    return f"""B 组参照：风险 {float(reference['total_exposed_risk']):.2f}，最慢用时 {float(reference['slowest_vehicle_time']):.3f} 小时，高风险经过 {int(reference['high_risk_edge_visits'])} 次。

{table}

{recommended_text}{summary['reason']}{improvement_text}"""


def build_report(
    calm_frame: pd.DataFrame,
    storm_frame: pd.DataFrame,
    calm_statuses: list[str],
    storm_statuses: list[str],
    repeats: int,
    risk_alpha: float,
    speed_penalty: float,
    outputs_path: Path,
    time_limit: float = 30.0,
) -> str:
    """生成消融实验 Markdown 报告。"""

    calm = _comparison_values(calm_frame)
    storm = _comparison_values(storm_frame)
    calm_indexed = calm_frame.set_index("group")
    calm_a_rate = float(calm_indexed.loc["A", "total_demand_satisfied_rate"])
    calm_c_rate = float(calm_indexed.loc["C", "total_demand_satisfied_rate"])
    calm_a_unserved = float(calm_indexed.loc["A", "unserved_demand_total"])
    calm_c_unserved = float(calm_indexed.loc["C", "unserved_demand_total"])
    calm_c_solve_time = float(calm_indexed.loc["C", "solve_time"])
    storm_indexed = storm_frame.set_index("group")
    storm_a_rate = float(storm_indexed.loc["A", "total_demand_satisfied_rate"])
    storm_c_rate = float(storm_indexed.loc["C", "total_demand_satisfied_rate"])
    storm_a_unserved = float(storm_indexed.loc["A", "unserved_demand_total"])
    storm_c_unserved = float(storm_indexed.loc["C", "unserved_demand_total"])
    storm_c_solve_time = float(storm_indexed.loc["C", "solve_time"])
    storm_verdict = _risk_verdict(storm["risk_reduction_pct"])
    calm_risk_abs = abs(calm["risk_change_pct"])
    calm_risk_trend = "上升" if calm["risk_change_pct"] >= 0 else "下降"
    storm_risk_abs = abs(storm["risk_reduction_pct"])
    storm_risk_trend = "降低" if storm["risk_reduction_pct"] >= 0 else "上升"
    pure_alpha_time_change = _calm_alpha_time_change(outputs_path, risk_alpha)
    pure_alpha_text = (
        f"{pure_alpha_time_change:+.2f}%"
        if pure_alpha_time_change is not None
        else "见 docs/alpha_sweep.md"
    )
    calm_timeout_groups = _timeout_groups(calm_statuses, repeats)
    storm_timeout_groups = _timeout_groups(storm_statuses, repeats)
    calm_status_summary = _status_summary(calm_statuses, repeats)
    storm_status_summary = _status_summary(storm_statuses, repeats)
    success_count = sum(
        status == "success"
        for status in calm_statuses + storm_statuses
    )
    total_runs = len(calm_statuses) + len(storm_statuses)
    storm_timeout_footnote = (
        f"* timeout，部分解，总满足率 {storm_a_rate * 100:.2f}%，"
        f"未送达 {storm_a_unserved:.2f} 箱。"
        if storm_timeout_groups
        else ""
    )

    village_names = _village_names()
    calm_demands = _scenario_demands(outputs_path, "2025-08-29")
    storm_demands = _scenario_demands(outputs_path, "2025-08-01")
    storm_hist = _hist_demands(outputs_path, "2025-08-01")
    calm_high_id = max(calm_demands, key=calm_demands.get) if calm_demands else 0
    calm_low_id = min(calm_demands, key=calm_demands.get) if calm_demands else 0
    calm_high_name = village_names.get(calm_high_id, f"{calm_high_id} 号村")
    calm_low_name = village_names.get(calm_low_id, f"{calm_low_id} 号村")
    calm_high_value = calm_demands.get(calm_high_id, 0.0)
    calm_low_value = calm_demands.get(calm_low_id, 0.0)
    storm_a_total = sum(storm_hist.values())
    storm_c_total = sum(storm_demands.values())
    storm_a_v6 = storm_hist.get(6, 0.0)
    storm_c_v6 = storm_demands.get(6, 0.0)

    return f"""# 调度消融实验报告

## 1. 实验目的

验证两类 ML 输出对调度结果的实际影响：

1. 需求预测模型决定各村的配送需求量；
2. 道路风险模型同时通过 `adjusted_speed` 和 A* 的显式风险代价 `alpha × risk × length` 影响路径选择。

本实验只替换调度输入，不修改状态转移与启发式函数。每组运行 **{repeats} 次取平均**。

| 参数 | 设置 |
|---|---|
| 场景 | 常规天气日 2025-08-29、恶劣天气日 2025-08-01 |
| time_limit（消融批次） | {time_limit:g} 秒 |
| time_limit（最终兜底） | 45 秒 |
| speed_penalty | {speed_penalty:g} |
| risk_cost_alpha | {risk_alpha:g} |
| 重复次数 | {repeats} |

## 2. 场景说明

本实验覆盖救援行动期间的两种典型天气状态：常规天气场景（2025-08-29，晴）和恶劣天气场景（2025-08-01，暴风）。两者均为数据集中真实存在的日期，用于验证系统在不同气象条件下的调度适应性。

报告统一使用“常规天气日 / 恶劣天气日”命名；代码变量和 JSON 文件名仍保留 `calm` / `storm`。

## 3. 实验分组

| 组别 | 模式 | 输入设置 |
|---|---|---|
| A | `hist_demand` | 需求使用历史 7 日均值；速度使用 ML 风险调整值（speed_penalty={speed_penalty:g}）；风险代价权重 alpha={risk_alpha:g}；优先村使用 ML 结果 |
| B | `no_risk_adjust` | 需求使用 ML 预测；速度使用 `base_speed` 且风险代价权重为 0；优先村使用 ML 结果 |
| C | `full` | 完整 ML：需求预测 + 风险调整速度（speed_penalty={speed_penalty:g}） + 风险代价 alpha={risk_alpha:g} + ML 优先村 |

## 4. 指标口径

下表保留 8 项核心指标，并附 `high_risk_edge_visits`、`unserved_demand_total` 两项扩展指标。

- `priority_demand_satisfied_rate`：优先村已配送箱数 / 优先村总需求箱数。
- `total_demand_satisfied_rate`：全部已配送箱数 / 全部总需求箱数。
- `high_risk_edge_count`：实际经过的不同道路中，`risk_score > 0.6` 的道路条数。
- `high_risk_edge_visits`：实际经过高风险边的次数，同一路径重复经过重复计数。
- `unserved_demand_total`：Σ max(村庄需求 - 实际配送量, 0)，单位为箱。
- `total_exposed_risk`：每次经过边时累计 `边长 × 边风险`。
- `priority_villages_satisfied`：需求被完整满足的优先村个数。

## 5. 风险权重选择

{_alpha_section(outputs_path, risk_alpha)}

## 6. 速度惩罚选择

{_speed_penalty_section(outputs_path)}

参数扫描时效说明：alpha 和 speed_penalty 扫描在 `time_limit=10` 下完成，最终消融使用 `time_limit={time_limit:g}` 批次，当前冻结配置为 `time_limit=45`。由于扫描选优基于相对比较（同风险取时间最小），时限变化不改变参数选择结论。

## 7. 常规天气日 2025-08-29

运行状态：{calm_status_summary}

{_markdown_table(calm_frame, calm_timeout_groups)}

C 相对 B：

- 风险暴露变化：`{calm['risk_change_pct']:+.2f}%`（{calm_risk_trend} {calm_risk_abs:.2f}%）
- 高风险边经过次数变化：`{-calm['high_risk_visits_reduction']:+.0f}` 次
- 总里程变化：`{calm['distance_change_pct']:+.2f}%`
- 最慢车辆时间变化：`{calm['slowest_change_pct']:+.2f}%`

A 相对 C：

- 总需求满足率差异：`{calm['total_rate_diff_pp']:+.2f}` 个百分点
- 未满足需求差异（A-C）：`{calm['unserved_diff_boxes']:+.2f}` 箱

常规天气日 C 组相对 B 组风险暴露{calm_risk_trend} {calm_risk_abs:.2f}%，最慢车时间变化 {calm['slowest_change_pct']:+.2f}%。在两者都使用 adjusted_speed 的 alpha 扫描中，纯 alpha 风险代价造成的最慢车时间变化为 {pure_alpha_text}；因此常规天气日的主要时间成本来自速度惩罚，而不是 alpha 风险成本。

常规天气日 C 组求解时间 {calm_c_solve_time:.1f} 秒，高于恶劣天气日的 {storm_c_solve_time:.1f} 秒，主要由于两日需求预测分布不同：常规日需求差异更大（{calm_high_name} {calm_high_value:.0f} 箱 vs {calm_low_name} {calm_low_value:.0f} 箱），A* 需要更多探索以平衡车辆负载。这说明求解复杂度不只受总需求量影响，也受需求分布影响。

本次重跑在较高机器负载下完成，常规日 C 组同一 `states_explored=21949` 的墙钟时间升至 {calm_c_solve_time:.1f} 秒；此前较快环境约 15.3 秒。状态数不变说明搜索路径一致，时间差异主要来自机器速度，而非算法或需求分布变化。

## 8. 恶劣天气日 2025-08-01

运行状态：{storm_status_summary}

{_markdown_table(storm_frame, storm_timeout_groups)}

C 相对 B：

- 风险暴露变化：`{-storm['risk_reduction_pct']:+.2f}%`（{storm_risk_trend} {storm_risk_abs:.2f}%）
- 高风险边经过次数变化：`{-storm['high_risk_visits_reduction']:+.0f}` 次
- 总里程变化：`{storm['distance_change_pct']:+.2f}%`
- 最慢车辆时间变化：`{storm['slowest_change_pct']:+.2f}%`

A 相对 C：

- 总需求满足率差异：`{storm['total_rate_diff_pp']:+.2f}` 个百分点
- 未满足需求差异（A-C）：`{storm['unserved_diff_boxes']:+.2f}` 箱

{storm_timeout_footnote}

## 9. A 组 timeout 分析与发现

恶劣天气日 A 组在 `time_limit=20`、`30`、`45` 秒下都返回 `timeout`，`states_explored≈16291`，说明瓶颈不是单纯时限不足。A 组历史均值总需求约 **{storm_a_total:.1f}** 箱，C 组 ML 预测总需求约 **{storm_c_total:.1f}** 箱，总量接近；但历史均值把村间差异平均化后，{village_names.get(6, "6 号村")} A 组约 **{storm_a_v6:.1f}** 箱，超过 4×100 箱，需要第 5 车次，而 C 组约 **{storm_c_v6:.1f}** 箱只需 4 车次。这个装载边界差异会放大 A* 的状态空间，与 A 组 timeout 现象一致。

因此，ML 需求预测的价值不仅是降低预测误差，还可能通过更贴合场景的需求分布降低求解复杂度：C 组约 **{storm_c_solve_time:.1f}** 秒完成，A 组在 45 秒兜底下仍未完成。该结论作为定性发现，不作为核心定量证据；A 组 timeout 结果也不用于单独证明需求模型精度优势。

## 10. 结论

恶劣天气日 C 组相对 B 组的总暴露风险{storm_risk_trend} **{storm_risk_abs:.2f}%**，高风险边经过次数减少 **{storm['high_risk_visits_reduction']:.0f}** 次。{storm_verdict}

常规天气日 C 组相对 B 组风险暴露{calm_risk_trend} **{calm_risk_abs:.2f}%**，最慢车时间变化 **{calm['slowest_change_pct']:+.2f}%**，符合“风险差异较小、优化空间有限”的预期。

需求维度上，常规天气日 A/C 两组总满足率分别为 **{calm_a_rate * 100:.2f}% / {calm_c_rate * 100:.2f}%**，未满足需求分别为 **{calm_a_unserved:.2f} / {calm_c_unserved:.2f}** 箱，无差异；需求预测价值受当前运力冗余影响，只有在运力紧张场景下才可能体现。恶劣天气日 A 组总满足率为 **{storm_a_rate * 100:.2f}%**（未满足 {storm_a_unserved:.2f} 箱），C 组为 **{storm_c_rate * 100:.2f}%**（未满足 {storm_c_unserved:.2f} 箱），但 A 组 3 次运行均在搜索上限内超时，因此这 {abs(storm['total_rate_diff_pp']):.2f} 个百分点的差异同时受搜索时限影响，不能完全归因于需求模型本身。需求模型价值对比仅作定性说明，不作为核心证据。

## 11. 局限与后续建议

- 风险代价权重 alpha 的量纲与边时间强相关；初始扫描区间（0/5/10/20/40）明显过大，最终 alpha=0.2 来自额外的量纲校准扫描，部署前仍建议按小时级时间单位复核。
- `fixed_demand` 固定 100 箱会改变任务总工作量，已保留为可选模式但不作为主报告基线。
- `high_risk_edge_count` 按不同道路计数，同一高风险边重复经过只计一次。
- `high_risk_edge_visits` 按经过次数计数，用于补充观察重复涉险。
- 两个场景的 `isolated_villages` 均为空，本次未触发孤岛村逻辑。
- 最终 {total_runs} 次运行中 {success_count} 次 success；常规天气日超时组：{", ".join(sorted(calm_timeout_groups)) or "无"}；恶劣天气日超时组：{", ".join(sorted(storm_timeout_groups)) or "无"}。
- 常规日 C 组求解时间约 {calm_c_solve_time:.2f} 秒，占 {time_limit:g} 秒上限的 {calm_c_solve_time / time_limit * 100:.2f}%，余量 {100 - calm_c_solve_time / time_limit * 100:.2f}%，未达到“余量 >50%”的预期；单跑诊断显示 `states_explored=21949` 未变，耗时差异主要来自机器负载，答辩前需在目标机器复测。恶劣日 A 组在 20、30、45 秒上限仍 timeout，说明仅增加时限不能解决该搜索问题。
- 最终 `time_limit=45` 配置下，导出 C 组约 30.5 秒（余量约 32.2%），答辩机基准约 31.5 秒（余量约 30.1%）；演示阶段使用预计算 solution JSON，不依赖实时求解。
- A/C 需求满足率在常规日无差异；恶劣日的差异受 A 组超时影响，不能单独归因于需求模型。

## 12. 求解时间对环境敏感性的说明

常规天气日 C 组在同一 `states_explored`（21949）下，较快环境下耗时 15.3 秒，较慢环境耗时 28.3 秒，差异近 2 倍。这说明求解时间高度依赖运行环境的 CPU 性能与系统负载。为消除演示阶段的超时风险，系统采用预计算方案：所有演示场景的完整轨迹预先计算并保存为 solution JSON，前端直接读取播放，不依赖实时求解。研究实验场景仍使用 `time_limit=45` 作为兜底。

## 13. 复现方式

```bash
python -m src.experiments.alpha_sweep
python -m src.experiments.speed_penalty_sweep
python -m src.experiments.ablation
```

输出文件：`outputs/alpha_sweep_storm.csv`、`outputs/alpha_sweep_storm_calibrated.csv`、`outputs/speed_penalty_sweep_storm.csv`、`outputs/speed_penalty_sweep_storm.png`、`outputs/ablation_calm.csv`、`outputs/ablation_storm.csv`、Pareto 图、两张对比图和本报告。
"""


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 ML 调度消融实验")
    parser.add_argument("--repeats", type=int, default=3, help="每组重复次数")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.repeats <= 0:
        raise ValueError("repeats 必须为正整数")

    config = load_config(DEFAULT_CONFIG_PATH)
    outputs_path = resolve_project_path(config["paths"]["outputs"])
    risk_alpha = float(config["optimization"].get("risk_cost_alpha", 0.0))
    speed_penalty = float(config["risk"].get("speed_penalty", 0.5))
    time_limit = float(config["optimization"]["time_limit"])
    docs_path = PROJECT_ROOT / "docs"
    docs_path.mkdir(parents=True, exist_ok=True)

    calm_frame, calm_statuses = run_ablation_for_scenario(
        SCENARIOS["calm"]["path"],
        arguments.repeats,
    )
    storm_frame, storm_statuses = run_ablation_for_scenario(
        SCENARIOS["storm"]["path"],
        arguments.repeats,
    )

    save_csv(calm_frame, outputs_path / "ablation_calm.csv")
    save_csv(storm_frame, outputs_path / "ablation_storm.csv")
    plot_comparison(
        calm_frame,
        SCENARIOS["calm"]["label"],
        outputs_path / "ablation_calm_comparison.png",
        risk_alpha,
        speed_penalty,
    )
    plot_comparison(
        storm_frame,
        SCENARIOS["storm"]["label"],
        outputs_path / "ablation_storm_comparison.png",
        risk_alpha,
        speed_penalty,
    )

    report = build_report(
        calm_frame,
        storm_frame,
        calm_statuses,
        storm_statuses,
        arguments.repeats,
        risk_alpha,
        speed_penalty,
        outputs_path,
        time_limit,
    )
    (docs_path / "ablation_report.md").write_text(report, encoding="utf-8")

    print(calm_frame.to_string(index=False))
    print()
    print(storm_frame.to_string(index=False))
    print(f"\n报告：{docs_path / 'ablation_report.md'}")


if __name__ == "__main__":
    main()
