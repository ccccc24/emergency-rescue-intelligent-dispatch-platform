# -*- coding: utf-8 -*-
"""扫描风险代价权重 alpha，并输出 Pareto 曲线与选型建议。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.ml.data_gen import configure_chinese_font, load_config, resolve_project_path
from src.optimization.run import solve_with_ablation


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH: Final[Path] = PROJECT_ROOT / "configs" / "default.yaml"
STORM_SCENARIO: Final[Path] = PROJECT_ROOT / "outputs" / "scenario_2025-08-01.json"
CALM_SCENARIO: Final[Path] = PROJECT_ROOT / "outputs" / "scenario_2025-08-29.json"

REQUIRED_ALPHAS: Final[tuple[float, ...]] = (0.0, 5.0, 10.0, 20.0, 40.0)
CALIBRATED_ALPHAS: Final[tuple[float, ...]] = (
    0.0,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.4,
)

METRIC_COLUMNS: Final[list[str]] = [
    "slowest_vehicle_time",
    "total_distance",
    "total_exposed_risk",
    "high_risk_edge_count",
    "high_risk_edge_visits",
    "priority_demand_satisfied_rate",
    "total_demand_satisfied_rate",
    "solve_time",
    "unserved_demand_total",
]


def run_scan(
    scenario_path: str | Path,
    alphas: tuple[float, ...],
) -> pd.DataFrame:
    """对每个 alpha 单次求解完整 ML 场景，记录指标与完成状态。"""

    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        solution = solve_with_ablation(
            scenario_path,
            "full",
            risk_alpha=alpha,
        )
        metrics = solution.metrics
        rows.append(
            {
                "alpha": float(alpha),
                **{
                    metric: metrics[metric]
                    for metric in METRIC_COLUMNS
                },
                "states_explored": int(metrics["states_explored"]),
                "status": str(metrics["status"]),
            }
        )
    return pd.DataFrame(rows)


def recommend_alpha(frame: pd.DataFrame) -> pd.Series:
    """在完整配送且时间增幅不超过 20% 的候选中选风险降幅最大者。"""

    base = frame.loc[frame["alpha"] == 0.0].iloc[0]
    candidates = frame[
        (frame["alpha"] > 0)
        & (frame["status"] == "success")
        & (frame["total_demand_satisfied_rate"] >= 0.999)
    ].copy()
    if candidates.empty:
        raise RuntimeError("没有满足完整配送约束的 alpha 候选")

    candidates["risk_reduction_pct"] = (
        (float(base["total_exposed_risk"]) - candidates["total_exposed_risk"])
        / float(base["total_exposed_risk"])
        * 100
    )
    candidates["time_change_pct"] = (
        (candidates["slowest_vehicle_time"] - float(base["slowest_vehicle_time"]))
        / float(base["slowest_vehicle_time"])
        * 100
    )
    eligible = candidates[candidates["time_change_pct"] <= 20.0]
    pool = eligible if not eligible.empty else candidates
    return pool.sort_values(
        ["risk_reduction_pct", "alpha"],
        ascending=[False, True],
    ).iloc[0]


def plot_pareto(
    frame: pd.DataFrame,
    output_path: Path,
    title: str,
    recommended_alpha: float | None = None,
) -> None:
    """绘制风险-时间 Pareto 散点图，超时点用叉号标出。"""

    configure_chinese_font()
    figure, axis = plt.subplots(figsize=(9, 6))
    completed_mask = (
        (frame["status"] == "success")
        & (frame["total_demand_satisfied_rate"] >= 0.999)
    )
    completed = frame[completed_mask]
    unfinished = frame[~completed_mask]

    axis.scatter(
        completed["total_exposed_risk"],
        completed["slowest_vehicle_time"],
        s=90,
        color="#4C78A8",
        label="完整配送",
        zorder=3,
    )
    if not unfinished.empty:
        axis.scatter(
            unfinished["total_exposed_risk"],
            unfinished["slowest_vehicle_time"],
            s=90,
            marker="x",
            color="#E45756",
            label="超时/未完成配送",
            zorder=3,
        )

    annotation_offsets = {
        0.0: (8, 8),
        0.01: (8, 6),
        0.02: (8, -15),
        0.05: (-44, 6),
        0.1: (8, 6),
        0.2: (-16, -18),
        0.4: (8, 8),
        5.0: (8, 8),
        10.0: (8, -15),
        20.0: (-42, 8),
        40.0: (-42, -15),
    }
    for row in frame.itertuples(index=False):
        offset = annotation_offsets.get(round(float(row.alpha), 4), (7, 6))
        axis.annotate(
            f"α={row.alpha:g}",
            (row.total_exposed_risk, row.slowest_vehicle_time),
            textcoords="offset points",
            xytext=offset,
            fontsize=9,
        )

    if recommended_alpha is not None:
        recommended = frame.loc[frame["alpha"] == recommended_alpha]
        if not recommended.empty:
            axis.scatter(
                recommended["total_exposed_risk"],
                recommended["slowest_vehicle_time"],
                s=240,
                facecolors="none",
                edgecolors="#54A24B",
                linewidths=2.0,
                label=f"推荐 α={recommended_alpha:g}",
                zorder=4,
            )

    axis.set_xlabel("总暴露风险 Σ(边长 × 风险)")
    axis.set_ylabel("最慢车辆用时（小时）")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _markdown_table(frame: pd.DataFrame) -> str:
    """把扫描结果格式化为紧凑的 Markdown 表。"""

    lines = [
        "| alpha | 最慢车用时 | 总暴露风险 | 高风险边数 | 高风险经过次数 | "
        "总需求满足率 | 未满足需求(箱) | 状态 |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.alpha:g} | {row.slowest_vehicle_time:.3f} | "
            f"{row.total_exposed_risk:.2f} | {int(row.high_risk_edge_count)} | "
            f"{int(row.high_risk_edge_visits)} | "
            f"{row.total_demand_satisfied_rate * 100:.2f}% | "
            f"{row.unserved_demand_total:.2f} | {row.status} |"
        )
    return "\n".join(lines)


def build_docs(
    required: pd.DataFrame,
    calibrated: pd.DataFrame,
    chosen: pd.Series,
    calm_check: pd.DataFrame,
    speed_penalty: float,
) -> str:
    """生成 alpha 扫描与选型说明。"""

    base = calibrated.loc[calibrated["alpha"] == 0.0].iloc[0]
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
    calm_base = calm_check.loc[calm_check["alpha"] == 0.0].iloc[0]
    calm_alpha = calm_check.loc[
        calm_check["alpha"] == float(chosen["alpha"])
    ].iloc[0]
    calm_risk_change = (
        (float(calm_alpha["total_exposed_risk"]) - float(calm_base["total_exposed_risk"]))
        / float(calm_base["total_exposed_risk"])
        * 100
    )
    calm_time_change = (
        (float(calm_alpha["slowest_vehicle_time"]) - float(calm_base["slowest_vehicle_time"]))
        / float(calm_base["slowest_vehicle_time"])
        * 100
    )

    return f"""# 风险代价权重 alpha 扫描

本次扫描固定 `speed_penalty={speed_penalty:g}`、`risk_cost_alpha` 推荐值来自下方 Pareto 选点。

## 1. 结论先行

- 推荐 `risk_cost_alpha = {float(chosen['alpha']):g}`。
- 恶劣天气日相对 `alpha=0`：总暴露风险降低 **{risk_reduction:.2f}%**，最慢车辆用时变化 **{time_change:+.2f}%**，且配送完成率保持 **{float(chosen['total_demand_satisfied_rate']) * 100:.2f}%**。
- 指定区间 `alpha ∈ {{0, 5, 10, 20, 40}}` 的正值全部超时且未完成配送。原因是风险项 `alpha × risk × length` 的量纲与小时级 `edge_time` 不匹配：典型边长 10~20 km、风险 0.3~0.7，alpha=5 会把单边风险代价放大到几十小时。
- 为保证公式仍是 `edge_time + alpha × risk × length`，没有偷偷缩放 alpha，而是增加了一组量纲校准扫描用于选点；指定区间结果原样保留。

## 2. 指定 alpha 区间扫描（恶劣天气日）

{_markdown_table(required)}

所有 `alpha > 0` 的结果都不是完整配送解。它们的“低风险、低用时”来自提前终止后返回的部分路径，不能用于证明风险优化的效果。

## 3. 量纲校准后的扫描（恶劣天气日）

{_markdown_table(calibrated)}

校准范围为 0~0.4：典型单边风险代价与边时间处于同一数量级。`alpha=0.4` 仍然超时，因此推荐点只能落在 0.2 以内。

## 4. Pareto 曲线读法

- 从 `alpha=0` 到 `0.2`：总暴露风险从 {float(base['total_exposed_risk']):.2f} 降到 {float(chosen['total_exposed_risk']):.2f}，降幅 {risk_reduction:.2f}%；最慢车辆用时从 {float(base['slowest_vehicle_time']):.3f} 小时变为 {float(chosen['slowest_vehicle_time']):.3f} 小时，变化 {time_change:+.2f}%。
- 高风险边经过次数从 {int(base['high_risk_edge_visits'])} 次降到 {int(chosen['high_risk_edge_visits'])} 次。
- 继续增大到 `alpha=0.4` 时搜索在 10 秒内无法完成完整配送，说明风险权重已经开始压过配送目标，不能作为部署参数。
- 选点规则：只在“状态 success 且总需求满足率 ≥ 99.9%”的候选中，选择时间增幅不超过 20% 且风险降幅最大的点。推荐 `alpha={float(chosen['alpha']):g}`。

## 5. 常规天气日复核

| alpha | 最慢车用时 | 总暴露风险 | 高风险经过次数 | 总需求满足率 | 状态 |
|---:|---:|---:|---:|---:|---|
| 0 | {float(calm_base['slowest_vehicle_time']):.3f} | {float(calm_base['total_exposed_risk']):.2f} | {int(calm_base['high_risk_edge_visits'])} | {float(calm_base['total_demand_satisfied_rate']) * 100:.2f}% | {calm_base['status']} |
| {float(chosen['alpha']):g} | {float(calm_alpha['slowest_vehicle_time']):.3f} | {float(calm_alpha['total_exposed_risk']):.2f} | {int(calm_alpha['high_risk_edge_visits'])} | {float(calm_alpha['total_demand_satisfied_rate']) * 100:.2f}% | {calm_alpha['status']} |

常规天气日风险暴露变化 **{calm_risk_change:+.2f}%**，最慢车辆用时变化 **{calm_time_change:+.2f}%**。风险普遍较低时，风险代价没有造成明显的时间牺牲。

## 6. 复现方式

```bash
python -m src.experiments.alpha_sweep
```
"""


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="扫描风险代价权重 alpha")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="配置文件路径",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config = load_config(resolve_project_path(arguments.config))
    outputs_path = resolve_project_path(config["paths"]["outputs"])
    docs_path = PROJECT_ROOT / "docs"
    docs_path.mkdir(parents=True, exist_ok=True)

    required = run_scan(STORM_SCENARIO, REQUIRED_ALPHAS)
    calibrated = run_scan(STORM_SCENARIO, CALIBRATED_ALPHAS)
    chosen = recommend_alpha(calibrated)

    calm_check = run_scan(CALM_SCENARIO, (0.0, float(chosen["alpha"])))

    required.to_csv(
        outputs_path / "alpha_sweep_storm.csv",
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )
    calibrated.to_csv(
        outputs_path / "alpha_sweep_storm_calibrated.csv",
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )
    calm_check.to_csv(
        outputs_path / "alpha_calm_check.csv",
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )

    plot_pareto(
        required,
        outputs_path / "pareto_storm.png",
        "恶劣天气日：指定 alpha 区间的风险-时间分布",
    )
    plot_pareto(
        calibrated,
        outputs_path / "pareto_storm_calibrated.png",
        "恶劣天气日：量纲校准后的 Pareto 曲线",
        recommended_alpha=float(chosen["alpha"]),
    )

    (docs_path / "alpha_sweep.md").write_text(
        build_docs(
            required,
            calibrated,
            chosen,
            calm_check,
            float(config["risk"].get("speed_penalty", 0.5)),
        ),
        encoding="utf-8",
    )

    print(required.to_string(index=False))
    print()
    print(calibrated.to_string(index=False))
    print()
    print(calm_check.to_string(index=False))
    print(f"\n推荐 alpha：{float(chosen['alpha']):g}")


if __name__ == "__main__":
    main()
