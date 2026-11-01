# -*- coding: utf-8 -*-
"""扫描道路风险速度惩罚系数，并输出安全-时间权衡结果。"""

from __future__ import annotations

import argparse
import json
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
PENALTIES: Final[tuple[float, ...]] = (0.5, 0.3, 0.2, 0.1)

METRIC_COLUMNS: Final[list[str]] = [
    "total_exposed_risk",
    "slowest_vehicle_time",
    "high_risk_edge_visits",
    "total_distance",
    "total_demand_satisfied_rate",
    "status",
]


def _select_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """提取扫描表所需的六个指标。"""

    return {metric: metrics[metric] for metric in METRIC_COLUMNS}


def run_sweep(risk_alpha: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    """运行 B 组参照和四个速度惩罚下的 C 组。"""

    reference_solution = solve_with_ablation(
        STORM_SCENARIO,
        "no_risk_adjust",
        risk_alpha=risk_alpha,
        speed_penalty=0.5,
    )
    reference = _select_metrics(reference_solution.metrics)

    rows: list[dict[str, Any]] = []
    for penalty in PENALTIES:
        solution = solve_with_ablation(
            STORM_SCENARIO,
            "full",
            risk_alpha=risk_alpha,
            speed_penalty=penalty,
        )
        metrics = _select_metrics(solution.metrics)
        risk_reduction_pct = (
            (float(reference["total_exposed_risk"]) - float(metrics["total_exposed_risk"]))
            / float(reference["total_exposed_risk"])
            * 100
        )
        time_change_pct = (
            (float(metrics["slowest_vehicle_time"]) - float(reference["slowest_vehicle_time"]))
            / float(reference["slowest_vehicle_time"])
            * 100
        )
        rows.append(
            {
                "speed_penalty": float(penalty),
                **metrics,
                "risk_reduction_pct": risk_reduction_pct,
                "time_change_pct": time_change_pct,
            }
        )

    frame = pd.DataFrame(rows)
    eligible = frame[
        (frame["risk_reduction_pct"] >= 20.0)
        & (frame["time_change_pct"] <= 10.0)
        & (frame["status"] == "success")
        & (frame["total_demand_satisfied_rate"] >= 0.999)
    ].copy()

    if eligible.empty:
        recommended = None
        reason = (
            "四个 speed_penalty 都无法同时满足风险降幅 ≥ 20% 且时间增幅 ≤ 10%，"
            "trade-off 不可避免。"
        )
    else:
        chosen = eligible.sort_values(
            ["risk_reduction_pct", "speed_penalty"],
            ascending=[False, True],
        ).iloc[0]
        recommended = float(chosen["speed_penalty"])
        reason = (
            f"speed_penalty={recommended:g} 满足风险降幅 ≥ 20%、时间增幅 ≤ 10%，"
            "且在合格候选中风险降幅最大。"
        )

    summary = {
        "scenario": "2025-08-01",
        "risk_alpha": float(risk_alpha),
        "reference_group": "B_no_risk_adjust",
        "reference": reference,
        "constraints": {
            "min_risk_reduction_pct": 20.0,
            "max_time_increase_pct": 10.0,
        },
        "recommended_speed_penalty": recommended,
        "reason": reason,
        "rows": frame.to_dict(orient="records"),
    }
    return frame, summary


def plot_sweep(
    frame: pd.DataFrame,
    reference: dict[str, Any],
    output_path: Path,
    recommended: float | None,
) -> None:
    """绘制速度惩罚与风险/时间的双 Y 轴关系。"""

    configure_chinese_font()
    ordered = frame.sort_values("speed_penalty")
    figure, left_axis = plt.subplots(figsize=(9, 6))
    right_axis = left_axis.twinx()

    risk_line = left_axis.plot(
        ordered["speed_penalty"],
        ordered["total_exposed_risk"],
        marker="o",
        linewidth=2,
        color="#4C78A8",
        label="总暴露风险",
    )[0]
    time_line = right_axis.plot(
        ordered["speed_penalty"],
        ordered["slowest_vehicle_time"],
        marker="s",
        linewidth=2,
        color="#F58518",
        label="最慢车辆用时",
    )[0]

    left_axis.axhline(
        float(reference["total_exposed_risk"]),
        color="#4C78A8",
        linestyle="--",
        linewidth=1.2,
        alpha=0.6,
        label="B 组暴露风险",
    )
    right_axis.axhline(
        float(reference["slowest_vehicle_time"]),
        color="#F58518",
        linestyle="--",
        linewidth=1.2,
        alpha=0.6,
        label="B 组最慢用时",
    )

    if recommended is not None:
        selected = ordered.loc[ordered["speed_penalty"] == recommended]
        left_axis.scatter(
            selected["speed_penalty"],
            selected["total_exposed_risk"],
            s=220,
            facecolors="none",
            edgecolors="#54A24B",
            linewidths=2.2,
            label=f"推荐 penalty={recommended:g}",
            zorder=4,
        )

    for row in ordered.itertuples(index=False):
        left_axis.annotate(
            f"{row.total_exposed_risk:.1f}",
            (row.speed_penalty, row.total_exposed_risk),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
            color="#4C78A8",
        )
        right_axis.annotate(
            f"{row.slowest_vehicle_time:.2f}",
            (row.speed_penalty, row.slowest_vehicle_time),
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            fontsize=8,
            color="#F58518",
        )

    left_axis.set_xlabel("速度惩罚系数 speed_penalty")
    left_axis.set_ylabel("总暴露风险 Σ(边长 × 风险)", color="#4C78A8")
    right_axis.set_ylabel("最慢车辆用时（小时）", color="#F58518")
    left_axis.set_title("恶劣天气日：速度惩罚参数的安全-时间权衡")
    left_axis.grid(alpha=0.25)
    left_axis.set_xticks(sorted(ordered["speed_penalty"].unique()))

    handles = risk_line, time_line
    labels = ["总暴露风险", "最慢车辆用时"]
    extra_handles = [
        plt.Line2D([], [], color="#4C78A8", linestyle="--", label="B 组暴露风险"),
        plt.Line2D([], [], color="#F58518", linestyle="--", label="B 组最慢用时"),
    ]
    if recommended is not None:
        extra_handles.append(
            plt.Line2D(
                [],
                [],
                marker="o",
                markerfacecolor="none",
                markeredgecolor="#54A24B",
                linestyle="",
                markersize=12,
                label=f"推荐 penalty={recommended:g}",
            )
        )
    left_axis.legend(handles + tuple(extra_handles), labels + [item.get_label() for item in extra_handles], loc="best")

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def build_docs(frame: pd.DataFrame, summary: dict[str, Any]) -> str:
    """生成速度惩罚扫描说明。"""

    lines = [
        "# 速度惩罚参数扫描",
        "",
        "## 结果",
        "",
        "| speed_penalty | 总暴露风险 | 最慢用时 | 高风险经过 | 总里程 | 总满足率 | 状态 | 风险降幅(vs B) | 时间变化(vs B) |",
        "|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.speed_penalty:g} | {row.total_exposed_risk:.2f} | "
            f"{row.slowest_vehicle_time:.3f} | {int(row.high_risk_edge_visits)} | "
            f"{row.total_distance:.1f} | {row.total_demand_satisfied_rate * 100:.2f}% | "
            f"{row.status} | {row.risk_reduction_pct:.2f}% | {row.time_change_pct:+.2f}% |"
        )
    lines.extend(
        [
            "",
            "## 推荐",
            "",
            summary["reason"],
            "",
            f"- B 组风险：{float(summary['reference']['total_exposed_risk']):.2f}",
            f"- B 组最慢用时：{float(summary['reference']['slowest_vehicle_time']):.3f} 小时",
            f"- alpha：{float(summary['risk_alpha']):g}",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="扫描速度惩罚系数")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config = load_config(resolve_project_path(arguments.config))
    outputs_path = resolve_project_path(config["paths"]["outputs"])
    docs_path = PROJECT_ROOT / "docs"
    docs_path.mkdir(parents=True, exist_ok=True)

    risk_alpha = float(config["optimization"]["risk_cost_alpha"])
    frame, summary = run_sweep(risk_alpha)

    frame.to_csv(
        outputs_path / "speed_penalty_sweep_storm.csv",
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )
    (outputs_path / "speed_penalty_sweep_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_sweep(
        frame,
        summary["reference"],
        outputs_path / "speed_penalty_sweep_storm.png",
        summary["recommended_speed_penalty"],
    )
    (docs_path / "speed_penalty_sweep.md").write_text(
        build_docs(frame, summary),
        encoding="utf-8",
    )

    print(frame.to_string(index=False))
    print()
    print(summary["reason"])
    print("推荐 speed_penalty：", summary["recommended_speed_penalty"])


if __name__ == "__main__":
    main()
