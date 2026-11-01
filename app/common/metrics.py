# -*- coding: utf-8 -*-
"""当前方案的统一配送指标口径。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DeliveryMetrics:
    total_demand: float
    delivered: float
    satisfaction_rate: float | None
    unserved_demand: float
    source: str

    @property
    def has_demand(self) -> bool:
        return self.total_demand > 0


def _non_negative_float(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return default


def format_satisfaction_rate(metrics: DeliveryMetrics) -> str:
    """零需求时返回明确文案，避免把空任务误显示为 100%。"""

    if metrics.satisfaction_rate is None:
        return "无可配送需求"
    return f"{metrics.satisfaction_rate * 100:.2f}%"


def compute_delivery_metrics(
    solution: dict[str, Any],
    fallback_scenario: dict[str, Any] | None = None,
) -> DeliveryMetrics:
    """按当前方案自身需求口径计算满足率。

    分母包含全部村庄：优先取 ``village_delivery_summary[*].demand``，
    对 summary 中缺失的村庄再用 fallback_scenario 的 demand 补齐，未配送量按 0
    计。分子为 ``Σ min(max(delivered, 0), demand)``。
    """

    summary = solution.get("village_delivery_summary")
    summary = summary if isinstance(summary, dict) else {}
    scenario_demands: dict[str, float] = {}
    if fallback_scenario:
        raw_demands = fallback_scenario.get("demands", {})
        if isinstance(raw_demands, dict):
            scenario_demands = {
                str(village_id): _non_negative_float(value)
                for village_id, value in raw_demands.items()
            }

    all_village_ids = set(scenario_demands) | {str(key) for key in summary}
    if all_village_ids:
        total_demand = 0.0
        delivered = 0.0
        for village_id in all_village_ids:
            row = summary.get(village_id, {})
            row = row if isinstance(row, dict) else {}
            if "demand" in row:
                demand = _non_negative_float(row.get("demand"))
            else:
                demand = scenario_demands.get(village_id, 0.0)
            delivered_at_village = _non_negative_float(row.get("delivered", 0.0))
            total_demand += demand
            delivered += min(delivered_at_village, demand)
        satisfaction_rate = delivered / total_demand if total_demand > 0 else None
        return DeliveryMetrics(
            total_demand=total_demand,
            delivered=delivered,
            satisfaction_rate=satisfaction_rate,
            unserved_demand=max(total_demand - delivered, 0.0),
            source="village_delivery_summary" if summary else "scenario",
        )

    # 兼容没有 summary 的旧结果：只使用 solution 已提供的指标，不臆造配送量。
    metrics = solution.get("metrics", {})
    metrics = metrics if isinstance(metrics, dict) else {}
    rate_value = metrics.get("total_demand_satisfied_rate")
    unserved_value = metrics.get("unserved_demand_total")
    total_demand = _non_negative_float(unserved_value)
    if rate_value is not None:
        rate = min(max(_non_negative_float(rate_value), 0.0), 1.0)
        if rate < 1.0:
            total_demand = total_demand / (1.0 - rate) if rate < 1.0 else 0.0
        delivered = total_demand * rate
        return DeliveryMetrics(
            total_demand=total_demand,
            delivered=delivered,
            satisfaction_rate=rate if total_demand > 0 else None,
            unserved_demand=max(total_demand - delivered, 0.0),
            source="metrics",
        )
    return DeliveryMetrics(
        total_demand=0.0,
        delivered=0.0,
        satisfaction_rate=None,
        unserved_demand=0.0,
        source="empty",
    )
