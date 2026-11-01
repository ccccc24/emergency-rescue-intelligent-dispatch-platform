# -*- coding: utf-8 -*-
"""不依赖网络的方案解释规则模板。"""

from __future__ import annotations

from typing import Any

SUGGESTED_QUESTIONS = (
    "为什么优先配送这几个村？",
    "当前方案的主要风险在哪里？",
    "如果某条道路中断，可能有什么影响？",
    "用一句话总结当前调度方案。",
)


def _satisfaction_text(metrics: dict[str, Any]) -> str:
    text = metrics.get("satisfaction_text")
    if text:
        return str(text)
    rate = metrics.get("total_demand_satisfied_rate")
    if rate is None:
        return "无可配送需求"
    return f"{float(rate) * 100:.2f}%"


def _priority_text(summary: dict[str, Any]) -> str:
    rows = summary.get("priority_villages", [])
    if not rows:
        return "当前场景没有标记优先村"
    parts = [
        f"{row['name']}（需求 {float(row['demand']):.2f} 箱，已送达 {float(row['delivered']):.2f} 箱）"
        for row in rows
    ]
    return "、".join(parts)


def _risk_text(summary: dict[str, Any]) -> str:
    metrics = summary.get("metrics", {})
    top_roads = summary.get("top_risk_roads", [])
    top_text = "无"
    if top_roads:
        top = top_roads[0]
        top_text = f"{top['road_id']}（{top['from']}→{top['to']}，风险 {float(top['risk_score']):.3f}）"
    return (
        f"总暴露风险为 {float(metrics.get('total_exposed_risk', 0.0)):.1f}，"
        f"高风险边访问 {int(metrics.get('high_risk_edge_visits', 0))} 次；"
        f"当前最高风险道路是 {top_text}"
    )


def answer_question(question: str, summary: dict[str, Any]) -> str:
    """根据四个建议问题返回确定性的规则回答。"""

    question = str(question).strip()
    metrics = summary.get("metrics", {})
    scenario = summary.get("scenario_label") or summary.get("scenario_date", "")
    group = summary.get("group_label") or summary.get("group", "")
    status = summary.get("solver_status", "unknown")

    if "为什么优先" in question or "优先配送" in question:
        return (
            f"{scenario}的优先村标记为：{_priority_text(summary)}。"
            "这些村庄是在 bridge 场景中按灾情、需求和道路风险综合标记的；"
            "调度器会优先安排它们，但不能保证所有优先村都在最短时间内完成。"
        )

    if "主要风险" in question or "风险在哪里" in question:
        top_roads = summary.get("top_risk_roads", [])
        names = "、".join(
            f"{row['road_id']}（{float(row['risk_score']):.3f}）" for row in top_roads[:3]
        )
        if not names:
            names = "无可用道路风险数据"
        return (
            f"{_risk_text(summary)}。主要风险集中在 {names}。"
            "该解释只引用当前方案与场景中的已有数值，不重新运行调度。"
        )

    if "道路中断" in question or "中断" in question:
        top_roads = summary.get("top_risk_roads", [])
        if top_roads:
            top = top_roads[0]
            road_text = f"{top['road_id']}（{top['from']}→{top['to']}，风险 {float(top['risk_score']):.3f}）"
        else:
            road_text = "当前未识别出可用道路"
        return (
            f"最值得关注的是 {road_text}。如果该道路中断，车辆可能需要绕行，"
            "最慢车辆时间、总里程和高风险边访问次数都可能上升；"
            "但具体影响必须重新运行 A* 才能确认，本页不做数值推断。"
        )

    if "一句话" in question or "总结" in question:
        satisfaction = _satisfaction_text(metrics)
        return (
            f"{scenario}采用{group}，状态为 {status}，需求满足率 {satisfaction}，"
            f"最慢车辆时间 {float(metrics.get('slowest_vehicle_time', 0.0)):.2f} 小时，"
            f"总暴露风险 {float(metrics.get('total_exposed_risk', 0.0)):.1f}。"
        )

    return (
        f"当前方案为{scenario}的{group}，状态 {status}，"
        f"需求满足率 {_satisfaction_text(metrics)}。"
        "可选择上方四个建议问题获取更具体的解释。"
    )



