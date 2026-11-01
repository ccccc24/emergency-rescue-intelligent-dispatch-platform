# -*- coding: utf-8 -*-
"""实时调度 CLI：兼容示例场景和用户场景，并消费场景调度参数。"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.common.scenario_builder import compute_scenario_hash, normalize_scheduler_config  # noqa: E402
from src.ml.data_gen import load_config  # noqa: E402
from src.optimization.export_solution import export_solution  # noqa: E402

GROUP_MODES = {
    "A": "hist_demand",
    "B": "no_risk_adjust",
    "C": "full",
    "user": "full",
}
REQUIRED_SOLUTION_FIELDS = (
    "village_delivery_summary",
    "metrics",
    "vehicle_trajectories",
    "solver_status",
    "scenario_hash",
    "group",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实时运行翼援调度方案")
    parser.add_argument("--scenario-json", type=Path, default=None, help="与 scenario_*.json 兼容的场景文件")
    parser.add_argument("--date", choices=("2025-08-29", "2025-08-01"), help="兼容旧入口的示例场景日期")
    parser.add_argument("--group", choices=tuple(GROUP_MODES), help="示例场景组别；用户场景固定为 user")
    parser.add_argument("--scenario-hash", default=None, help="调用方持有的场景指纹，用于一致性校验")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "live_demo.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "live",
    )
    return parser.parse_args()


def _resolve_scenario(arguments: argparse.Namespace) -> tuple[Path, str, dict[str, Any]]:
    if arguments.scenario_json is not None:
        scenario_path = arguments.scenario_json
        if not scenario_path.is_absolute():
            scenario_path = PROJECT_ROOT / scenario_path
        if not scenario_path.is_file():
            raise FileNotFoundError(f"场景文件不存在：{scenario_path}")
        scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
        source = str(scenario.get("source") or scenario.get("metadata", {}).get("source") or "")
        if source in {"manual", "excel"}:
            group = "user"
            if arguments.group not in (None, "user"):
                raise ValueError("用户场景只能使用 group=user，不能借用 A/B/C 消融组别")
        else:
            group = str(arguments.group or "C")
        return scenario_path, group, scenario

    if arguments.date is None or arguments.group not in GROUP_MODES or arguments.group == "user":
        raise ValueError("旧入口必须提供 --date 和 A/B/C 组别")
    scenario_path = PROJECT_ROOT / "outputs" / f"scenario_{arguments.date}.json"
    if not scenario_path.is_file():
        raise FileNotFoundError(f"场景文件不存在：{scenario_path}")
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    return scenario_path, arguments.group, scenario


def _scenario_config(scenario: dict[str, Any], config_path: Path) -> dict[str, int | float]:
    base = load_config(config_path)
    optimization = base["optimization"]
    raw_config: dict[str, Any] = {
        "num_vehicles": optimization["num_vehicles"],
        "vehicle_capacity": optimization["vehicle_capacity"],
        "vehicle_time_limit": optimization["vehicle_time_limit"],
        "risk_cost_alpha": optimization.get("risk_cost_alpha", 0.2),
        "speed_penalty": base.get("risk", {}).get("speed_penalty", 0.1),
        "time_limit": optimization["time_limit"],
        "max_states": optimization["max_states"],
    }
    top_level_config = scenario.get("config")
    if isinstance(top_level_config, dict):
        raw_config.update(top_level_config)
    metadata = scenario.get("metadata", {})
    metadata_config = metadata.get("scheduler_config") if isinstance(metadata, dict) else None
    if isinstance(metadata_config, dict):
        raw_config.update(metadata_config)
    return normalize_scheduler_config(raw_config)


def _write_runtime_config(base_config_path: Path, config: dict[str, int | float], target: Path) -> None:
    text = base_config_path.read_text(encoding="utf-8")
    for key, value in config.items():
        pattern = rf"(?m)^(\s*{re.escape(key)}:\s*)[^#\r\n]*"
        text, count = re.subn(pattern, lambda match: match.group(1) + str(value), text, count=1)
        if count != 1:
            raise ValueError(f"配置文件中未找到可覆盖的参数：{key}")
    target.write_text(text, encoding="utf-8")


def _resolve_scenario_hash(
    scenario: dict[str, Any],
    config: dict[str, int | float],
    requested_hash: str | None,
) -> str:
    computed = compute_scenario_hash(scenario, config)
    declared = str(
        scenario.get("scenario_hash")
        or scenario.get("metadata", {}).get("scenario_hash")
        or ""
    )
    expected = str(requested_hash or declared or computed)
    if expected != computed:
        raise ValueError(f"场景指纹不一致：调用方={expected[:16]}，实际={computed[:16]}")
    return computed


def _verify_solution_schema(payload: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_SOLUTION_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"实时 solution 缺少字段：{missing}")


def main() -> int:
    try:
        arguments = parse_arguments()
        scenario_path, group, scenario = _resolve_scenario(arguments)
        config_path = arguments.config if arguments.config.is_absolute() else PROJECT_ROOT / arguments.config
        config = _scenario_config(scenario, config_path)
        scenario_hash = _resolve_scenario_hash(scenario, config, arguments.scenario_hash)
        source = str(scenario.get("source") or scenario.get("metadata", {}).get("source") or "example")
        output_dir = arguments.output_dir if arguments.output_dir.is_absolute() else PROJECT_ROOT / arguments.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = output_dir / f"{scenario_hash}_{group}_{timestamp}.json"

        with tempfile.TemporaryDirectory(prefix="wingaid_live_") as temp_dir:
            runtime_config = Path(temp_dir) / "live_config.yaml"
            _write_runtime_config(config_path, config, runtime_config)
            payload = export_solution(
                scenario_path,
                output_path,
                ablation_mode=GROUP_MODES[group],
                config_path=runtime_config,
            )

        payload["scenario_hash"] = scenario_hash
        payload["group"] = group
        payload["scenario_source"] = source
        payload["scenario_config"] = dict(config)
        payload["config_snapshot"] = {
            **dict(payload.get("config_snapshot", {})),
            **dict(config),
        }
        payload["config_schema_version"] = int(
            scenario.get("config_schema_version")
            or scenario.get("metadata", {}).get("config_schema_version", 1)
        )
        _verify_solution_schema(payload)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 1

    print(
        json.dumps(
            {
                "ok": True,
                "output_path": str(output_path.resolve()),
                "solver_status": payload.get("solver_status"),
                "solve_time_actual": payload.get("solve_time_actual"),
                "scenario_hash": payload.get("scenario_hash"),
                "group": payload.get("group"),
                "config_snapshot": payload.get("config_snapshot"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

