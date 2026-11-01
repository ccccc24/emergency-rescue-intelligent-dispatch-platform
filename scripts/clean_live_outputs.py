# -*- coding: utf-8 -*-
"""清理 outputs/live 中过期的实时 JSON，默认仅预览。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIVE_DIR = PROJECT_ROOT / "outputs" / "live"
ACTIVE_SCENARIO_NAME = "active_scenario.json"


def clean_live_outputs(
    live_dir: str | Path = DEFAULT_LIVE_DIR,
    *,
    keep_last: int = 20,
    older_than_days: float = 7.0,
    execute: bool = False,
) -> dict[str, Any]:
    """保留最新 N 份，且只清理早于时间阈值的文件。"""

    if keep_last < 0:
        raise ValueError("keep_last 不能小于 0")
    if older_than_days < 0:
        raise ValueError("older_than_days 不能小于 0")
    directory = Path(live_dir)
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for path in directory.glob("*.json"):
        if not path.is_file() or path.is_symlink() or path.name == ACTIVE_SCENARIO_NAME:
            continue
        files.append(path)
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    recent = files[:keep_last] if keep_last > 0 else []
    protected = set(recent)
    cutoff = time.time() - older_than_days * 86400.0
    candidates = [
        path
        for path in files
        if path not in protected and path.stat().st_mtime < cutoff
    ]

    deleted: list[str] = []
    errors: list[str] = []
    if execute:
        for path in candidates:
            try:
                path.unlink()
                deleted.append(str(path))
            except OSError as error:
                errors.append(f"{path}: {error}")

    return {
        "dry_run": not execute,
        "keep_last": keep_last,
        "older_than_days": older_than_days,
        "total_json_files": len(files),
        "protected_recent": [str(path) for path in recent],
        "candidates": [str(path) for path in candidates],
        "deleted": deleted,
        "errors": errors,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="清理 outputs/live 中的旧实时结果")
    parser.add_argument("--keep-last", type=int, default=20, help="保留最新 N 个 live JSON")
    parser.add_argument("--older-than", type=float, default=7.0, help="仅清理超过 D 天的文件")
    parser.add_argument("--yes", action="store_true", help="确认执行删除；默认只预览")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        result = clean_live_outputs(
            DEFAULT_LIVE_DIR,
            keep_last=arguments.keep_last,
            older_than_days=arguments.older_than,
            execute=arguments.yes,
        )
    except Exception as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps({"ok": True, **result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
