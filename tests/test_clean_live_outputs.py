# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import time
from pathlib import Path

from scripts.clean_live_outputs import clean_live_outputs


def _make_live_files(directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    now = time.time()
    ages = [10, 9, 8, 0, 0]
    for index, age_days in enumerate(ages, start=1):
        path = directory / f"live_{index}.json"
        path.write_text("{}", encoding="utf-8")
        timestamp = now - age_days * 86400
        os.utime(path, (timestamp, timestamp))
        files.append(path)
    active = directory / "active_scenario.json"
    active.write_text('{"source": "example"}', encoding="utf-8")
    os.utime(active, (now - 20 * 86400, now - 20 * 86400))
    return files


def test_clean_live_outputs_dry_run_does_not_delete(tmp_path) -> None:
    files = _make_live_files(tmp_path)
    result = clean_live_outputs(tmp_path, keep_last=2, older_than_days=1, execute=False)
    assert result["dry_run"] is True
    assert len(result["candidates"]) == 3
    assert result["deleted"] == []
    assert all(path.exists() for path in files)
    assert (tmp_path / "active_scenario.json").exists()


def test_clean_live_outputs_execute_deletes_only_candidates(tmp_path) -> None:
    files = _make_live_files(tmp_path)
    result = clean_live_outputs(tmp_path, keep_last=2, older_than_days=1, execute=True)
    assert result["dry_run"] is False
    assert len(result["deleted"]) == 3
    assert len(list(tmp_path.glob("live_*.json"))) == 2
    assert (tmp_path / "active_scenario.json").exists()
    assert sum(path.exists() for path in files) == 2
