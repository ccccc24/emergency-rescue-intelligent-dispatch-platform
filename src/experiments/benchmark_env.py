# -*- coding: utf-8 -*-
"""在答辩机器上测量 C 组完整求解耗时，并给出 time_limit 余量。"""

from __future__ import annotations

import argparse
import ctypes
import os
import platform
from pathlib import Path
from typing import Any, Final

from src.ml.data_gen import load_config, resolve_project_path
from src.optimization.run import DEFAULT_CONFIG_PATH, solve_with_ablation


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
SCENARIOS: Final[tuple[tuple[str, str], ...]] = (
    ("常规天气日", "outputs/scenario_2025-08-29.json"),
    ("恶劣天气日", "outputs/scenario_2025-08-01.json"),
)


def _cpu_model() -> str:
    """优先从 Windows 注册表读取 CPU 型号，失败时回退到 platform。"""

    if os.name == "nt":
        try:
            import winreg

            key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                if str(value).strip():
                    return str(value).strip()
        except OSError:
            pass
    return platform.processor() or platform.machine() or "未知 CPU"


def _total_memory_gb() -> float:
    """跨平台读取物理内存总量，单位 GB。"""

    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return float(status.ullTotalPhys) / (1024 ** 3)
    else:
        try:
            return (
                float(os.sysconf("SC_PAGE_SIZE"))
                * float(os.sysconf("SC_PHYS_PAGES"))
                / (1024 ** 3)
            )
        except (AttributeError, ValueError, OSError):
            pass
    return float("nan")


def _run_case(
    label: str,
    scenario_path: Path,
    config_path: Path,
    time_limit: float,
) -> dict[str, Any]:
    """完整重跑一次 C 组求解，不使用任何缓存。"""

    solution = solve_with_ablation(
        scenario_path,
        "full",
        config_path=config_path,
    )
    metrics = solution.metrics
    solve_time = float(metrics["solve_time"])
    margin = (1.0 - solve_time / time_limit) * 100.0
    return {
        "label": label,
        "status": str(metrics["status"]),
        "states_explored": int(metrics["states_explored"]),
        "solve_time": solve_time,
        "ratio": solve_time / time_limit * 100.0,
        "margin": margin,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="答辩机调度性能基准测试")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="配置文件路径",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config_path = resolve_project_path(arguments.config)
    config = load_config(config_path)
    time_limit = float(config["optimization"]["time_limit"])

    print("=== 答辩机环境 ===")
    print(f"CPU: {_cpu_model()}")
    print(f"逻辑核心数: {os.cpu_count()}")
    print(f"物理内存: {_total_memory_gb():.2f} GB")
    print(f"time_limit: {time_limit:g} 秒")
    print()

    for label, relative_path in SCENARIOS:
        scenario_path = resolve_project_path(relative_path)
        result = _run_case(label, scenario_path, config_path, time_limit)
        print(
            f"{label}: status={result['status']}, "
            f"states_explored={result['states_explored']}, "
            f"耗时={result['solve_time']:.3f}s, "
            f"占 {time_limit:g}s 的 {result['ratio']:.2f}%, "
            f"余量={result['margin']:.2f}%"
        )


if __name__ == "__main__":
    main()
