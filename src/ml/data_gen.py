# -*- coding: utf-8 -*-
"""生成需求预测与道路风险评估所需的模拟历史数据。

数据生成完全离线，只依赖 pandas / numpy。随机种子、输出目录和噪声参数从
configs/default.yaml 读取；命令行参数 --seed / --out-dir 可覆盖默认值。
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH: Final[Path] = PROJECT_ROOT / "configs" / "default.yaml"
START_DATE: Final[pd.Timestamp] = pd.Timestamp("2025-06-01")
HISTORY_DAYS: Final[int] = 90
WARMUP_DAYS: Final[int] = 7

DEMAND_COLUMNS: Final[list[str]] = [
    "village_id",
    "date",
    "population",
    "disaster_level",
    "distance_to_center",
    "road_accessibility",
    "weather",
    "is_priority",
    "hist_demand_mean",
    "demand",
]

ROAD_COLUMNS: Final[list[str]] = [
    "road_id",
    "date",
    "from",
    "to",
    "length",
    "base_speed",
    "weather",
    "terrain",
    "geology_risk",
    "weather_sensitivity",
    "hist_break_count",
    "is_broken",
]

# 村庄静态属性：人口、灾情等级、距离和可达性在 90 天内保持不变。
# 灾情等级越高，道路可达性系统性越低，保证后续特征重要性分析有业务含义。
VILLAGES: Final[tuple[dict[str, float | int], ...]] = (
    {
        "village_id": 1,
        "population": 1200,
        "disaster_level": 1,
        "distance_to_center": 10.0,
        "road_accessibility": 0.92,
    },
    {
        "village_id": 2,
        "population": 1800,
        "disaster_level": 2,
        "distance_to_center": 16.0,
        "road_accessibility": 0.80,
    },
    {
        "village_id": 3,
        "population": 2600,
        "disaster_level": 3,
        "distance_to_center": 22.0,
        "road_accessibility": 0.68,
    },
    {
        "village_id": 4,
        "population": 3800,
        "disaster_level": 4,
        "distance_to_center": 29.0,
        "road_accessibility": 0.55,
    },
    {
        "village_id": 5,
        "population": 5000,
        "disaster_level": 4,
        "distance_to_center": 35.0,
        "road_accessibility": 0.48,
    },
    {
        "village_id": 6,
        "population": 6200,
        "disaster_level": 5,
        "distance_to_center": 42.0,
        "road_accessibility": 0.35,
    },
)

# 固定路网：0 为指挥中心，1~6 为村庄，7~14 为中间节点。
# 12 条平原边构成低风险骨干网，12 条丘陵边提供备用路径，6 条山区边构成高风险捷径。
ROADS: Final[tuple[dict[str, float | int | str], ...]] = (
    {"road_id": "R01", "from_node": 0, "to_node": 1, "length": 11.5, "base_speed": 55, "terrain": "平原"},
    {"road_id": "R02", "from_node": 0, "to_node": 2, "length": 17.2, "base_speed": 50, "terrain": "平原"},
    {"road_id": "R03", "from_node": 0, "to_node": 3, "length": 27.8, "base_speed": 45, "terrain": "平原"},
    {"road_id": "R04", "from_node": 1, "to_node": 7, "length": 8.6, "base_speed": 50, "terrain": "平原"},
    {"road_id": "R05", "from_node": 7, "to_node": 4, "length": 10.4, "base_speed": 48, "terrain": "平原"},
    {"road_id": "R06", "from_node": 2, "to_node": 8, "length": 9.8, "base_speed": 50, "terrain": "平原"},
    {"road_id": "R07", "from_node": 8, "to_node": 5, "length": 11.2, "base_speed": 46, "terrain": "平原"},
    {"road_id": "R08", "from_node": 3, "to_node": 9, "length": 8.9, "base_speed": 48, "terrain": "平原"},
    {"road_id": "R09", "from_node": 9, "to_node": 6, "length": 12.1, "base_speed": 45, "terrain": "平原"},
    {"road_id": "R10", "from_node": 7, "to_node": 8, "length": 7.5, "base_speed": 52, "terrain": "平原"},
    {"road_id": "R11", "from_node": 8, "to_node": 9, "length": 8.2, "base_speed": 50, "terrain": "平原"},
    {"road_id": "R12", "from_node": 1, "to_node": 3, "length": 14.6, "base_speed": 48, "terrain": "平原"},
    {"road_id": "R13", "from_node": 0, "to_node": 4, "length": 33.6, "base_speed": 40, "terrain": "丘陵"},
    {"road_id": "R14", "from_node": 0, "to_node": 5, "length": 36.4, "base_speed": 38, "terrain": "丘陵"},
    {"road_id": "R15", "from_node": 0, "to_node": 6, "length": 42.8, "base_speed": 36, "terrain": "丘陵"},
    {"road_id": "R16", "from_node": 1, "to_node": 2, "length": 9.4, "base_speed": 45, "terrain": "丘陵"},
    {"road_id": "R17", "from_node": 2, "to_node": 3, "length": 12.7, "base_speed": 42, "terrain": "丘陵"},
    {"road_id": "R18", "from_node": 4, "to_node": 5, "length": 12.9, "base_speed": 38, "terrain": "丘陵"},
    {"road_id": "R19", "from_node": 5, "to_node": 6, "length": 10.6, "base_speed": 36, "terrain": "丘陵"},
    {"road_id": "R20", "from_node": 1, "to_node": 10, "length": 16.3, "base_speed": 40, "terrain": "丘陵"},
    {"road_id": "R21", "from_node": 10, "to_node": 4, "length": 14.8, "base_speed": 38, "terrain": "丘陵"},
    {"road_id": "R22", "from_node": 3, "to_node": 11, "length": 17.6, "base_speed": 40, "terrain": "丘陵"},
    {"road_id": "R23", "from_node": 11, "to_node": 6, "length": 15.9, "base_speed": 38, "terrain": "丘陵"},
    {"road_id": "R24", "from_node": 2, "to_node": 4, "length": 15.8, "base_speed": 40, "terrain": "丘陵"},
    {"road_id": "R25", "from_node": 2, "to_node": 12, "length": 18.5, "base_speed": 32, "terrain": "山区"},
    {"road_id": "R26", "from_node": 12, "to_node": 5, "length": 16.8, "base_speed": 30, "terrain": "山区"},
    {"road_id": "R27", "from_node": 3, "to_node": 13, "length": 19.2, "base_speed": 32, "terrain": "山区"},
    {"road_id": "R28", "from_node": 13, "to_node": 6, "length": 17.4, "base_speed": 30, "terrain": "山区"},
    {"road_id": "R29", "from_node": 4, "to_node": 14, "length": 22.5, "base_speed": 34, "terrain": "山区"},
    {"road_id": "R30", "from_node": 14, "to_node": 1, "length": 24.0, "base_speed": 36, "terrain": "山区"},
)

WEATHER_STATES: Final[tuple[str, ...]] = ("晴", "雨", "暴风")
WEATHER_INITIAL_PROBABILITY: Final[np.ndarray] = np.array([0.65, 0.25, 0.10])
WEATHER_TRANSITION_MATRIX: Final[np.ndarray] = np.array(
    [
        [0.65, 0.28, 0.07],
        [0.35, 0.50, 0.15],
        [0.20, 0.45, 0.35],
    ]
)
WEATHER_DEMAND_FACTOR: Final[dict[str, float]] = {"晴": 1.00, "雨": 1.15, "暴风": 1.30}
WEATHER_RISK: Final[dict[str, float]] = {"晴": 0.00, "雨": 0.50, "暴风": 1.00}
WEATHER_CODE: Final[dict[str, int]] = {"晴": 0, "雨": 1, "暴风": 2}

# 图表统一使用中文标签；特征名保留英文仅用于代码和 CSV 字段。
DEMAND_FEATURE_LABELS: Final[dict[str, str]] = {
    "population": "人口",
    "disaster_level": "灾情等级",
    "distance_to_center": "距指挥中心距离",
    "road_accessibility": "道路可达性",
    "is_priority": "是否优先村",
    "weather_code": "天气等级",
    "hist_demand_mean": "历史7天需求均值",
    "demand": "需求量",
}
ROAD_FEATURE_LABELS: Final[dict[str, str]] = {
    "length": "道路长度",
    "base_speed": "基准速度",
    "weather_code": "天气等级",
    "terrain": "地形",
    "geology_risk": "地质风险",
    "weather_sensitivity": "天气敏感度",
    "hist_break_count": "近7天中断次数",
}
CHINESE_FONT_CANDIDATES: Final[tuple[str, ...]] = (
    "Microsoft YaHei",
    "SimHei",
    "Source Han Serif SC",
    "Noto Sans CJK SC",
    "WenQuanYi Zen Hei",
)

# 非线性需求结构：高灾情触发应急响应升级，重灾且断路造成物资积压。
EMERGENCY_RESPONSE_THRESHOLD: Final[int] = 4
EMERGENCY_RESPONSE_MULTIPLIER: Final[float] = 1.25
ACCESS_INTERACTION_COEFFICIENT: Final[float] = 0.35
ACCESS_INTERACTION_DENOMINATOR: Final[float] = 4.0

# 道路地形分层：平原/丘陵/山区分别决定地质风险范围、天气敏感度和坡度因子。
TERRAIN_TYPES: Final[tuple[str, ...]] = ("平原", "丘陵", "山区")
TERRAIN_EXPECTED_COUNTS: Final[dict[str, int]] = {"平原": 12, "丘陵": 12, "山区": 6}
TERRAIN_RISK_RANGES: Final[dict[str, tuple[float, float]]] = {
    "平原": (0.05, 0.25),
    "丘陵": (0.25, 0.55),
    "山区": (0.55, 0.90),
}
TERRAIN_SENSITIVITY_RANGES: Final[dict[str, tuple[float, float]]] = {
    "平原": (0.60, 0.90),
    "丘陵": (0.85, 1.15),
    "山区": (1.10, 1.60),
}
TERRAIN_SLOPE_FACTOR: Final[dict[str, float]] = {"平原": 0.20, "丘陵": 0.60, "山区": 1.00}

# 道路中断概率：
# p = sigmoid(a*geology + b*weather*weather_sensitivity + c*history
#             + d*terrain_slope + 2.0*geology*weather - bias)
ROAD_GEOLOGY_COEFFICIENT: Final[float] = 1.30
ROAD_WEATHER_COEFFICIENT: Final[float] = 1.80
ROAD_HISTORY_COEFFICIENT: Final[float] = 0.18
ROAD_SLOPE_COEFFICIENT: Final[float] = 0.80
ROAD_INTERACTION_COEFFICIENT: Final[float] = 2.00
ROAD_BIAS: Final[float] = 4.13


def configure_chinese_font() -> None:
    """为 Matplotlib 选择可用中文字体，并修复负号显示。"""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    selected_font = next(
        (font_name for font_name in CHINESE_FONT_CANDIDATES if font_name in available_fonts),
        "DejaVu Sans",
    )
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [selected_font, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _parse_scalar(raw_value: str) -> Any:
    """解析 YAML 中当前配置需要的简单标量。"""
    value = raw_value.strip().strip('"').strip("'")
    lowered = value.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """读取只包含一层嵌套的简单 YAML 配置，避免引入额外依赖。"""
    config: dict[str, Any] = {}
    current_section: str | None = None

    for line_number, raw_line in enumerate(config_path.read_text(encoding="utf-8").splitlines(), start=1):
        content = raw_line.split("#", 1)[0].rstrip()
        if not content.strip():
            continue

        indentation = len(content) - len(content.lstrip(" "))
        key, separator, raw_value = content.strip().partition(":")
        if not separator or not key.strip():
            raise ValueError(f"配置格式错误：{config_path}:{line_number}")

        key = key.strip()
        if indentation == 0:
            current_section = None
            if raw_value.strip():
                config[key] = _parse_scalar(raw_value)
            else:
                config[key] = {}
                current_section = key
            continue

        if current_section is None or not isinstance(config.get(current_section), dict):
            raise ValueError(f"配置缩进错误：{config_path}:{line_number}")
        config[current_section][key] = _parse_scalar(raw_value)

    return config


def resolve_project_path(raw_path: str | Path) -> Path:
    """以项目根目录为基准解析配置中的相对路径。"""
    path = Path(raw_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def generate_weather_sequence(days: int, rng: np.random.Generator) -> list[str]:
    """用一阶马尔可夫链生成区域天气，使连续恶劣天气更符合现实。"""
    weather: list[str] = []
    state_index = int(rng.choice(len(WEATHER_STATES), p=WEATHER_INITIAL_PROBABILITY))

    for day_index in range(days):
        weather.append(WEATHER_STATES[state_index])
        if day_index < days - 1:
            state_index = int(rng.choice(len(WEATHER_STATES), p=WEATHER_TRANSITION_MATRIX[state_index]))

    return weather


def generate_demand_data(
    weather_sequence: list[str],
    noise_std: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """生成 90 天需求，前 7 天只作 warm-up，不写入输出数据。"""
    rows: list[dict[str, Any]] = []
    dates = pd.date_range(start=START_DATE, periods=len(weather_sequence), freq="D")
    median_population = float(np.median([float(village["population"]) for village in VILLAGES]))

    for village in VILLAGES:
        population = int(village["population"])
        disaster_level = int(village["disaster_level"])
        distance_to_center = float(village["distance_to_center"])
        road_accessibility = float(village["road_accessibility"])
        is_priority = int(disaster_level >= 4 and population > median_population)

        demand_values = np.empty(len(weather_sequence), dtype=np.int64)
        for day_index, weather in enumerate(weather_sequence):
            base_demand = population * 0.02
            disaster_factor = 1.0 + 0.4 * (disaster_level - 1)
            access_factor = 0.5 + 0.5 * road_accessibility
            weather_factor = WEATHER_DEMAND_FACTOR[weather]
            response_multiplier = (
                EMERGENCY_RESPONSE_MULTIPLIER
                if disaster_level >= EMERGENCY_RESPONSE_THRESHOLD
                else 1.0
            )
            access_interaction = 1.0 + (
                ACCESS_INTERACTION_COEFFICIENT
                * (disaster_level - 1)
                / ACCESS_INTERACTION_DENOMINATOR
                * (1.0 - road_accessibility)
            )
            noise = float(rng.normal(loc=1.0, scale=noise_std))
            raw_demand = (
                base_demand
                * disaster_factor
                * access_factor
                * weather_factor
                * response_multiplier
                * access_interaction
                * noise
            )
            demand_values[day_index] = max(1, int(np.rint(raw_demand)))

        # 前 7 天仅用于累积历史；从第 8 天起才有完整的 7 天滞后均值。
        history_series = (
            pd.Series(demand_values, dtype="float64")
            .shift(1)
            .rolling(window=WARMUP_DAYS, min_periods=WARMUP_DAYS)
            .mean()
        )
        history_values = history_series.to_numpy()

        for day_index in range(WARMUP_DAYS, len(dates)):
            date_value = dates[day_index]
            rows.append(
                {
                    "village_id": int(village["village_id"]),
                    "date": date_value.strftime("%Y-%m-%d"),
                    "population": population,
                    "disaster_level": disaster_level,
                    "distance_to_center": round(distance_to_center, 1),
                    "road_accessibility": round(road_accessibility, 2),
                    "weather": weather_sequence[day_index],
                    "is_priority": is_priority,
                    "hist_demand_mean": round(float(history_values[day_index]), 6),
                    "demand": int(demand_values[day_index]),
                }
            )

    return pd.DataFrame(rows, columns=DEMAND_COLUMNS)


def _sample_road_specs(rng: np.random.Generator) -> list[dict[str, float | int | str]]:
    """按地形分层为每条道路采样地质风险和天气敏感度。"""
    sampled_roads: list[dict[str, float | int | str]] = []
    for road in ROADS:
        terrain = str(road["terrain"])
        geology_low, geology_high = TERRAIN_RISK_RANGES[terrain]
        sensitivity_low, sensitivity_high = TERRAIN_SENSITIVITY_RANGES[terrain]
        sampled_roads.append(
            {
                **road,
                "geology_risk": float(rng.uniform(geology_low, geology_high)),
                "weather_sensitivity": float(rng.uniform(sensitivity_low, sensitivity_high)),
            }
        )
    return sampled_roads


def _road_probability(
    geology_risk: float,
    weather: str,
    history_count: int,
    weather_sensitivity: float,
    terrain: str,
) -> float:
    """计算一条道路当天发生中断的概率，包含地质与天气交互项。"""
    weather_risk = WEATHER_RISK[weather]
    interaction = ROAD_INTERACTION_COEFFICIENT * geology_risk * weather_risk
    logit = (
        ROAD_GEOLOGY_COEFFICIENT * geology_risk
        + ROAD_WEATHER_COEFFICIENT * weather_risk * weather_sensitivity
        + ROAD_HISTORY_COEFFICIENT * history_count
        + ROAD_SLOPE_COEFFICIENT * TERRAIN_SLOPE_FACTOR[terrain]
        + interaction
        - ROAD_BIAS
    )
    return float(1.0 / (1.0 + np.exp(-logit)))


def generate_road_data(weather_sequence: list[str], rng: np.random.Generator) -> pd.DataFrame:
    """按地形分层、天气敏感度、交互项和历史中断生成道路标签。"""
    rows: list[dict[str, Any]] = []
    dates = pd.date_range(start=START_DATE, periods=len(weather_sequence), freq="D")
    road_specs = _sample_road_specs(rng)

    for road in road_specs:
        break_history: list[int] = []
        for day_index, date_value in enumerate(dates):
            weather = weather_sequence[day_index]
            history_count = int(sum(break_history[-7:]))
            terrain = str(road["terrain"])
            geology_risk = float(road["geology_risk"])
            weather_sensitivity = float(road["weather_sensitivity"])
            probability = _road_probability(
                geology_risk=geology_risk,
                weather=weather,
                history_count=history_count,
                weather_sensitivity=weather_sensitivity,
                terrain=terrain,
            )
            is_broken = int(rng.random() < probability)
            rows.append(
                {
                    "road_id": str(road["road_id"]),
                    "date": date_value.strftime("%Y-%m-%d"),
                    "from": int(road["from_node"]),
                    "to": int(road["to_node"]),
                    "length": round(float(road["length"]), 1),
                    "base_speed": int(road["base_speed"]),
                    "weather": weather,
                    "terrain": terrain,
                    "geology_risk": round(geology_risk, 3),
                    "weather_sensitivity": round(weather_sensitivity, 3),
                    "hist_break_count": history_count,
                    "is_broken": is_broken,
                }
            )
            break_history.append(is_broken)

    return pd.DataFrame(rows, columns=ROAD_COLUMNS)


def _count_simple_paths(
    adjacency: dict[int, set[int]],
    target: int,
    max_count: int,
) -> int:
    """从指挥中心 0 到目标节点计数简单路径，达到上限即停止。"""
    path_count = 0
    stack: list[tuple[int, frozenset[int]]] = [(0, frozenset({0}))]
    while stack and path_count < max_count:
        node, visited = stack.pop()
        if node == target:
            path_count += 1
            continue
        for neighbor in adjacency.get(node, set()):
            if neighbor not in visited:
                stack.append((neighbor, visited | {neighbor}))
    return path_count


def validate_road_topology() -> None:
    """检查 30 条边、地形配比、连通性、村庄双路径和低风险备用路径。"""
    if len(ROADS) != 30:
        raise ValueError(f"路网应为 30 条边，当前为 {len(ROADS)} 条")

    road_ids = [str(road["road_id"]) for road in ROADS]
    if len(set(road_ids)) != len(road_ids):
        raise ValueError("存在重复 road_id")

    edge_keys = {
        tuple(sorted((int(road["from_node"]), int(road["to_node"]))))
        for road in ROADS
    }
    if len(edge_keys) != len(ROADS):
        raise ValueError("存在重复无向边")

    terrain_counts = Counter(str(road["terrain"]) for road in ROADS)
    if dict(terrain_counts) != TERRAIN_EXPECTED_COUNTS:
        raise ValueError(
            f"地形配比错误：期望 {TERRAIN_EXPECTED_COUNTS}，实际 {dict(terrain_counts)}"
        )

    adjacency: dict[int, set[int]] = {}
    safe_adjacency: dict[int, set[int]] = {}
    for road in ROADS:
        from_node = int(road["from_node"])
        to_node = int(road["to_node"])
        terrain = str(road["terrain"])
        adjacency.setdefault(from_node, set()).add(to_node)
        adjacency.setdefault(to_node, set()).add(from_node)
        if terrain != "山区":
            safe_adjacency.setdefault(from_node, set()).add(to_node)
            safe_adjacency.setdefault(to_node, set()).add(from_node)

    required_nodes = set(range(7))
    if not required_nodes.issubset(adjacency):
        raise ValueError("路网必须包含指挥中心 0 和村庄 1~6")

    visited: set[int] = set()
    stack = [0]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        stack.extend(adjacency.get(node, set()) - visited)

    if visited != set(adjacency):
        raise ValueError("路网不连通，请检查 ROADS 拓扑")

    for village_id in range(1, 7):
        if _count_simple_paths(adjacency, target=village_id, max_count=2) < 2:
            raise ValueError(f"村庄 {village_id} 到指挥中心的路径不足 2 条")
        if _count_simple_paths(safe_adjacency, target=village_id, max_count=1) < 1:
            raise ValueError(f"村庄 {village_id} 没有避开山区高风险边的备用路径")


def generate_datasets(seed: int, noise_std: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """使用独立随机流生成两份共享同一区域天气的数据集。"""
    validate_road_topology()
    seed_sequence = np.random.SeedSequence(seed)
    weather_seed, demand_seed, road_seed = seed_sequence.spawn(3)

    weather_rng = np.random.default_rng(weather_seed)
    demand_rng = np.random.default_rng(demand_seed)
    road_rng = np.random.default_rng(road_seed)

    weather_sequence = generate_weather_sequence(HISTORY_DAYS, weather_rng)
    demand_data = generate_demand_data(weather_sequence, noise_std, demand_rng)
    road_data = generate_road_data(weather_sequence, road_rng)
    return demand_data, road_data


def save_csv(data_frame: pd.DataFrame, output_path: Path) -> None:
    """以带 BOM 的 UTF-8 写出 CSV，保证 Excel 直接打开时中文正常。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_frame.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )


def calculate_md5(file_path: Path) -> str:
    """计算输出文件的 MD5，用于验证同种子复现性。"""
    return hashlib.md5(file_path.read_bytes()).hexdigest()


def create_plots(demand_data: pd.DataFrame, output_dir: Path) -> None:
    """生成需求分布、灾情箱线图和特征相关热力图；matplotlib 缺失时跳过。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过 PNG 图表输出；CSV 不受影响。")
        return

    configure_chinese_font()
    output_dir.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(demand_data["demand"], bins=20, color="#2E86AB", edgecolor="white")
    axis.set_title("需求量分布")
    axis.set_xlabel("需求量（箱）")
    axis.set_ylabel("记录数")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "demand_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(figure)

    disaster_levels = sorted(int(level) for level in demand_data["disaster_level"].unique())
    grouped_demands = [
        demand_data.loc[demand_data["disaster_level"] == level, "demand"].to_numpy()
        for level in disaster_levels
    ]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.boxplot(grouped_demands, tick_labels=[str(level) for level in disaster_levels])
    axis.set_title("不同灾情等级的需求量分布")
    axis.set_xlabel("灾情等级")
    axis.set_ylabel("需求量（箱）")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "demand_by_disaster.png", dpi=150, bbox_inches="tight")
    plt.close(figure)

    correlation_columns = [
        "population",
        "disaster_level",
        "distance_to_center",
        "road_accessibility",
        "is_priority",
        "weather_code",
        "hist_demand_mean",
        "demand",
    ]
    correlation_frame = demand_data[
        [
            "population",
            "disaster_level",
            "distance_to_center",
            "road_accessibility",
            "is_priority",
            "hist_demand_mean",
            "demand",
        ]
    ].copy()
    correlation_frame["weather_code"] = demand_data["weather"].map(WEATHER_CODE)
    correlation_matrix = correlation_frame[correlation_columns].corr()
    correlation_labels = [DEMAND_FEATURE_LABELS[column] for column in correlation_columns]

    figure, axis = plt.subplots(figsize=(9, 7))
    image = axis.imshow(correlation_matrix.to_numpy(), cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axis.set_xticks(range(len(correlation_columns)))
    axis.set_yticks(range(len(correlation_columns)))
    axis.set_xticklabels(correlation_labels, rotation=45, ha="right")
    axis.set_yticklabels(correlation_labels)
    axis.set_title("特征相关性热力图")
    for row_index in range(len(correlation_columns)):
        for column_index in range(len(correlation_columns)):
            value = float(correlation_matrix.iloc[row_index, column_index])
            axis.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=7)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_dir / "feature_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def print_summary(demand_data: pd.DataFrame, road_data: pd.DataFrame, demand_path: Path, road_path: Path) -> None:
    """打印生成规模、标签分布和关键特征关系，便于人工自检。"""
    demand_description = demand_data["demand"].describe()
    group_means = demand_data.groupby("disaster_level", sort=True)["demand"].mean()
    broken_rate = float(road_data["is_broken"].mean())
    disaster_access_correlation = float(
        demand_data[["disaster_level", "road_accessibility"]].corr().iloc[0, 1]
    )

    print("\n[生成结果]")
    print(f"demand.csv：{len(demand_data)} 行 × {len(demand_data.columns)} 列")
    print(f"road.csv：{len(road_data)} 行 × {len(road_data.columns)} 列")
    print(
        "demand min/max/mean/std："
        f"{demand_description['min']:.2f} / {demand_description['max']:.2f} / "
        f"{demand_description['mean']:.2f} / {demand_description['std']:.2f}"
    )
    print("按 disaster_level 分组的需求均值：")
    print(group_means.to_string(float_format=lambda value: f"{value:.2f}"))
    print(f"road.csv 中断正例率：{broken_rate:.2%}")
    terrain_counts = road_data.groupby("terrain")["road_id"].nunique().to_dict()
    print(f"道路地形分布：{terrain_counts}")
    print(
        "weather_sensitivity 范围："
        f"{road_data['weather_sensitivity'].min():.3f}~{road_data['weather_sensitivity'].max():.3f}"
    )
    print(f"disaster_level 与 road_accessibility 相关系数：{disaster_access_correlation:.4f}")
    print(f"demand.csv MD5：{calculate_md5(demand_path)}")
    print(f"road.csv MD5：{calculate_md5(road_path)}")


def parse_arguments() -> argparse.Namespace:
    """解析命令行覆盖参数。"""
    parser = argparse.ArgumentParser(description="生成航空应急救援模拟历史数据")
    parser.add_argument("--seed", type=int, default=None, help="覆盖 configs/default.yaml 中的随机种子")
    parser.add_argument("--out-dir", type=Path, default=None, help="覆盖 CSV 输出目录")
    return parser.parse_args()


def main() -> None:
    """读取配置、生成两份数据并输出自检摘要。"""
    arguments = parse_arguments()
    config = load_config()

    seed = int(arguments.seed if arguments.seed is not None else config["seed"])
    noise_std = float(config["ml"]["noise_std"])
    synthetic_path = resolve_project_path(
        arguments.out_dir if arguments.out_dir is not None else config["paths"]["data_synthetic"]
    )
    outputs_path = resolve_project_path(config["paths"]["outputs"])

    if noise_std <= 0:
        raise ValueError("ml.noise_std 必须为正数")

    demand_data, road_data = generate_datasets(seed=seed, noise_std=noise_std)
    demand_path = synthetic_path / "demand.csv"
    road_path = synthetic_path / "road.csv"
    save_csv(demand_data, demand_path)
    save_csv(road_data, road_path)
    create_plots(demand_data, outputs_path)

    print(f"随机种子：{seed}；需求噪声标准差：{noise_std}")
    print_summary(demand_data, road_data, demand_path, road_path)
    print(f"CSV 输出目录：{synthetic_path}")
    print(f"PNG 输出目录：{outputs_path}")


if __name__ == "__main__":
    main()
