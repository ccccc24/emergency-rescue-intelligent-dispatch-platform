# -*- coding: utf-8 -*-
"""把需求预测和道路风险模型的输出转换为调度器可读取的场景。

本模块只负责特征准备、模型推理和结构化输出，不在 bridge 层加入封路、
地形禁行等业务判断。所有概率和速度均以连续值传给调度器。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd

from src.ml.data_gen import (
    TERRAIN_TYPES,
    WEATHER_CODE,
    load_config,
    resolve_project_path,
)
from src.ml.demand_model import load_demand_data
from src.ml.risk_model import load_road_data


DEFAULT_CONFIG_PATH: Final[Path] = Path("configs/default.yaml")
PRIORITY_DISASTER_LEVEL: Final[int] = 4
DEFAULT_PRIORITY_EDGE_MEAN_THRESHOLD: Final[float] = 0.6
DEFAULT_ISOLATION_EDGE_THRESHOLD: Final[float] = 0.75


@dataclass(frozen=True)
class VillageInfo:
    """场景中的村庄静态和当日信息。"""

    id: int
    name: str
    population: int
    disaster_level: int
    is_priority: int
    distance_to_center: float
    road_accessibility: float
    weather_code: int
    hist_demand_mean: float


@dataclass(frozen=True)
class RoadInfo:
    """场景中的道路静态和当日信息。

    ``from`` / ``to`` 是 Python 保留字，因此内部字段使用 ``from_node`` /
    ``to_node``；写出 JSON 时恢复为调度器约定的 ``from`` / ``to``。
    """

    id: int
    code: str
    from_node: int
    to_node: int
    length: float
    base_speed: float
    terrain: str
    weather_code: int
    weather: str
    geology_risk: float
    weather_sensitivity: float
    hist_break_count: int


@dataclass
class Scenario:
    """一次完整调度场景。"""

    date: str
    villages: dict[int, VillageInfo]
    roads: dict[int, RoadInfo]
    demands: dict[int, float]
    risk_scores: dict[int, float]
    priority_villages: set[int]
    adjusted_speed: dict[int, float]
    isolated_villages: set[int]
    metadata: dict[str, Any] = field(default_factory=dict)


def _resolve_model_path(raw_path: str | Path) -> Path:
    """将模型路径解析为绝对路径，并校验文件存在。"""

    path = resolve_project_path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(f"模型文件不存在：{path}")
    return path


def _load_model_metadata(model_path: Path) -> dict[str, Any]:
    """读取与模型同名的元数据文件，保证特征顺序与训练时一致。"""

    metadata_path = model_path.with_name(f"{model_path.stem}_meta.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"模型元数据不存在：{metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_order = metadata.get("feature_order")
    if not isinstance(feature_order, list) or not feature_order:
        raise ValueError(f"模型元数据缺少有效的 feature_order：{metadata_path}")
    return metadata


def _target_timestamp(target_date: str | Path | pd.Timestamp | None, demand_data: pd.DataFrame) -> pd.Timestamp:
    """确定场景日期；未指定时使用需求数据中的最后一天。"""

    if target_date is None:
        return pd.Timestamp(pd.to_datetime(demand_data["date"]).max()).normalize()
    timestamp = pd.Timestamp(pd.to_datetime(target_date)).normalize()
    if pd.isna(timestamp):
        raise ValueError(f"无法解析目标日期：{target_date!r}")
    return timestamp


def _filter_target_day(
    data_frame: pd.DataFrame,
    target_timestamp: pd.Timestamp,
    entity_column: str,
) -> pd.DataFrame:
    """筛选目标日期的记录，并校验实体没有重复。"""

    dates = pd.to_datetime(data_frame["date"]).dt.normalize()
    day_data = data_frame.loc[dates == target_timestamp].copy()
    if day_data.empty:
        raise ValueError(f"目标日期 {target_timestamp.date()} 在 {entity_column} 数据中不存在")
    if day_data[entity_column].duplicated().any():
        duplicated = sorted(day_data.loc[day_data[entity_column].duplicated(), entity_column].unique())
        raise ValueError(f"目标日期存在重复 {entity_column}：{duplicated}")
    return day_data.sort_values(entity_column).reset_index(drop=True)


def _build_feature_frame(
    rows: pd.DataFrame,
    feature_order: list[str],
    data_name: str,
) -> pd.DataFrame:
    """按模型元数据顺序生成数值特征矩阵。"""

    feature_frame = pd.DataFrame(index=rows.index)
    for feature_name in feature_order:
        if feature_name == "weather_code":
            feature_frame[feature_name] = rows["weather"].map(WEATHER_CODE)
        elif feature_name in {"terrain", "terrain_code"}:
            terrain_mapping = {name: index for index, name in enumerate(TERRAIN_TYPES)}
            feature_frame[feature_name] = rows["terrain"].map(terrain_mapping)
        elif feature_name in rows.columns:
            feature_frame[feature_name] = rows[feature_name]
        else:
            raise ValueError(f"{data_name} 缺少模型所需特征：{feature_name}")

    if feature_frame.isna().any().any():
        missing_columns = feature_frame.columns[feature_frame.isna().any()].tolist()
        raise ValueError(f"{data_name} 特征存在空值：{missing_columns}")
    numeric_frame = feature_frame.astype(float)
    if not np.isfinite(numeric_frame.to_numpy()).all():
        raise ValueError(f"{data_name} 特征包含无穷大或非数值")
    return numeric_frame


def _road_numeric_id(road_code: str, used_ids: set[int]) -> int:
    """把 ``R01`` 形式的道路编号转换为稳定的整数主键。"""

    road_text = str(road_code).strip()
    numeric_text = road_text[1:] if road_text.upper().startswith("R") else road_text
    if not numeric_text.isdigit():
        raise ValueError(f"道路编号无法转换为整数：{road_code!r}")
    road_id = int(numeric_text)
    if road_id in used_ids:
        raise ValueError(f"道路编号转换后重复：{road_code!r} -> {road_id}")
    return road_id


def _positive_probability(model: Any, feature_frame: pd.DataFrame) -> np.ndarray:
    """取分类模型的正类（道路中断）概率。"""

    probability = np.asarray(model.predict_proba(feature_frame), dtype=float)
    if probability.ndim != 2 or probability.shape[1] < 2:
        raise ValueError("风险模型 predict_proba 未返回二分类概率")
    classes = list(getattr(model, "classes_", [0, 1]))
    if 1 not in classes:
        raise ValueError(f"风险模型缺少正类标签 1：classes={classes}")
    return probability[:, classes.index(1)]


def _file_md5(path: Path) -> str:
    """计算数据文件 MD5，写入场景元数据。"""

    return hashlib.md5(path.read_bytes()).hexdigest()


def _build_metadata(
    demand_model: Any,
    risk_model: Any,
    demand_metadata: dict[str, Any],
    risk_metadata: dict[str, Any],
    config: dict[str, Any],
    demand_path: Path,
    road_path: Path,
) -> dict[str, Any]:
    """汇总模型、配置和数据版本信息。"""

    demand_metrics = demand_metadata.get("test_metrics", {})
    risk_metrics = risk_metadata.get("test_metrics", {})
    demand_mae = demand_metrics.get("MAE", demand_metrics.get("mae"))
    risk_auc = risk_metrics.get("auc", risk_metrics.get("AUC"))
    return {
        "demand_model": {
            "type": demand_metadata.get("model_type", type(demand_model).__name__),
            "trained_date": demand_metadata.get("trained_at"),
            "test_mae": None if demand_mae is None else float(demand_mae),
        },
        "risk_model": {
            "type": risk_metadata.get("model_type", type(risk_model).__name__),
            "trained_date": risk_metadata.get("trained_at"),
            "test_auc": None if risk_auc is None else float(risk_auc),
            "features_used": list(risk_metadata["feature_order"]),
        },
        "config_seed": int(config["seed"]),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "data_md5": {
            "demand": _file_md5(demand_path),
            "road": _file_md5(road_path),
        },
    }


def prepare_scenario(
    demand_model_path: str | Path,
    risk_model_path: str | Path,
    config_path: str | Path,
    target_date: str | None = None,
) -> Scenario:
    """加载两个模型，生成指定日期的调度场景并写出 JSON。

    Parameters
    ----------
    demand_model_path:
        需求预测模型路径，模型元数据文件需要与模型同目录、同主文件名。
    risk_model_path:
        道路风险分类模型路径，元数据要求同上。
    config_path:
        YAML 配置文件路径。
    target_date:
        ``YYYY-MM-DD`` 格式日期；为空时使用需求数据最后一天。
    """

    config_file = resolve_project_path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"配置文件不存在：{config_file}")
    config = load_config(config_file)

    demand_model_file = _resolve_model_path(demand_model_path)
    risk_model_file = _resolve_model_path(risk_model_path)
    demand_metadata = _load_model_metadata(demand_model_file)
    risk_metadata = _load_model_metadata(risk_model_file)

    demand_path = resolve_project_path(config["paths"]["data_synthetic"]) / "demand.csv"
    road_path = resolve_project_path(config["paths"]["data_synthetic"]) / "road.csv"
    demand_data = load_demand_data(demand_path)
    road_data = load_road_data(road_path)

    target_timestamp = _target_timestamp(target_date, demand_data)
    date_text = target_timestamp.date().isoformat()
    demand_day = _filter_target_day(demand_data, target_timestamp, "village_id")
    road_day = _filter_target_day(road_data, target_timestamp, "road_id")

    demand_model = joblib.load(demand_model_file)
    risk_model = joblib.load(risk_model_file)

    demand_features = _build_feature_frame(
        demand_day,
        list(demand_metadata["feature_order"]),
        "需求数据",
    )
    risk_features = _build_feature_frame(
        road_day,
        list(risk_metadata["feature_order"]),
        "道路数据",
    )

    predicted_demands = np.asarray(demand_model.predict(demand_features), dtype=float)
    if predicted_demands.shape[0] != len(demand_day):
        raise ValueError("需求模型输出数量与村庄数量不一致")

    villages: dict[int, VillageInfo] = {}
    demands: dict[int, float] = {}
    for row, prediction in zip(demand_day.itertuples(index=False), predicted_demands, strict=True):
        village_id = int(row.village_id)
        villages[village_id] = VillageInfo(
            id=village_id,
            name=f"村庄{village_id}",
            population=int(row.population),
            disaster_level=int(row.disaster_level),
            is_priority=int(row.is_priority),
            distance_to_center=float(row.distance_to_center),
            road_accessibility=float(row.road_accessibility),
            weather_code=int(row.weather_code),
            hist_demand_mean=float(row.hist_demand_mean),
        )
        demands[village_id] = float(prediction)

    roads: dict[int, RoadInfo] = {}
    used_road_ids: set[int] = set()
    for _, row in road_day.iterrows():
        road_code = str(row["road_id"])
        road_id = _road_numeric_id(road_code, used_road_ids)
        used_road_ids.add(road_id)
        roads[road_id] = RoadInfo(
            id=road_id,
            code=road_code,
            from_node=int(row["from"]),
            to_node=int(row["to"]),
            length=float(row["length"]),
            base_speed=float(row["base_speed"]),
            terrain=str(row["terrain"]),
            weather_code=int(row["weather_code"]),
            weather=str(row["weather"]),
            geology_risk=float(row["geology_risk"]),
            weather_sensitivity=float(row["weather_sensitivity"]),
            hist_break_count=int(row["hist_break_count"]),
        )

    positive_probability = _positive_probability(risk_model, risk_features)
    risk_scores = {
        road_id: float(score)
        for road_id, score in zip(roads.keys(), positive_probability, strict=True)
    }

    incident_risks: dict[int, list[float]] = {village_id: [] for village_id in villages}
    for road in roads.values():
        for node_id in (road.from_node, road.to_node):
            if node_id in incident_risks:
                incident_risks[node_id].append(risk_scores[road.id])

    priority_threshold = float(
        config.get("risk", {}).get(
            "priority_village_edge_mean_threshold",
            DEFAULT_PRIORITY_EDGE_MEAN_THRESHOLD,
        )
    )
    isolation_threshold = float(
        config.get("risk", {}).get(
            "isolation_edge_threshold",
            DEFAULT_ISOLATION_EDGE_THRESHOLD,
        )
    )
    priority_villages: set[int] = set()
    isolated_villages: set[int] = set()
    for village_id, village in villages.items():
        edge_risks = incident_risks[village_id]
        edge_mean = float(np.mean(edge_risks)) if edge_risks else 0.0
        if village.disaster_level >= PRIORITY_DISASTER_LEVEL or edge_mean >= priority_threshold:
            priority_villages.add(village_id)
        if not edge_risks or all(score > isolation_threshold for score in edge_risks):
            isolated_villages.add(village_id)

    speed_penalty = float(config["risk"]["speed_penalty"])
    adjusted_speed = {
        road_id: roads[road_id].base_speed * (1.0 - risk_scores[road_id] * speed_penalty)
        for road_id in roads
    }

    metadata = _build_metadata(
        demand_model=demand_model,
        risk_model=risk_model,
        demand_metadata=demand_metadata,
        risk_metadata=risk_metadata,
        config=config,
        demand_path=demand_path,
        road_path=road_path,
    )
    scenario = Scenario(
        date=date_text,
        villages=villages,
        roads=roads,
        demands=demands,
        risk_scores=risk_scores,
        priority_villages=priority_villages,
        adjusted_speed=adjusted_speed,
        isolated_villages=isolated_villages,
        metadata=metadata,
    )
    validate_scenario(scenario)
    save_scenario(scenario, resolve_project_path(config["paths"]["outputs"]))
    return scenario


def validate_scenario(scenario: Scenario) -> None:
    """检查 bridge 输出满足调度器前置契约。"""

    village_ids = set(scenario.villages)
    road_ids = set(scenario.roads)
    if not scenario.demands or set(scenario.demands) != village_ids:
        raise AssertionError("demands 必须覆盖全部村庄")
    if any(value <= 0 for value in scenario.demands.values()):
        raise AssertionError("demands 存在非正值")
    if set(scenario.risk_scores) != road_ids:
        raise AssertionError("risk_scores 必须覆盖全部道路")
    if any(value < 0.0 or value > 1.0 for value in scenario.risk_scores.values()):
        raise AssertionError("risk_scores 存在超出 [0, 1] 的值")
    if not scenario.priority_villages <= village_ids:
        raise AssertionError("priority_villages 不是 villages 的子集")
    if not scenario.isolated_villages <= village_ids:
        raise AssertionError("isolated_villages 不是 villages 的子集")
    if set(scenario.adjusted_speed) != road_ids:
        raise AssertionError("adjusted_speed 必须覆盖全部道路")


def _scenario_to_dict(scenario: Scenario) -> dict[str, Any]:
    """把 dataclass 场景转换为可直接写入 JSON 的字典。"""

    villages = {
        str(village_id): asdict(village)
        for village_id, village in sorted(scenario.villages.items())
    }
    roads = {
        str(road_id): {
            "id": road.id,
            "road_id": road.code,
            "from": road.from_node,
            "to": road.to_node,
            "length": road.length,
            "base_speed": road.base_speed,
            "terrain": road.terrain,
            "weather": road.weather,
            "weather_code": road.weather_code,
            "geology_risk": road.geology_risk,
            "weather_sensitivity": road.weather_sensitivity,
            "hist_break_count": road.hist_break_count,
        }
        for road_id, road in sorted(scenario.roads.items())
    }
    return {
        "date": scenario.date,
        "villages": villages,
        "roads": roads,
        "demands": {
            str(village_id): demand
            for village_id, demand in sorted(scenario.demands.items())
        },
        "risk_scores": {
            str(road_id): risk
            for road_id, risk in sorted(scenario.risk_scores.items())
        },
        "priority_villages": sorted(scenario.priority_villages),
        "adjusted_speed": {
            str(road_id): speed
            for road_id, speed in sorted(scenario.adjusted_speed.items())
        },
        "isolated_villages": sorted(scenario.isolated_villages),
        "metadata": scenario.metadata,
    }


def save_scenario(scenario: Scenario, output_dir: Path) -> tuple[Path, Path]:
    """写出日期场景和 latest 副本，均使用 UTF-8 JSON。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_path = output_dir / f"scenario_{scenario.date}.json"
    latest_path = output_dir / "scenario_latest.json"
    payload = json.dumps(
        _scenario_to_dict(scenario),
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )
    scenario_path.write_text(payload + "\n", encoding="utf-8")
    shutil.copyfile(scenario_path, latest_path)
    return scenario_path, latest_path


def print_scenario_summary(scenario: Scenario) -> None:
    """打印场景摘要，便于联调和答辩时快速检查。"""

    print(f"场景日期：{scenario.date}")
    print("\n[村庄需求预测]")
    for village_id, demand in sorted(scenario.demands.items()):
        village = scenario.villages[village_id]
        print(
            f"  {village.name}（id={village_id}）：{demand:.2f} 箱；"
            f"灾情等级={village.disaster_level}，优先村={village.is_priority}"
        )

    risk_values = np.array(list(scenario.risk_scores.values()), dtype=float)
    print("\n[道路风险分布]")
    print(
        f"  min={risk_values.min():.4f}，median={np.median(risk_values):.4f}，"
        f"max={risk_values.max():.4f}"
    )
    print(f"  优先村：{sorted(scenario.priority_villages)}")
    print(
        "  孤岛村："
        f"{sorted(scenario.isolated_villages) if scenario.isolated_villages else '无（路网状态良好）'}"
    )

    slowest_road_id = min(scenario.adjusted_speed, key=scenario.adjusted_speed.get)
    fastest_road_id = max(scenario.adjusted_speed, key=scenario.adjusted_speed.get)
    slowest_road = scenario.roads[slowest_road_id]
    fastest_road = scenario.roads[fastest_road_id]
    print("\n[调整后速度]")
    print(
        f"  最低：{slowest_road.code}（{slowest_road.from_node}->{slowest_road.to_node}，"
        f"{slowest_road.terrain}）={scenario.adjusted_speed[slowest_road_id]:.2f} km/h；"
        f"risk={scenario.risk_scores[slowest_road_id]:.4f}"
    )
    print(
        f"  最高：{fastest_road.code}（{fastest_road.from_node}->{fastest_road.to_node}，"
        f"{fastest_road.terrain}）={scenario.adjusted_speed[fastest_road_id]:.2f} km/h；"
        f"risk={scenario.risk_scores[fastest_road_id]:.4f}"
    )


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="生成航空应急救援调度场景 JSON")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="配置文件路径")
    parser.add_argument("--target-date", type=str, default=None, help="场景日期，默认使用数据最后一天")
    parser.add_argument("--demand-model", type=Path, default=None, help="需求模型路径")
    parser.add_argument("--risk-model", type=Path, default=None, help="风险模型路径")
    return parser.parse_args()


def main() -> None:
    """生成场景文件并打印摘要。"""

    arguments = parse_arguments()
    config = load_config(resolve_project_path(arguments.config))
    models_path = resolve_project_path(config["paths"]["models"])
    demand_model_path = arguments.demand_model or models_path / "demand_model.pkl"
    risk_model_path = arguments.risk_model or models_path / "risk_model.pkl"
    scenario = prepare_scenario(
        demand_model_path=demand_model_path,
        risk_model_path=risk_model_path,
        config_path=arguments.config,
        target_date=arguments.target_date,
    )
    print_scenario_summary(scenario)
    print(f"\n场景文件：{resolve_project_path(config['paths']['outputs']) / f'scenario_{scenario.date}.json'}")
    print("最新场景：outputs/scenario_latest.json")


if __name__ == "__main__":
    main()
