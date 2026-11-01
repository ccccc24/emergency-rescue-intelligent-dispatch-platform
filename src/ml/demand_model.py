# -*- coding: utf-8 -*-
"""训练并评估村庄需求预测模型。

只处理需求预测：按日期做 70/30 时间切分，比较基线、线性回归和随机森林。
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.ml.data_gen import (
    DEMAND_FEATURE_LABELS,
    WEATHER_CODE,
    configure_chinese_font,
    load_config,
    resolve_project_path,
)


FEATURE_COLUMNS: Final[list[str]] = [
    "population",
    "disaster_level",
    "distance_to_center",
    "road_accessibility",
    "weather_code",
    "is_priority",
    "hist_demand_mean",
]
TARGET_COLUMN: Final[str] = "demand"
METRIC_COLUMNS: Final[list[str]] = ["model", "MAE", "RMSE", "R²"]
MODEL_LABELS: Final[dict[str, str]] = {
    "history_mean": "历史均值",
    "linear_regression": "线性回归",
    "rf_default": "随机森林（默认）",
    "rf_tuned": "随机森林（调参）",
    "linear_regression_log1p": "线性回归 + 对数变换",
    "rf_tuned_log1p": "随机森林（调参）+ 对数变换",
}
METRIC_LABELS: Final[dict[str, str]] = {
    "MAE": "平均绝对误差（MAE）",
    "RMSE": "均方根误差（RMSE）",
    "R²": "决定系数（R²）",
}


def load_demand_data(data_path: Path) -> pd.DataFrame:
    """读取需求数据并完成天气有序编码。"""
    demand_data = pd.read_csv(data_path, encoding="utf-8-sig", parse_dates=["date"])
    required_columns = {
        "village_id",
        "date",
        "weather",
        "demand",
        *(set(FEATURE_COLUMNS) - {"weather_code"}),
    }
    missing_columns = required_columns - set(demand_data.columns)
    if missing_columns:
        raise ValueError(f"demand.csv 缺少字段：{sorted(missing_columns)}")

    demand_data = demand_data.sort_values(["date", "village_id"]).reset_index(drop=True)
    weather_code = demand_data["weather"].map(WEATHER_CODE)
    if weather_code.isna().any():
        unknown_weather = sorted(demand_data.loc[weather_code.isna(), "weather"].unique())
        raise ValueError(f"存在未知天气类型：{unknown_weather}")
    demand_data["weather_code"] = weather_code.astype(int)
    return demand_data


def chronological_split(
    demand_data: pd.DataFrame,
    train_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按日期升序做时间切分，并校验训练集和测试集覆盖全部村庄。"""
    unique_dates = np.sort(demand_data["date"].unique())
    if len(unique_dates) < 2:
        raise ValueError("日期数量不足，无法切分训练集和测试集")

    train_date_count = int(len(unique_dates) * train_ratio)
    if train_date_count <= 0 or train_date_count >= len(unique_dates):
        raise ValueError("train_ratio 必须产生非空的训练日期和测试日期")

    train_dates = unique_dates[:train_date_count]
    test_dates = unique_dates[train_date_count:]
    train_data = demand_data[demand_data["date"].isin(train_dates)].copy()
    test_data = demand_data[demand_data["date"].isin(test_dates)].copy()

    if set(train_dates) & set(test_dates):
        raise ValueError("训练集和测试集存在日期重叠")

    expected_villages = set(demand_data["village_id"].astype(int).unique())
    train_villages = set(train_data["village_id"].astype(int).unique())
    test_villages = set(test_data["village_id"].astype(int).unique())
    if train_villages != expected_villages or test_villages != expected_villages:
        raise ValueError(
            "时间切分后村庄覆盖不完整："
            f"训练集={sorted(train_villages)}，测试集={sorted(test_villages)}"
        )

    return train_data, test_data


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """计算 MAE、RMSE 和 R²。"""
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R²": float(r2_score(actual, predicted)),
    }


def build_random_forest(config: dict[str, Any], tuned: bool = False) -> RandomForestRegressor:
    """按配置构造默认或调参后的随机森林，不做超参数搜索。"""
    ml_config = config["ml"]
    if tuned:
        max_depth = ml_config["rf_tuned_max_depth"]
        min_samples_leaf = ml_config["rf_tuned_min_samples_leaf"]
    else:
        max_depth = ml_config["rf_max_depth"]
        min_samples_leaf = ml_config["rf_min_samples_leaf"]
    return RandomForestRegressor(
        n_estimators=int(ml_config["rf_n_estimators"]),
        max_depth=None if max_depth is None else int(max_depth),
        min_samples_leaf=int(min_samples_leaf),
        max_features=float(ml_config["rf_max_features"]),
        n_jobs=int(ml_config["rf_n_jobs"]),
        random_state=int(config["seed"]),
    )


def build_log1p_model(base_model: Any) -> TransformedTargetRegressor:
    """包装模型，在 log1p(demand) 上训练并自动用 expm1 还原预测。"""
    return TransformedTargetRegressor(
        regressor=base_model,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def get_base_model_type(model: Any) -> str:
    """返回部署模型的实际回归器类型，便于写入 metadata。"""
    if isinstance(model, TransformedTargetRegressor):
        return type(model.regressor_).__name__
    return type(model).__name__


def save_model_metadata(
    metadata_path: Path,
    model: Any,
    metrics: dict[str, float],
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    config: dict[str, Any],
    data_path: Path,
    selected_model_name: str,
) -> None:
    """写出 bridge.py 加载模型所需的元数据。"""
    is_log1p = isinstance(model, TransformedTargetRegressor)
    metadata = {
        "model_file": "demand_model.pkl",
        "model_type": get_base_model_type(model),
        "target_transform": "log1p" if is_log1p else "none",
        "feature_order": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "selected_candidate": selected_model_name,
        "trained_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "training_date_range": {
            "start": pd.to_datetime(train_data["date"]).min().date().isoformat(),
            "end": pd.to_datetime(train_data["date"]).max().date().isoformat(),
        },
        "test_date_range": {
            "start": pd.to_datetime(test_data["date"]).min().date().isoformat(),
            "end": pd.to_datetime(test_data["date"]).max().date().isoformat(),
        },
        "test_metrics": metrics,
        "random_seed": int(config["seed"]),
        "dataset_md5": hashlib.md5(data_path.read_bytes()).hexdigest(),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_evaluation(metrics_frame: pd.DataFrame, output_path: Path) -> None:
    """写出所有接受候选模型的统一测试集指标。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(output_path, index=False, encoding="utf-8-sig", lineterminator="\n")


def plot_predictions(
    test_data: pd.DataFrame,
    predictions: np.ndarray,
    output_path: Path,
) -> None:
    """绘制测试集预测值与实际值散点图。"""
    configure_chinese_font()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 6))
    for village_id in sorted(test_data["village_id"].unique()):
        mask = test_data["village_id"] == village_id
        axis.scatter(
            test_data.loc[mask, TARGET_COLUMN],
            predictions[mask.to_numpy()],
            alpha=0.7,
            s=28,
            label=f"村庄 {int(village_id)}",
        )

    minimum = float(min(test_data[TARGET_COLUMN].min(), predictions.min()))
    maximum = float(max(test_data[TARGET_COLUMN].max(), predictions.max()))
    axis.plot([minimum, maximum], [minimum, maximum], "k--", linewidth=1.2, label="理想预测线")
    axis.set_title("测试集需求量预测值与实际值")
    axis.set_xlabel("实际需求量（箱）")
    axis.set_ylabel("预测需求量（箱）")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_feature_importance(
    feature_importance: pd.Series,
    output_path: Path,
) -> None:
    """绘制随机森林特征重要性排序图。"""
    configure_chinese_font()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_importance = feature_importance.sort_values(ascending=True)
    display_labels = [DEMAND_FEATURE_LABELS.get(name, name) for name in sorted_importance.index]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.barh(display_labels, sorted_importance.to_numpy(), color="#2E86AB")
    axis.set_title("调参随机森林特征重要性")
    axis.set_xlabel("重要性")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_model_comparison(metrics_frame: pd.DataFrame, output_path: Path) -> None:
    """为 MAE、RMSE、R² 分别绘制候选模型对比柱状图。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metric_names = ["MAE", "RMSE", "R²"]
    colors = plt.get_cmap("tab10")(np.arange(len(metrics_frame)))
    configure_chinese_font()
    model_labels = metrics_frame["model"].map(MODEL_LABELS).fillna(metrics_frame["model"]).tolist()
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    for axis, metric_name in zip(axes, metric_names):
        values = metrics_frame[metric_name].to_numpy(dtype=float)
        bars = axis.bar(model_labels, values, color=colors)
        axis.set_title(METRIC_LABELS[metric_name])
        axis.set_ylabel("指标值")
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=25, labelsize=8)
        if metric_name == "R²":
            axis.axhline(0.0, color="black", linewidth=0.8)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    figure.suptitle("需求模型在时间测试集上的指标对比")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def print_report(
    demand_data: pd.DataFrame,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    metrics_frame: pd.DataFrame,
    feature_importance: pd.Series,
    test_with_prediction: pd.DataFrame,
    selected_model_name: str,
) -> None:
    """打印切分信息、指标、特征重要性和最大误差样本。"""
    train_dates = pd.to_datetime(train_data["date"])
    test_dates = pd.to_datetime(test_data["date"])
    train_villages = sorted(train_data["village_id"].astype(int).unique())
    test_villages = sorted(test_data["village_id"].astype(int).unique())
    largest_error = test_with_prediction.sort_values("absolute_error", ascending=False).iloc[0]

    print("\n[数据切分]")
    print(
        f"总行数 {len(demand_data)}；训练集 {len(train_data)} 行（{len(train_data) / len(demand_data):.2%}），"
        f"测试集 {len(test_data)} 行（{len(test_data) / len(demand_data):.2%}）"
    )
    print(f"训练日期：{train_dates.min().date()} 至 {train_dates.max().date()}")
    print(f"测试日期：{test_dates.min().date()} 至 {test_dates.max().date()}")
    print(f"训练集村庄：{train_villages}")
    print(f"测试集村庄：{test_villages}")

    print("\n[模型指标]")
    print(metrics_frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    print("\n[调参 Random Forest Top 3 特征重要性]")
    for rank, (feature_name, importance) in enumerate(feature_importance.head(3).items(), start=1):
        print(f"{rank}. {feature_name}: {importance:.4f}")

    print("\n[测试集最大绝对误差]")
    print(
        f"村庄 {int(largest_error['village_id'])}，日期 {largest_error['date'].date()}，"
        f"实际 {largest_error[TARGET_COLUMN]:.2f}，预测 {largest_error['prediction']:.2f}，"
        f"绝对误差 {largest_error['absolute_error']:.2f}"
    )

    selected_metrics = metrics_frame.loc[metrics_frame["model"] == selected_model_name].iloc[0]
    print(f"\n[最终模型 {selected_model_name} 相对基线改进]")
    for model_name in ["history_mean", "linear_regression"]:
        if model_name == selected_model_name:
            continue
        baseline_metrics = metrics_frame.loc[metrics_frame["model"] == model_name].iloc[0]
        mae_improvement = (
            (baseline_metrics["MAE"] - selected_metrics["MAE"])
            / baseline_metrics["MAE"]
            * 100.0
        )
        rmse_improvement = (
            (baseline_metrics["RMSE"] - selected_metrics["RMSE"])
            / baseline_metrics["RMSE"]
            * 100.0
        )
        print(
            f"相对 {model_name}：MAE 改善 {mae_improvement:.2f}%，"
            f"RMSE 改善 {rmse_improvement:.2f}%"
        )


def main() -> None:
    """训练候选模型，按测试 MAE 选型，并持久化部署模型和元数据。"""
    # Windows 默认 GBK 控制台可能无法编码 R²，替换而不中断图表和文件输出。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    config = load_config()
    data_path = resolve_project_path(config["paths"]["data_synthetic"]) / "demand.csv"
    models_path = resolve_project_path(config["paths"]["models"])
    outputs_path = resolve_project_path(config["paths"]["outputs"])

    demand_data = load_demand_data(data_path)
    train_data, test_data = chronological_split(demand_data, float(config["ml"]["train_ratio"]))
    x_train = train_data[FEATURE_COLUMNS]
    y_train = train_data[TARGET_COLUMN].to_numpy(dtype=float)
    x_test = test_data[FEATURE_COLUMNS]
    y_test = test_data[TARGET_COLUMN].to_numpy(dtype=float)

    linear_model = LinearRegression().fit(x_train, y_train)
    rf_default_model = build_random_forest(config, tuned=False).fit(x_train, y_train)
    rf_tuned_model = build_random_forest(config, tuned=True).fit(x_train, y_train)
    lr_log1p_model = build_log1p_model(LinearRegression()).fit(x_train, y_train)
    rf_log1p_model = build_log1p_model(build_random_forest(config, tuned=True)).fit(x_train, y_train)

    predictions_by_model = {
        "history_mean": test_data["hist_demand_mean"].to_numpy(dtype=float),
        "linear_regression": linear_model.predict(x_test),
        "rf_default": rf_default_model.predict(x_test),
        "rf_tuned": rf_tuned_model.predict(x_test),
        "linear_regression_log1p": lr_log1p_model.predict(x_test),
        "rf_tuned_log1p": rf_log1p_model.predict(x_test),
    }
    metrics_by_model = {
        model_name: calculate_metrics(y_test, predictions)
        for model_name, predictions in predictions_by_model.items()
    }

    log1p_threshold = float(config["ml"]["log1p_min_mae_improvement"])
    lr_log1p_improvement = (
        (metrics_by_model["linear_regression"]["MAE"] - metrics_by_model["linear_regression_log1p"]["MAE"])
        / metrics_by_model["linear_regression"]["MAE"]
    )
    rf_log1p_improvement = (
        (metrics_by_model["rf_tuned"]["MAE"] - metrics_by_model["rf_tuned_log1p"]["MAE"])
        / metrics_by_model["rf_tuned"]["MAE"]
    )
    accepted_log1p_models: set[str] = set()
    if lr_log1p_improvement > log1p_threshold:
        accepted_log1p_models.add("linear_regression_log1p")
    if rf_log1p_improvement > log1p_threshold:
        accepted_log1p_models.add("rf_tuned_log1p")

    evaluated_model_names = ["history_mean", "linear_regression", "rf_default", "rf_tuned"]
    evaluated_model_names.extend(sorted(accepted_log1p_models))
    metrics_frame = pd.DataFrame(
        [
            {"model": model_name, **metrics_by_model[model_name]}
            for model_name in evaluated_model_names
        ],
        columns=METRIC_COLUMNS,
    )

    deployment_candidates = ["linear_regression", "rf_tuned", *sorted(accepted_log1p_models)]
    selected_model_name = min(
        deployment_candidates,
        key=lambda model_name: metrics_by_model[model_name]["MAE"],
    )
    model_registry = {
        "linear_regression": linear_model,
        "rf_tuned": rf_tuned_model,
        "linear_regression_log1p": lr_log1p_model,
        "rf_tuned_log1p": rf_log1p_model,
    }
    selected_model = model_registry[selected_model_name]
    selected_predictions = predictions_by_model[selected_model_name]

    feature_importance = pd.Series(
        rf_tuned_model.feature_importances_,
        index=FEATURE_COLUMNS,
        name="importance",
    ).sort_values(ascending=False)
    test_with_prediction = test_data.copy()
    test_with_prediction["prediction"] = selected_predictions
    test_with_prediction["absolute_error"] = (
        test_with_prediction[TARGET_COLUMN] - test_with_prediction["prediction"]
    ).abs()

    evaluation_path = outputs_path / "demand_eval.csv"
    deployment_model_path = models_path / "demand_model.pkl"
    rf_comparison_path = models_path / "demand_rf.pkl"
    metadata_path = models_path / "demand_model_meta.json"
    save_evaluation(metrics_frame, evaluation_path)
    models_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_model, deployment_model_path)
    joblib.dump(rf_tuned_model, rf_comparison_path)
    save_model_metadata(
        metadata_path=metadata_path,
        model=selected_model,
        metrics=metrics_by_model[selected_model_name],
        train_data=train_data,
        test_data=test_data,
        config=config,
        data_path=data_path,
        selected_model_name=selected_model_name,
    )

    plot_predictions(
        test_data,
        selected_predictions,
        outputs_path / "demand_pred_vs_actual.png",
    )
    plot_feature_importance(
        feature_importance,
        outputs_path / "demand_feature_importance.png",
    )
    plot_model_comparison(metrics_frame, outputs_path / "demand_model_compare.png")

    rf_default_train_mae = float(mean_absolute_error(y_train, rf_default_model.predict(x_train)))
    rf_tuned_train_mae = float(mean_absolute_error(y_train, rf_tuned_model.predict(x_train)))
    print(f"数据文件：{data_path}")
    print_report(
        demand_data,
        train_data,
        test_data,
        metrics_frame,
        feature_importance,
        test_with_prediction,
        selected_model_name,
    )
    print("\n[选型与过拟合]")
    print(f"最终部署模型：{selected_model_name}")
    print(
        f"LR log1p MAE 改善：{lr_log1p_improvement:.2%}，"
        f"{'接受' if 'linear_regression_log1p' in accepted_log1p_models else '放弃'}"
    )
    print(
        f"RF tuned log1p MAE 改善：{rf_log1p_improvement:.2%}，"
        f"{'接受' if 'rf_tuned_log1p' in accepted_log1p_models else '放弃'}"
    )
    print(
        f"RF 默认训练/测试 MAE：{rf_default_train_mae:.4f} / "
        f"{metrics_by_model['rf_default']['MAE']:.4f}，差距 "
        f"{metrics_by_model['rf_default']['MAE'] - rf_default_train_mae:.4f}"
    )
    print(
        f"RF 调参训练/测试 MAE：{rf_tuned_train_mae:.4f} / "
        f"{metrics_by_model['rf_tuned']['MAE']:.4f}，差距 "
        f"{metrics_by_model['rf_tuned']['MAE'] - rf_tuned_train_mae:.4f}"
    )
    print(f"\n部署模型：{deployment_model_path}")
    print(f"RF 对照：{rf_comparison_path}")
    print(f"模型元数据：{metadata_path}")
    print(f"指标文件：{evaluation_path}")
    print(f"图表目录：{outputs_path}")


if __name__ == "__main__":
    main()
