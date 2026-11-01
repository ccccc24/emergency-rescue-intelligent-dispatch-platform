# -*- coding: utf-8 -*-
"""训练并评估道路中断风险分类模型。

数据按日期做 70/30 时间切分，比较全 0 基线、逻辑回归和随机森林。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.ml.data_gen import (
    ROAD_FEATURE_LABELS,
    WEATHER_CODE,
    configure_chinese_font,
    load_config,
    resolve_project_path,
)


FEATURE_COLUMNS: Final[list[str]] = [
    "length",
    "base_speed",
    "weather_code",
    "geology_risk",
    "hist_break_count",
]
TARGET_COLUMN: Final[str] = "is_broken"
METRIC_COLUMNS: Final[list[str]] = ["model", "accuracy", "precision", "recall", "f1", "auc"]


def load_road_data(data_path: Path) -> pd.DataFrame:
    """读取道路数据并完成天气有序编码。"""
    road_data = pd.read_csv(data_path, encoding="utf-8-sig", parse_dates=["date"])
    required_columns = {
        "road_id",
        "date",
        "weather",
        "is_broken",
        *(set(FEATURE_COLUMNS) - {"weather_code"}),
    }
    missing_columns = required_columns - set(road_data.columns)
    if missing_columns:
        raise ValueError(f"road.csv 缺少字段：{sorted(missing_columns)}")

    road_data = road_data.sort_values(["date", "road_id"]).reset_index(drop=True)
    weather_code = road_data["weather"].map(WEATHER_CODE)
    if weather_code.isna().any():
        unknown_weather = sorted(road_data.loc[weather_code.isna(), "weather"].unique())
        raise ValueError(f"存在未知天气类型：{unknown_weather}")
    road_data["weather_code"] = weather_code.astype(int)
    return road_data


def chronological_split(
    road_data: pd.DataFrame,
    train_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按日期升序切分，并校验训练集和测试集覆盖全部道路。"""
    unique_dates = np.sort(road_data["date"].unique())
    if len(unique_dates) < 2:
        raise ValueError("日期数量不足，无法切分训练集和测试集")

    train_date_count = int(np.floor(len(unique_dates) * train_ratio + 1e-9))
    if train_date_count <= 0 or train_date_count >= len(unique_dates):
        raise ValueError("train_ratio 必须产生非空的训练日期和测试日期")

    train_dates = unique_dates[:train_date_count]
    test_dates = unique_dates[train_date_count:]
    train_data = road_data[road_data["date"].isin(train_dates)].copy()
    test_data = road_data[road_data["date"].isin(test_dates)].copy()

    if set(train_dates) & set(test_dates):
        raise ValueError("训练集和测试集存在日期重叠")

    expected_roads = set(road_data["road_id"].astype(str).unique())
    train_roads = set(train_data["road_id"].astype(str).unique())
    test_roads = set(test_data["road_id"].astype(str).unique())
    if train_roads != expected_roads or test_roads != expected_roads:
        raise ValueError(
            "时间切分后道路覆盖不完整："
            f"训练集={sorted(train_roads)}，测试集={sorted(test_roads)}"
        )

    return train_data, test_data


def classification_metrics(
    actual: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray | None = None,
) -> dict[str, float]:
    """计算分类指标；无有效概率分数时 AUC 返回 NaN。"""
    auc_value = float("nan")
    if scores is not None and len(np.unique(actual)) == 2:
        auc_value = float(roc_auc_score(actual, scores))
    return {
        "accuracy": float(accuracy_score(actual, labels)),
        "precision": float(precision_score(actual, labels, zero_division=0)),
        "recall": float(recall_score(actual, labels, zero_division=0)),
        "f1": float(f1_score(actual, labels, zero_division=0)),
        "auc": auc_value,
    }


def metrics_at_threshold(actual: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """按给定概率阈值计算分类指标。"""
    labels = (scores >= threshold).astype(int)
    return classification_metrics(actual, labels, scores)


def save_evaluation(metrics_frame: pd.DataFrame, output_path: Path) -> None:
    """写出三个模型的测试集指标。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(output_path, index=False, encoding="utf-8-sig", lineterminator="\n")


def validate_hist_break_count(road_data: pd.DataFrame) -> int:
    """复核 hist_break_count 是否严格只统计当天之前的 7 天中断次数。"""
    ordered_data = road_data.sort_values(["road_id", "date"]).copy()
    expected_history = (
        ordered_data.groupby("road_id", sort=False)["is_broken"]
        .transform(lambda series: series.shift(1).rolling(window=7, min_periods=1).sum())
        .fillna(0)
        .astype(int)
    )
    return int((ordered_data["hist_break_count"].astype(int) != expected_history).sum())


def build_logistic_regression(config: dict[str, Any]) -> LogisticRegression:
    """按配置构造带类别平衡权重的逻辑回归。"""
    ml_config = config["ml"]
    return LogisticRegression(
        class_weight="balanced",
        max_iter=int(ml_config["risk_lr_max_iter"]),
        random_state=int(config["seed"]),
    )


def build_random_forest(config: dict[str, Any]) -> RandomForestClassifier:
    """按配置构造轻量随机森林，不做额外超参数搜索。"""
    ml_config = config["ml"]
    return RandomForestClassifier(
        n_estimators=int(ml_config["risk_rf_n_estimators"]),
        max_depth=int(ml_config["risk_rf_max_depth"]),
        class_weight="balanced",
        random_state=int(config["seed"]),
        n_jobs=int(ml_config["risk_rf_n_jobs"]),
    )


def positive_probability(model: Any, features: pd.DataFrame) -> np.ndarray:
    """返回二分类模型对正类（道路中断）的预测概率。"""
    return model.predict_proba(features)[:, 1].astype(float)


def plot_roc_curve(
    actual: np.ndarray,
    lr_scores: np.ndarray,
    rf_scores: np.ndarray,
    output_path: Path,
) -> None:
    """在同一张图中绘制逻辑回归与随机森林的 ROC 曲线。"""
    configure_chinese_font()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lr_fpr, lr_tpr, _ = roc_curve(actual, lr_scores)
    rf_fpr, rf_tpr, _ = roc_curve(actual, rf_scores)
    lr_auc = float(roc_auc_score(actual, lr_scores))
    rf_auc = float(roc_auc_score(actual, rf_scores))

    figure, axis = plt.subplots(figsize=(7.5, 6))
    axis.plot(lr_fpr, lr_tpr, color="#2E86AB", linewidth=2, label=f"逻辑回归（AUC={lr_auc:.3f}）")
    axis.plot(rf_fpr, rf_tpr, color="#D1495B", linewidth=2, label=f"随机森林（AUC={rf_auc:.3f}）")
    axis.plot([0, 1], [0, 1], "k--", linewidth=1, label="随机猜测")
    axis.set_title("道路中断风险 ROC 曲线")
    axis.set_xlabel("假阳性率")
    axis.set_ylabel("真阳性率")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right", frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_confusion_matrix(
    actual: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    """绘制随机森林在指定业务阈值下的混淆矩阵。"""
    configure_chinese_font()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(actual, labels, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(6.2, 5.4))
    image = axis.imshow(matrix, cmap="Blues")
    class_names = ["未中断", "中断"]
    axis.set_xticks([0, 1], labels=class_names)
    axis.set_yticks([0, 1], labels=class_names)
    axis.set_xlabel("预测标签")
    axis.set_ylabel("真实标签")
    axis.set_title(f"随机森林混淆矩阵（阈值={threshold:.2f}）")

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = int(matrix[row_index, column_index])
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > matrix.max() / 2 else "black",
                fontsize=12,
            )

    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_feature_importance(feature_importance: pd.Series, output_path: Path) -> None:
    """绘制随机森林特征重要性排序图。"""
    configure_chinese_font()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_importance = feature_importance.sort_values(ascending=True)
    display_labels = [ROAD_FEATURE_LABELS.get(name, name) for name in sorted_importance.index]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.barh(display_labels, sorted_importance.to_numpy(), color="#D1495B")
    axis.set_title("随机森林特征重要性：道路中断风险")
    axis.set_xlabel("重要性")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def build_latest_risk_scores(
    test_data: pd.DataFrame,
    rf_scores: np.ndarray,
) -> tuple[str, list[dict[str, Any]]]:
    """取出测试集最后一天每条道路的中断概率，作为 bridge.py 可用的 risk_score。"""
    latest_date = pd.to_datetime(test_data["date"]).max()
    latest_mask = pd.to_datetime(test_data["date"]).to_numpy() == latest_date.to_datetime64()
    latest_rows = test_data.loc[latest_mask, ["road_id", "from", "to"]].copy()
    latest_rows["risk_score"] = rf_scores[latest_mask]
    latest_rows = latest_rows.sort_values("road_id")

    risk_scores = [
        {
            "road_id": str(row["road_id"]),
            "from": int(row["from"]),
            "to": int(row["to"]),
            "risk_score": round(float(row["risk_score"]), 6),
        }
        for _, row in latest_rows.iterrows()
    ]
    return latest_date.date().isoformat(), risk_scores


def save_model_metadata(metadata_path: Path, metadata: dict[str, Any]) -> None:
    """以 UTF-8 JSON 保存模型契约和最新风险分数。"""
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def print_report(
    road_data: pd.DataFrame,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    metrics_frame: pd.DataFrame,
    feature_importance: pd.Series,
    rf_scores: np.ndarray,
    default_threshold: float,
    candidate_threshold: float,
    threshold_analysis: dict[str, dict[str, float]],
    latest_date: str,
    latest_risk_scores: list[dict[str, Any]],
    leakage_mismatches: int,
) -> None:
    """打印切分、泄漏校验、指标、阈值和特征重要性报告。"""
    train_dates = pd.to_datetime(train_data["date"])
    test_dates = pd.to_datetime(test_data["date"])
    first_day = pd.to_datetime(road_data["date"]).min()
    first_day_history = road_data.loc[road_data["date"] == first_day, "hist_break_count"]

    print("\n[前置检查：hist_break_count]")
    print("定义：截至当天之前 7 天的 is_broken 之和，不包含当天标签。")
    print(
        f"逐行复算不一致数：{leakage_mismatches}；"
        f"首日取值：{sorted(first_day_history.astype(int).unique().tolist())}；"
        f"全量范围：{int(road_data['hist_break_count'].min())}~{int(road_data['hist_break_count'].max())}"
    )

    print("\n[数据切分]")
    print(
        f"总行数 {len(road_data)}；训练集 {len(train_data)} 行，测试集 {len(test_data)} 行；"
        f"训练日期 {train_dates.min().date()} 至 {train_dates.max().date()}；"
        f"测试日期 {test_dates.min().date()} 至 {test_dates.max().date()}"
    )
    print(
        f"道路覆盖：训练集 {train_data['road_id'].nunique()} 条，"
        f"测试集 {test_data['road_id'].nunique()} 条，全量 {road_data['road_id'].nunique()} 条"
    )
    print(
        f"正例率：训练集 {train_data[TARGET_COLUMN].mean():.2%}，"
        f"测试集 {test_data[TARGET_COLUMN].mean():.2%}"
    )

    print("\n[测试集指标，默认阈值 0.50]")
    print(metrics_frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    rf_auc = float(metrics_frame.loc[metrics_frame["model"] == "random_forest", "auc"].iloc[0])
    lr_auc = float(metrics_frame.loc[metrics_frame["model"] == "logistic_regression", "auc"].iloc[0])
    print(f"\nAUC：LR={lr_auc:.4f}，RF={rf_auc:.4f}，差值={rf_auc - lr_auc:+.4f}")

    print("\n[RF 阈值分析]")
    default_metrics = threshold_analysis[f"{default_threshold:.2f}"]
    candidate_metrics = threshold_analysis[f"{candidate_threshold:.2f}"]
    print(
        f"阈值 {default_threshold:.2f}：Precision={default_metrics['precision']:.4f}，"
        f"Recall={default_metrics['recall']:.4f}，F1={default_metrics['f1']:.4f}，"
        f"Accuracy={default_metrics['accuracy']:.4f}"
    )
    print(
        f"阈值 {candidate_threshold:.2f}：Precision={candidate_metrics['precision']:.4f}，"
        f"Recall={candidate_metrics['recall']:.4f}，F1={candidate_metrics['f1']:.4f}，"
        f"Accuracy={candidate_metrics['accuracy']:.4f}"
    )
    print(
        f"Recall 变化：{candidate_metrics['recall'] - default_metrics['recall']:+.4f}；"
        f"Precision 变化：{candidate_metrics['precision'] - default_metrics['precision']:+.4f}"
    )

    print("\n[RF 特征重要性 Top 3]")
    for feature_name, importance in feature_importance.head(3).items():
        print(f"{feature_name}: {importance:.4f}")

    print(f"\n[测试集最后一天风险分数：{latest_date}]")
    for item in sorted(latest_risk_scores, key=lambda row: row["risk_score"], reverse=True)[:5]:
        print(
            f"{item['road_id']} ({item['from']}->{item['to']}): "
            f"{item['risk_score']:.4f}"
        )
    print(f"最后一天风险分数数量：{len(latest_risk_scores)}")


def main() -> None:
    """训练 LR/RF、评估阈值、保存 RF 模型与 bridge.py 所需元数据。"""
    config = load_config()
    data_path = resolve_project_path(config["paths"]["data_synthetic"]) / "road.csv"
    models_path = resolve_project_path(config["paths"]["models"])
    outputs_path = resolve_project_path(config["paths"]["outputs"])

    road_data = load_road_data(data_path)
    leakage_mismatches = validate_hist_break_count(road_data)
    if leakage_mismatches:
        raise ValueError(
            f"hist_break_count 存在目标泄漏：{leakage_mismatches} 行与历史标签复算不一致"
        )

    train_data, test_data = chronological_split(
        road_data,
        float(config["ml"]["train_ratio"]),
    )
    x_train = train_data[FEATURE_COLUMNS]
    y_train = train_data[TARGET_COLUMN].to_numpy(dtype=int)
    x_test = test_data[FEATURE_COLUMNS]
    y_test = test_data[TARGET_COLUMN].to_numpy(dtype=int)

    lr_model = build_logistic_regression(config).fit(x_train, y_train)
    rf_model = build_random_forest(config).fit(x_train, y_train)
    lr_scores = positive_probability(lr_model, x_test)
    rf_scores = positive_probability(rf_model, x_test)

    default_threshold = float(config["ml"]["risk_default_threshold"])
    candidate_threshold = float(config["ml"]["risk_candidate_threshold"])
    all_zero_labels = np.zeros_like(y_test)
    lr_labels = (lr_scores >= default_threshold).astype(int)
    rf_labels = (rf_scores >= default_threshold).astype(int)

    metrics_by_model = {
        "all_zero": classification_metrics(y_test, all_zero_labels),
        "logistic_regression": classification_metrics(y_test, lr_labels, lr_scores),
        "random_forest": classification_metrics(y_test, rf_labels, rf_scores),
    }
    metrics_frame = pd.DataFrame(
        [{"model": model_name, **metrics} for model_name, metrics in metrics_by_model.items()],
        columns=METRIC_COLUMNS,
    )
    evaluation_path = outputs_path / "risk_eval.csv"
    save_evaluation(metrics_frame, evaluation_path)

    models_path.mkdir(parents=True, exist_ok=True)
    model_path = models_path / "risk_model.pkl"
    metadata_path = models_path / "risk_model_meta.json"
    joblib.dump(rf_model, model_path)

    rf_threshold_analysis = {
        f"{threshold:.2f}": metrics_at_threshold(y_test, rf_scores, threshold)
        for threshold in (default_threshold, candidate_threshold)
    }
    latest_date, latest_risk_scores = build_latest_risk_scores(test_data, rf_scores)
    feature_importance = pd.Series(
        rf_model.feature_importances_,
        index=FEATURE_COLUMNS,
        name="importance",
    ).sort_values(ascending=False)

    metadata: dict[str, Any] = {
        "model_file": "risk_model.pkl",
        "model_type": type(rf_model).__name__,
        "target": TARGET_COLUMN,
        "feature_order": FEATURE_COLUMNS,
        "weather_encoding": WEATHER_CODE,
        "trained_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "training_date_range": {
            "start": pd.to_datetime(train_data["date"]).min().date().isoformat(),
            "end": pd.to_datetime(train_data["date"]).max().date().isoformat(),
        },
        "test_date_range": {
            "start": pd.to_datetime(test_data["date"]).min().date().isoformat(),
            "end": pd.to_datetime(test_data["date"]).max().date().isoformat(),
        },
        "test_metrics": metrics_by_model["random_forest"],
        "threshold_analysis": rf_threshold_analysis,
        "default_threshold": default_threshold,
        "recommended_threshold": candidate_threshold,
        "recommended_threshold_reason": (
            "道路中断的漏报成本高于误报成本；降低阈值以提高 Recall，"
            "便于调度器提前绕行或在人工复核后封路。"
        ),
        "priority_threshold_default": float(config["risk"]["priority_threshold"]),
        "feature_importance": {
            feature_name: round(float(importance), 6)
            for feature_name, importance in feature_importance.items()
        },
        "latest_risk_score_date": latest_date,
        "latest_risk_scores": latest_risk_scores,
        "random_seed": int(config["seed"]),
        "dataset_md5": hashlib.md5(data_path.read_bytes()).hexdigest(),
    }
    save_model_metadata(metadata_path, metadata)

    plot_roc_curve(y_test, lr_scores, rf_scores, outputs_path / "risk_roc_curve.png")
    plot_confusion_matrix(
        y_test,
        (rf_scores >= candidate_threshold).astype(int),
        candidate_threshold,
        outputs_path / "risk_confusion_matrix.png",
    )
    plot_feature_importance(feature_importance, outputs_path / "risk_feature_importance.png")

    print_report(
        road_data=road_data,
        train_data=train_data,
        test_data=test_data,
        metrics_frame=metrics_frame,
        feature_importance=feature_importance,
        rf_scores=rf_scores,
        default_threshold=default_threshold,
        candidate_threshold=candidate_threshold,
        threshold_analysis=rf_threshold_analysis,
        latest_date=latest_date,
        latest_risk_scores=latest_risk_scores,
        leakage_mismatches=leakage_mismatches,
    )
    print(f"\nRF 模型：{model_path}")
    print(f"模型元数据：{metadata_path}")
    print(f"指标文件：{evaluation_path}")
    print(f"图表目录：{outputs_path}")


if __name__ == "__main__":
    main()
