# -*- coding: utf-8 -*-
"""FastAPI backend deployed as a Vercel Python Function.

Every browser request targets ``/api/index`` and selects an operation through
the ``action`` query parameter. This avoids Vercel dynamic-route ambiguity.
"""

from __future__ import annotations

import base64
import copy
import json
import os
import re
import sys
import tempfile
from datetime import date as date_cls
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.common.data_loader import (  # noqa: E402
    build_plan_summary,
    load_network_layout,
    load_scenario,
    load_solution,
    load_solution_index,
    load_village_names,
)
from app.common.scenario_builder import (  # noqa: E402
    ScenarioValidationError,
    build_user_active_scenario,
    compute_scenario_hash,
    default_scheduler_config,
    normalize_scheduler_config,
    read_excel_scenario,
)
from scripts.run_live_schedule import GROUP_MODES  # noqa: E402
from src.llm import SUGGESTED_QUESTIONS, explain_question, is_llm_configured  # noqa: E402
from src.ml.data_gen import load_config  # noqa: E402
from src.ml.user_prediction import UserPredictionError, predict_user_inputs  # noqa: E402
from src.optimization.export_solution import export_solution  # noqa: E402

BASE_CONFIG_PATH = PROJECT_ROOT / "configs" / "live_demo.yaml"
TEMPLATE_PATH = PROJECT_ROOT / "app" / "assets" / "scenario_template.xlsx"
MODE_TO_GROUP = {"hist_demand": "A", "no_risk_adjust": "B", "full": "C"}

app = FastAPI(title="驰援应急救援智能调度 API", version="1.0.0")
_origins = os.getenv("CORS_ORIGINS", "*").strip()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _origins == "*" else [value.strip() for value in _origins.split(",") if value.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _success(data: Any) -> dict[str, Any]:
    return {"ok": True, "data": jsonable_encoder(data)}


def _failure(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"ok": False, "error": message})


def _normalize_scenario_maps(scenario: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(scenario)
    for field in ("villages", "roads", "demands", "risk_scores", "adjusted_speed"):
        value = normalized.get(field)
        if isinstance(value, dict):
            normalized[field] = {int(key): item for key, item in value.items()}
    for field in ("priority_villages", "isolated_villages"):
        value = normalized.get(field)
        if isinstance(value, list):
            normalized[field] = [int(item) for item in value]
    return normalized


def _config_from_payload(payload: dict[str, Any], scenario: dict[str, Any] | None = None) -> dict[str, Any]:
    config = dict(default_scheduler_config())
    if scenario:
        raw = scenario.get("config")
        if isinstance(raw, dict):
            config.update(raw)
        metadata = scenario.get("metadata")
        if isinstance(metadata, dict) and isinstance(metadata.get("scheduler_config"), dict):
            config.update(metadata["scheduler_config"])
    requested = payload.get("config")
    if isinstance(requested, dict):
        config.update(requested)
    return normalize_scheduler_config(config)


def _example_index_payload() -> dict[str, Any]:
    index = load_solution_index()
    scenarios = [
        {
            "date": scenario_date,
            "label": item.get("label", scenario_date),
            "weather": item.get("weather", ""),
            "modes": [str(mode).upper() for mode in item.get("modes", ["A", "B", "C"])],
        }
        for scenario_date, item in index["scenarios"].items()
    ]
    scenarios.sort(key=lambda item: item["date"], reverse=True)
    return {
        "default_date": index["default_scenario"],
        "default_group": index["default_mode"],
        "scenarios": scenarios,
    }


def _safe_example_date(value: str) -> str:
    if value not in load_solution_index()["scenarios"]:
        raise ValueError(f"未知示例场景：{value}")
    return value


def _example_payload(scenario_date: str, group: str) -> dict[str, Any]:
    scenario_date = _safe_example_date(scenario_date)
    group = str(group or "C").upper()
    index = load_solution_index()
    if group not in index["scenarios"][scenario_date].get("modes", []):
        raise ValueError(f"场景 {scenario_date} 不支持方案 {group}")
    scenario = load_scenario(scenario_date, index)
    solution = load_solution(scenario_date, group)
    return {
        "date": scenario_date,
        "group": group,
        "scenario": scenario,
        "solution": solution,
        "summary": build_plan_summary(scenario, solution, load_village_names(), group),
        "layout": load_network_layout(),
    }


def _write_runtime_config(config: dict[str, Any], target: Path) -> None:
    text = BASE_CONFIG_PATH.read_text(encoding="utf-8")
    for key, value in config.items():
        pattern = rf"(?m)^(\s*{re.escape(key)}:\s*)[^#\r\n]*"
        text, count = re.subn(pattern, lambda match: match.group(1) + str(value), text, count=1)
        if count != 1:
            raise ValueError(f"配置文件中找不到可覆盖参数：{key}")
    target.write_text(text, encoding="utf-8")


def _run_scheduler(
    scenario: dict[str, Any],
    *,
    config: dict[str, Any],
    mode: str,
    group: str | None = None,
    scenario_date: str | None = None,
) -> dict[str, Any]:
    if mode not in MODE_TO_GROUP:
        raise ValueError(f"未知调度模式：{mode}")
    source = str(scenario.get("source") or scenario.get("metadata", {}).get("source") or "example")
    resolved_group = group or ("user" if source in {"manual", "excel"} else MODE_TO_GROUP[mode])
    if resolved_group != "user" and resolved_group != MODE_TO_GROUP[mode]:
        raise ValueError("方案组与消融模式不一致")

    with tempfile.TemporaryDirectory(prefix="wingaid_api_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        scenario_path = temp_dir / "scenario.json"
        output_path = temp_dir / "solution.json"
        runtime_config = temp_dir / "live_config.yaml"
        scenario_copy = _normalize_scenario_maps(scenario)
        scenario_copy.setdefault("date", scenario_date or date_cls.today().isoformat())
        scenario_path.write_text(json.dumps(scenario_copy, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_runtime_config(config, runtime_config)
        result = export_solution(scenario_path, output_path, ablation_mode=mode, config_path=runtime_config)

    result["scenario"] = scenario_copy
    result["scenario_hash"] = compute_scenario_hash(scenario_copy, config)
    result["group"] = resolved_group
    result["scenario_source"] = source
    result["scenario_config"] = dict(config)
    result["config_snapshot"] = {**dict(result.get("config_snapshot", {})), **dict(config)}
    result["summary"] = build_plan_summary(scenario_copy, result, load_village_names(), resolved_group)
    return result


def _build_active_scenario(payload: dict[str, Any]) -> dict[str, Any]:
    villages = payload.get("villages")
    roads = payload.get("roads")
    if not isinstance(villages, list) or not isinstance(roads, list):
        raise ValueError("villages 和 roads 必须是数组")
    config = _config_from_payload(payload)
    weather = str(payload.get("weather") or "晴")
    input_mode = str(payload.get("input_mode") or "minimal")
    prediction = None
    if bool(payload.get("predict_missing", False)):
        prediction = predict_user_inputs(
            villages,
            roads,
            source="excel" if payload.get("source") == "excel" else "manual",
            weather=weather,
            overwrite_existing=bool(payload.get("overwrite_existing", False)),
        )
        villages = prediction["villages"]
        roads = prediction["roads"]
    return build_user_active_scenario(
        villages,
        roads,
        config,
        source="excel" if payload.get("source") == "excel" else "manual",
        label=str(payload.get("label") or "用户自定义场景"),
        input_mode=input_mode,
        weather=weather,
        prediction=prediction,
        ml_unused=prediction is None,
    )


def _comparison_payload(scenario_date: str) -> dict[str, Any]:
    scenario_date = _safe_example_date(scenario_date)
    index = load_solution_index()
    scenario = load_scenario(scenario_date, index)
    groups = []
    for group in [str(mode).upper() for mode in index["scenarios"][scenario_date].get("modes", ["A", "B", "C"])]:
        solution = load_solution(scenario_date, group)
        groups.append(
            {
                "group": group,
                "label": {"A": "历史需求基线", "B": "无风险修正", "C": "完整 ML + 风险"}.get(group, group),
                "solution": solution,
                "summary": build_plan_summary(scenario, solution, load_village_names(), group),
            }
        )
    return {"date": scenario_date, "scenario": scenario, "groups": groups, "layout": load_network_layout()}


def _model_metadata() -> dict[str, Any]:
    demand = json.loads((PROJECT_ROOT / "models" / "demand_model_meta.json").read_text(encoding="utf-8"))
    risk = json.loads((PROJECT_ROOT / "models" / "risk_model_meta.json").read_text(encoding="utf-8"))
    return {
        "demand": demand,
        "risk": risk,
        "llm_configured": is_llm_configured(),
        "suggested_questions": list(SUGGESTED_QUESTIONS),
    }


def _get_action(action: str, params: dict[str, Any]) -> Any:
    index = load_solution_index()
    if action == "health":
        return {"status": "ok", "service": "chiyuan-api", "version": app.version}
    if action == "meta":
        return _model_metadata()
    if action == "examples":
        return _example_index_payload()
    if action == "example":
        return _example_payload(str(params.get("date") or index["default_scenario"]), str(params.get("group") or index["default_mode"]))
    if action == "compare":
        return _comparison_payload(str(params.get("date") or index["default_scenario"]))
    if action == "template":
        if not TEMPLATE_PATH.is_file():
            raise FileNotFoundError("Excel 模板不存在")
        return Response(
            content=TEMPLATE_PATH.read_bytes(),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": 'attachment; filename="scenario_template.xlsx"'},
        )
    raise ValueError(f"未知 GET action：{action}")


def _post_action(action: str, payload: dict[str, Any]) -> Any:
    if action == "predict":
        return predict_user_inputs(
            payload.get("villages") or [],
            payload.get("roads") or [],
            source="excel" if payload.get("source") == "excel" else "manual",
            weather=str(payload.get("weather") or "晴"),
            overwrite_existing=bool(payload.get("overwrite_existing", False)),
        )
    if action == "build":
        return _build_active_scenario(payload)
    if action == "schedule":
        scenario = payload.get("scenario")
        if not isinstance(scenario, dict):
            raise ValueError("scenario 必须是对象")
        return _run_scheduler(
            scenario,
            config=_config_from_payload(payload, scenario),
            mode=str(payload.get("mode") or "full"),
            group=payload.get("group"),
            scenario_date=payload.get("date"),
        )
    if action == "explain":
        scenario = payload.get("scenario")
        solution = payload.get("solution")
        if not isinstance(scenario, dict) or not isinstance(solution, dict):
            raise ValueError("scenario 和 solution 必须是对象")
        normalized = _normalize_scenario_maps(scenario)
        summary = build_plan_summary(normalized, solution, load_village_names(), str(payload.get("group") or "C"))
        return explain_question(str(payload.get("question") or "请解释当前方案"), summary)
    if action == "import-excel":
        encoded = str(payload.get("data_base64") or "")
        if not encoded:
            raise ValueError("缺少 Excel 文件内容")
        parsed = read_excel_scenario(base64.b64decode(encoded))
        return _build_active_scenario(
            {
                **payload,
                "villages": parsed["villages"],
                "roads": parsed["roads"],
                "source": "excel",
                "label": payload.get("label") or "Excel 上传场景",
                "weather": payload.get("weather") or parsed["weather"],
                "input_mode": payload.get("input_mode") or parsed["detected_input_mode"],
                "config": payload.get("config") or parsed["scheduler_config"],
            }
        )
    raise ValueError(f"未知 POST action：{action}")


_ROOT_PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>驰援应急救援智能调度 API</title>
<style>
body{margin:0;font-family:system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;background:#f6f7f9;color:#1f2933;display:flex;min-height:100vh;align-items:center;justify-content:center;padding:24px;box-sizing:border-box}
main{max-width:560px;width:100%;background:#fff;border:1px solid #e4e7eb;border-radius:12px;padding:32px}
h1{font-size:20px;margin:0 0 8px}p{line-height:1.7;margin:8px 0;color:#52606d}
.ok{display:inline-block;background:#e3f9ee;color:#0f7a4a;border-radius:999px;padding:2px 10px;font-size:13px;margin-bottom:12px}
ul{padding-left:20px;line-height:1.9}a{color:#c2410c}code{background:#f1f3f5;padding:1px 6px;border-radius:4px}
</style></head><body><main>
<span class="ok">后端运行正常</span>
<h1>驰援应急救援智能调度 API</h1>
<p>这是 FastAPI 后端服务，所有接口统一位于 <code>/api/index</code>，通过 <code>action</code> 参数区分操作。</p>
<ul>
<li><a href="/api/index?action=health">健康检查 action=health</a></li>
<li><a href="/api/index?action=examples">示例场景 action=examples</a></li>
<li><a href="/docs">接口文档 /docs</a></li>
</ul>
<p>可视化前端位于仓库的 <code>frontend/</code> 目录，需要单独部署为 Next.js 项目。设置环境变量 <code>FRONTEND_URL</code> 后，访问本页会自动跳转到前端。</p>
</main></body></html>"""


@app.get("/", include_in_schema=False)
def root() -> Response:
    frontend_url = os.getenv("FRONTEND_URL", "").strip()
    if frontend_url:
        return RedirectResponse(frontend_url, status_code=307)
    return HTMLResponse(_ROOT_PAGE)


@app.get("/api/index")
def api_get(request: Request, action: str = Query(default="health")) -> Any:
    try:
        result = _get_action(action, dict(request.query_params))
        return result if isinstance(result, Response) else _success(result)
    except (ValueError, KeyError, FileNotFoundError, ScenarioValidationError, UserPredictionError) as error:
        return _failure(str(error), 400)
    except Exception as error:
        return _failure(f"服务端异常：{type(error).__name__}: {error}", 500)


@app.post("/api/index")
async def api_post(request: Request, action: str = Query(...)) -> Any:
    try:
        try:
            payload = await request.json()
        except Exception as error:
            raise ValueError("请求体必须是 JSON") from error
        if not isinstance(payload, dict):
            raise ValueError("请求体必须是 JSON 对象")
        return _success(_post_action(action, payload))
    except (ValueError, KeyError, FileNotFoundError, ScenarioValidationError, UserPredictionError) as error:
        return _failure(str(error), 400)
    except Exception as error:
        return _failure(f"服务端异常：{type(error).__name__}: {error}", 500)
