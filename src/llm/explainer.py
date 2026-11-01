# -*- coding: utf-8 -*-
"""可选真实 LLM 调用；配置动态读取，失败时返回可诊断的规则兜底结果。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

try:
    from dotenv import dotenv_values
except ImportError:  # pragma: no cover - 无 python-dotenv 时仍支持环境变量
    dotenv_values = None

from src.llm.mock import answer_question

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_BASE_URL = "https://api.deepseek.com/v1/chat/completions"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_TIMEOUT = (3.0, 10.0)


@dataclass(frozen=True)
class LLMConfig:
    api_key: str = field(repr=False, default="")
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    env_path: Path = field(repr=False, default=DEFAULT_ENV_PATH)

    @property
    def configured(self) -> bool:
        return bool(self.api_key.strip())


def _dotenv_file_values(env_path: Path) -> dict[str, str]:
    if dotenv_values is None or not env_path.is_file():
        return {}
    try:
        values = dotenv_values(env_path)
    except Exception:
        return {}
    return {
        str(key): str(value)
        for key, value in values.items()
        if value is not None and str(value).strip()
    }


def _pick(file_values: dict[str, str], *names: str) -> str:
    for name in names:
        file_value = str(file_values.get(name, "") or "").strip()
        if file_value:
            return file_value
    for name in names:
        env_value = str(os.getenv(name) or "").strip()
        if env_value:
            return env_value
    return ""


def get_llm_config(env_path: str | Path | None = None) -> LLMConfig:
    """每次调用都重新读取环境变量，避免运行期间配置陈旧。"""

    path = Path(env_path) if env_path is not None else DEFAULT_ENV_PATH
    file_values = _dotenv_file_values(path)
    return LLMConfig(
        api_key=_pick(file_values, "LLM_API_KEY", "ALIYUN_API_KEY"),
        base_url=_pick(file_values, "LLM_BASE_URL") or DEFAULT_BASE_URL,
        model=_pick(file_values, "LLM_MODEL") or DEFAULT_MODEL,
        env_path=path,
    )


def is_llm_configured() -> bool:
    return get_llm_config().configured


def _mock_result(
    question: str,
    summary: dict[str, Any],
    *,
    status: str,
    error_type: str,
    user_message: str,
    detail: str | None = None,
) -> dict[str, Any]:
    fallback_reason = f"{error_type}: {detail}" if detail else error_type
    return {
        "mode": "mock",
        "status": status,
        "answer": answer_question(question, summary),
        "fallback_reason": fallback_reason[:400],
        "error_type": error_type,
        "user_message": user_message,
        "detail": (detail or "")[:400] or None,
    }


def _extract_text(payload: dict[str, Any]) -> str:
    output = payload.get("output")
    if isinstance(output, dict):
        text = output.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    raise ValueError("LLM 返回中没有可识别的文本")


def _request_url(base_url: str) -> str:
    """兼容完整 chat/completions 地址、OpenAI base URL 和旧 DashScope 地址。"""

    base = base_url.strip().rstrip("/")
    if "dashscope.aliyuncs.com" in base and "services/aigc" in base:
        return base
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def _request_payload(
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if "dashscope.aliyuncs.com" in base_url and "services/aigc" in base_url:
        return {"model": model, "input": {"messages": messages}, "parameters": {}}
    return {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 500,
        "stream": False,
    }


def _normalize_timeout(value: float | tuple[float, float] | None) -> float | tuple[float, float]:
    if value is None:
        return DEFAULT_TIMEOUT
    if isinstance(value, (tuple, list)) and len(value) == 2:
        connect, read = (float(value[0]), float(value[1]))
        if connect <= 0 or read <= 0:
            raise ValueError("LLM timeout 必须为正数")
        return connect, read
    seconds = float(value)
    if seconds <= 0:
        raise ValueError("LLM timeout 必须为正数")
    return min(3.0, seconds), seconds


def _classify_error(error: Exception) -> tuple[str, str, str]:
    """返回 error_type、用户提示、技术详情，不包含请求头或 API Key。"""

    detail = f"{type(error).__name__}: {error}"
    if isinstance(error, requests.exceptions.ProxyError):
        return "proxy", "无法连接到 DeepSeek，请检查网络或代理设置。", detail
    if isinstance(error, requests.exceptions.Timeout):
        return "timeout", "LLM 请求超时，请稍后重试。", detail
    if isinstance(error, requests.exceptions.HTTPError):
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
        if status_code in {401, 403}:
            return f"http_{status_code}", f"LLM 鉴权失败（HTTP {status_code}），请检查 API Key。", detail
        if status_code is not None:
            return f"http_{status_code}", f"LLM 服务返回错误（HTTP {status_code}）。", detail
        return "http_error", "LLM 服务返回 HTTP 错误。", detail
    if isinstance(error, requests.exceptions.ConnectionError):
        return "connection", "无法连接到 DeepSeek，请检查网络连接。", detail
    if isinstance(error, (ValueError, json.JSONDecodeError)):
        return "invalid_response", "LLM 返回格式无效，已回退规则模式。", detail
    return "unknown", "LLM 调用失败，已回退规则兜底模式。", detail


def explain_question(
    question: str,
    summary: dict[str, Any],
    *,
    timeout_seconds: float | tuple[float, float] | None = None,
) -> dict[str, Any]:
    """调用真实 LLM 解释摘要；任何失败都返回带错误分类的 mock 结果。"""

    config = get_llm_config()
    if not config.configured:
        return _mock_result(
            question,
            summary,
            status="missing_key",
            error_type="missing_key",
            user_message="未配置 API Key，当前为规则兜底模式。",
            detail="LLM_API_KEY 不在 .env 或进程环境变量中",
        )

    request_url = _request_url(config.base_url)
    system_prompt = (
        "你是车辆运输应急救援调度方案解释助手。只能使用用户提供的 JSON 摘要中的数字和事实，"
        "不得编造或推算未给出的数值。回答简洁、中文、最多四句话，并明确这是对现有方案的解释。"
    )
    user_prompt = (
        f"用户问题：{question}\n\n"
        "方案摘要（唯一事实来源）：\n"
        f"{json.dumps(summary, ensure_ascii=False, indent=2)}"
    )

    request_payload = _request_payload(request_url, config.model, system_prompt, user_prompt)
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            request_url,
            headers=headers,
            json=request_payload,
            timeout=_normalize_timeout(timeout_seconds),
        )
        status_code = getattr(response, "status_code", None)
        if status_code is not None and status_code >= 400:
            detail = str(getattr(response, "text", "") or "")[:300]
            raise requests.HTTPError(
                f"HTTP {status_code}: {detail}",
                response=response,
            )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("LLM 返回 JSON 不是对象")
        text = _extract_text(payload)
    except Exception as error:
        error_type, user_message, detail = _classify_error(error)
        return _mock_result(
            question,
            summary,
            status="failed",
            error_type=error_type,
            user_message=user_message,
            detail=detail,
        )

    return {
        "mode": "ai",
        "status": "success",
        "answer": text,
        "fallback_reason": None,
        "error_type": None,
        "user_message": None,
        "detail": None,
    }

