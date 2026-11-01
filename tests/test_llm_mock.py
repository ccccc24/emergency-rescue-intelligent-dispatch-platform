# -*- coding: utf-8 -*-
from __future__ import annotations

import requests

from app.common.data_loader import (
    build_plan_summary,
    get_default_selection,
    load_scenario,
    load_solution,
    load_solution_index,
    load_village_names,
)
from src.llm.explainer import explain_question
from src.llm.mock import SUGGESTED_QUESTIONS, answer_question


def _summary() -> dict:
    index = load_solution_index()
    scenario_date, group = get_default_selection(index)
    return build_plan_summary(
        load_scenario(scenario_date, index),
        load_solution(scenario_date, group),
        load_village_names(),
        group,
    )


def test_four_suggested_questions_have_answers() -> None:
    summary = _summary()
    answers = [answer_question(question, summary) for question in SUGGESTED_QUESTIONS]
    assert len(answers) == 4
    assert all(answer.strip() for answer in answers)
    assert all("100.00%" in answer or "风险" in answer or "优先村" in answer for answer in answers)


def test_no_key_uses_mock(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("ALIYUN_API_KEY", raising=False)
    monkeypatch.setattr("src.llm.explainer.DEFAULT_ENV_PATH", tmp_path / "missing.env")
    result = explain_question(SUGGESTED_QUESTIONS[0], _summary())
    assert result["mode"] == "mock"
    assert result["status"] == "missing_key"
    assert result["answer"]


def test_llm_failure_uses_mock(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key-not-real")
    monkeypatch.delenv("ALIYUN_API_KEY", raising=False)
    monkeypatch.setattr("src.llm.explainer.DEFAULT_ENV_PATH", tmp_path / "missing.env")

    def fake_post(*args, **kwargs):
        raise requests.Timeout("synthetic timeout")

    monkeypatch.setattr("src.llm.explainer.requests.post", fake_post)
    result = explain_question(SUGGESTED_QUESTIONS[1], _summary())
    assert result["mode"] == "mock"
    assert result["status"] == "failed"
    assert result["error_type"] == "timeout"
    assert "超时" in result["user_message"]
    assert result["answer"]
    assert "Timeout" in result["fallback_reason"]

def test_invalid_llm_response_uses_mock(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key-not-real")
    monkeypatch.delenv("ALIYUN_API_KEY", raising=False)
    monkeypatch.setattr("src.llm.explainer.DEFAULT_ENV_PATH", tmp_path / "missing.env")

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"output": {}}

    monkeypatch.setattr("src.llm.explainer.requests.post", lambda *args, **kwargs: FakeResponse())
    result = explain_question(SUGGESTED_QUESTIONS[2], _summary())
    assert result["mode"] == "mock"
    assert result["status"] == "failed"
    assert result["error_type"] == "invalid_response"
    assert result["answer"]
    assert "ValueError" in result["fallback_reason"]


def test_mock_answers_do_not_contain_double_percent() -> None:
    summary = _summary()
    answers = [answer_question(question, summary) for question in SUGGESTED_QUESTIONS]
    assert all("%%" not in answer for answer in answers)


def test_openai_compatible_chat_payload(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key-not-real")
    monkeypatch.setattr("src.llm.explainer.DEFAULT_ENV_PATH", tmp_path / "missing.env")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("LLM_MODEL", "deepseek-v4-flash")
    captured = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"choices": [{"message": {"content": "这是测试回答。"}}]}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["payload"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("src.llm.explainer.requests.post", fake_post)
    result = explain_question(SUGGESTED_QUESTIONS[3], _summary())
    assert result["mode"] == "ai"
    assert result["status"] == "success"
    assert captured["url"] == "https://api.deepseek.com/v1/chat/completions"
    assert captured["timeout"] == (3.0, 10.0)
    assert captured["payload"]["model"] == "deepseek-v4-flash"
    assert "messages" in captured["payload"]
    assert "input" not in captured["payload"]
    assert SUGGESTED_QUESTIONS[3] in captured["payload"]["messages"][1]["content"]

def test_summary_contains_full_current_plan_data() -> None:
    summary = _summary()
    assert summary["village_count"] == 6
    assert summary["road_count"] == 30
    assert summary["vehicle_count"] == 3
    assert len(summary["villages"]) == 6
    assert len(summary["roads"]) == 30
    assert len(summary["vehicles"]) == 3
    assert all("route_locations" in vehicle for vehicle in summary["vehicles"])

def test_proxy_error_is_classified(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key-not-real")
    monkeypatch.setattr("src.llm.explainer.DEFAULT_ENV_PATH", tmp_path / "missing.env")
    monkeypatch.setattr(
        "src.llm.explainer.requests.post",
        lambda *args, **kwargs: (_ for _ in ()).throw(requests.exceptions.ProxyError("synthetic proxy")),
    )
    result = explain_question(SUGGESTED_QUESTIONS[1], _summary())
    assert result["status"] == "failed"
    assert result["error_type"] == "proxy"
    assert "代理" in result["user_message"]
    assert "ProxyError" in result["fallback_reason"]


def test_http_401_is_classified(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key-not-real")
    monkeypatch.setattr("src.llm.explainer.DEFAULT_ENV_PATH", tmp_path / "missing.env")

    class FakeResponse:
        status_code = 401
        text = "unauthorized"

    monkeypatch.setattr("src.llm.explainer.requests.post", lambda *args, **kwargs: FakeResponse())
    result = explain_question(SUGGESTED_QUESTIONS[1], _summary())
    assert result["status"] == "failed"
    assert result["error_type"] == "http_401"
    assert result["user_message"] == "LLM 鉴权失败（HTTP 401），请检查 API Key。"
