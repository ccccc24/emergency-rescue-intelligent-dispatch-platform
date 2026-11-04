# -*- coding: utf-8 -*-
from __future__ import annotations

from src.llm.explainer import DEFAULT_BASE_URL, DEFAULT_MODEL, explain_question, test_llm_connection as check_llm_connection


def _summary() -> dict:
    return {
        "scenario_label": "测试场景",
        "group_label": "C",
        "solver_status": "success",
        "metrics": {
            "total_demand_satisfied_rate": 1.0,
            "unserved_demand_total": 0.0,
            "slowest_vehicle_time": 1.2,
            "total_exposed_risk": 3.4,
            "high_risk_edge_visits": 1,
        },
        "priority_villages": [],
        "top_risk_roads": [],
    }


class _SuccessResponse:
    status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"choices": [{"message": {"content": "这是 DeepSeek 返回的解释。"}}]}


def test_user_api_key_uses_deepseek_without_server_configuration(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("ALIYUN_API_KEY", raising=False)
    monkeypatch.setattr("src.llm.explainer.DEFAULT_ENV_PATH", tmp_path / "missing.env")
    captured: dict = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(url=url, headers=headers, payload=json, timeout=timeout)
        return _SuccessResponse()

    monkeypatch.setattr("src.llm.explainer.requests.post", fake_post)
    result = explain_question("为什么这样调度？", _summary(), api_key="sk-user-test")

    assert result["mode"] == "ai"
    assert result["status"] == "success"
    assert captured["url"] == DEFAULT_BASE_URL
    assert captured["headers"]["Authorization"] == "Bearer sk-user-test"
    assert captured["payload"]["model"] == DEFAULT_MODEL
    assert captured["payload"]["thinking"] == {"type": "disabled"}
    assert "sk-user-test" not in result["answer"]


def test_connection_test_uses_the_supplied_key(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setattr("src.llm.explainer.DEFAULT_ENV_PATH", tmp_path / "missing.env")
    captured: dict = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["authorization"] = headers["Authorization"]
        captured["max_tokens"] = json["max_tokens"]
        return _SuccessResponse()

    monkeypatch.setattr("src.llm.explainer.requests.post", fake_post)
    result = check_llm_connection(api_key="sk-connection-test")

    assert result["ok"] is True
    assert result["provider"] == "DeepSeek"
    assert result["model"] == DEFAULT_MODEL
    assert captured["authorization"] == "Bearer sk-connection-test"
    assert captured["max_tokens"] == 8


def test_connection_test_rejects_invalid_key(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setattr("src.llm.explainer.DEFAULT_ENV_PATH", tmp_path / "missing.env")

    class _UnauthorizedResponse:
        status_code = 401
        text = "unauthorized"

    monkeypatch.setattr(
        "src.llm.explainer.requests.post",
        lambda *args, **kwargs: _UnauthorizedResponse(),
    )
    result = check_llm_connection(api_key="sk-invalid")

    assert result["ok"] is False
    assert result["error_type"] == "http_401"
    assert "API Key" in result["message"]


def test_api_endpoint_forwards_user_key_header(monkeypatch) -> None:
    from fastapi.testclient import TestClient
    from api.index import app

    captured: dict = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["authorization"] = headers["Authorization"]
        return _SuccessResponse()

    monkeypatch.setattr("src.llm.explainer.requests.post", fake_post)
    response = TestClient(app).post(
        "/api/index?action=test-llm",
        headers={"X-DeepSeek-API-Key": "sk-from-browser-header"},
        json={},
    )

    assert response.status_code == 200
    assert response.json()["data"]["ok"] is True
    assert captured["authorization"] == "Bearer sk-from-browser-header"
