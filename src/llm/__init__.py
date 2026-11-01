# -*- coding: utf-8 -*-
"""可选 LLM 解释层：配置动态读取，失败时自动使用规则 mock。"""

from src.llm.explainer import explain_question, get_llm_config, is_llm_configured
from src.llm.mock import SUGGESTED_QUESTIONS, answer_question

__all__ = [
    "SUGGESTED_QUESTIONS",
    "answer_question",
    "explain_question",
    "get_llm_config",
    "is_llm_configured",
]
