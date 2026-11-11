"""OpenWeightsLLM: keyless open-weight models, against a fake post()."""

from __future__ import annotations

import asyncio
import json

import pytest
from pydantic import BaseModel

from zerkalo.config import Models
from zerkalo.llm import OpenWeightsLLM, ZerkaloLLMError, make_llm
from zerkalo.schemas import AgentAction


def run(coro):
    return asyncio.run(coro)


class Note(BaseModel):
    headline: str


def make(calls, reply):
    def post(path, body):
        calls.append({"path": path, "body": body})
        return reply(calls[-1]["body"])

    models = Models(agent="qwen3:8b", analyst="qwen3:32b")
    return OpenWeightsLLM(models, url="http://m/v1", post=post)


def test_text_sends_system_and_user_with_no_key():
    calls = []
    llm = make(calls, lambda b: {"choices": [{"message": {"content": "hello"}}], "usage": {"prompt_tokens": 3, "completion_tokens": 2}})
    out = run(llm.text("sys", [{"role": "user", "content": "hi"}], role="analyst"))
    assert out == "hello"
    assert calls[0]["path"] == "/chat/completions"
    assert calls[0]["body"]["model"] == "qwen3:32b"  # analyst role
    assert calls[0]["body"]["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    assert "api_key" not in json.dumps(calls[0]["body"])
    assert llm.usage.input_tokens == 3 and llm.usage.output_tokens == 2


def test_structured_parses_schema_json_and_strips_think():
    calls = []
    llm = make(calls, lambda b: {"choices": [{"message": {"content": "<think>x</think>{\"headline\": \"ok\"}"}}]})
    note = run(llm.structured("s", "p", Note, role="agent"))
    assert isinstance(note, Note)
    assert note.headline == "ok"
    assert calls[0]["body"]["response_format"]["type"] == "json_schema"
    assert calls[0]["body"]["model"] == "qwen3:8b"  # agent role


def test_structured_raises_on_garbage():
    llm = make([], lambda b: {"choices": [{"message": {"content": "not json"}}]})
    with pytest.raises(ZerkaloLLMError):
        run(llm.structured("s", "p", Note, role="agent"))


def test_agent_action_schema_round_trip():
    llm = make([], lambda b: {"choices": [{"message": {"content": json.dumps({"action": "comment", "target_post_id": 7, "text": "hm", "new_stance": 0.1, "reasoning": "r"})}}]})
    act = run(llm.structured("s", "p", AgentAction, role="agent"))
    assert act.action == "comment" and act.target_post_id == 7


def test_make_llm_selects_open_weights_from_env(monkeypatch):
    monkeypatch.delenv("ZERKALO_OPEN_WEIGHTS_URL", raising=False)
    monkeypatch.delenv("ZERKALO_LLM", raising=False)
    from zerkalo.llm import MockLLM

    assert isinstance(make_llm(Models(), True), MockLLM)
    monkeypatch.setenv("ZERKALO_OPEN_WEIGHTS_URL", "http://m:11434/v1")
    assert isinstance(make_llm(Models(), False), OpenWeightsLLM)
    monkeypatch.delenv("ZERKALO_OPEN_WEIGHTS_URL", raising=False)
    monkeypatch.setenv("ZERKALO_LLM", "open-weights")
    assert isinstance(make_llm(Models(), False), OpenWeightsLLM)


def test_transport_failure_becomes_zerkalo_error():
    def post(path, body):
        raise OSError("connection refused")

    llm = OpenWeightsLLM(Models(), url="http://m/v1", post=post)
    with pytest.raises(ZerkaloLLMError):
        run(llm.text("s", [{"role": "user", "content": "go"}], role="agent"))
