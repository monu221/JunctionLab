"""The Groq path, on real recorded responses (tests/data/llm/record_groq.py).

No test here makes a network call, except the live test, which needs JUNCTIONLAB_LIVE_LLM=1.
Provider-independent limits are tested in test_llm_anthropic.py.
"""

import json
import os
from pathlib import Path

import httpx2
import pytest

from junctionlab import llm
from junctionlab.config import get_config
from junctionlab.examples import list_examples
from junctionlab.llm import (
    CallLimiter,
    GroqClient,
    InterpretationError,
    LlmSession,
    LlmUnavailable,
    ReplayTransport,
    interpret,
    load_recording,
    make_client,
)

RECORDINGS = Path(__file__).parent / "data" / "llm"
EXAMPLES = {e.key: e for e in list_examples()}
KEY_NAME = llm.API_KEY_NAMES["groq"]
FAKE_KEY = "gsk_test_not_a_real_key"


@pytest.fixture
def limiter(tmp_path):
    return CallLimiter(get_config(), counter_file=tmp_path / "calls.json")


def run(rec, limiter):
    transport = ReplayTransport(rec)
    client = make_client(FAKE_KEY, transport=transport, max_retries=0, provider="groq")
    return interpret(rec["text"], LlmSession(), client=client, limiter=limiter), transport


def replay(name, limiter):
    return run(load_recording(RECORDINGS / f"{name}.json"), limiter)


def inline(status, body, headers=None):
    response = {"status": status, "body": body}
    if headers:
        response["headers"] = headers
    return {"source": "test", "text": "a crossroads", "response": response}


def completion(message, finish_reason="stop"):
    return {
        "choices": [{"message": {"role": "assistant", **message}, "finish_reason": finish_reason}]
    }


# ------------------------------------------------------------------ provider and request


def test_default_provider_is_groq():
    assert get_config().llm.provider == "groq"
    assert isinstance(make_client(FAKE_KEY), GroqClient)


def test_all_committed_groq_recordings_are_live():
    files = list(RECORDINGS.glob("*.json"))
    assert len(files) == 10
    for path in files:
        assert load_recording(path)["source"] == "live", path.name


def test_request_uses_strict_structured_output(limiter):
    _, transport = replay("four_arm_signal", limiter)
    (body,) = transport.requests
    cfg = get_config().llm.groq
    assert body["model"] == cfg.model
    assert body["max_completion_tokens"] == cfg.max_tokens
    assert body["reasoning_effort"] == cfg.reasoning_effort
    assert body["messages"][0] == {"role": "system", "content": llm.SYSTEM_PROMPT}
    assert body["messages"][1] == {"role": "user", "content": transport.recording["text"]}
    fmt = body["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"] == json.loads(json.dumps(llm.OUTPUT_FORMAT["schema"]))


def test_key_is_sent_only_as_a_header():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx2.Response(200, json=completion({"content": "{}"}))

    client = GroqClient(FAKE_KEY, transport=httpx2.MockTransport(handler), max_retries=0)
    client.post_chat({"model": "m"})
    assert seen[0].headers["authorization"] == f"Bearer {FAKE_KEY}"
    assert FAKE_KEY not in seen[0].read().decode() and FAKE_KEY not in str(seen[0].url)


# ------------------------------------------------------------------ real responses


@pytest.mark.parametrize("key", ["four_arm_signal", "t_junction_priority", "four_arm_bus_lane"])
def test_example_descriptions_give_the_example_scenario(key, limiter):
    result, _ = replay(key, limiter)
    expected = EXAMPLES[key].scenario()
    got = result.scenario
    assert got.junction_type == expected.junction_type
    assert got.approaches == expected.approaches
    assert got.control == expected.control
    assert set(got.assumed) == set(expected.assumed)


def test_movement_counts(limiter):
    s = replay("movement_counts", limiter)[0].scenario
    assert s.approaches["north"].demand_vph == 800
    assert s.approaches["north"].turning_shares_pct == {"left": 12.5, "through": 75, "right": 12.5}
    assert s.approaches["east"].turning_shares_pct["through"] == pytest.approx(66.67, abs=0.05)
    assert s.control.phase_plan == "four_phase_split" and s.control.cycle_s == 100
    assert "approaches.west.turning_shares_pct" in s.assumed


def test_partial_shares_on_t_junction(limiter):
    s = replay("partial_shares", limiter)[0].scenario
    assert s.stem == "north"
    assert s.approaches["west"].turning_shares_pct == {"left": 10, "through": 90, "right": 0}
    assert "approaches.east.turning_shares_pct" in s.assumed


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("unclear_demand", "traffic"),
        ("not_a_junction", "road junction"),
        ("invalid_cycle", "too short"),
        ("max_tokens", "did not fit"),  # Groq answers 400 when strict output is cut off
    ],
)
def test_rephrase_cases(name, reason, limiter):
    with pytest.raises(InterpretationError) as e:
        replay(name, limiter)
    assert reason in str(e.value)


def test_rejected_key(limiter):
    with pytest.raises(LlmUnavailable, match="rejected"):
        replay("auth_error", limiter)


# ------------------------------------------------------------------ errors (constructed)


@pytest.mark.parametrize(
    ("retry_after", "wait"),
    [("7", "7 s"), ("600", "10 min"), ("7200", "2 h"), (None, "in a minute"), ("soon", "minute")],
)
def test_rate_limit_gives_a_clear_message(retry_after, wait, limiter):
    headers = {"retry-after": retry_after} if retry_after else None
    rec = inline(429, {"error": {"message": "Rate limit reached", "code": "rate_limit"}}, headers)
    with pytest.raises(LlmUnavailable) as e:
        run(rec, limiter)
    assert "rate limit" in str(e.value) and wait in str(e.value)


def test_rate_limit_is_not_retried(limiter):
    rec = inline(429, {"error": {"message": "limit"}}, {"retry-after": "1"})
    transport = ReplayTransport(rec)
    client = make_client(FAKE_KEY, transport=transport, max_retries=2, provider="groq")
    with pytest.raises(LlmUnavailable):
        interpret(rec["text"], LlmSession(), client=client, limiter=limiter)
    assert len(transport.requests) == 1


def test_server_error_is_retried(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)
    answers = [httpx2.Response(503, json={}), httpx2.Response(200, json=completion({}))]
    client = GroqClient(
        FAKE_KEY, transport=httpx2.MockTransport(lambda _: answers.pop(0)), max_retries=1
    )
    assert client.post_chat({}).status_code == 200


def test_connection_error(monkeypatch, limiter):
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)

    def fail(request):
        raise httpx2.ConnectError("offline", request=request)

    client = GroqClient(FAKE_KEY, transport=httpx2.MockTransport(fail), max_retries=1)
    with pytest.raises(LlmUnavailable, match="Could not reach"):
        interpret("a crossroads", LlmSession(), client=client, limiter=limiter)


@pytest.mark.parametrize(
    ("message", "finish", "reason"),
    [
        ({"content": None, "refusal": "no"}, "stop", "declined"),
        ({"content": '{"understood"'}, "length", "cut off"),
        ({"content": None}, "stop", "no structured answer"),
        ({"content": '{"understood": true}'}, "stop", "did not fit"),
    ],
)
def test_unusable_answers(message, finish, reason, limiter):
    with pytest.raises(InterpretationError, match=reason):
        run(inline(200, completion(message, finish)), limiter)


def test_error_message_does_not_echo_the_key(limiter):
    rec = inline(400, {"error": {"message": f"bad request for {FAKE_KEY}"}})
    with pytest.raises(LlmUnavailable) as e:
        run(rec, limiter)
    assert FAKE_KEY not in str(e.value)


def test_no_api_key(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "REPO_ROOT", tmp_path)
    monkeypatch.delenv(KEY_NAME, raising=False)
    assert llm.api_key("groq") is None
    with pytest.raises(LlmUnavailable, match=KEY_NAME):
        make_client(provider="groq")


def test_api_key_from_dotenv(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "REPO_ROOT", tmp_path)
    monkeypatch.delenv(KEY_NAME, raising=False)
    (tmp_path / ".env").write_text(f"{KEY_NAME}={FAKE_KEY}\n", encoding="utf-8")
    assert llm.api_key("groq") == FAKE_KEY
    monkeypatch.delenv(KEY_NAME, raising=False)  # load_dotenv set it


def test_cli_interpret_without_key(monkeypatch, tmp_path, capsys):
    from junctionlab.__main__ import main

    monkeypatch.setattr(llm, "REPO_ROOT", tmp_path)
    monkeypatch.delenv(KEY_NAME, raising=False)
    assert main(["interpret", "a crossroads"]) == 3
    assert "No API key" in capsys.readouterr().out


@pytest.mark.skipif(
    os.environ.get("JUNCTIONLAB_LIVE_LLM") != "1" or not llm.api_key("groq"),
    reason="live API test: needs GROQ_API_KEY and JUNCTIONLAB_LIVE_LLM=1",
)
def test_live_example_description(tmp_path):
    """One real call (not in CI). Checks the model reads example 1 as the example."""
    limiter = CallLimiter(get_config(), counter_file=tmp_path / "calls.json")
    text = EXAMPLES["four_arm_signal"].description
    result = interpret(text, LlmSession(), limiter=limiter)
    expected = EXAMPLES["four_arm_signal"].scenario()
    assert {a: x.demand_vph for a, x in result.scenario.approaches.items()} == {
        a: x.demand_vph for a, x in expected.approaches.items()
    }
    assert result.scenario.control.kind == "signal" and result.scenario.control.cycle_s == 90
