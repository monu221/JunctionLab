"""The Anthropic path, on synthetic responses in the Messages API format (no key was
available to record real ones). No test here makes a network call."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from junctionlab import llm
from junctionlab.config import get_config
from junctionlab.examples import list_examples
from junctionlab.llm import (
    CallLimiter,
    InterpretationError,
    LimitReached,
    LlmSession,
    LlmUnavailable,
    ReplayTransport,
    interpret,
    load_recording,
    make_client,
)

RECORDINGS = Path(__file__).parent / "data" / "llm_anthropic"
EXAMPLES = {e.key: e for e in list_examples()}
KEY_NAME = llm.API_KEY_NAMES["anthropic"]


@pytest.fixture
def limiter(tmp_path):
    return CallLimiter(get_config(), counter_file=tmp_path / "calls.json")


def replay(name, limiter, session=None):
    """Run interpret() on a recording; returns (result, transport)."""
    rec = load_recording(RECORDINGS / f"{name}.json")
    transport = ReplayTransport(rec)
    client = make_client("sk-test", transport=transport, max_retries=0, provider="anthropic")
    result = interpret(rec["text"], session or LlmSession(), client=client, limiter=limiter)
    return result, transport


def replay_error(name, limiter):
    rec = load_recording(RECORDINGS / f"{name}.json")
    client = make_client(
        "sk-test", transport=ReplayTransport(rec), max_retries=0, provider="anthropic"
    )
    return lambda: interpret(rec["text"], LlmSession(), client=client, limiter=limiter)


# ------------------------------------------------------------------ the request


def test_request_uses_config_model_and_structured_output(limiter):
    _, transport = replay("four_arm_signal", limiter)
    (body,) = transport.requests
    cfg = get_config()
    assert body["model"] == cfg.llm.anthropic.model == "claude-haiku-5-5"
    assert body["max_tokens"] == cfg.llm.anthropic.max_tokens
    assert body["output_config"]["effort"] == cfg.llm.anthropic.effort
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["system"] == llm.SYSTEM_PROMPT
    assert body["messages"] == [{"role": "user", "content": transport.recording["text"]}]
    assert "thinking" not in body and "temperature" not in body


def _walk(node, path="$"):
    if isinstance(node, dict):
        yield path, node
        for k, v in node.items():
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, f"{path}[{i}]")


def test_schema_meets_structured_output_rules(limiter):
    """Every object closed and fully required; no unsupported constraints."""
    _, transport = replay("four_arm_signal", limiter)
    schema = transport.requests[0]["output_config"]["format"]["schema"]
    for path, node in _walk(schema):
        if node.get("type") == "object":
            assert node.get("additionalProperties") is False, path
            assert set(node.get("required", [])) == set(node.get("properties", {})), path
        for key in ("minimum", "maximum", "minLength", "maxLength", "minItems", "maxItems"):
            assert key not in node, (path, key)


def test_llm_schema_matches_what_the_sdk_sends(limiter):
    _, transport = replay("four_arm_signal", limiter)
    sent = transport.requests[0]["output_config"]["format"]["schema"]
    props = TypeAdapter(llm.Interpretation).json_schema()["properties"]
    assert set(sent["properties"]) == set(props)


# ------------------------------------------------------------------ bundled examples


@pytest.mark.parametrize("key", ["four_arm_signal", "t_junction_priority", "four_arm_bus_lane"])
def test_example_descriptions_give_the_example_scenario(key, limiter):
    """The example's description, through the LLM path, gives the example's own scenario."""
    result, _ = replay(key, limiter)
    expected = EXAMPLES[key].scenario()
    got = result.scenario
    assert got.junction_type == expected.junction_type
    assert got.approaches == expected.approaches
    assert got.control == expected.control
    assert set(got.assumed) == set(expected.assumed)


# ------------------------------------------------------------------ conversions


def test_movement_counts_and_remainder_share(limiter):
    result, _ = replay("movement_counts", limiter)
    s = result.scenario
    assert s.approaches["north"].demand_vph == 800
    assert s.approaches["north"].turning_shares_pct == {"left": 12.5, "through": 75, "right": 12.5}
    # east: left 0 and right 33.3 stated, through is the rest
    assert s.approaches["east"].turning_shares_pct["through"] == pytest.approx(66.7)
    assert s.control.phase_plan == "four_phase_split" and s.control.cycle_s == 100
    assert len(s.control.green_s) == 4 and "control.green_s" in s.assumed
    assert s.speed_limit_kmh == 50 and "speed_limit_kmh" not in s.assumed
    assert "approaches.west.turning_shares_pct" in s.assumed
    assert any("pedestrian" in n for n in result.notes)


def test_partial_shares_on_t_junction(limiter):
    """West: 10% left into the stem, right leads nowhere (0), so through is 90%."""
    result, _ = replay("partial_shares", limiter)
    s = result.scenario
    assert s.stem == "north"
    assert s.approaches["west"].turning_shares_pct == {"left": 10, "through": 90, "right": 0}
    assert "approaches.west.turning_shares_pct" not in s.assumed
    assert "approaches.east.turning_shares_pct" in s.assumed


def test_two_missing_shares_fall_back_to_defaults():
    notes: list[str] = []
    s = llm.TurningShares(left=20, through=None, right=None)
    assert llm._shares(s, ("left", "through", "right"), "north", notes) is None
    assert "partly stated" in notes[0]


def _interp(**kw):
    base = {
        "understood": True,
        "unclear": [],
        "notes": [],
        "interpreted": [],
        "name": None,
        "junction_type": "four_arm",
        "speed_limit_kmh": None,
        "approach_length_m": None,
        "approaches": {
            arm: {
                "demand_vph": 300,
                "lane_count": None,
                "lanes": None,
                "turning_shares_pct": None,
                "heavy_vehicle_share_pct": None,
                "bus_lane": None,
                "bus_demand_bph": None,
                "bus_right_share_pct": None,
            }
            for arm in ("north", "east", "south", "west")
        },
        "control": {
            "kind": "all_way_stop",
            "cycle_s": None,
            "phase_plan": None,
            "green_s": None,
            "major_road": None,
        },
    }
    base.update(kw)
    return llm.Interpretation.model_validate(base)


def test_required_fields_are_reported_together():
    i = _interp(junction_type=None)
    i.approaches.east.demand_vph = None
    i.control.kind = None
    with pytest.raises(InterpretationError) as e:
        llm.to_draft(i)
    text = str(e.value)
    assert "junction type" in text and "east approach" in text and "control" in text


def test_wrong_arm_count_is_an_interpretation_error():
    i = _interp(junction_type="t_junction")
    with pytest.raises(InterpretationError, match="exactly three approaches"):
        llm.to_draft(i)


# ------------------------------------------------------------------ asking to rephrase


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("unclear_demand", "traffic"),
        ("not_a_junction", "does not describe a road junction"),
        ("invalid_cycle", "too short"),
        ("refusal", "declined"),
        ("max_tokens", "cut off"),
    ],
)
def test_rephrase_cases(name, reason, limiter):
    with pytest.raises(InterpretationError) as e:
        replay_error(name, limiter)()
    assert reason in str(e.value)


def test_empty_and_too_long_text_make_no_call(limiter):
    session = LlmSession()
    with pytest.raises(InterpretationError, match="empty"):
        interpret("   ", session, client=object(), limiter=limiter)
    with pytest.raises(InterpretationError, match="longer than"):
        interpret("x" * 5000, session, client=object(), limiter=limiter)
    assert session.calls == 0 and limiter.used_today() == 0


# ------------------------------------------------------------------ API failures


@pytest.mark.parametrize(("name", "reason"), [("auth_error", "rejected"), ("overloaded", "529")])
def test_api_errors(name, reason, limiter):
    with pytest.raises(LlmUnavailable, match=reason):
        replay_error(name, limiter)()


def test_no_api_key(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "REPO_ROOT", tmp_path)  # no .env there
    monkeypatch.delenv(KEY_NAME, raising=False)
    assert llm.api_key("anthropic") is None
    with pytest.raises(LlmUnavailable, match="No API key"):
        make_client(provider="anthropic")


def test_api_key_from_dotenv(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "REPO_ROOT", tmp_path)
    monkeypatch.delenv(KEY_NAME, raising=False)
    (tmp_path / ".env").write_text(f"{KEY_NAME}=sk-from-dotenv\n", encoding="utf-8")
    assert llm.api_key("anthropic") == "sk-from-dotenv"
    monkeypatch.delenv(KEY_NAME, raising=False)  # load_dotenv set it


# ------------------------------------------------------------------ limits


def test_session_limit(limiter):
    session = LlmSession()
    for _ in range(get_config().llm.calls_per_session):
        limiter.reserve(session)
    with pytest.raises(LimitReached, match="session"):
        limiter.reserve(session)


def test_daily_limit_and_reset(tmp_path):
    cfg = get_config()
    day = [datetime(2026, 11, 18, 23, 0, tzinfo=UTC)]
    limiter = CallLimiter(cfg, counter_file=tmp_path / "calls.json", now=lambda: day[0])
    for _ in range(cfg.llm.calls_per_day):
        limiter.reserve(LlmSession())
    assert limiter.used_today() == cfg.llm.calls_per_day
    with pytest.raises(LimitReached, match="daily"):
        limiter.reserve(LlmSession())
    # A new limiter on the same file (server restart) still sees the count.
    again = CallLimiter(cfg, counter_file=tmp_path / "calls.json", now=lambda: day[0])
    with pytest.raises(LimitReached):
        again.reserve(LlmSession())
    day[0] = datetime(2026, 11, 19, 0, 5, tzinfo=UTC)
    limiter.reserve(LlmSession())
    assert json.loads((tmp_path / "calls.json").read_text())["calls"] == 1


def test_failed_calls_count(limiter):
    session = LlmSession()
    with pytest.raises(LlmUnavailable):
        rec = load_recording(RECORDINGS / "overloaded.json")
        client = make_client(
            "sk-test", transport=ReplayTransport(rec), max_retries=0, provider="anthropic"
        )
        interpret("four arms", session, client=client, limiter=limiter)
    assert session.calls == 1 and limiter.used_today() == 1


def test_limit_checked_before_calling(limiter):
    session = LlmSession(calls=get_config().llm.calls_per_session)
    with pytest.raises(LimitReached):
        interpret("four arms", session, client=object(), limiter=limiter)


def test_answer_that_does_not_fit_the_schema(limiter):
    rec = load_recording(RECORDINGS / "four_arm_signal.json")
    rec["response"]["body"]["content"][1]["text"] = '{"understood": true}'
    client = make_client(
        "sk-test", transport=ReplayTransport(rec), max_retries=0, provider="anthropic"
    )
    with pytest.raises(InterpretationError, match="did not fit"):
        interpret(rec["text"], LlmSession(), client=client, limiter=limiter)
