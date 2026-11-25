"""Signal-design agent: recorded and scripted responses, a fake simulator, no live calls."""

import json
import random
from pathlib import Path

import pytest
from agent_fake import (
    SequenceTransport,
    fake_summary,
    propose,
    recorded_session,
    simulate,
    text_turn,
    tool_turn,
)
from pydantic import ValidationError
from streamlit.testing.v1 import AppTest

from junctionlab import agent, agent_eval, llm
from junctionlab.agent import ArmResult, Goal, Outcome
from junctionlab.config import get_config
from junctionlab.examples import EXAMPLES_DIR, load_example
from junctionlab.scenario import UnconfirmedDemand

APP = str(Path(__file__).parents[1] / "app" / "streamlit_app.py")
GATE = Path(__file__).parent / "data" / "agent" / "tool_call_gate.json"
GOAL = Goal("All approaches LOS C or better, shortest cycle.", "shortest_cycle", "C")
RATE_LIMITED = (
    429,
    {"error": {"message": "rate limit", "code": "rate_limit_exceeded"}},
    {"retry-after": "15"},
)
NO_RESET = (429, {"error": {"message": "rate limit", "code": "rate_limit_exceeded"}}, {})
DAILY = (
    429,
    {"error": {"message": "Rate limit reached on tokens per day (TPD)", "code": "rate_limit"}},
    {"retry-after": "600"},
)


def base():
    return load_example(EXAMPLES_DIR / "four_arm_signal.json").scenario()


def client(responses):
    transport = SequenceTransport(responses)
    return llm.make_client("test-key", transport=transport, max_retries=0, provider="groq")


@pytest.fixture
def limiter(tmp_path):
    return agent.AgentLimiter(get_config(), counter_file=tmp_path / "calls.json")


def run(responses, limiter, scenario=None, goal=GOAL):
    return agent.run_agent(
        scenario or base(),
        goal,
        client(responses),
        simulate=fake_summary,
        limiter=limiter,
        pause_s=0,
    )


def outcome(grades, mean="C", delay=30.0, lower=False):
    return Outcome(
        {
            "north": ArmResult(delay, delay + 2, mean, tuple(grades), lower),
            "south": ArmResult(delay, delay + 2, "B", ("B", "B", "B"), False),
        },
        delay,
    )


# ------------------------------------------------------------------ recorded responses


def test_gate_recording_is_a_live_tool_call():
    recording = json.loads(GATE.read_text(encoding="utf-8"))
    assert recording["source"] == "live"
    call = recording["response"]["body"]["choices"][0]["message"]["tool_calls"][0]
    assert call["function"]["name"] == "propose_change"
    args = agent.ProposeArgs.model_validate_json(call["function"]["arguments"])
    assert agent.apply_change(base(), args.cycle_s, args.phase_plan, args.green_s)


def test_recorded_session_replays_to_a_design_that_meets_the_goal(limiter):
    responses = recorded_session()
    c = client(responses)
    goal = agent.read_goal("All approaches LOS C or better, shortest cycle.", c, limiter=limiter)
    assert (goal.max_los, goal.objective) == ("C", "shortest_cycle")
    before = base()
    result = agent.run_agent(before, goal, c, simulate=fake_summary, limiter=limiter, pause_s=0)
    assert result.status == "met"
    assert result.runs_used <= get_config().agent.max_simulations_per_goal
    assert result.llm_calls + 1 <= get_config().agent.max_llm_calls_per_goal
    assert result.best.control.cycle_s < before.control.cycle_s
    assert result.best.approaches == before.approaches
    assert all(t.reason for t in result.trace)


# ------------------------------------------------------------------ goal


def test_goal_status_separates_met_borderline_and_not_met():
    assert agent.goal_status(GOAL, outcome(("B", "C", "C"))) == "met"
    assert agent.goal_status(GOAL, outcome(("B", "C", "D"))) == "borderline"
    assert agent.goal_status(GOAL, outcome(("D", "D", "D"), mean="D")) == "not_met"
    assert agent.goal_status(GOAL, outcome(("B", "C", "C"), lower=True)) == "not_met"


def test_best_design_prefers_met_over_borderline_even_with_a_longer_cycle():
    s = base()
    short = agent.apply_change(s, cycle_s=50)
    long = agent.apply_change(s, cycle_s=70)
    cfg = get_config()
    assert agent.rank(GOAL, long, outcome(("C",) * 3), cfg) < agent.rank(
        GOAL, short, outcome(("C", "C", "D")), cfg
    )


def test_goal_that_needs_other_changes_is_refused():
    reading = agent.GoalReading(
        understood=False, problem="It changes demand.", max_los=None, objective=None
    )
    with pytest.raises(agent.GoalError, match="changes demand"):
        agent.goal_from_reading("Halve the traffic", reading)


def test_shortest_cycle_needs_a_los_limit():
    reading = agent.GoalReading(
        understood=True, problem=None, max_los=None, objective="shortest_cycle"
    )
    with pytest.raises(agent.GoalError, match="LOS limit"):
        agent.goal_from_reading("Shortest cycle", reading)


# ------------------------------------------------------------------ changes


@pytest.mark.parametrize(
    "kw",
    [
        {"cycle_s": 70, "green_s": [10, 10]},  # greens do not add up
        {"cycle_s": 70.5},
        {"cycle_s": 30},
        {"cycle_s": 200},
        {"green_s": [49, 31, 5]},
        {"green_s": [2, 78]},  # below the minimum green
        {},
    ],
)
def test_invalid_change_is_rejected(kw):
    with pytest.raises((ValueError, ValidationError)):
        agent.apply_change(base(), **kw)


def test_t_junction_cannot_take_four_phases():
    t = agent_eval.junctions()["t_junction_signal"]
    with pytest.raises(ValidationError):
        agent.apply_change(t, phase_plan="four_phase_split")


def test_change_keeps_demand_and_geometry():
    s = base()
    new = agent.apply_change(s, cycle_s=60, phase_plan="four_phase_split")
    assert new.approaches == s.approaches
    assert new.approach_length_m == s.approach_length_m
    assert new.speed_limit_kmh == s.speed_limit_kmh
    assert sum(new.control.green_s) + 4 * get_config().control.intergreen_s == 60


def test_tool_cannot_carry_demand_or_scenario_fields(limiter):
    args = {**propose(70)[1], "demand_vph": 100}
    result = run([tool_turn(("propose_change", args)), text_turn(), text_turn()], limiter)
    assert result.runs_used == 1
    assert any("Rejected" in t.result for t in result.trace)
    assert result.best.approaches == base().approaches


def test_invalid_change_goes_back_to_the_agent_as_an_error(limiter):
    transport_turns = [
        tool_turn(propose(70, green=[10, 10])),
        tool_turn(propose(70), simulate("D1")),
        text_turn(),
    ]
    c = client(transport_turns)
    transport = c._http._transport
    result = agent.run_agent(base(), GOAL, c, simulate=fake_summary, limiter=limiter, pause_s=0)
    tool_message = transport.requests[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert "error" in json.loads(tool_message["content"])
    assert result.runs_used == 2


# ------------------------------------------------------------------ guardrails


def test_simulation_budget_is_never_exceeded(limiter):
    turns = [
        tool_turn(propose(60 + i), simulate(f"D{i + 1}"))
        for i in range(get_config().agent.max_llm_calls_per_goal)
    ]
    runs = []

    def counted(s):
        runs.append(s)
        return fake_summary(s)

    result = agent.run_agent(
        base(), GOAL, client(turns), simulate=counted, limiter=limiter, pause_s=0
    )
    assert len(runs) == result.runs_used == get_config().agent.max_simulations_per_goal
    assert result.stop_reason == "simulation_budget"


def test_extra_simulation_in_the_last_turn_is_refused(limiter):
    calls = []
    for i in range(1, 10):
        calls += [propose(60 + i), simulate(f"D{i}")]
    result = run([tool_turn(*calls)], limiter)
    assert result.runs_used == get_config().agent.max_simulations_per_goal
    assert any("budget is used up" in t.result for t in result.trace)


def test_llm_call_budget_stops_the_loop(limiter):
    turns = [tool_turn(propose(cycle=70, reason="same")) for _ in range(20)]
    result = run(turns, limiter)
    assert result.stop_reason == "llm_budget"
    assert result.llm_calls == get_config().agent.max_llm_calls_per_goal - 1


def test_repeated_design_costs_no_run(limiter):
    turns = [
        tool_turn(propose(70), simulate("D1")),
        tool_turn(propose(70), simulate("D1")),
        text_turn(),
    ]
    assert run(turns, limiter).runs_used == 2


def test_goals_per_session_limit():
    session = agent.AgentSession()
    for _ in range(get_config().agent.max_goals_per_session):
        agent.start_goal(session)
    with pytest.raises(llm.LimitReached, match="agent goals"):
        agent.start_goal(session)


def test_daily_limit_stops_with_the_best_design(tmp_path):
    cfg = get_config()
    agent_cfg = cfg.agent.model_copy(update={"llm_calls_per_day": 1})
    small = cfg.model_copy(update={"agent": agent_cfg})
    limiter = agent.AgentLimiter(small, counter_file=tmp_path / "calls.json")
    result = agent.run_agent(
        base(),
        GOAL,
        client([tool_turn(propose(70), simulate("D1"))]),
        simulate=fake_summary,
        config=small,
        limiter=limiter,
        pause_s=0,
    )
    assert result.stop_reason == "limit"
    assert result.runs_used == 2


def test_unconfirmed_demand_blocks_the_agent(limiter):
    s = base().model_copy(update={"interpreted": {"approaches.north.demand_vph": "per road?"}})
    with pytest.raises(UnconfirmedDemand):
        run([], limiter, scenario=s)


def test_rate_limit_stops_cleanly_with_the_best_design(limiter):
    turns = [tool_turn(propose(60), simulate("D1")), RATE_LIMITED]
    c = client(turns)
    result = agent.run_agent(base(), GOAL, c, simulate=fake_summary, limiter=limiter, pause_s=0)
    assert result.stop_reason == "rate_limited"
    assert "15 s" in result.message
    assert result.runs_used == 2
    assert len(c._http._transport.requests) == 2  # a 429 is not retried


def run_waiting(turns, limiter, waits=10):
    slept = []
    c = client(turns)
    result = agent.run_agent(
        base(),
        GOAL,
        c,
        simulate=fake_summary,
        limiter=limiter,
        pause_s=0,
        rate_limit_waits=waits,
        sleep=slept.append,
    )
    return result, slept, c


def test_evaluation_mode_waits_out_a_429_and_continues(limiter):
    turns = [tool_turn(propose(60), simulate("D1")), RATE_LIMITED, NO_RESET, text_turn()]
    turns.append(text_turn())
    result, slept, c = run_waiting(turns, limiter)
    assert slept[:2] == [15.0, 60.0]  # Groq's reset time, else the default
    assert result.rate_limit_waits == 2
    assert result.stop_reason == "agent_finished"
    assert result.llm_calls == len(c._http._transport.requests) - 2  # retries are not counted


def test_evaluation_mode_stops_after_the_last_allowed_wait(limiter):
    turns = [tool_turn(propose(60), simulate("D1")), RATE_LIMITED, RATE_LIMITED]
    result, slept, _c = run_waiting(turns, limiter, waits=1)
    assert (result.stop_reason, result.rate_limit_waits, slept) == ("rate_limited", 1, [15.0])


def test_evaluation_mode_does_not_wait_out_the_daily_quota(limiter):
    result, slept, _c = run_waiting([tool_turn(propose(60), simulate("D1")), DAILY], limiter)
    assert (result.stop_reason, result.rate_limit_waits, slept) == ("rate_limited", 0, [])


def test_server_error_stops_with_the_best_design(limiter):
    turns = [tool_turn(propose(60), simulate("D1")), (500, {"error": {"message": "x"}}, {})]
    result = run(turns, limiter)
    assert result.stop_reason == "llm_error"
    assert result.runs_used == 2


def test_model_stopping_early_is_asked_once_to_continue(limiter):
    impossible = Goal("LOS A", "min_junction_delay", "A")
    result = run([text_turn(), text_turn()], limiter, goal=impossible)
    assert result.stop_reason == "agent_finished"
    assert result.llm_calls == 2


@pytest.mark.parametrize(
    ("headers", "used", "expected"),
    [
        ({"x-ratelimit-remaining-tokens": "5000"}, 1000, 0),
        ({"x-ratelimit-remaining-tokens": "900", "x-ratelimit-reset-tokens": "7.5s"}, 1000, 8.5),
        ({"x-ratelimit-remaining-tokens": "10", "x-ratelimit-reset-tokens": "1m2s"}, 1000, 63),
        ({"x-ratelimit-remaining-tokens": "10", "x-ratelimit-reset-tokens": "500ms"}, 1000, 1.5),
        ({"x-ratelimit-remaining-tokens": "10", "x-ratelimit-reset-tokens": "9m"}, 1000, 65),
        ({}, 1000, 0),
    ],
)
def test_wait_for_the_token_budget(headers, used, expected):
    assert agent.token_wait_s(headers, used) == pytest.approx(expected)


def test_anthropic_messages_group_tool_results():
    call = {"id": "t1", "function": {"name": "simulate", "arguments": '{"design_id":"D0"}'}}
    second = {"id": "t2", "function": {"name": "simulate", "arguments": '{"design_id":"D1"}'}}
    out = agent._anthropic_messages(
        [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "", "tool_calls": [call, second]},
            {"role": "tool", "tool_call_id": "t1", "content": "{}"},
            {"role": "tool", "tool_call_id": "t2", "content": "{}"},
        ]
    )
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert out[1]["content"][0]["input"] == {"design_id": "D0"}
    assert [b["tool_use_id"] for b in out[2]["content"]] == ["t1", "t2"]


# ------------------------------------------------------------------ evaluation


def fake_outcome(s, _seeds):
    return Outcome.from_summary(fake_summary(s))


def test_webster_cycle_uses_the_formula():
    cfg = get_config()
    s = base()
    y = agent_eval.critical_flow_ratios(s, cfg)
    lost = 2 * (cfg.control.lost_time_s + cfg.control.all_red_s)
    assert agent_eval.webster_cycle(s, cfg) == pytest.approx((1.5 * lost + 5) / (1 - sum(y)))
    assert agent_eval.webster_design(s, cfg).control.cycle_s == 43


def test_holdout_seeds_are_unused():
    cfg = get_config()
    assert not set(cfg.simulation.seeds) & set(cfg.agent.evaluation.holdout_seeds)
    assert len(cfg.agent.evaluation.holdout_seeds) == 5


def test_t_junction_copy_keeps_demand_and_geometry():
    t = load_example(EXAMPLES_DIR / "t_junction_priority.json").scenario()
    copy = agent_eval.junctions()["t_junction_signal"]
    assert copy.control.kind == "signal"
    assert copy.approaches == t.approaches
    assert copy.approach_length_m == t.approach_length_m


@pytest.mark.parametrize("key", ["four_arm_signal", "t_junction_signal"])
def test_grid_designs_are_valid_and_distinct(key):
    s = agent_eval.junctions()[key]
    designs = agent_eval.grid_designs(s)
    assert len({agent.design_key(d) for d in designs}) == len(designs) >= 100
    assert all(d.approaches == s.approaches for d in designs)


def test_random_search_is_reproducible_and_valid():
    s = base()
    a = agent_eval.random_designs(s, random.Random(7))
    b = agent_eval.random_designs(s, random.Random(7))
    assert [agent.design_key(d) for d in a] == [agent.design_key(d) for d in b]
    assert len(a) == get_config().agent.evaluation.random_designs
    assert all(d.approaches == s.approaches for d in a)


def test_low_memory_stops_before_a_simulation():
    cfg = get_config()
    with pytest.raises(agent_eval.MemoryLow, match="Stopped cleanly"):
        agent_eval.check_memory(cfg, lambda: cfg.agent.evaluation.min_free_memory_gb - 0.1)
    agent_eval.check_memory(cfg, lambda: cfg.agent.evaluation.min_free_memory_gb + 0.1)
    agent_eval.check_memory(cfg, lambda: None)  # a system that does not report memory


def test_parallel_runs_are_limited(monkeypatch):
    seen = {}

    class Done:
        summary = fake_summary(base())

    def fake_run(_s, _dir, **kw):
        seen.update(kw)
        return Done

    monkeypatch.setattr(agent, "run_scenario", fake_run)
    monkeypatch.setattr(agent_eval, "check_memory", lambda _cfg: None)
    agent_eval.real_simulator(get_config())(base(), [1, 2, 3, 4, 5])
    assert seen["max_workers"] == get_config().agent.evaluation.max_parallel_runs == 3


def test_report_needs_every_agent_run(tmp_path):
    store = agent_eval.Store(tmp_path)
    with pytest.raises(ValueError, match="agent run"):
        agent_eval.build_report(store=store, simulator=fake_outcome)


def test_preliminary_report_uses_the_agent_runs_saved_so_far(tmp_path, limiter):
    cfg = get_config()
    store = agent_eval.Store(tmp_path)
    agent_eval.run_grid(cfg, store, fake_outcome, progress=lambda _m: None)
    agent_eval.run_baselines(cfg, store, fake_outcome, progress=lambda _m: None)
    agent_eval.run_agent_stage(
        client([text_turn(), text_turn()]),
        cfg,
        store,
        fake_outcome,
        limiter,
        progress=lambda _m: None,
        pause_s=0,
        summary_simulator=fake_summary,
        max_runs=1,
    )
    store.save_rate_limited("four_arm_signal|G1|2", {"stop_reason": "rate_limited"})
    store.save_rate_limited(
        "four_arm_signal|G1|2", {"stop_reason": "rate_limited", "message": "daily quota is used"}
    )
    report = agent_eval.build_report(cfg, store, fake_outcome, preliminary=True)
    assert "Preliminary, 1 of 18 agent runs" in report
    assert "1 of 3 repeats" in report and "not run yet" in report
    assert "ended 2 agent run(s) early (1 of them by the daily token quota; 1 were" in report


def test_report_compares_all_methods(tmp_path, limiter):
    cfg = get_config()
    store = agent_eval.Store(tmp_path)
    agent_eval.run_grid(cfg, store, fake_outcome, progress=lambda _m: None)
    agent_eval.run_baselines(cfg, store, fake_outcome, progress=lambda _m: None)
    n = 6 * cfg.agent.evaluation.repeats
    turns = []
    for _ in range(n):
        turns += [tool_turn(propose(60), simulate("D1")), text_turn(), text_turn()]
    done = agent_eval.run_agent_stage(
        client(turns),
        cfg,
        store,
        fake_outcome,
        limiter,
        progress=lambda _m: None,
        pause_s=0,
        summary_simulator=fake_summary,
    )
    assert done == n
    report = agent_eval.build_report(cfg, store, fake_outcome)
    for text in ("Agent", "Webster", "Random search", "Grid optimum", "Unused seeds"):
        assert text in report
    assert report.count("### G1") == 2
    assert "Note on the free tier" in report


def test_agent_stage_respects_max_runs(tmp_path, limiter):
    cfg = get_config()
    store = agent_eval.Store(tmp_path)
    turns = [text_turn(), text_turn()] * 2
    done = agent_eval.run_agent_stage(
        client(turns),
        cfg,
        store,
        fake_outcome,
        limiter,
        progress=lambda _m: None,
        pause_s=0,
        summary_simulator=fake_summary,
        max_runs=2,
    )
    assert done == len(store.agent_runs) == 2


def test_agent_stage_keeps_a_run_ended_by_a_429_out_of_the_results(tmp_path, limiter):
    cfg = get_config()
    store = agent_eval.Store(tmp_path)
    done = agent_eval.run_agent_stage(
        client([DAILY]),
        cfg,
        store,
        fake_outcome,
        limiter,
        progress=lambda _m: None,
        pause_s=0,
        summary_simulator=fake_summary,
        sleep=lambda _s: None,
    )
    assert done == 0
    assert store.agent_runs == {}
    [(key, run)] = store.rate_limited.items()
    assert key == "four_arm_signal|G1|1#1" and "daily quota" in run["message"]
    assert agent_eval.Store(tmp_path).rate_limited == store.rate_limited  # saved to disk


def stage_output(tmp_path, limiter, turns):
    printed = []
    agent_eval.run_agent_stage(
        client(turns),
        get_config(),
        agent_eval.Store(tmp_path),
        fake_outcome,
        limiter,
        progress=printed.append,
        pause_s=0,
        summary_simulator=fake_summary,
        max_runs=1,
        sleep=lambda _s: None,
    )
    return printed


def test_agent_stage_daily_quota_says_only_when_to_resume(tmp_path, limiter):
    printed = stage_output(tmp_path, limiter, [DAILY])
    assert printed[-1] == (
        "Stopped. The free Groq tier's daily quota is used up. "
        "Run this stage again in about 10 min."
    )
    assert not any("form" in line or "waiting" in line for line in printed)


def test_agent_stage_per_minute_429_says_it_is_waiting(tmp_path, limiter):
    turns = [tool_turn(propose(60), simulate("D1")), RATE_LIMITED, text_turn(), text_turn()]
    printed = stage_output(tmp_path, limiter, turns)
    waits = get_config().agent.evaluation.rate_limit_waits
    assert printed[1] == (
        "four_arm_signal|G1|1: the free Groq tier's per-minute rate limit is reached; "
        f"waiting 15 s (wait 1 of {waits})."
    )
    assert not any("form" in line or "Stopped" in line for line in printed)


def test_agent_stage_moves_old_rate_limited_runs_aside(tmp_path, limiter):
    cfg = get_config()
    store = agent_eval.Store(tmp_path)
    store.save_agent_run("four_arm_signal|G1|1", {"stop_reason": "agent_finished"})
    store.save_agent_run("four_arm_signal|G1|2", {"stop_reason": "rate_limited"})
    agent_eval.run_agent_stage(
        client([text_turn(), text_turn()]),
        cfg,
        store,
        fake_outcome,
        limiter,
        progress=lambda _m: None,
        pause_s=0,
        summary_simulator=fake_summary,
        max_runs=1,
        sleep=lambda _s: None,
    )
    assert list(store.rate_limited) == ["four_arm_signal|G1|2#1"]
    assert store.agent_runs["four_arm_signal|G1|2"]["stop_reason"] != "rate_limited"


def test_agent_stage_pauses_between_runs(tmp_path, limiter):
    cfg = get_config()
    slept = []
    agent_eval.run_agent_stage(
        client([text_turn(), text_turn()] * 2),
        cfg,
        agent_eval.Store(tmp_path),
        fake_outcome,
        limiter,
        progress=lambda _m: None,
        summary_simulator=fake_summary,
        max_runs=2,
        sleep=slept.append,
    )
    assert slept.count(cfg.agent.evaluation.pause_between_runs_s) >= 1


# ------------------------------------------------------------------ app


def app_with_result(monkeypatch):
    monkeypatch.setattr(llm, "make_client", lambda *_a, **_k: object())
    monkeypatch.setattr(agent, "read_goal", lambda *_a, **_k: GOAL)

    def finished(s, goal, _client, on_step=None, **_kw):
        best = agent.apply_change(s, cycle_s=60)
        return agent.AgentResult(
            goal=goal,
            before=s,
            best=best,
            best_id="D1",
            best_outcome=Outcome.from_summary(fake_summary(best)),
            before_summary=fake_summary(s),
            best_summary=fake_summary(best),
            status="borderline",
            stop_reason="rate_limited",
            runs_used=2,
            best_at_run=2,
            llm_calls=3,
            trace=[
                agent.TraceStep(1, "Before", "Met", "Measure"),
                agent.TraceStep(2, "60 s", "Met", "Why"),
            ],
            message="Try again in about 15 s.",
        )

    monkeypatch.setattr(agent, "run_agent", finished)
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["scenario"] = base()
    at.session_state["source"] = "example"
    at.run()
    at.text_input(key="agent_goal_text").set_value("LOS C, shortest cycle")
    next(b for b in at.button if b.key == "agent_read_goal").click().run()
    next(b for b in at.button if b.key == "agent_start").click().run()
    assert not at.exception
    return at


def test_app_shows_the_result_and_adopts_only_the_timing(monkeypatch):
    at = app_with_result(monkeypatch)
    text = "\n".join(str(x.value) for x in [*at.markdown, *at.caption, *at.warning])
    assert "Borderline" in text
    assert "Stopped by the provider's rate limit" in text
    before = at.session_state["scenario"]
    next(b for b in at.button if b.key == "agent_adopt").click().run()
    assert not at.exception
    after = at.session_state["scenario"]
    assert after.control.cycle_s == 60
    assert after.approaches == before.approaches


def test_app_offers_no_agent_for_a_priority_junction():
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["scenario"] = load_example(
        EXAMPLES_DIR / "t_junction_priority.json"
    ).scenario()
    at.session_state["source"] = "example"
    at.run()
    assert not at.exception
    assert not [t for t in at.text_input if t.key == "agent_goal_text"]
