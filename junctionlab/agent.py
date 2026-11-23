"""Signal-design agent (docs/method.md, section 6).

The LLM proposes signal timings through tools. Python keeps everything that must be trusted:
- the budget (simulation runs, LLM calls, goals);
- validation: only cycle, greens and phase plan can change, through `apply_change`;
- the goal check and the choice of the best design.

The simulate and compare tools take design ids, so the model never handles a scenario and has
no way to send changed demand or geometry.
"""

from __future__ import annotations

import json
import re
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import anthropic
from pydantic import BaseModel, ConfigDict, ValidationError

from junctionlab import llm
from junctionlab.config import Config, get_config
from junctionlab.metrics import Summary
from junctionlab.runner import run_scenario
from junctionlab.scenario import FixedTimeSignal, PhasePlan, Scenario

Objective = Literal["shortest_cycle", "min_junction_delay", "min_worst_delay"]
Status = Literal["met", "borderline", "not_met"]
StopReason = Literal[
    "agent_finished", "simulation_budget", "llm_budget", "rate_limited", "limit", "llm_error"
]
Simulate = Callable[[Scenario], Summary]

TOKEN_MARGIN = 1.5  # the conversation grows, so the next call needs more than the last one
MAX_TOKEN_WAIT_S = 65.0
_DURATION = re.compile(r"(?:\d+(?:\.\d+)?(?:h|ms|m|s))+")
_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(h|ms|m|s)")

OBJECTIVE_LABELS = {
    "shortest_cycle": "shortest cycle",
    "min_junction_delay": "lowest junction delay",
    "min_worst_delay": "lowest worst-approach delay",
}
STATUS_LABELS = {"met": "Met", "borderline": "Borderline", "not_met": "Not met"}
STOP_LABELS = {
    "agent_finished": "The agent finished.",
    "simulation_budget": "The simulation budget is used up.",
    "llm_budget": "The LLM call budget for this goal is used up.",
    "rate_limited": "Stopped by the provider's rate limit.",
    "limit": "The daily agent limit is reached.",
    "llm_error": "Stopped by an LLM error.",
}


class GoalError(llm.LlmError):
    """The goal text could not be used. The message is shown to the user."""


# --------------------------------------------------------------------------- goal


class GoalReading(BaseModel):
    model_config = ConfigDict(extra="forbid")

    understood: bool
    problem: str | None
    max_los: Literal["A", "B", "C", "D", "E"] | None
    objective: Objective | None


GOAL_PROMPT = """\
Read a traffic engineer's goal for a fixed-time signal. Only cycle length, green splits and \
phase plan can change.
- max_los: the worst level of service allowed on every approach, or null if none is stated.
- objective: shortest_cycle, min_junction_delay (lowest average delay over the junction) or \
min_worst_delay (lowest delay on the worst approach).
- If the goal needs anything else, such as changing traffic, lanes or road layout, set \
understood to false and say why in problem. Otherwise problem is null.
- If it states only a LOS limit, use min_junction_delay."""


@dataclass(frozen=True)
class Goal:
    text: str
    objective: Objective
    max_los: str | None = None

    @property
    def label(self) -> str:
        limit = f"every approach LOS {self.max_los} or better, in every seed; " * bool(self.max_los)
        text = f"{limit}{OBJECTIVE_LABELS[self.objective]}"
        return text[0].upper() + text[1:]


def goal_from_reading(text: str, reading: GoalReading) -> Goal:
    if not reading.understood or reading.objective is None:
        raise GoalError(reading.problem or "The goal could not be read.")
    if reading.objective == "shortest_cycle" and reading.max_los is None:
        raise GoalError("A shortest-cycle goal needs a LOS limit, such as 'LOS C or better'.")
    return Goal(text, reading.objective, reading.max_los)


def read_goal(
    text: str, client: llm.LlmClient, cfg: Config | None = None, limiter: AgentLimiter | None = None
) -> Goal:
    """One structured LLM call. Raises GoalError, or LimitReached / RateLimited / LlmError."""
    cfg = cfg or get_config()
    text = text.strip()
    if not text:
        raise GoalError("Write a goal first.")
    if len(text) > cfg.agent.max_goal_chars:
        raise GoalError(f"The goal is longer than {cfg.agent.max_goal_chars} characters.")
    (limiter or AgentLimiter(cfg)).reserve_call()
    schema = anthropic.transform_schema(GoalReading)
    if isinstance(client, llm.GroqClient):
        out = _structured_groq(text, schema, client, cfg)
    else:
        out = _structured_anthropic(text, schema, client, cfg)
    try:
        return goal_from_reading(text, GoalReading.model_validate_json(out))
    except ValidationError as e:
        raise GoalError("The model's reading of the goal was not valid.") from e


def _structured_groq(text: str, schema: dict, client: llm.GroqClient, cfg: Config) -> str:
    response = client.post_chat(
        {
            "model": cfg.llm.groq.model,
            "max_completion_tokens": cfg.llm.groq.max_tokens,
            "temperature": 0,
            "reasoning_effort": cfg.llm.groq.reasoning_effort,
            "include_reasoning": False,
            "messages": [
                {"role": "system", "content": GOAL_PROMPT},
                {"role": "user", "content": text},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "signal_goal", "strict": True, "schema": schema},
            },
        }
    )
    if response.status_code != 200:
        raise llm._groq_error(response)
    try:
        return response.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise llm.LlmUnavailable("The API returned an unreadable response.") from e


def _structured_anthropic(text: str, schema: dict, client: anthropic.Anthropic, cfg: Config) -> str:
    try:
        response = client.messages.create(
            model=cfg.llm.anthropic.model,
            max_tokens=cfg.llm.anthropic.max_tokens,
            system=GOAL_PROMPT,
            messages=[{"role": "user", "content": text}],
            output_config={
                "format": {"type": "json_schema", "schema": schema},
                "effort": cfg.llm.anthropic.effort,
            },
        )
    except anthropic.APIError as e:
        raise _anthropic_error(e) from e
    answer = next((b.text for b in response.content if b.type == "text"), None)
    if answer is None:
        raise GoalError("The model gave no reading of the goal.")
    return answer


def _anthropic_error(e: anthropic.APIError) -> llm.LlmError:
    if isinstance(e, anthropic.RateLimitError):
        return llm.RateLimited("The API is rate-limited right now. Try again in a minute.")
    return llm.LlmUnavailable("The Anthropic API call failed. Try again.")


# --------------------------------------------------------------------------- limits


class AgentLimiter(llm.CallLimiter):
    """Daily LLM calls for the agent, counted separately from text-to-scenario calls."""

    def __init__(self, cfg: Config, counter_file: Path | None = None, now=None):
        super().__init__(cfg, counter_file or cfg.agent.counter_file, now)
        self.per_day = cfg.agent.llm_calls_per_day
        self.per_session = cfg.agent.max_goals_per_session * cfg.agent.max_llm_calls_per_goal
        self._calls = llm.LlmSession()

    def reserve_call(self) -> None:
        self._calls.calls = 0  # the session limit is enforced through goals and calls per goal
        try:
            self.reserve(self._calls)
        except llm.LimitReached as e:
            raise llm.LimitReached(
                f"The daily limit of {self.per_day} agent calls is used up. Try again tomorrow."
            ) from e

    def remaining_today(self) -> int:
        return max(self.per_day - self.used_today(), 0)


@dataclass
class AgentSession:
    """Per-browser-session state."""

    goals: int = 0


# --------------------------------------------------------------------------- results


@dataclass(frozen=True)
class ArmResult:
    delay_s: float
    worst_seed_delay_s: float
    los: str
    los_by_seed: tuple[str, ...]
    lower_bound: bool


@dataclass(frozen=True)
class Outcome:
    """The measures the goal check needs. Plain data, so evaluation runs can be cached."""

    approaches: dict[str, ArmResult]
    junction_delay_s: float

    @classmethod
    def from_summary(cls, summary: Summary) -> Outcome:
        arms = {}
        for arm, a in summary.approaches.items():
            delay = a.control_delay_s
            arms[arm] = ArmResult(
                delay_s=delay.mean if delay else 0.0,
                worst_seed_delay_s=delay.max if delay else 0.0,
                los=a.los,
                los_by_seed=tuple(a.los_by_seed),
                lower_bound=a.delay_lower_bound,
            )
        j = summary.junction.control_delay_s
        return cls(arms, j.mean if j else 0.0)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Outcome:
        arms = {
            arm: ArmResult(**{**a, "los_by_seed": tuple(a["los_by_seed"])})
            for arm, a in data["approaches"].items()
        }
        return cls(arms, data["junction_delay_s"])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def worst_delay_s(self) -> float:
        return max(a.delay_s for a in self.approaches.values())

    @property
    def lower_bound(self) -> bool:
        return any(a.lower_bound for a in self.approaches.values())


def _within(grade: str, limit: str) -> bool:
    return grade == "-" or grade <= limit  # "A" < "F"


def goal_status(goal: Goal, outcome: Outcome) -> Status:
    """Met: every approach in every seed. Borderline: on the mean only."""
    if outcome.lower_bound:
        return "not_met"
    if goal.max_los is None:
        return "met"
    arms = outcome.approaches.values()
    if not all(_within(a.los, goal.max_los) for a in arms):
        return "not_met"
    every_seed = all(_within(g, goal.max_los) for a in arms for g in a.los_by_seed)
    return "met" if every_seed else "borderline"


def objective_value(goal: Goal, design: Scenario, outcome: Outcome) -> float:
    if goal.objective == "shortest_cycle":
        return design.control.cycle_s
    if goal.objective == "min_junction_delay":
        return outcome.junction_delay_s
    return outcome.worst_delay_s


def rank(goal: Goal, design: Scenario, outcome: Outcome, cfg: Config) -> tuple:
    """Lower is better: status, then how far from the limit, then the objective."""
    status = goal_status(goal, outcome)
    failing, excess = 0, 0.0
    if status != "met" and goal.max_los:
        limit = cfg.los.signalised[goal.max_los]
        for a in outcome.approaches.values():
            failing += sum(not _within(g, goal.max_los) for g in a.los_by_seed)
            excess = max(excess, a.worst_seed_delay_s - limit)
    order = {"met": 0, "borderline": 1, "not_met": 2}[status]
    return (
        order,
        failing,
        round(excess, 6),
        objective_value(goal, design, outcome),
        outcome.junction_delay_s,
    )


# --------------------------------------------------------------------------- changes


def design_key(s: Scenario) -> tuple:
    c = s.control
    return (c.phase_plan, c.cycle_s, tuple(c.green_s))


def design_label(s: Scenario) -> str:
    c = s.control
    plan = "2 phases" if c.phase_plan == "two_phase" else "4 phases"
    return f"{plan}, {c.cycle_s:g} s, greens {' / '.join(f'{g:g}' for g in c.green_s)} s"


def _fixed_part(s: Scenario) -> dict[str, Any]:
    keep = {"junction_type", "speed_limit_kmh", "approach_length_m", "approaches"}
    return s.model_dump(mode="json", include=keep)


def _whole(value: float) -> bool:
    return float(value).is_integer()


def apply_change(
    base: Scenario,
    cycle_s: float | None = None,
    phase_plan: PhasePlan | None = None,
    green_s: list[float] | None = None,
    config: Config | None = None,
) -> Scenario:
    """A new design with the given timing. Without greens, a new cycle or plan gets Webster
    splits. Raises ValueError or ValidationError; the agent sees these as tool errors."""
    cfg = config or get_config()
    if not isinstance(base.control, FixedTimeSignal):
        raise ValueError("only a fixed-time signal can be designed")
    if cycle_s is None and phase_plan is None and green_s is None:
        raise ValueError("give at least one of cycle_s, phase_plan, green_s")
    control = base.control.model_dump(mode="json")
    if cycle_s is not None:
        if not _whole(cycle_s):
            raise ValueError("cycle_s must be a whole number of seconds")
        if not cfg.agent.cycle_min_s <= cycle_s <= cfg.agent.cycle_max_s:
            raise ValueError(
                f"cycle_s must be {cfg.agent.cycle_min_s}-{cfg.agent.cycle_max_s} s, got "
                f"{cycle_s:g}"
            )
        control["cycle_s"] = float(cycle_s)
    if phase_plan is not None:
        control["phase_plan"] = phase_plan
    if green_s is not None:
        if not all(_whole(g) for g in green_s):
            raise ValueError("green_s must be whole seconds")
        control["green_s"] = [float(g) for g in green_s]
    elif cycle_s is not None or phase_plan is not None:
        control["green_s"] = None  # filled with Webster splits on validation
    new = base.with_values({"control": control}, cfg)
    if _fixed_part(new) != _fixed_part(base):  # cannot happen through this path; never allow it
        raise ValueError("a design may not change demand or geometry")
    return new


# --------------------------------------------------------------------------- tools


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str


class ProposeArgs(_Args):
    cycle_s: float | None
    phase_plan: PhasePlan | None
    green_s: list[float] | None


class SimulateArgs(_Args):
    design_id: str


class CompareArgs(_Args):
    a: str
    b: str


_REASON = {"type": "string", "description": "One short line: why this step."}


def _tool(name: str, description: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"reason": _REASON, **properties},
                "required": ["reason", *properties],
                "additionalProperties": False,
            },
        },
    }


TOOLS = [
    _tool(
        "propose_change",
        "Create a new design from the latest one. Use null for fields to keep. If green_s is "
        "null and the cycle or plan changes, Webster splits are used. Whole seconds only.",
        {
            "cycle_s": {"type": ["number", "null"]},
            "phase_plan": {
                "type": ["string", "null"],
                "enum": ["two_phase", "four_phase_split", None],
            },
            "green_s": {"type": ["array", "null"], "items": {"type": "number"}},
        },
    ),
    _tool(
        "simulate",
        "Simulate a design by id. Uses one run from the budget.",
        {"design_id": {"type": "string"}},
    ),
    _tool(
        "compare",
        "Compare two simulated designs per approach. Uses no run.",
        {"a": {"type": "string"}, "b": {"type": "string"}},
    ),
]
_ARGS = {"propose_change": ProposeArgs, "simulate": SimulateArgs, "compare": CompareArgs}

SYSTEM_PROMPT = """\
You design a fixed-time traffic signal to reach the user's goal. Work through the tools: \
propose a design, simulate it, read the result, then improve it. Demand and geometry are fixed.

Rules:
- Each phase has a green plus {intergreen:g} s of intergreen, and the greens plus intergreens \
must add up to the cycle. Each green is at least {min_green:g} s.
- The cycle is {cycle_min}-{cycle_max} s. A T-junction always has two phases.
- two_phase serves north+south, then east+west. four_phase_split serves one approach at a \
time, in the order north, east, south, west.
- You have {budget} simulation runs in total, so choose each one carefully. You may call \
propose_change and simulate in the same turn, using next_design_id.
- A goal is met only if every approach meets the LOS limit in every seed. "borderline" means \
it holds on the mean but not in every seed.
- Give a short, factual reason in every tool call.
- When the goal is met and more runs are unlikely to improve it, reply without a tool call."""


@dataclass
class TraceStep:
    step: int
    change: str
    result: str
    reason: str
    design_id: str | None = None
    ok: bool = True


@dataclass
class AgentResult:
    goal: Goal
    before: Scenario
    best: Scenario
    best_id: str
    best_outcome: Outcome
    before_summary: Summary
    best_summary: Summary
    status: Status
    stop_reason: StopReason
    runs_used: int
    best_at_run: int
    llm_calls: int
    trace: list[TraceStep]
    message: str = ""

    @property
    def changed(self) -> bool:
        return design_key(self.best) != design_key(self.before)


@dataclass
class Workbench:
    """The designs of one goal, and the tools that act on them."""

    goal: Goal
    before: Scenario
    simulator: Simulate
    cfg: Config
    on_step: Callable[[list[TraceStep]], None] = lambda _trace: None
    designs: dict[str, Scenario] = field(default_factory=dict)
    summaries: dict[str, Summary] = field(default_factory=dict)
    outcomes: dict[str, Outcome] = field(default_factory=dict)
    run_order: dict[str, int] = field(default_factory=dict)
    trace: list[TraceStep] = field(default_factory=list)
    current: str = "D0"

    def __post_init__(self) -> None:
        self.designs["D0"] = self.before

    @property
    def budget(self) -> int:
        return self.cfg.agent.max_simulations_per_goal

    @property
    def runs_used(self) -> int:
        return len(self.run_order)

    @property
    def budget_used(self) -> bool:
        return self.runs_used >= self.budget

    def _add(self, change: str, result: str, reason: str, design=None, ok=True) -> TraceStep:
        step = TraceStep(len(self.trace) + 1, change, result, reason, design, ok)
        self.trace.append(step)
        self.on_step(self.trace)
        return step

    def _result_line(self, design_id: str) -> str:
        o = self.outcomes[design_id]
        grades = " ".join(f"{arm[0].upper()}:{a.los}" for arm, a in o.approaches.items())
        status = STATUS_LABELS[goal_status(self.goal, o)]
        return f"{status} · {grades} · junction {o.junction_delay_s:.1f} s/veh"

    def _result_json(self, design_id: str) -> dict[str, Any]:
        o = self.outcomes[design_id]
        return {
            "design_id": design_id,
            "design": design_label(self.designs[design_id]),
            "goal_status": goal_status(self.goal, o),
            "objective_value": round(objective_value(self.goal, self.designs[design_id], o), 1),
            "junction_delay_s": round(o.junction_delay_s, 1),
            "approaches": {
                arm: {
                    "delay_s": round(a.delay_s, 1),
                    "worst_seed_s": round(a.worst_seed_delay_s, 1),
                    "los": a.los,
                    "seed_los": "".join(a.los_by_seed),
                }
                for arm, a in o.approaches.items()
            },
            "runs_used": self.runs_used,
            "runs_left": self.budget - self.runs_used,
        }

    def run_design(self, design_id: str, reason: str) -> dict[str, Any]:
        summary = self.simulator(self.designs[design_id])
        self.summaries[design_id] = summary
        self.outcomes[design_id] = Outcome.from_summary(summary)
        self.run_order[design_id] = self.runs_used + 1
        line = self._result_line(design_id)
        row = next((t for t in self.trace if t.design_id == design_id and not t.result), None)
        if row:
            row.result = line
            self.on_step(self.trace)
        else:
            self._add(design_label(self.designs[design_id]), line, reason, design_id)
        return self._result_json(design_id)

    def call(self, name: str, raw_arguments: str) -> str:
        """Run one tool call. Errors go back to the agent as JSON; they never raise."""
        model = _ARGS.get(name)
        if model is None:
            return self._error(name, f"unknown tool {name}", "")
        try:
            args = model.model_validate_json(raw_arguments)
        except ValidationError as e:
            return self._error(name, "; ".join(llm._messages(e)), "")
        try:
            return json.dumps(getattr(self, f"_tool_{name}")(args), separators=(",", ":"))
        except ValidationError as e:
            return self._error(name, "; ".join(llm._messages(e)), args.reason)
        except ValueError as e:
            return self._error(name, str(e), args.reason)

    def _error(self, name: str, message: str, reason: str) -> str:
        self._add(name, f"Rejected: {message}", reason, ok=False)
        return json.dumps({"error": message})

    def _tool_propose_change(self, args: ProposeArgs) -> dict[str, Any]:
        new = apply_change(
            self.designs[self.current], args.cycle_s, args.phase_plan, args.green_s, self.cfg
        )
        existing = next(
            (i for i, d in self.designs.items() if design_key(d) == design_key(new)), None
        )
        design_id = existing or f"D{len(self.designs)}"
        if existing is None:
            self.designs[design_id] = new
            self._add(design_label(new), "", args.reason, design_id)
        self.current = design_id
        return {
            "design_id": design_id,
            "design": design_label(new),
            "already_exists": existing is not None,
            "already_simulated": design_id in self.outcomes,
            "next_design_id": f"D{len(self.designs)}",
        }

    def _tool_simulate(self, args: SimulateArgs) -> dict[str, Any]:
        if args.design_id not in self.designs:
            raise ValueError(f"unknown design {args.design_id}")
        if args.design_id in self.outcomes:  # costs no run
            return self._result_json(args.design_id)
        if self.budget_used:
            raise ValueError("the simulation budget is used up")
        try:
            return self.run_design(args.design_id, args.reason)
        except Exception as e:  # SUMO failed: the agent can try another design
            raise ValueError(f"the simulation failed: {e}") from e

    def _tool_compare(self, args: CompareArgs) -> dict[str, Any]:
        for design_id in (args.a, args.b):
            if design_id not in self.outcomes:
                raise ValueError(f"design {design_id} has not been simulated")
        a, b = self.outcomes[args.a], self.outcomes[args.b]
        out = {
            arm: {
                "delay_change_s": round(b.approaches[arm].delay_s - x.delay_s, 1),
                "los": f"{x.los}->{b.approaches[arm].los}",
            }
            for arm, x in a.approaches.items()
        }
        change = round(b.junction_delay_s - a.junction_delay_s, 1)
        self._add(f"compare {args.a} and {args.b}", f"Junction {change:+.1f} s/veh", args.reason)
        return {"approaches": out, "junction_delay_change_s": change}

    def best_id(self) -> str:
        return min(
            self.outcomes,
            key=lambda i: rank(self.goal, self.designs[i], self.outcomes[i], self.cfg),
        )


# --------------------------------------------------------------------------- provider turns


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class Turn:
    text: str
    calls: tuple[ToolCall, ...]
    wait_s: float = 0.0  # time to wait so the next call fits the provider's token limit


class InvalidToolCall(llm.LlmError):
    """The provider rejected the model's tool call. The model can try again."""


def _duration_s(text: str | None) -> float:
    """Groq's reset times: '7.66s', '1m2.5s', '120ms'. Unreadable values count as 0."""
    if not text or not _DURATION.fullmatch(text):
        return 0.0
    units = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}
    return sum(float(v) * units[u] for v, u in _DURATION_PART.findall(text))


def token_wait_s(headers: Mapping[str, str], total_tokens: int) -> float:
    """Seconds to wait if the tokens left this minute may not cover a similar next call."""
    try:
        remaining = int(headers["x-ratelimit-remaining-tokens"])
    except (KeyError, ValueError):
        return 0.0
    if remaining >= total_tokens * TOKEN_MARGIN:
        return 0.0
    return min(_duration_s(headers.get("x-ratelimit-reset-tokens")) + 1, MAX_TOKEN_WAIT_S)


def _chat(messages: list[dict[str, Any]], client: llm.LlmClient, cfg: Config) -> Turn:
    if isinstance(client, llm.GroqClient):
        return _chat_groq(messages, client, cfg)
    return _chat_anthropic(messages, client, cfg)


def _chat_groq(messages: list[dict[str, Any]], client: llm.GroqClient, cfg: Config) -> Turn:
    response = client.post_chat(
        {
            "model": cfg.llm.groq.model,
            "max_completion_tokens": cfg.llm.groq.max_tokens,
            "temperature": 0,
            "reasoning_effort": cfg.llm.groq.reasoning_effort,
            "include_reasoning": False,
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        }
    )
    if response.status_code == 400:
        try:
            code = response.json().get("error", {}).get("code")
        except (ValueError, AttributeError):
            code = None
        if code == "tool_use_failed":
            raise InvalidToolCall("the tool call was not valid")
    if response.status_code != 200:
        raise llm._groq_error(response)
    try:
        choice = response.json()["choices"][0]
        message = choice["message"]
        calls = tuple(
            ToolCall(c["id"], c["function"]["name"], c["function"]["arguments"])
            for c in message.get("tool_calls") or []
        )
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise llm.LlmUnavailable("The API returned an unreadable response.") from e
    if choice.get("finish_reason") == "length":
        raise llm.LlmUnavailable("The model's answer was cut off.")
    used = (response.json().get("usage") or {}).get("total_tokens", 0)
    return Turn(message.get("content") or "", calls, token_wait_s(response.headers, used))


def _anthropic_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages[1:]:  # the first message is the system prompt
        if m["role"] == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": m["tool_call_id"],
                "content": m["content"],
            }
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        elif m["role"] == "assistant":
            blocks: list[dict[str, Any]] = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for c in m.get("tool_calls") or []:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": c["id"],
                        "name": c["function"]["name"],
                        "input": json.loads(c["function"]["arguments"]),
                    }
                )
            out.append({"role": "assistant", "content": blocks})
        else:
            out.append({"role": "user", "content": m["content"]})
    return out


def _chat_anthropic(messages: list[dict[str, Any]], client: anthropic.Anthropic, cfg: Config):
    tools = [
        {
            "name": t["function"]["name"],
            "description": t["function"]["description"],
            "input_schema": t["function"]["parameters"],
        }
        for t in TOOLS
    ]
    try:
        response = client.messages.create(
            model=cfg.llm.anthropic.model,
            max_tokens=cfg.llm.anthropic.max_tokens,
            system=messages[0]["content"],
            messages=_anthropic_messages(messages),
            tools=tools,
            output_config={"effort": cfg.llm.anthropic.effort},
        )
    except anthropic.APIError as e:
        raise _anthropic_error(e) from e
    if response.stop_reason == "max_tokens":
        raise llm.LlmUnavailable("The model's answer was cut off.")
    text = "".join(b.text for b in response.content if b.type == "text")
    calls = tuple(
        ToolCall(b.id, b.name, json.dumps(b.input))
        for b in response.content
        if b.type == "tool_use"
    )
    return Turn(text, calls)


# --------------------------------------------------------------------------- loop


def simulate_summary(
    scenario: Scenario, seeds: list[int] | None = None, config: Config | None = None
) -> Summary:
    """One run through the existing facade, in a temporary folder."""
    with tempfile.TemporaryDirectory(prefix="junctionlab-", ignore_cleanup_errors=True) as tmp:
        return run_scenario(scenario, Path(tmp) / "run", seeds=seeds, config=config).summary


def start_goal(session: AgentSession, config: Config | None = None) -> None:
    cfg = config or get_config()
    if session.goals >= cfg.agent.max_goals_per_session:
        raise llm.LimitReached(
            f"This session has used its {cfg.agent.max_goals_per_session} agent goals. Reload "
            "the page for a new session."
        )
    session.goals += 1


def run_agent(
    before: Scenario,
    goal: Goal,
    client: llm.LlmClient,
    simulate: Simulate | None = None,
    config: Config | None = None,
    limiter: AgentLimiter | None = None,
    on_step: Callable[[list[TraceStep]], None] | None = None,
    pause_s: float | None = None,
) -> AgentResult:
    """Run the loop for one goal. Any provider failure ends it with the best design so far.

    Raises only if the scenario cannot be designed or the "before" run fails, since there is
    then no design to return.
    """
    cfg = config or get_config()
    if not isinstance(before.control, FixedTimeSignal):
        raise ValueError("the agent only designs fixed-time signals")
    before.require_confirmed()
    limiter = limiter or AgentLimiter(cfg)
    pause = cfg.agent.pause_between_calls_s if pause_s is None else pause_s
    simulator = simulate or (lambda s: simulate_summary(s, config=cfg))
    bench = Workbench(goal, before, simulator, cfg, on_step or (lambda _trace: None))
    first = bench.run_design("D0", "Measure the current design.")
    bench.trace[0].change = f"Before: {design_label(before)}"

    system = SYSTEM_PROMPT.format(
        intergreen=cfg.control.intergreen_s,
        min_green=cfg.control.min_green_s,
        cycle_min=cfg.agent.cycle_min_s,
        cycle_max=cfg.agent.cycle_max_s,
        budget=bench.budget,
    )
    user = (
        f"Goal: {goal.text}\nRead as: {goal.label}.\nJunction: {before.junction_type}, demand "
        + ", ".join(f"{arm} {a.demand_vph:g} veh/h" for arm, a in before.approaches.items())
        + f".\nCurrent design, already simulated: {json.dumps(first, separators=(',', ':'))}\n"
        + "next_design_id: D1"
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    max_calls = cfg.agent.max_llm_calls_per_goal - 1  # one call reads the goal
    calls = 0
    nudged = False
    wait_s = 0.0
    stop: StopReason = "llm_budget"
    note = ""
    while calls < max_calls:
        if bench.budget_used:
            stop = "simulation_budget"
            break
        if calls and (delay := max(pause, wait_s)):
            time.sleep(delay)
        try:
            limiter.reserve_call()
            calls += 1
            turn = _chat(messages, client, cfg)
            wait_s = turn.wait_s
        except InvalidToolCall:
            messages.append(
                {"role": "user", "content": "Your last tool call was not valid. Try again."}
            )
            continue
        except llm.RateLimited as e:
            stop, note = "rate_limited", str(e)
            break
        except llm.LimitReached as e:
            stop, note = "limit", str(e)
            break
        except llm.LlmError as e:
            stop, note = "llm_error", str(e)
            break
        if not turn.calls:
            met = any(goal_status(goal, o) == "met" for o in bench.outcomes.values())
            if met or nudged:
                stop = "agent_finished"
                break
            nudged = True
            messages.append({"role": "assistant", "content": turn.text or "Done."})
            messages.append(
                {
                    "role": "user",
                    "content": f"The goal is not met and {bench.budget - bench.runs_used} runs "
                    "are left. Try another design, or reply without a tool call if none can help.",
                }
            )
            continue
        messages.append(
            {
                "role": "assistant",
                "content": turn.text,
                "tool_calls": [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": c.arguments},
                    }
                    for c in turn.calls
                ],
            }
        )
        for c in turn.calls:
            messages.append(
                {"role": "tool", "tool_call_id": c.id, "content": bench.call(c.name, c.arguments)}
            )
    else:
        if bench.budget_used:
            stop = "simulation_budget"

    best_id = bench.best_id()
    return AgentResult(
        goal=goal,
        before=before,
        best=bench.designs[best_id],
        best_id=best_id,
        best_outcome=bench.outcomes[best_id],
        before_summary=bench.summaries["D0"],
        best_summary=bench.summaries[best_id],
        status=goal_status(goal, bench.outcomes[best_id]),
        stop_reason=stop,
        runs_used=bench.runs_used,
        best_at_run=bench.run_order[best_id],
        llm_calls=calls,
        trace=bench.trace,
        message=note,
    )
