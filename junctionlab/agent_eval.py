"""Evaluation of the signal-design agent (docs/method.md, section 6).

Compares four methods per junction and goal: the agent, Webster's optimum cycle, random search
and a grid search. Every method's best design is re-checked on seeds that no method used to
choose it.

Stages, each resumable because every result is saved as it is produced:
    python -m junctionlab agent-eval grid       no LLM calls
    python -m junctionlab agent-eval baselines  no LLM calls
    python -m junctionlab agent-eval agent      live LLM calls, within the daily limit
    python -m junctionlab agent-eval report     writes docs/agent_evaluation.md
"""

from __future__ import annotations

import ctypes
import json
import random
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from junctionlab import agent, llm
from junctionlab.agent import Goal, Outcome
from junctionlab.config import REPO_ROOT, Config, get_config
from junctionlab.examples import EXAMPLES_DIR, load_example
from junctionlab.scenario import ControlDraft, Scenario, resolve
from junctionlab.signal import critical_flow_ratios

REPORT_PATH = REPO_ROOT / "docs" / "agent_evaluation.md"
JUNCTION_LABELS = {
    "four_arm_signal": "Example 1: 4-arm signal",
    "t_junction_signal": "T-junction, signalised copy",
}
GOALS = {
    "G1": Goal("All approaches LOS C or better, shortest cycle possible.", "shortest_cycle", "C"),
    "G2": Goal("Lowest average delay over the junction.", "min_junction_delay"),
    "G3": Goal("Lowest delay on the worst approach.", "min_worst_delay"),
}
Simulator = Callable[[Scenario, list[int]], Outcome]


def junctions(config: Config | None = None) -> dict[str, Scenario]:
    """Example 1, and the T-junction example with a signal instead of its priority rule.

    The copy has the same demand and geometry and exists only for this evaluation.
    """
    cfg = config or get_config()
    four = load_example(EXAMPLES_DIR / "four_arm_signal.json").scenario(cfg)
    t = load_example(EXAMPLES_DIR / "t_junction_priority.json").draft
    control = ControlDraft(kind="signal", cycle_s=cfg.control.cycle_default_s)
    copy = t.model_copy(update={"name": "T-junction signal (evaluation)", "control": control})
    return {"four_arm_signal": four, "t_junction_signal": resolve(copy, cfg)}


# --------------------------------------------------------------------------- designs


def _with_timing(
    s: Scenario, plan: str, cycle: int, greens: list[int] | None, cfg: Config
) -> Scenario | None:
    """A valid design, or None if these values break a validation rule."""
    try:
        return agent.apply_change(
            s,
            cycle_s=cycle,
            phase_plan=plan,
            green_s=None if greens is None else [float(g) for g in greens],
            config=cfg,
        )
    except ValueError:  # pydantic's ValidationError is a ValueError
        return None


def _cycles(cfg: Config) -> range:
    a = cfg.agent
    return range(a.cycle_min_s, a.cycle_max_s + 1, a.evaluation.cycle_step_s)


def _plans(s: Scenario) -> list[str]:
    return ["two_phase"] if s.junction_type == "t_junction" else ["two_phase", "four_phase_split"]


def grid_designs(s: Scenario, config: Config | None = None) -> list[Scenario]:
    """Two-phase: every cycle with shares around Webster's. Four-phase: Webster splits."""
    cfg = config or get_config()
    out: dict[tuple, Scenario] = {}
    for cycle in _cycles(cfg):
        webster = _with_timing(s, "two_phase", cycle, None, cfg)
        if webster is None:
            continue
        total = sum(webster.control.green_s)
        centre = 5 * round(100 * webster.control.green_s[0] / total / 5)
        for offset in cfg.agent.evaluation.split_offsets_pct:
            first = round(total * (centre + offset) / 100)
            design = _with_timing(s, "two_phase", cycle, [first, int(total) - first], cfg)
            if design is not None:
                out.setdefault(agent.design_key(design), design)
        if "four_phase_split" in _plans(s):
            design = _with_timing(s, "four_phase_split", cycle, None, cfg)
            if design is not None:
                out.setdefault(agent.design_key(design), design)
    return list(out.values())


def webster_cycle(s: Scenario, config: Config | None = None) -> float:
    """Co = (1.5 L + 5) / (1 - Y), with the measured saturation flow and lost time.

    L is the lost time per cycle: per phase, the measured lost time plus the all-red.
    """
    cfg = config or get_config()
    y = critical_flow_ratios(s, cfg)
    big_y = sum(y)
    if big_y >= 1:
        return float("inf")
    lost = len(y) * (cfg.control.lost_time_s + cfg.control.all_red_s)
    return (1.5 * lost + 5) / (1 - big_y)


def webster_design(s: Scenario, config: Config | None = None) -> Scenario:
    """Webster's cycle in whole seconds, held to the shared search space, with Webster splits."""
    cfg = config or get_config()
    co = webster_cycle(s, cfg)
    cycle = cfg.agent.cycle_max_s if co == float("inf") else round(co)
    cycle = min(max(cycle, cfg.agent.cycle_min_s), cfg.agent.cycle_max_s)
    design = _with_timing(s, s.control.phase_plan, cycle, None, cfg)
    if design is None:
        raise ValueError(f"Webster's cycle of {cycle} s is not valid for this junction")
    return design


def random_designs(s: Scenario, rng: random.Random, config: Config | None = None) -> list[Scenario]:
    """Distinct random valid designs in the shared search space, in whole seconds."""
    cfg = config or get_config()
    wanted = cfg.agent.evaluation.random_designs
    plans = _plans(s)
    out: dict[tuple, Scenario] = {}
    for _ in range(wanted * 200):
        if len(out) == wanted:
            break
        plan = rng.choice(plans)
        cycle = rng.randint(cfg.agent.cycle_min_s, cfg.agent.cycle_max_s)
        phases = 4 if plan == "four_phase_split" else 2
        spare = cycle - phases * (cfg.control.intergreen_s + cfg.control.min_green_s)
        if spare < 0:
            continue
        cuts = sorted(rng.randint(0, int(spare)) for _ in range(phases - 1))
        parts = [b - a for a, b in zip([0, *cuts], [*cuts, int(spare)], strict=True)]
        greens = [int(cfg.control.min_green_s) + p for p in parts]
        design = _with_timing(s, plan, cycle, greens, cfg)
        if design is not None:
            out.setdefault(agent.design_key(design), design)
    return list(out.values())


# --------------------------------------------------------------------------- storage


def _key(junction: str, design: Scenario, seeds: list[int]) -> str:
    plan, cycle, greens = agent.design_key(design)
    timing = "/".join(f"{g:g}" for g in greens)
    return f"{junction}|{plan}|{cycle:g}|{timing}|{','.join(map(str, seeds))}"


class Store:
    """Simulated outcomes and agent runs, saved after every result so a stage can resume."""

    def __init__(self, directory: Path):
        self.dir = directory if directory.is_absolute() else REPO_ROOT / directory
        self.dir.mkdir(parents=True, exist_ok=True)
        self.outcomes = self._read("outcomes.json")
        self.agent_runs: dict[str, Any] = self._read("agent_runs.json")
        # Runs a 429 ended. Kept apart from the results, which only hold complete runs.
        self.rate_limited: dict[str, Any] = self._read("rate_limited.json")

    def _read(self, name: str) -> dict[str, Any]:
        path = self.dir / name
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    def _write(self, name: str, data: dict[str, Any]) -> None:
        tmp = self.dir / f"{name}.tmp"
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(self.dir / name)

    def outcome(
        self, junction: str, design: Scenario, seeds: list[int], simulator: Simulator
    ) -> Outcome:
        key = _key(junction, design, seeds)
        if key not in self.outcomes:
            self.outcomes[key] = simulator(design, seeds).to_dict()
            self._write("outcomes.json", self.outcomes)
        return Outcome.from_dict(self.outcomes[key])

    def save_agent_run(self, key: str, run: dict[str, Any]) -> None:
        self.agent_runs[key] = run
        self._write("agent_runs.json", self.agent_runs)

    def save_rate_limited(self, key: str, run: dict[str, Any]) -> None:
        """Record a run that a 429 ended, under a new id so earlier ones are kept."""
        self.rate_limited[f"{key}#{len(self.rate_limited) + 1}"] = run
        self._write("rate_limited.json", self.rate_limited)

    def move_rate_limited_runs(self) -> list[str]:
        """Move saved agent runs that a 429 ended into the rate-limited record, so they run
        again. Returns their keys."""
        moved = [k for k, r in self.agent_runs.items() if r.get("stop_reason") == "rate_limited"]
        for key in moved:
            self.save_rate_limited(key, self.agent_runs.pop(key))
        if moved:
            self._write("agent_runs.json", self.agent_runs)
        return moved


class MemoryLow(RuntimeError):
    """Free memory is below the configured minimum. Saved results are kept."""


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def available_memory_gb() -> float | None:
    """Available physical memory, or None if this system does not report it."""
    if sys.platform == "win32":
        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.ullAvailPhys / 1024**3
        return None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024**2
    except (OSError, ValueError, IndexError):
        return None
    return None


def check_memory(cfg: Config, available: Callable[[], float | None] = available_memory_gb) -> None:
    free = available()
    minimum = cfg.agent.evaluation.min_free_memory_gb
    if free is not None and free < minimum:
        raise MemoryLow(
            f"Stopped cleanly: {free:.1f} GB of memory is free, below the {minimum:g} GB minimum. "
            "All finished results are saved."
        )


def real_simulator(cfg: Config) -> Simulator:
    def simulate(s: Scenario, seeds: list[int]) -> Outcome:
        check_memory(cfg)
        workers = cfg.agent.evaluation.max_parallel_runs
        return Outcome.from_summary(agent.simulate_summary(s, seeds, cfg, workers))

    return simulate


def _train(cfg: Config) -> list[int]:
    return list(cfg.simulation.seeds)


def _holdout(cfg: Config) -> list[int]:
    return list(cfg.agent.evaluation.holdout_seeds)


def _best(
    goal: Goal,
    junction: str,
    designs: list[Scenario],
    store: Store,
    simulator: Simulator,
    cfg: Config,
) -> Scenario:
    seeds = _train(cfg)
    return min(
        designs,
        key=lambda d: agent.rank(goal, d, store.outcome(junction, d, seeds, simulator), cfg),
    )


# --------------------------------------------------------------------------- stages


def run_grid(
    config: Config | None = None,
    store: Store | None = None,
    simulator: Simulator | None = None,
    progress: Callable[[str], None] = print,
) -> int:
    cfg = config or get_config()
    store = store or Store(cfg.agent.evaluation.results_dir)
    simulator = simulator or real_simulator(cfg)
    total = 0
    for junction, s in junctions(cfg).items():
        designs = grid_designs(s, cfg)
        for i, design in enumerate(designs, 1):
            store.outcome(junction, design, _train(cfg), simulator)
            if i % 10 == 0 or i == len(designs):
                progress(f"{junction}: {i}/{len(designs)}")
        total += len(designs)
    return total


def _random_sets(junction: str, s: Scenario, cfg: Config) -> list[list[Scenario]]:
    """Fixed random seeds per junction and repeat, so the baseline can be reproduced."""
    return [
        random_designs(s, random.Random(f"{junction}-{repeat}"), cfg)
        for repeat in range(cfg.agent.evaluation.repeats)
    ]


def run_baselines(
    config: Config | None = None,
    store: Store | None = None,
    simulator: Simulator | None = None,
    progress: Callable[[str], None] = print,
) -> None:
    cfg = config or get_config()
    store = store or Store(cfg.agent.evaluation.results_dir)
    simulator = simulator or real_simulator(cfg)
    for junction, s in junctions(cfg).items():
        store.outcome(junction, webster_design(s, cfg), _train(cfg), simulator)
        for repeat, designs in enumerate(_random_sets(junction, s, cfg), 1):
            for design in designs:
                store.outcome(junction, design, _train(cfg), simulator)
            progress(f"{junction}: random repeat {repeat} done")


def _agent_key(junction: str, goal: str, repeat: int) -> str:
    return f"{junction}|{goal}|{repeat}"


def _about(seconds: float) -> str:
    if seconds < 90:
        return f"about {max(1, round(seconds))} s"
    if seconds < 5400:
        return f"about {round(seconds / 60)} min"
    return f"about {round(seconds / 3600)} h"


def waiting_message(wait_s: float, wait: int, ev: Any) -> str:
    """A per-minute 429 that the evaluation waits out. Not the app's message: nothing here
    can be edited in a form."""
    return (
        f"the free Groq tier's per-minute rate limit is reached; waiting {wait_s:.0f} s "
        f"(wait {wait} of {ev.rate_limit_waits})."
    )


def stop_message(e: llm.RateLimited, waits: int) -> str:
    """Why a 429 ended the stage, and when to run it again."""
    when = (
        f"Run this stage again in {_about(e.wait_s)}."
        if e.wait_s is not None
        else "Run this stage again later."
    )
    if e.daily:
        return f"The free Groq tier's daily quota is used up. {when}"
    return (
        f"The free Groq tier's per-minute rate limit was still reached after {waits} "
        f"wait(s). {when}"
    )


def run_agent_stage(
    client: llm.LlmClient | None = None,
    config: Config | None = None,
    store: Store | None = None,
    simulator: Simulator | None = None,
    limiter: agent.AgentLimiter | None = None,
    progress: Callable[[str], None] = print,
    pause_s: float | None = None,
    summary_simulator: agent.Simulate | None = None,
    max_runs: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Run missing agent cases one at a time, at most `max_runs` of them, with a pause between
    runs. Stops before a case that the daily limit might cut short.

    A 429 waits for the reported reset and the run continues (config `rate_limit_waits`). A run
    that a 429 still ends (a per-day quota, or too many waits) is not saved as a result: it goes
    to the rate-limited record, the stage stops, and the run starts again on the next call."""
    cfg = config or get_config()
    store = store or Store(cfg.agent.evaluation.results_dir)
    simulator = simulator or real_simulator(cfg)
    limiter = limiter or agent.AgentLimiter(cfg)
    client = client or llm.make_client()
    train = _train(cfg)
    workers = cfg.agent.evaluation.max_parallel_runs
    simulate = summary_simulator or (lambda d: agent.simulate_summary(d, train, cfg, workers))
    for key in store.move_rate_limited_runs():
        progress(f"{key}: ended by a 429 earlier; moved to the rate-limited record, runs again.")
    ev = cfg.agent.evaluation
    between = ev.pause_between_runs_s if pause_s is None else pause_s
    done = 0
    for junction, s in junctions(cfg).items():
        for name, goal in GOALS.items():
            for repeat in range(1, cfg.agent.evaluation.repeats + 1):
                key = _agent_key(junction, name, repeat)
                if key in store.agent_runs:
                    continue
                if summary_simulator is None:
                    check_memory(cfg)
                if limiter.remaining_today() < cfg.agent.max_llm_calls_per_goal - 1:
                    progress("Daily agent limit is near. Run this stage again tomorrow.")
                    return done
                if done:
                    sleep(between)
                progress(f"{key}: started")
                result = agent.run_agent(
                    s,
                    goal,
                    client,
                    simulate=simulate,
                    config=cfg,
                    limiter=limiter,
                    pause_s=pause_s,
                    rate_limit_waits=ev.rate_limit_waits,
                    default_wait_s=ev.rate_limit_default_wait_s,
                    on_wait=lambda s_, n, k=key: progress(f"{k}: {waiting_message(s_, n, ev)}"),
                    sleep=sleep,
                )
                record = {
                    "plan": result.best.control.phase_plan,
                    "cycle_s": result.best.control.cycle_s,
                    "green_s": list(result.best.control.green_s),
                    "runs_used": result.runs_used,
                    "best_at_run": result.best_at_run,
                    "llm_calls": result.llm_calls,
                    "stop_reason": result.stop_reason,
                    "rate_limit_waits": result.rate_limit_waits,
                }
                if result.stop_reason == "rate_limited":
                    message = (
                        stop_message(result.rate_limit, result.rate_limit_waits)
                        if result.rate_limit
                        else result.message
                    )
                    record["message"] = message
                    record["date"] = datetime.now(UTC).date().isoformat()
                    store.save_rate_limited(key, record)
                    progress(f"{key}: ended by the rate limit, not saved as a result.")
                    progress(f"Stopped. {message}")
                    return done
                store.outcomes[_key(junction, result.best, train)] = result.best_outcome.to_dict()
                store._write("outcomes.json", store.outcomes)
                store.save_agent_run(key, record)
                done += 1
                progress(
                    f"{key}: {result.status}, {result.runs_used} runs, {result.stop_reason}, "
                    f"{result.rate_limit_waits} rate-limit waits"
                )
                if max_runs is not None and done >= max_runs:
                    return done
    return done


# --------------------------------------------------------------------------- report


@dataclass
class MethodRow:
    method: str
    result: str
    recheck: str
    goal: str
    runs: str


def _value_text(goal: Goal, value: float) -> str:
    return f"{value:.0f} s cycle" if goal.objective == "shortest_cycle" else f"{value:.1f} s/veh"


def _spread(goal: Goal, values: list[float]) -> str:
    mid = statistics.median(values)
    if len(values) == 1 or min(values) == max(values):
        return _value_text(goal, mid)
    digits = 0 if goal.objective == "shortest_cycle" else 1
    return f"{_value_text(goal, mid)} ({min(values):.{digits}f}–{max(values):.{digits}f})"


def _status_text(statuses: list[str]) -> str:
    if len(statuses) == 1:
        return agent.STATUS_LABELS[statuses[0]]
    parts = [
        f"{statuses.count(k)}/{len(statuses)} {label.lower()}"
        for k, label in agent.STATUS_LABELS.items()
        if statuses.count(k)
    ]
    return ", ".join(parts)


def _row(
    method: str,
    goal: Goal,
    junction: str,
    designs: list[Scenario],
    runs: str,
    store: Store,
    simulator: Simulator,
    cfg: Config,
) -> MethodRow:
    train, holdout, statuses = [], [], []
    for design in designs:
        first = store.outcome(junction, design, _train(cfg), simulator)
        check = store.outcome(junction, design, _holdout(cfg), simulator)
        train.append(agent.objective_value(goal, design, first))
        holdout.append(agent.objective_value(goal, design, check))
        statuses.append(agent.goal_status(goal, check))
    return MethodRow(
        method, _spread(goal, train), _spread(goal, holdout), _status_text(statuses), runs
    )


def missing_agent_runs(store: Store, cfg: Config) -> list[str]:
    return [
        _agent_key(junction, name, repeat)
        for junction in JUNCTION_LABELS
        for name in GOALS
        for repeat in range(1, cfg.agent.evaluation.repeats + 1)
        if _agent_key(junction, name, repeat) not in store.agent_runs
    ]


def build_report(
    config: Config | None = None,
    store: Store | None = None,
    simulator: Simulator | None = None,
    preliminary: bool = False,
) -> str:
    """The evaluation as Markdown. Raises ValueError if any agent case has not been run, unless
    `preliminary`: the report is then marked as such and uses the agent runs saved so far."""
    cfg = config or get_config()
    store = store or Store(cfg.agent.evaluation.results_dir)
    simulator = simulator or real_simulator(cfg)
    missing = missing_agent_runs(store, cfg)
    if missing and not preliminary:
        raise ValueError(f"{len(missing)} agent run(s) missing, for example {missing[0]}")
    n = cfg.agent.evaluation.repeats
    total = len(JUNCTION_LABELS) * len(GOALS) * n
    lines = ["# Signal-design agent: evaluation", ""]
    if missing:
        lines += [
            f"> **Preliminary, {total - len(missing)} of {total} agent runs.** The other agent "
            "runs wait for the free tier's daily quota. Webster, random search and the grid "
            "are complete.",
            "",
        ]
    lines += [
        f"Each method chose its best design on seeds {_train(cfg)}. **Unused seeds** re-runs "
        f"that design on seeds {_holdout(cfg)}, and **Goal** is judged there: met means every "
        "approach in every seed, borderline means on the mean only. Repeated methods show the "
        f"median and range over {n} repeats. Runs are simulation runs used to choose a design.",
        "",
    ]
    for junction, s in junctions(cfg).items():
        lines += [f"## {JUNCTION_LABELS[junction]}", ""]
        grid = grid_designs(s, cfg)
        for name, goal in GOALS.items():
            keys = [_agent_key(junction, name, r) for r in range(1, n + 1)]
            agent_runs = [store.agent_runs[k] for k in keys if k in store.agent_runs]
            agent_designs = [
                agent.apply_change(s, r["cycle_s"], r["plan"], r["green_s"], cfg)
                for r in agent_runs
            ]
            if agent_runs:
                used = statistics.fmean(r["runs_used"] for r in agent_runs)
                found = statistics.fmean(r["best_at_run"] for r in agent_runs)
                agent_text = f"{used:.1f} (best at run {found:.1f})"
                if len(agent_runs) < n:
                    agent_text += f"; {len(agent_runs)} of {n} repeats"
            randoms = [
                _best(goal, junction, designs, store, simulator, cfg)
                for designs in _random_sets(junction, s, cfg)
            ]
            rows = [
                _row("Agent", goal, junction, agent_designs, agent_text, store, simulator, cfg)
                if agent_runs
                else MethodRow("Agent", "not run yet", "–", "–", f"0 of {n} repeats"),
                _row(
                    "Webster",
                    goal,
                    junction,
                    [webster_design(s, cfg)],
                    "1",
                    store,
                    simulator,
                    cfg,
                ),
                _row(
                    "Random search",
                    goal,
                    junction,
                    randoms,
                    str(cfg.agent.evaluation.random_designs),
                    store,
                    simulator,
                    cfg,
                ),
                _row(
                    "Grid optimum",
                    goal,
                    junction,
                    [_best(goal, junction, grid, store, simulator, cfg)],
                    str(len(grid)),
                    store,
                    simulator,
                    cfg,
                ),
            ]
            lines += [
                f"### {name}: {goal.text}",
                "",
                "| Method | Best result | Unused seeds | Goal | Runs |",
                "| --- | --- | --- | --- | --- |",
                *[f"| {r.method} | {r.result} | {r.recheck} | {r.goal} | {r.runs} |" for r in rows],
                "",
            ]
    lines += _free_tier_note(store)
    return "\n".join(lines)


def _free_tier_note(store: Store) -> list[str]:
    """How often the free tier's rate limit got in the way. Not an agent result."""
    ended = list(store.rate_limited.values())
    daily = sum("daily quota" in r.get("message", "") for r in ended)
    unknown = sum("message" not in r for r in ended)
    waits = sum(r.get("rate_limit_waits", 0) for r in store.agent_runs.values())
    runs_waited = sum(bool(r.get("rate_limit_waits")) for r in store.agent_runs.values())
    kinds = f"{daily} of them by the daily token quota"
    if unknown:
        kinds += (
            f"; {unknown} were recorded before the kind of limit was saved, so for those it is "
            "not known"
        )
    return [
        "## Note on the free tier",
        "",
        f"A 429 from Groq's free tier ended {len(ended)} agent run(s) early ({kinds}). Those "
        "runs were set aside and run again from the start; they are not in the tables above. "
        + (
            f"Of the runs that are, {runs_waited} waited for a per-minute limit to reset "
            f"({waits} wait(s) in total) and then continued the same run. "
            if waits
            else "None of the runs in the tables needed a wait for a per-minute limit. "
        )
        + "A wait, when one happens, does not change what the model sees or how many calls it "
        "gets. The daily quota is not waited out: the stage stops and resumes on a later day.",
        "",
    ]


SUMMARY_START = "<!-- summary -->"
SUMMARY_END = "<!-- /summary -->"


def write_report(config: Config | None = None, preliminary: bool = False) -> Path:
    """Write the report, keeping the hand-written summary of an earlier report, if any."""
    report = build_report(config, preliminary=preliminary)
    old = REPORT_PATH.read_text(encoding="utf-8") if REPORT_PATH.is_file() else ""
    if SUMMARY_START in old and SUMMARY_END in old:
        summary = old[old.index(SUMMARY_START) : old.index(SUMMARY_END) + len(SUMMARY_END)]
        title, _, rest = report.partition("\n\n")
        report = f"{title}\n\n{summary}\n\n{rest}"
    REPORT_PATH.write_text(report, encoding="utf-8")
    return REPORT_PATH
