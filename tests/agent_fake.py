"""Test helpers for the agent: a deterministic simulator and scripted Groq responses."""

import json
from pathlib import Path

import httpx2

from junctionlab.config import get_config
from junctionlab.metrics import ApproachSummary, JunctionSummary, Stat, Summary, los_grade
from junctionlab.scenario import Scenario
from junctionlab.signal import phase_arms

SESSION = Path(__file__).parent / "data" / "agent" / "session_los_c.json"
SEED_SPREAD_S = 3.0


def fake_summary(s: Scenario) -> Summary:
    """Delay grows with red time and cycle. No SUMO, same answer every time."""
    cfg = get_config()
    c = s.control
    greens = {arm: g for arms, g in zip(phase_arms(s), c.green_s, strict=True) for arm in arms}
    arms = {}
    for arm in s.approaches:
        mean = 8 + 0.45 * c.cycle_s * (1 - greens[arm] / c.cycle_s) ** 2
        seeds = [mean - SEED_SPREAD_S, mean, mean + SEED_SPREAD_S]
        stat = Stat.of(seeds)
        volume = Stat.of([s.approaches[arm].demand_vph] * 3)
        arms[arm] = ApproachSummary(
            arm=arm,
            control_delay_s=stat,
            delay_s=stat,
            queue_m=Stat(40, 30, 50),
            queue_veh=Stat(6, 5, 7),
            throughput_vph=volume,
            demand_vph=volume,
            los=los_grade(mean, cfg.los.signalised),
            los_by_seed=[los_grade(d, cfg.los.signalised) for d in seeds],
            f_reasons=[],
            v_c=0.6,
            delay_lower_bound=False,
            spillback_seeds=[],
        )
    junction = sum(a.control_delay_s.mean for a in arms.values()) / len(arms)
    return Summary(
        approaches=arms,
        junction=JunctionSummary(
            Stat.of([junction]), Stat.of([junction]), Stat(2400, 2400, 2400), "C"
        ),
        seeds=[1, 2, 3],
        warnings=[],
    )


class SequenceTransport(httpx2.BaseTransport):
    """Answers each request with the next (status, body, headers). Keeps the requests."""

    def __init__(self, responses: list[tuple[int, dict, dict]]):
        self.responses = list(responses)
        self.requests: list[dict] = []

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.read()))
        if not self.responses:
            raise AssertionError("the agent made more calls than were scripted")
        status, body, headers = self.responses.pop(0)
        return httpx2.Response(status, json=body, headers=headers)


def recorded_session() -> list[tuple[int, dict, dict]]:
    calls = json.loads(SESSION.read_text(encoding="utf-8"))["calls"]
    return [(c["status"], c["body"], {}) for c in calls]


def tool_turn(*calls: tuple[str, dict]) -> tuple[int, dict, dict]:
    tool_calls = [
        {
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }
        for i, (name, args) in enumerate(calls)
    ]
    message = {"role": "assistant", "content": None, "tool_calls": tool_calls}
    body = {
        "choices": [{"message": message, "finish_reason": "tool_calls"}],
        "usage": {"total_tokens": 500},
    }
    return 200, body, {}


def text_turn(text: str = "Done.") -> tuple[int, dict, dict]:
    message = {"role": "assistant", "content": text}
    body = {"choices": [{"message": message, "finish_reason": "stop"}], "usage": {}}
    return 200, body, {}


def propose(cycle=None, plan=None, green=None, reason="try this") -> tuple[str, dict]:
    args = {"reason": reason, "cycle_s": cycle, "phase_plan": plan, "green_s": green}
    return "propose_change", args


def simulate(design_id: str) -> tuple[str, dict]:
    return "simulate", {"reason": "measure it", "design_id": design_id}
