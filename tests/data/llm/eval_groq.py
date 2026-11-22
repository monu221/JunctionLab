"""Run 10 varied descriptions through interpret() and score them. Makes live API calls.

Expected values are written here by hand from each description. Run from the repo root:
    python tests/data/llm/eval_groq.py OUTPUT.json
"""

import json
import sys
import tempfile
import time
from pathlib import Path

from junctionlab import llm
from junctionlab.config import get_config

PAUSE_S = 15


def shares(left, through, right):
    return {"left": left, "through": through, "right": right}


CASES = [
    {
        "kind": "clear",
        "text": "Crossroads with traffic signals, 120 s cycle. 900 veh/h from the north, 750 from "
        "the south, 400 from the east, 350 from the west. Three lanes on the north and south "
        "approaches, one lane on east and west. 8% heavy vehicles on all approaches. 60 km/h.",
        "expect": {
            "junction_type": "four_arm",
            "control": "signal",
            "cycle_s": 120,
            "speed_limit_kmh": 60,
            "demand": {"north": 900, "south": 750, "east": 400, "west": 350},
            "lane_count": {"north": 3, "south": 3, "east": 1, "west": 1},
            "heavy": {"north": 8, "south": 8, "east": 8, "west": 8},
        },
    },
    {
        "kind": "clear",
        "text": "All-way stop at a four-way intersection: 150 vehicles per hour from each of "
        "north and south, 120 from the east and 100 from the west.",
        "expect": {
            "junction_type": "four_arm",
            "control": "all_way_stop",
            "demand": {"north": 150, "south": 150, "east": 120, "west": 100},
        },
    },
    {
        "kind": "clear",
        "text": "T-junction with traffic lights, the side road joins from the east. 500 veh/h "
        "from the north, 450 from the south, 200 from the east. 60 second cycle.",
        "expect": {
            "junction_type": "t_junction",
            "control": "signal",
            "cycle_s": 60,
            "demand": {"north": 500, "south": 450, "east": 200},
        },
    },
    {
        "kind": "vague",
        "text": "A quiet crossroads in a residential area with a give-way sign.",
        "expect": {"rejected": True},
    },
    {
        "kind": "vague",
        "text": "Traffic lights at a crossroads, about 600 cars an hour on each road.",
        # "Each road" could mean 600 per approach or 300: rejected, or demand interpreted.
        "expect": {"rejected_or_flagged": True},
    },
    {
        "kind": "per-movement",
        "text": "Signalised crossroads, 80 s cycle. From the north: 100 left, 400 straight, "
        "100 right. From the south: 50 left, 300 straight, 50 right. From the east: 60 left, "
        "200 straight, 40 right. From the west: 40 left, 240 straight, 120 right.",
        "expect": {
            "junction_type": "four_arm",
            "control": "signal",
            "cycle_s": 80,
            "interpreted_demand": True,  # the counts have no time period
            "demand": {"north": 600, "south": 400, "east": 300, "west": 400},
            "shares": {
                "north": shares(16.67, 66.67, 16.67),
                "south": shares(12.5, 75, 12.5),
                "east": shares(20, 66.67, 13.33),
                "west": shares(10, 60, 30),
            },
        },
    },
    {
        "kind": "per-movement",
        "text": "T-junction, the main road runs north-south and has priority, the side road "
        "joins from the west. 15-minute counts: from the north 100 straight and 25 turning "
        "right; from the south 80 straight and 20 turning left; from the west 30 left and "
        "10 right.",
        "expect": {
            "junction_type": "t_junction",
            "control": "priority",
            "major_road": "north-south",
            "demand": {"north": 500, "south": 400, "west": 160},
            "shares": {
                "north": shares(0, 80, 20),
                "south": shares(20, 80, 0),
                "west": shares(75, 0, 25),
            },
        },
    },
    {
        "kind": "bus lane",
        "text": "Signalised crossroads, 90 s cycle, two lanes per approach. 600 veh/h from the "
        "north, 500 from the south, 450 from the east, 550 from the west. The east approach "
        "also has a bus lane with 20 buses per hour, a quarter of which turn right.",
        "expect": {
            "junction_type": "four_arm",
            "control": "signal",
            "cycle_s": 90,
            "demand": {"north": 600, "south": 500, "east": 450, "west": 550},
            "general_lanes": {"north": 2, "south": 2, "east": 2, "west": 2},
            "bus": {"east": (20, 25)},
        },
    },
    {
        "kind": "impossible",
        "text": "A five-arm roundabout with 500 veh/h on every arm.",
        "expect": {"rejected": True},
    },
    {
        "kind": "impossible",
        "text": "T-junction with four approaches: 300 veh/h each from the north, east, south "
        "and west, with a 90 s signal.",
        "expect": {"rejected": True},
    },
]


def check(s, e) -> list[str]:
    bad = []

    def same(label, got, want, tol=0.01):
        ok = abs(got - want) <= tol if isinstance(want, int | float) else got == want
        if not ok:
            bad.append(f"{label}: got {got}, expected {want}")

    if "junction_type" in e:
        same("junction type", s.junction_type, e["junction_type"])
    if "control" in e:
        same("control", s.control.kind, e["control"])
    if "cycle_s" in e:
        same("cycle", s.control.cycle_s, e["cycle_s"])
    if "major_road" in e:
        same("major road", s.control.major_road, e["major_road"])
    if "speed_limit_kmh" in e:
        same("speed", s.speed_limit_kmh, e["speed_limit_kmh"])
    if "demand" in e:
        same("arms", set(s.approaches), set(e["demand"]))
        for arm, v in e["demand"].items():
            if arm in s.approaches:
                same(f"{arm} demand", s.approaches[arm].demand_vph, v)
    for arm, v in e.get("lane_count", {}).items():
        same(f"{arm} lanes", len(s.approaches[arm].lanes), v)
    for arm, v in e.get("general_lanes", {}).items():
        same(f"{arm} general lanes", len(s.approaches[arm].lanes), v)
    for arm, v in e.get("heavy", {}).items():
        same(f"{arm} heavy %", s.approaches[arm].heavy_vehicle_share_pct, v)
    for arm, want in e.get("shares", {}).items():
        got = s.approaches[arm].turning_shares_pct
        for m, v in want.items():
            same(f"{arm} {m} %", got.get(m, 0), v, tol=0.5)
    for arm, (bph, right) in e.get("bus", {}).items():
        a = s.approaches[arm]
        same(f"{arm} bus lane", a.bus_lane, True)
        same(f"{arm} buses/h", a.bus_demand_bph, bph)
        same(f"{arm} bus right %", a.bus_right_share_pct, right)
    return bad


def run_case(case, client, cfg, limiter) -> dict:
    e = case["expect"]
    out = {"kind": case["kind"], "text": case["text"]}
    try:
        r = llm.interpret(
            case["text"], llm.LlmSession(), client=client, config=cfg, limiter=limiter
        )
    except llm.InterpretationError as err:
        out |= {"outcome": "rejected", "reasons": err.reasons}
        out["correct"] = bool(e.get("rejected") or e.get("rejected_or_flagged"))
        out["problems"] = [] if out["correct"] else ["valid description was rejected"]
        return out
    s = r.scenario
    out |= {
        "outcome": "parsed",
        "assumed": list(s.assumed),
        "notes": r.notes,
        "interpreted": s.interpreted,
        "needs_confirmation": list(s.needs_confirmation),
        "scenario": json.loads(s.model_dump_json()),
    }
    if e.get("rejected"):
        out["problems"] = ["should have been rejected"]
    elif e.get("rejected_or_flagged"):
        flagged = len(s.needs_confirmation) == len(s.approaches)
        out["problems"] = [] if flagged else ["ambiguous demand not marked interpreted"]
    else:
        out["problems"] = check(s, e)
        if e.get("interpreted_demand") and len(s.needs_confirmation) != len(s.approaches):
            out["problems"].append("ambiguous demand not marked interpreted")
        if not e.get("interpreted_demand") and s.needs_confirmation:
            out["problems"].append("clear demand wrongly marked interpreted")
    out["correct"] = not out["problems"]
    return out


def main(path: str) -> None:
    cfg = get_config()
    client = llm.make_client(max_retries=1, provider="groq")
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        limiter = llm.CallLimiter(
            cfg.model_copy(
                update={"llm": cfg.llm.model_copy(update={"calls_per_day": 1000})},
            ),
            counter_file=Path(tmp) / "calls.json",
        )
        for i, case in enumerate(CASES):
            if i:
                time.sleep(PAUSE_S)
            for attempt in range(4):
                try:
                    results.append(run_case(case, client, cfg, limiter))
                    break
                except llm.LlmUnavailable as err:
                    if "rate limit" not in str(err) or attempt == 3:
                        results.append(
                            {**case, "outcome": "api error", "correct": False, "error": str(err)}
                        )
                        break
                    time.sleep(45)
            r = results[-1]
            print(i + 1, r["kind"], r["outcome"], "OK" if r["correct"] else "WRONG")
    Path(path).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
