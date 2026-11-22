"""Write synthetic recordings in the Messages API response format.

Used until live recordings exist: `python -m junctionlab interpret --record FILE "text"`
replaces a file with a real response ("source": "live"). Run from the repo root:
    python tests/data/llm/make_synthetic.py
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
MODEL = "claude-haiku-5-5"


def approach(demand=None, **kw):
    a = {
        "demand_vph": demand,
        "lane_count": None,
        "lanes": None,
        "turning_shares_pct": None,
        "heavy_vehicle_share_pct": None,
        "bus_lane": None,
        "bus_demand_bph": None,
        "bus_right_share_pct": None,
    }
    a.update(kw)
    return a


def shares(left, through, right):
    return {"left": left, "through": through, "right": right}


def control(kind, **kw):
    c = {"kind": kind, "cycle_s": None, "phase_plan": None, "green_s": None, "major_road": None}
    c.update(kw)
    return c


def interp(approaches, ctrl, *, jt, name=None, understood=True, unclear=(), notes=(), **kw):
    arms = {arm: None for arm in ("north", "east", "south", "west")}
    arms.update(approaches)
    out = {
        "understood": understood,
        "unclear": list(unclear),
        "notes": list(notes),
        "interpreted": [],
        "name": name,
        "junction_type": jt,
        "speed_limit_kmh": None,
        "approach_length_m": None,
        "approaches": arms,
        "control": ctrl,
    }
    out.update(kw)
    return out


def message(content, stop_reason="end_turn", n=1, stop_details=None):
    return {
        "id": f"msg_synthetic_{n:02d}",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "stop_details": stop_details,
        "usage": {
            "input_tokens": 1850,
            "output_tokens": 420,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


def ok(text, data, n):
    content = [
        {"type": "thinking", "thinking": "", "signature": "synthetic"},
        {"type": "text", "text": json.dumps(data)},
    ]
    return {"status": 200, "body": message(content, n=n)}


def error(status, kind, msg):
    return {"status": status, "body": {"type": "error", "error": {"type": kind, "message": msg}}}


def write(name, text, response):
    rec = {"source": "synthetic", "text": text, "response": response}
    (HERE / f"{name}.json").write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")


examples = {
    e.stem: json.loads(e.read_text(encoding="utf-8"))["description"]
    for e in (HERE.parents[2] / "examples").glob("*.json")
}

write(
    "four_arm_signal",
    examples["four_arm_signal"],
    ok(
        examples["four_arm_signal"],
        interp(
            {
                "north": approach(800, lane_count=2),
                "south": approach(600, lane_count=2),
                "east": approach(500, lane_count=2),
                "west": approach(500, lane_count=2),
            },
            control("signal", cycle_s=90),
            jt="four_arm",
            name="Four-arm fixed-time signal",
        ),
        1,
    ),
)
write(
    "t_junction_priority",
    examples["t_junction_priority"],
    ok(
        examples["t_junction_priority"],
        interp(
            {
                "east": approach(350, lane_count=1),
                "west": approach(350, lane_count=1),
                "south": approach(140, lane_count=1),
            },
            control("priority", major_road="east-west"),
            jt="t_junction",
            name="T-junction with priority rule",
        ),
        2,
    ),
)
write(
    "four_arm_bus_lane",
    examples["four_arm_bus_lane"],
    ok(
        examples["four_arm_bus_lane"],
        interp(
            {
                "north": approach(450, lane_count=2),
                "east": approach(600, lane_count=2),
                "south": approach(400, lane_count=2),
                "west": approach(700, lane_count=2, bus_lane=True, bus_demand_bph=12),
            },
            control("signal", cycle_s=90),
            jt="four_arm",
            name="Four-arm signal with west bus lane",
        ),
        3,
    ),
)

COUNTS = (
    "Signalised crossroads, 100 second cycle, one approach at a time. Peak 15-minute counts: "
    "from the north 25 left, 150 straight, 25 right; from the south 20 left, 120 straight, "
    "20 right; from the east 100 vehicles, a third of them turning right and the rest going "
    "straight; from the west 90 vehicles. Single lane on each arm, 50 km/h. There is a "
    "pedestrian crossing on the west arm."
)
write(
    "movement_counts",
    COUNTS,
    ok(
        COUNTS,
        interp(
            {
                "north": approach(800, lane_count=1, turning_shares_pct=shares(12.5, 75, 12.5)),
                "south": approach(640, lane_count=1, turning_shares_pct=shares(12.5, 75, 12.5)),
                "east": approach(400, lane_count=1, turning_shares_pct=shares(0, None, 33.3)),
                "west": approach(360, lane_count=1),
            },
            control("signal", cycle_s=100, phase_plan="four_phase_split"),
            jt="four_arm",
            name="Signalised crossroads (from peak counts)",
            speed_limit_kmh=50,
            notes=[
                "15-minute counts multiplied by 4 to get vehicles per hour.",
                "The pedestrian crossing on the west arm is not modelled.",
            ],
        ),
        4,
    ),
)

PARTIAL = (
    "T-junction with a stop-controlled side road from the north. Main road east-west, "
    "600 veh/h from the east and 550 veh/h from the west, side road 200 veh/h. "
    "On the main road 10% turn into the side road from the west."
)
write(
    "partial_shares",
    PARTIAL,
    ok(
        PARTIAL,
        interp(
            {
                "east": approach(600),
                "west": approach(550, turning_shares_pct=shares(10, None, None)),
                "north": approach(200),
            },
            control("priority", major_road="east-west"),
            jt="t_junction",
            notes=["The side-road stop sign is modelled as a priority rule."],
        ),
        5,
    ),
)

UNCLEAR = "A busy signalised crossroads near the station, worst in the evening rush hour."
write(
    "unclear_demand",
    UNCLEAR,
    ok(
        UNCLEAR,
        interp(
            {arm: approach(None) for arm in ("north", "east", "south", "west")},
            control("signal"),
            jt="four_arm",
            unclear=["The traffic volumes on the approaches are not stated."],
        ),
        6,
    ),
)

OFF_TOPIC = "Write me a poem about the sea."
write(
    "not_a_junction",
    OFF_TOPIC,
    ok(
        OFF_TOPIC,
        interp(
            {},
            control(None),
            jt=None,
            understood=False,
            unclear=["The text does not describe a road junction."],
        ),
        7,
    ),
)

SHORT_CYCLE = "Crossroads with traffic lights on a 15 second cycle, 300 veh/h on every approach."
write(
    "invalid_cycle",
    SHORT_CYCLE,
    ok(
        SHORT_CYCLE,
        interp(
            {arm: approach(300) for arm in ("north", "east", "south", "west")},
            control("signal", cycle_s=15),
            jt="four_arm",
        ),
        8,
    ),
)

write(
    "refusal",
    "(text withheld)",
    {
        "status": 200,
        "body": message(
            [],
            stop_reason="refusal",
            n=9,
            stop_details={"type": "refusal", "category": None, "explanation": None},
        ),
    },
)
write(
    "max_tokens",
    examples["four_arm_signal"],
    {
        "status": 200,
        "body": message(
            [{"type": "text", "text": '{"understood": true, "unclear": [], "notes": ["'}],
            stop_reason="max_tokens",
            n=10,
        ),
    },
)
write("auth_error", "any", error(401, "authentication_error", "invalid x-api-key"))
write("overloaded", "any", error(529, "overloaded_error", "Overloaded"))
