import pytest
from pydantic import ValidationError

from junctionlab.config import get_config
from junctionlab.examples import list_examples
from junctionlab.scenario import (
    Scenario,
    ScenarioDraft,
    destination,
    resolve,
    standard_lanes,
)


def four_arm(**overrides) -> dict:
    draft = {
        "junction_type": "four_arm",
        "approaches": {arm: {"demand_vph": 500} for arm in ("north", "east", "south", "west")},
        "control": {"kind": "signal"},
    }
    draft.update(overrides)
    return draft


def t_junction(**overrides) -> dict:
    """Main road east-west, stem from the south (north arm missing)."""
    draft = {
        "junction_type": "t_junction",
        "approaches": {arm: {"demand_vph": 400} for arm in ("east", "south", "west")},
        "control": {"kind": "priority"},
    }
    draft.update(overrides)
    return draft


def build(draft: dict) -> Scenario:
    return resolve(ScenarioDraft.model_validate(draft))


# ------------------------------------------------------------------ geometry


@pytest.mark.parametrize(
    ("arm", "movement", "expected"),
    [
        ("north", "left", "east"),
        ("north", "through", "south"),
        ("north", "right", "west"),
        ("west", "left", "north"),
        ("south", "right", "east"),
    ],
)
def test_destination_right_hand_traffic(arm, movement, expected):
    assert destination(arm, movement) == expected


def test_standard_lane_use():
    assert standard_lanes(1) == [["left", "through", "right"]]
    assert standard_lanes(3) == [["left", "through"], ["through"], ["through", "right"]]


# ------------------------------------------------------------------ defaults and assumed fields


def test_defaults_fill_gaps_and_are_marked_assumed():
    cfg = get_config()
    s = build(four_arm())
    north = s.approaches["north"]
    assert north.lanes == (("left", "through"), ("through", "right"))
    assert north.heavy_vehicle_share_pct == cfg.scenario_defaults.heavy_vehicle_share_pct
    assert north.turning_shares_pct == dict(cfg.scenario_defaults.turning_shares_pct.four_arm)
    assert s.control.cycle_s == cfg.control.cycle_default_s
    assert s.control.phase_plan == "two_phase"
    assert s.control.green_s is None
    assert s.speed_limit_kmh == 50
    assert s.approach_length_m == 300
    for path in (
        "approaches.north.lanes",
        "approaches.north.turning_shares_pct",
        "approaches.north.heavy_vehicle_share_pct",
        "control.cycle_s",
        "control.phase_plan",
        "control.green_s",
        "speed_limit_kmh",
        "approach_length_m",
    ):
        assert s.is_assumed(path), path


def test_stated_values_are_not_assumed():
    draft = four_arm(control={"kind": "signal", "cycle_s": 80, "green_s": [35, 35]})
    draft["approaches"]["north"] = {
        "demand_vph": 800,
        "lanes": [["left"], ["through", "right"]],
        "turning_shares_pct": {"left": 20, "through": 65, "right": 15},
        "heavy_vehicle_share_pct": 8,
    }
    s = build(draft)
    for path in (
        "control.cycle_s",
        "control.green_s",
        "approaches.north.lanes",
        "approaches.north.turning_shares_pct",
        "approaches.north.heavy_vehicle_share_pct",
    ):
        assert not s.is_assumed(path), path
    assert s.is_assumed("approaches.east.lanes")


def test_absent_bus_lane_is_not_an_assumption():
    s = build(four_arm())
    assert not any("bus" in path for path in s.assumed)


def test_bus_direction_is_assumed_only_when_buses_run():
    draft = four_arm()
    draft["approaches"]["west"].update(bus_lane=True, bus_demand_bph=12)
    s = build(draft)
    assert s.approaches["west"].lane_count == 3
    assert s.is_assumed("approaches.west.bus_right_share_pct")


def test_lane_count_gives_standard_lane_use():
    draft = four_arm()
    draft["approaches"]["north"]["lane_count"] = 3
    lanes = build(draft).approaches["north"].lanes
    assert lanes == (("left", "through"), ("through",), ("through", "right"))


# ------------------------------------------------------------------ T-junction


def test_t_junction_geometry_and_defaults():
    s = build(t_junction())
    assert s.missing_arm == "north"
    assert s.stem == "south"
    assert s.control.major_road == "east-west"
    assert not s.is_assumed("control.major_road")  # fixed by geometry
    assert s.approaches["south"].turning_shares_pct == {"left": 50, "right": 50}
    assert s.approaches["east"].turning_shares_pct == {"through": 80, "left": 20}
    assert s.approaches["west"].turning_shares_pct == {"through": 80, "right": 20}
    # Default lanes lose movements that lead to the missing arm.
    assert s.approaches["south"].lanes == (("left",), ("right",))
    assert s.approaches["east"].lanes == (("left", "through"), ("through",))


def test_t_junction_rejects_stem_as_major_road():
    with pytest.raises(ValidationError, match="stem is always the minor road"):
        build(t_junction(control={"kind": "priority", "major_road": "north-south"}))


def test_t_junction_rejects_movement_to_missing_arm():
    draft = t_junction()
    draft["approaches"]["east"]["turning_shares_pct"] = {"left": 20, "through": 70, "right": 10}
    draft["approaches"]["east"]["lanes"] = [["left", "through", "right"]]
    with pytest.raises(ValidationError, match="lead to no arm"):
        build(draft)


def test_t_junction_signal_must_be_two_phase():
    with pytest.raises(ValidationError, match="two-phase"):
        build(t_junction(control={"kind": "signal", "phase_plan": "four_phase_split"}))


def test_t_junction_rejects_buses_turning_into_missing_arm():
    draft = t_junction()
    draft["approaches"]["east"].update(bus_demand_bph=10, bus_right_share_pct=50)
    with pytest.raises(ValidationError, match="buses go straight or right"):
        build(draft)


def test_four_arm_needs_all_arms():
    draft = four_arm()
    del draft["approaches"]["west"]
    with pytest.raises(ValidationError, match="4-arm"):
        build(draft)


# ------------------------------------------------------------------ approach rules


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"turning_shares_pct": {"left": 20, "through": 70, "right": 20}}, "sum to 100"),
        ({"lanes": [["through"], ["through", "right"]]}, "no lane serves"),
        ({"lanes": [["through", "right"], ["left"]]}, "cross"),
        ({"lanes": [["through", "left"], ["right"]]}, "left to right"),
        ({"heavy_vehicle_share_pct": 40}, "less than or equal to 30"),
        ({"lane_count": 4, "bus_lane": True}, "at most 4 lanes"),
        ({"demand_vph": -1}, "greater than or equal to 0"),
    ],
)
def test_approach_rules(change, message):
    draft = four_arm()
    draft["approaches"]["north"].update(change)
    with pytest.raises(ValidationError, match=message):
        build(draft)


# ------------------------------------------------------------------ signal rules


@pytest.mark.parametrize(
    ("control", "message"),
    [
        ({"cycle_s": 90, "green_s": [40, 40, 0]}, "needs 2 green times"),
        ({"cycle_s": 90, "green_s": [50, 40]}, "the cycle is 90"),
        ({"cycle_s": 90, "green_s": [4, 76]}, "at least 5"),
        ({"cycle_s": 15}, "too short"),
        ({"cycle_s": 30, "phase_plan": "four_phase_split"}, "too short"),
    ],
)
def test_signal_rules(control, message):
    with pytest.raises(ValidationError, match=message):
        build(four_arm(control={"kind": "signal", **control}))


def test_valid_explicit_greens():
    # 2 phases x (green + 3 s amber + 2 s all-red) = 90 s
    s = build(four_arm(control={"kind": "signal", "cycle_s": 90, "green_s": [45, 35]}))
    assert s.control.green_s == (45, 35)
    s = build(
        four_arm(
            control={
                "kind": "signal",
                "cycle_s": 100,
                "phase_plan": "four_phase_split",
                "green_s": [20, 20, 20, 20],
            }
        )
    )
    assert s.control.phase_plan == "four_phase_split"


def test_priority_on_four_arm_defaults_major_road():
    s = build(four_arm(control={"kind": "priority"}))
    assert s.control.major_road == "north-south"
    assert s.is_assumed("control.major_road")


# ------------------------------------------------------------------ form edits


def test_edit_clears_assumed_mark():
    s = build(four_arm())
    edited = s.with_values({"control.cycle_s": 100, "approaches.north.heavy_vehicle_share_pct": 10})
    assert edited.control.cycle_s == 100
    assert not edited.is_assumed("control.cycle_s")
    assert not edited.is_assumed("approaches.north.heavy_vehicle_share_pct")
    assert edited.is_assumed("approaches.east.heavy_vehicle_share_pct")


def test_editing_parent_clears_children():
    s = build(four_arm())
    north = s.approaches["north"].model_dump(mode="json")
    edited = s.with_values({"approaches.north": north})
    assert not any(p.startswith("approaches.north.") for p in edited.assumed)


def test_invalid_edit_is_rejected():
    s = build(four_arm())
    with pytest.raises(ValidationError):
        s.with_values({"approaches.north.turning_shares_pct": {"left": 50, "through": 70}})


def test_json_round_trip():
    s = build(t_junction())
    assert Scenario.model_validate_json(s.model_dump_json()) == s


def test_draft_json_schema_available_for_llm():
    schema = ScenarioDraft.model_json_schema()
    assert set(schema["required"]) == {"junction_type", "approaches", "control"}


# ------------------------------------------------------------------ bundled examples


def test_three_bundled_examples_resolve():
    examples = {e.key: e for e in list_examples()}
    assert set(examples) == {"four_arm_signal", "t_junction_priority", "four_arm_bus_lane"}

    signal = examples["four_arm_signal"].scenario()
    assert signal.control.kind == "signal" and signal.control.cycle_s == 90
    assert not signal.is_assumed("control.cycle_s")
    assert signal.approaches["north"].demand_vph == 800

    t = examples["t_junction_priority"].scenario()
    assert t.stem == "south" and t.control.major_road == "east-west"

    bus = examples["four_arm_bus_lane"].scenario()
    west = bus.approaches["west"]
    assert west.bus_lane and west.bus_demand_bph == 12 and west.lane_count == 3
    for e in examples.values():
        assert e.description
