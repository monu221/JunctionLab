"""Stated, interpreted and assumed fields, and the confirmation gate before a run."""

import pytest

from junctionlab import llm
from junctionlab.__main__ import main
from junctionlab.runner import run_scenario
from junctionlab.scenario import UnconfirmedDemand, resolve

NOTE = "'each road' read as per approach; it could mean both directions together."
ARMS = ("north", "east", "south", "west")


def interpretation(**kw):
    base = {
        "understood": True,
        "unclear": [],
        "notes": [],
        "not_simulated": [],
        "interpreted": [],
        "name": None,
        "junction_type": "four_arm",
        "speed_limit_kmh": 50,
        "approach_length_m": None,
        "approaches": {
            arm: {
                "demand_vph": 600,
                "lane_count": 2,
                "lanes": None,
                "turning_shares_pct": None,
                "heavy_vehicle_share_pct": None,
                "bus_lane": None,
                "bus_demand_bph": None,
                "bus_right_share_pct": None,
            }
            for arm in ARMS
        },
        "control": {
            "kind": "signal",
            "cycle_s": 90,
            "phase_plan": None,
            "green_s": None,
            "major_road": None,
        },
    }
    base.update(kw)
    return llm.Interpretation.model_validate(base)


def scenario(items=()):
    draft, _ = llm.to_draft(interpretation(interpreted=list(items)))
    return resolve(draft)


ALL_DEMAND = {"arm": None, "field_name": "demand_vph", "note": NOTE}


def test_every_field_has_exactly_one_state():
    s = scenario([ALL_DEMAND])
    states = s.field_states()
    assert set(states.values()) == {"stated", "interpreted", "assumed"}
    assert states["speed_limit_kmh"] == "stated"
    assert states["control.cycle_s"] == "stated"
    assert states["approaches.north.demand_vph"] == "interpreted"
    assert states["approach_length_m"] == "assumed"
    assert states["approaches.north.heavy_vehicle_share_pct"] == "assumed"
    assert states["approaches.north.bus_lane"] == "assumed"  # default, not highlighted
    assert "approaches.north.bus_lane" not in s.assumed


def test_interpreted_is_not_also_assumed():
    s = scenario([{"arm": "east", "field_name": "lane_count", "note": NOTE}])
    assert s.field_state("approaches.east.lanes") == "interpreted"
    assert "approaches.east.lanes" not in s.assumed
    assert "approaches.west.lanes" in s.assumed


def test_null_arm_covers_every_approach_and_keeps_the_note():
    s = scenario([ALL_DEMAND])
    assert len(s.needs_confirmation) == 4
    assert all(s.interpreted[p] == NOTE for p in s.needs_confirmation)


def test_interpreted_entry_without_a_value_is_ignored():
    s = scenario([{"arm": "north", "field_name": "heavy_vehicle_share_pct", "note": NOTE}])
    assert s.field_state("approaches.north.heavy_vehicle_share_pct") == "assumed"


def test_non_demand_interpretation_does_not_block():
    s = scenario([{"arm": None, "field_name": "control.cycle_s", "note": NOTE}])
    assert s.field_state("control.cycle_s") == "interpreted"
    assert s.needs_confirmation == ()
    s.require_confirmed()


def test_run_is_blocked_until_all_demand_is_confirmed(tmp_path):
    s = scenario([ALL_DEMAND])
    with pytest.raises(UnconfirmedDemand) as e:
        run_scenario(s, run_dir=tmp_path / "run")
    assert len(e.value.paths) == 4
    assert not (tmp_path / "run").exists()  # stopped before any file is written

    partly = s.confirm(["approaches.north.demand_vph"])
    assert len(partly.needs_confirmation) == 3
    with pytest.raises(UnconfirmedDemand):
        partly.require_confirmed()
    s.confirm().require_confirmed()


def test_confirming_keeps_the_state_and_note():
    s = scenario([ALL_DEMAND]).confirm()
    assert s.field_state("approaches.north.demand_vph") == "interpreted"
    assert s.interpreted["approaches.north.demand_vph"] == NOTE


def test_confirm_rejects_a_field_that_is_not_interpreted():
    with pytest.raises(ValueError, match="not interpreted"):
        scenario().confirm(["approaches.north.demand_vph"])


def test_editing_demand_makes_it_stated():
    s = scenario([ALL_DEMAND]).with_values({"approaches.north.demand_vph": 300})
    assert s.field_state("approaches.north.demand_vph") == "stated"
    assert "approaches.north.demand_vph" not in s.needs_confirmation
    assert len(s.needs_confirmation) == 3


def test_unrelated_edit_does_not_confirm_demand():
    s = scenario([ALL_DEMAND]).with_values({"control.cycle_s": 100})
    assert len(s.needs_confirmation) == 4


def test_gate_survives_a_json_round_trip():
    s = scenario([ALL_DEMAND])
    again = type(s).model_validate_json(s.model_dump_json())
    assert again.needs_confirmation == s.needs_confirmation


def test_cli_run_refuses_unconfirmed_demand(tmp_path, capsys):
    path = tmp_path / "s.json"
    path.write_text(scenario([ALL_DEMAND]).model_dump_json(), encoding="utf-8")
    assert main(["run", str(path), "--out", str(tmp_path / "run")]) == 4
    out = capsys.readouterr().out
    assert "Not run" in out and NOTE in out
    assert not (tmp_path / "run").exists()


# ------------------------------------------------------------------ counts with no period


@pytest.mark.parametrize(
    "text",
    [
        "Crossroads with signals. North 100, south 90, east 80, west 70.",
        "Signals with a 2 minute cycle. North 100, south 90, east 80, west 70.",
        "Signals with a 90 s cycle. Counts: north 100, south 90, east 80, west 70.",
        "Signals, evening rush hour. North 100, south 90, east 80, west 70.",
    ],
)
def test_counts_without_a_period_need_confirmation(text):
    draft, _ = llm.to_draft(interpretation())
    s = resolve(llm._flag_missing_period(text, draft))
    assert len(s.needs_confirmation) == 4
    assert s.interpreted["approaches.west.demand_vph"] == llm.NO_PERIOD_NOTE


@pytest.mark.parametrize(
    "text",
    [
        "600 veh/h on every approach",
        "600 vph on every approach",
        "600 vehicles per hour on every approach",
        "about 600 cars an hour",
        "15-minute counts: 100 on every approach",
        "counts over 15 min: 100 on every approach",
        "hourly flow 600 on every approach",
    ],
)
def test_a_stated_period_is_not_flagged(text):
    draft, _ = llm.to_draft(interpretation())
    assert llm._flag_missing_period(text, draft).interpreted == {}


def test_model_note_is_kept_over_the_period_note():
    draft, _ = llm.to_draft(interpretation(interpreted=[ALL_DEMAND]))
    s = resolve(llm._flag_missing_period("north 100, south 90, east 80, west 70", draft))
    assert s.interpreted["approaches.north.demand_vph"] == NOTE


# ------------------------------------------------------------------ rejection messages


def test_missing_demand_is_reported_once():
    i = interpretation(unclear=[f"Demand not provided for {arm} approach" for arm in ARMS])
    for arm in ARMS:
        getattr(i.approaches, arm).demand_vph = None
    with pytest.raises(llm.InterpretationError) as e:
        llm.to_draft(i)
    assert e.value.reasons == [
        "the traffic on the north, east, south and west approaches is not stated"
    ]


def test_one_missing_demand_names_the_approach():
    i = interpretation()
    i.approaches.east.demand_vph = None
    with pytest.raises(llm.InterpretationError) as e:
        llm.to_draft(i)
    assert e.value.reasons == ["the traffic on the east approach is not stated"]


def test_contradictory_junction_is_not_reported_as_missing():
    reason = "T-junction should have three arms but four approaches are given"
    i = interpretation(junction_type=None, unclear=[reason])
    with pytest.raises(llm.InterpretationError) as e:
        llm.to_draft(i)
    assert e.value.reasons == [reason]


def test_truly_missing_junction_type_is_still_reported():
    i = interpretation(junction_type=None)
    with pytest.raises(llm.InterpretationError, match="junction type is not stated"):
        llm.to_draft(i)
