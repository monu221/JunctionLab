import pytest

from junctionlab.config import get_config
from junctionlab.examples import list_examples
from junctionlab.scenario import ScenarioDraft, resolve
from junctionlab.signal import critical_flow_ratios, lane_flows_vph, phase_arms, webster_greens

EXAMPLES = {e.key: e for e in list_examples()}


def signal(demand: dict[str, float], **control) -> object:
    return resolve(
        ScenarioDraft.model_validate(
            {
                "junction_type": "four_arm",
                "approaches": {arm: {"demand_vph": q} for arm, q in demand.items()},
                "control": {"kind": "signal", "cycle_s": 90, **control},
            }
        )
    )


def test_lane_flows_split_shared_movements_equally():
    s = signal({"north": 800, "east": 500, "south": 600, "west": 500})
    # 15/70/15 over [L+T, T+R]: each lane gets 120 + 280 = 400
    assert lane_flows_vph(s, "north") == pytest.approx([400, 400])


def test_bus_lane_carries_buses():
    s = EXAMPLES["four_arm_bus_lane"].scenario()
    flows = lane_flows_vph(s, "west")
    assert flows[-1] == 12 and len(flows) == 3


def test_greens_proportional_to_critical_ratios():
    s = signal({"north": 800, "east": 500, "south": 600, "west": 500})
    y = critical_flow_ratios(s)
    assert y == pytest.approx([400 / 1800, 250 / 1800])
    greens = webster_greens(s)
    assert sum(greens) == 80  # 90 s - 2 x 5 s intergreen
    assert greens == (49, 31)  # 80 x 0.222/0.361 = 49.2
    assert all(g == int(g) for g in greens)


def test_minimum_green_is_enforced():
    s = signal({"north": 1500, "east": 1, "south": 1500, "west": 1})
    assert webster_greens(s) == (75, get_config().control.min_green_s)


def test_zero_demand_splits_equally():
    s = signal({"north": 0, "east": 0, "south": 0, "west": 0})
    assert webster_greens(s) == (40, 40)


def test_t_junction_phases_major_then_stem():
    draft = EXAMPLES["t_junction_priority"].draft.model_dump()
    draft["control"] = {"kind": "signal"}
    s = resolve(ScenarioDraft.model_validate(draft))
    assert phase_arms(s) == [("east", "west"), ("south",)]


def test_edit_rederives_assumed_greens():
    s = signal({"north": 800, "east": 500, "south": 600, "west": 500})
    assert s.is_assumed("control.green_s")
    edited = s.with_values({"approaches.east.demand_vph": 800, "approaches.west.demand_vph": 800})
    assert edited.control.green_s == (40, 40)
    assert edited.is_assumed("control.green_s")


def test_stated_greens_are_kept():
    s = signal({"north": 800, "east": 500, "south": 600, "west": 500}, green_s=[40, 40])
    assert s.control.green_s == (40, 40)
    assert not s.is_assumed("control.green_s")
