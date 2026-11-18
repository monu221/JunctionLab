"""End-to-end SUMO runs with a shortened timeline to keep the suite fast."""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from junctionlab.__main__ import main
from junctionlab.config import get_config
from junctionlab.examples import list_examples, load_example
from junctionlab.runner import run_scenario
from junctionlab.scenario import ScenarioDraft, resolve

EXAMPLES = {e.key: e for e in list_examples()}


def short_config(**simulation):
    cfg = get_config()
    sim = {"warmup_s": 60, "measurement_s": 300, "runout_max_s": 300, **simulation}
    return cfg.model_copy(update={"simulation": cfg.simulation.model_copy(update=sim)})


def scenario(key, **approach_demands):
    draft = EXAMPLES[key].draft.model_dump(exclude_none=True)
    for arm, q in approach_demands.items():
        draft["approaches"][arm]["demand_vph"] = q
    return resolve(ScenarioDraft.model_validate(draft))


@pytest.mark.parametrize("key", ["four_arm_signal", "four_arm_bus_lane"])
def test_signal_examples_run_cleanly(key, tmp_path):
    cfg = short_config()
    (r,) = run_scenario(EXAMPLES[key].scenario(), tmp_path, seeds=[1], config=cfg).seeds
    assert r.ended_by == "measured_vehicles_left"
    assert r.end_time_s < cfg.simulation.end_s
    assert r.vehicles_arrived > 100
    assert (r.collisions, r.teleports, r.gridlock) == (0, 0, [])
    assert r.warnings == []
    trips = list(ET.parse(r.tripinfo).getroot().iter("tripinfo"))
    assert all(float(t.get("timeLoss")) >= 0 for t in trips)
    intervals = list(ET.parse(r.e2).getroot().iter("interval"))
    assert intervals and "maxJamLengthInMeters" in intervals[0].attrib
    assert "maxJamLengthInVehicles" in intervals[0].attrib
    log = (r.dir / "sumo.log").read_text(encoding="utf-8").lower()
    assert "unsafe green" not in log


def test_low_demand_priority_junction_runs(tmp_path):
    s = scenario("t_junction_priority", east=300, west=300, south=100)
    (r,) = run_scenario(s, tmp_path, seeds=[1], config=short_config()).seeds
    assert r.ended_by == "measured_vehicles_left" and r.warnings == []


def test_overload_hits_time_cap(tmp_path):
    s = scenario("t_junction_priority", east=900, west=900, south=600)
    cfg = short_config(runout_max_s=60)
    (r,) = run_scenario(s, tmp_path, seeds=[1], config=cfg).seeds
    assert r.ended_by == "time_cap"
    assert r.end_time_s == cfg.simulation.end_s
    assert r.measured_vehicles_remaining > 0
    assert r.teleports == 0  # vehicles wait, they are never teleported
    assert any("time cap" in w for w in r.warnings)
    unfinished = [
        t for t in ET.parse(r.tripinfo).getroot().iter("tripinfo") if float(t.get("arrival")) < 0
    ]
    assert unfinished


def test_original_overloaded_t_junction_warns(tmp_path):
    """The first T-junction example volumes overload the stem: full timeline, all 5 seeds."""
    example = load_example(Path(__file__).parent / "data" / "t_junction_overloaded.json")
    result = run_scenario(example.scenario(), tmp_path)
    assert all(r.ended_by == "time_cap" for r in result.seeds)
    assert all(r.teleports == 0 for r in result.seeds)
    assert any(r.gridlock for r in result.seeds)
    assert all(r.warnings for r in result.seeds)


def test_gridlock_warning(tmp_path):
    # With a 4-phase split and a 120 s cycle, red lasts far longer than 20 s.
    draft = EXAMPLES["four_arm_signal"].draft.model_dump(exclude_none=True)
    draft["control"] = {"kind": "signal", "cycle_s": 120, "phase_plan": "four_phase_split"}
    s = resolve(ScenarioDraft.model_validate(draft))
    (r,) = run_scenario(s, tmp_path, seeds=[1], config=short_config(gridlock_stop_s=20)).seeds
    assert r.gridlock
    assert all(e.stopped_for_s > 20 for e in r.gridlock)
    assert any("gridlock" in w for w in r.warnings)


def test_seeds_differ_and_are_reproducible(tmp_path):
    s = EXAMPLES["four_arm_signal"].scenario()
    cfg = short_config()
    a = run_scenario(s, tmp_path / "a", seeds=[1, 2], config=cfg).seeds
    b = run_scenario(s, tmp_path / "b", seeds=[1], config=cfg, parallel=False).seeds
    assert a[0].vehicles_inserted != a[1].vehicles_inserted
    assert _trips(a[0].tripinfo) == _trips(b[0].tripinfo)


def _trips(path):
    """Trip records without SUMO's header comment (which holds a timestamp)."""
    return [t.attrib for t in ET.parse(path).getroot().iter("tripinfo")]


def test_cli_run(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("junctionlab.runner.get_config", lambda: short_config())
    assert main(["run", "four_arm_signal", "--seeds", "1", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "greens 49 / 31 s" in out
    assert "measured_vehicles_left" in out
    assert "Results over 1 seed(s): mean (min-max)" in out
    for arm in ("north", "east", "south", "west", "junction"):
        assert any(line.startswith(arm) for line in out.splitlines()), arm
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "seed_1" / "tripinfo.xml").is_file()
