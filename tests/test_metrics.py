"""Metrics on hand-written SUMO output (exact values) and on short real runs."""

import json
from pathlib import Path

import pytest

from junctionlab.config import get_config
from junctionlab.examples import list_examples, load_example
from junctionlab.metrics import (
    Stat,
    grade,
    los_grade,
    measurement_window,
    seed_metrics,
    signal_v_c,
    summarize,
)
from junctionlab.runner import run_scenario

EXAMPLES = {e.key: e for e in list_examples()}


def cfg_with(**simulation):
    cfg = get_config()
    return cfg.model_copy(update={"simulation": cfg.simulation.model_copy(update=simulation)})


# ------------------------------------------------------------------ LOS


@pytest.mark.parametrize(
    ("delay", "expected"),
    [(0, "A"), (10, "A"), (10.01, "B"), (20, "B"), (35, "C"), (55, "D"), (80, "E"), (80.1, "F")],
)
def test_signalised_los_boundaries(delay, expected):
    assert los_grade(delay, get_config().los.signalised) == expected


@pytest.mark.parametrize(
    ("delay", "expected"), [(15, "B"), (25, "C"), (35, "D"), (50, "E"), (51, "F")]
)
def test_unsignalised_los_boundaries(delay, expected):
    assert los_grade(delay, get_config().los.unsignalised) == expected


def test_signal_v_c_uses_measured_saturation_flow_and_lost_time():
    cfg = get_config()
    s = EXAMPLES["four_arm_signal"].scenario()
    v_c = signal_v_c(s, cfg)
    sat, lost, amber = (
        cfg.control.saturation_flow_vphpl,
        cfg.control.lost_time_s,
        cfg.control.amber_s,
    )
    # north: 400 veh/h per lane / (s x effective green / cycle)
    assert v_c["north"] == pytest.approx(400 / (sat * (49 + amber - lost) / 90))
    assert v_c["east"] == pytest.approx(250 / (sat * (31 + amber - lost) / 90))


def test_grade_overload_rules():
    cfg = get_config()
    signal = EXAMPLES["four_arm_signal"].scenario()
    priority = EXAMPLES["t_junction_priority"].scenario()

    assert grade(signal, cfg, 30, 500, 500, 0, 0.9) == ("C", [])
    los, reasons = grade(signal, cfg, 30, 500, 500, 0, 1.1)
    assert los == "F" and "v/c 1.10 > 1" in reasons[0]
    # The throughput and backlog rules apply to unsignalised approaches only.
    assert grade(signal, cfg, 30, 100, 500, 50, 0.9)[0] == "C"

    assert grade(priority, cfg, 20, 96, 100, 0, None) == ("C", [])
    los, reasons = grade(priority, cfg, 20, 94, 100, 0, None)
    assert los == "F" and "throughput" in reasons[0]
    assert grade(priority, cfg, 20, 100, 100, 4, None)[0] == "C"
    los, reasons = grade(priority, cfg, 20, 100, 100, 5, None)
    assert los == "F" and "backlog" in reasons[0]


def test_stat():
    assert Stat.of([1, 2, 6]) == Stat(3, 1, 6)
    assert Stat.of([None, None]) is None


# ------------------------------------------------------------------ synthetic seed output


def write_seed(d: Path, cfg, trips, never_inserted=(), backlog=None, e1=(), e2=(), end=None):
    """Write minimal SUMO-like outputs for one seed."""
    d.mkdir(parents=True)
    rows = "".join(
        f'<tripinfo id="{vid}" depart="{dep}" departDelay="{dd}" timeLoss="{tl}" arrival="{arr}"/>'
        for vid, dep, dd, tl, arr in trips
    )
    (d / "tripinfo.xml").write_text(f"<tripinfos>{rows}</tripinfos>", encoding="utf-8")
    e1_rows = "".join(
        f'<interval begin="{b}" end="{b + 60}" id="e1_{lane}" nVehContrib="{n}"/>'
        for b, lane, n in e1
    )
    (d / "e1.xml").write_text(f"<detector>{e1_rows}</detector>", encoding="utf-8")
    e2_rows = "".join(
        f'<interval begin="{b}" end="{b + 60}" id="e2_{lane}" maxJamLengthInMeters="{m}" '
        f'maxJamLengthInVehicles="{v}"/>'
        for b, lane, m, v in e2
    )
    (d / "e2.xml").write_text(f"<detector>{e2_rows}</detector>", encoding="utf-8")
    times, counts = backlog or ([], {})
    (d / "backlog.json").write_text(
        json.dumps(
            {
                "interval_s": 10,
                "times_s": times,
                "counts_by_arm": counts,
                "never_inserted": [{"id": v, "first_seen_s": t} for v, t in never_inserted],
            }
        ),
        encoding="utf-8",
    )
    end = end if end is not None else cfg.simulation.end_s
    (d / "run.json").write_text(json.dumps({"end_time_s": end}), encoding="utf-8")


def test_seed_metrics_on_known_output(tmp_path):
    cfg = cfg_with(warmup_s=60, measurement_s=120, runout_max_s=60)  # window [60, 180)
    s = EXAMPLES["t_junction_priority"].scenario()
    write_seed(
        tmp_path / "s1",
        cfg,
        trips=[
            # id, depart, departDelay, timeLoss, arrival
            ("south_left_car.0", 30, 0, 99, 80),  # warm-up: not measured
            ("south_left_car.1", 70, 2, 10, 120),  # measured: delay 12
            ("south_right_car.2", 100, 0, 20, 150),  # measured: delay 20
            ("south_left_car.3", 185, 10, 30, -1),  # intended 175: unfinished, delay 40
            ("south_left_car.4", 200, 0, 5, 240),  # intended 200: after window
            ("ro_south_left_car.0", 100, 0, 500, 200),  # run-out flow: never measured
            ("east_through_car.0", 90, 0, 3, 120),
        ],
        never_inserted=[("south_left_car.9", 170), ("ro_south_left_car.5", 170)],
        backlog=([50, 60, 170, 180], {"south": [0, 1, 3, 4], "east": [0, 0, 0, 0]}),
        e1=[
            (0, "south_in_0", 9),
            (60, "south_in_0", 3),
            (120, "south_in_0", 2),
            (180, "south_in_0", 7),
        ],
        e2=[(60, "south_in_0", 40.0, 6), (120, "south_in_0", 25.0, 4), (180, "south_in_0", 99, 15)],
        end=240,
    )
    m = seed_metrics(s, tmp_path / "s1", cfg)["south"]
    assert measurement_window(cfg).hours == pytest.approx(120 / 3600)
    # delays: 12, 20, 40 (unfinished), 70 (never inserted: 240 - 170)
    assert m.vehicles == 4 and m.finished == 2 and m.unfinished == 1 and m.never_inserted == 1
    assert m.delay_s == pytest.approx((12 + 20 + 40 + 70) / 4)
    assert m.delay_lower_bound
    assert m.throughput_vph == pytest.approx(5 * 30)  # 3 + 2 vehicles in 2 minutes
    assert m.demand_vph == pytest.approx(4 * 30)
    assert (m.queue_m, m.queue_veh) == (40.0, 6)  # interval at 180 is outside the window
    assert (m.backlog_start, m.backlog_end) == (1, 4)
    assert m.v_c is None

    east = seed_metrics(s, tmp_path / "s1", cfg)["east"]
    assert east.delay_s == 3 and not east.delay_lower_bound


def test_queue_is_longest_lane_not_sum(tmp_path):
    cfg = cfg_with(warmup_s=60, measurement_s=120, runout_max_s=60)
    s = EXAMPLES["four_arm_signal"].scenario()
    write_seed(
        tmp_path / "s1",
        cfg,
        trips=[],
        e2=[
            (60, "north_in_0", 30.0, 4),
            (60, "north_in_1", 50.0, 7),
            (120, "north_in_0", 45.0, 6),
        ],
    )
    m = seed_metrics(s, tmp_path / "s1", cfg)["north"]
    assert (m.queue_m, m.queue_veh) == (50.0, 7)
    assert m.delay_s is None and m.los in "ABCDEF-"


def test_summary_mean_and_range(tmp_path):
    cfg = cfg_with(warmup_s=60, measurement_s=120, runout_max_s=60)
    s = EXAMPLES["t_junction_priority"].scenario()
    for seed, delay in ((1, 10), (2, 30)):
        write_seed(
            tmp_path / f"s{seed}",
            cfg,
            trips=[(f"south_left_car.{i}", 70 + i, 0, delay, 150) for i in range(4)],
            e1=[(60, "south_in_0", 2), (120, "south_in_0", 2)],
            e2=[(60, "south_in_0", 10.0 * seed, seed)],
        )
    summary = summarize(s, {1: tmp_path / "s1", 2: tmp_path / "s2"}, cfg)
    south = summary.approaches["south"]
    assert south.delay_s == Stat(20, 10, 30)
    assert south.los == "C"  # mean 20 s on the unsignalised table
    assert south.los_by_seed == ["A", "D"] and south.los_range == "A-D"
    assert south.queue_m == Stat(15, 10, 20)
    assert summary.junction.los == "-"  # no junction LOS for a priority rule
    json.loads(summary.to_json())


def test_spillback_warning(tmp_path):
    cfg = cfg_with(warmup_s=60, measurement_s=120, runout_max_s=60)
    s = EXAMPLES["t_junction_priority"].scenario()
    write_seed(tmp_path / "s1", cfg, trips=[], e2=[(60, "east_in_0", 241.0, 33)])
    summary = summarize(s, {1: tmp_path / "s1"}, cfg)
    assert summary.approaches["east"].spillback_seeds == [1]
    assert any("east: queue reached 80%" in w for w in summary.warnings)


# ------------------------------------------------------------------ real runs


def test_short_run_throughput_matches_demand(tmp_path):
    cfg = cfg_with(warmup_s=120, measurement_s=600, runout_max_s=300)
    result = run_scenario(
        EXAMPLES["four_arm_signal"].scenario(), tmp_path, seeds=[1, 2], config=cfg
    )
    summary = result.summary
    assert (tmp_path / "summary.json").is_file()
    for a in summary.approaches.values():
        assert a.throughput_vph.mean == pytest.approx(a.demand_vph.mean, rel=0.15)
        assert a.delay_s.mean > 0 and a.los in "ABCDEF"
        assert not a.f_reasons and not a.delay_lower_bound
    assert summary.junction.los in "ABCDEF"


def test_overloaded_stem_graded_f(tmp_path):
    example = load_example(Path(__file__).parent / "data" / "t_junction_overloaded.json")
    cfg = cfg_with(warmup_s=60, measurement_s=900, runout_max_s=60)
    south = run_scenario(example.scenario(), tmp_path, seeds=[1], config=cfg).summary.approaches[
        "south"
    ]
    assert south.los == "F"
    assert any("throughput" in r for r in south.f_reasons)
    assert south.delay_lower_bound
