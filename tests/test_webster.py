import pytest

from junctionlab.__main__ import main
from junctionlab.config import get_config
from junctionlab.webster import run_check, saturation_from_crossings, webster_delay


def test_webster_formula_by_hand():
    # c = 90 s, lambda = 0.4, q = 360 veh/h = 0.1 veh/s, x = 0.5
    c, lam, q, x = 90, 0.4, 0.1, 0.5
    expected = (
        c * (1 - lam) ** 2 / (2 * (1 - lam * x))
        + x**2 / (2 * q * (1 - x))
        - 0.65 * (c / q**2) ** (1 / 3) * x ** (2 + 5 * lam)
    )
    assert webster_delay(90, 0.4, 360, 0.5) == pytest.approx(expected)
    # uniform 90 x 0.36 / 1.6 = 20.25; random 0.25 / 0.1 = 2.5; correction 0.65 x 9000^(1/3) x 0.5^4
    assert webster_delay(90, 0.4, 360, 0.5) == pytest.approx(20.25 + 2.5 - 0.845, abs=0.01)


def test_delay_rises_steeply_towards_capacity():
    d = [webster_delay(90, 0.41, x * 1861 * 0.41, x) for x in (0.5, 0.7, 0.85, 0.95)]
    assert d == sorted(d) and d[-1] > 2 * d[0]


def test_saturation_from_known_crossings():
    cfg = get_config()
    cfg = cfg.model_copy(
        update={
            "simulation": cfg.simulation.model_copy(update={"warmup_s": 180, "measurement_s": 180})
        }
    )
    # Window [180, 360) = cycles starting at 180 and 270. Green 40 s, amber 3 s, all-red 2 s.
    # Per cycle: first crossing 3 s after green start, then the first four 2.5 s apart (start-up),
    # then every 2.0 s; the last crossings fall into the amber.
    crossings = []
    for start in (0, 90, 180, 270):
        t = start + 3.0
        for i in range(20):
            crossings.append(t)
            t += 2.5 if i < 4 else 2.0
    headways, per_cycle = saturation_from_crossings(crossings, cfg)
    assert per_cycle == [20, 20]  # the cycles at 0 and 90 are warm-up
    # In-green crossings: 3, 5.5, 8, 10.5, 13, 15, ... 39 -> 18 vehicles; headways from the 5th on
    assert headways == pytest.approx(([2.5] + [2.0] * 13) * 2)


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    """The full check with the configured timeline: this is the M4 acceptance test."""
    return run_check(tmp_path_factory.mktemp("webster"))


def test_measured_values_match_config(report):
    cfg = get_config()
    sat = report.saturation
    assert sat.saturation_flow_vph == pytest.approx(cfg.control.saturation_flow_vphpl, rel=0.02)
    assert sat.lost_time_s == pytest.approx(cfg.control.lost_time_s, abs=0.5)
    assert sat.cycles >= 150


def test_tested_cases_within_margin(report):
    tested = [c for c in report.cases if c.tested]
    assert [c.x for c in tested] == [0.5, 0.7, 0.85]
    assert report.all_passed, [(c.x, round(c.diff_s, 1)) for c in tested]
    assert [c.passed for c in report.cases if not c.tested] == [None]


def test_free_flow_background_loss_is_small(report):
    assert 0 < report.free_flow_delay_s.mean < 5


def test_compared_delay_is_control_delay(report):
    for c in report.cases:
        assert c.simulated_s.mean == pytest.approx(
            c.simulated_total_s.mean - report.free_flow_delay_s.mean
        )
        assert c.diff_s == pytest.approx(c.simulated_s.mean - c.webster_s)


def test_cli(tmp_path, capsys, report, monkeypatch):
    monkeypatch.setattr("junctionlab.webster.run_check", lambda out, cfg: report)
    assert main(["webster", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "saturation flow" in out and "Overall: PASS" in out
    assert out.count("pass") >= 3 and "info only" in out
