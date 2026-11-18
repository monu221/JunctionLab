"""Measures per approach from SUMO outputs, and their aggregation over seeds (method §3).

Measured vehicles: warm-up/measurement flows (no "ro_" prefix) whose intended departure
(depart - departDelay) falls in the measurement window.

Per approach and seed:
- delay: mean of timeLoss + departDelay over measured vehicles. Vehicles still driving at
  the time cap count with their delay so far, and vehicles still waiting to enter count
  with their waiting time so far; both make the value a lower bound (flagged).
- max queue: largest maxJamLengthInMeters / maxJamLengthInVehicles of any lane's E2
  detector in any 60 s interval of the measurement window (longest lane, not the sum).
- throughput: vehicles crossing the stop-line E1 loops during the window, per hour.
- demand: measured vehicles generated in the window, per hour (the Poisson realisation).
- LOS: HCM table by control type from delay; F when overloaded (see `grade`).
"""

from __future__ import annotations

import json
import statistics
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from junctionlab.build import DETECTOR_PERIOD_S, RUNOUT_PREFIX
from junctionlab.config import LOS_GRADES, Config, get_config
from junctionlab.scenario import FixedTimeSignal, PriorityRule, Scenario
from junctionlab.signal import lane_flows_vph, phase_arms


@dataclass(frozen=True)
class Window:
    start_s: float
    end_s: float

    @property
    def hours(self) -> float:
        return (self.end_s - self.start_s) / 3600

    def __contains__(self, t: float) -> bool:
        return self.start_s <= t < self.end_s


def measurement_window(cfg: Config) -> Window:
    sim = cfg.simulation
    if sim.warmup_s % DETECTOR_PERIOD_S or sim.measurement_s % DETECTOR_PERIOD_S:
        raise ValueError(f"warm-up and measurement must be multiples of {DETECTOR_PERIOD_S} s")
    return Window(sim.warmup_s, sim.warmup_s + sim.measurement_s)


def _arm_of_vehicle(vehicle_id: str) -> str:
    return vehicle_id.split("_", 1)[0]


def _arm_of_lane(lane_id: str) -> str:
    return lane_id.split("_", 1)[0]  # "north_in_0" -> "north"


# --------------------------------------------------------------------------- LOS


def los_grade(delay_s: float, thresholds: dict[str, float]) -> str:
    for grade in LOS_GRADES:
        if delay_s <= thresholds[grade]:
            return grade
    return "F"


def signal_v_c(scenario: Scenario, cfg: Config) -> dict[str, float]:
    """Degree of saturation per approach: critical lane flow / (s x effective green / cycle).

    Uses the saturation flow and lost time measured in M4, and the nominal demand.
    """
    c = scenario.control
    assert isinstance(c, FixedTimeSignal) and c.green_s is not None
    s = cfg.control.saturation_flow_vphpl
    result = {}
    for arms, green in zip(phase_arms(scenario), c.green_s, strict=True):
        capacity_per_lane = s * cfg.control.effective_green_s(green) / c.cycle_s
        for arm in arms:
            result[arm] = max(lane_flows_vph(scenario, arm)) / capacity_per_lane
    return result


def grade(
    scenario: Scenario,
    cfg: Config,
    delay_s: float | None,
    throughput_vph: float,
    demand_vph: float,
    backlog_growth: float,
    v_c: float | None,
) -> tuple[str, list[str]]:
    """LOS grade and the reasons for an overload F (empty if not overloaded)."""
    signalised = isinstance(scenario.control, FixedTimeSignal)
    table = cfg.los.signalised if signalised else cfg.los.unsignalised
    reasons = []
    if signalised:
        if v_c is not None and v_c > 1:
            reasons.append(f"v/c {v_c:.2f} > 1")
    else:
        share = cfg.los.overload_throughput_share
        if demand_vph > 0 and throughput_vph < share * demand_vph:
            reasons.append(
                f"throughput {throughput_vph:.0f} < {share:.0%} of demand {demand_vph:.0f} veh/h"
            )
        if backlog_growth >= cfg.los.backlog_growth_min_veh:
            reasons.append(f"insertion backlog grew by {backlog_growth:.0f} vehicles")
    if reasons:
        return "F", reasons
    if delay_s is None:
        return "-", []
    return los_grade(delay_s, table), []


# --------------------------------------------------------------------------- one seed


@dataclass
class SeedApproach:
    arm: str
    vehicles: int
    finished: int
    unfinished: int
    never_inserted: int
    delay_s: float | None
    throughput_vph: float
    demand_vph: float
    queue_m: float
    queue_veh: float
    backlog_start: int
    backlog_end: int
    v_c: float | None
    los: str = "-"
    f_reasons: list[str] = field(default_factory=list)

    @property
    def delay_lower_bound(self) -> bool:
        return self.unfinished + self.never_inserted > 0

    @property
    def backlog_growth(self) -> int:
        return self.backlog_end - self.backlog_start


def _delays_from_tripinfo(path: Path, win: Window) -> dict[str, list[tuple[float, bool]]]:
    """Per approach: (delay, finished) of each measured vehicle that has entered."""
    out: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    for t in ET.parse(path).getroot().iter("tripinfo"):
        vid = t.get("id")
        if vid.startswith(RUNOUT_PREFIX):
            continue
        depart_delay = float(t.get("departDelay"))
        if float(t.get("depart")) - depart_delay not in win:
            continue
        delay = float(t.get("timeLoss")) + depart_delay
        out[_arm_of_vehicle(vid)].append((delay, float(t.get("arrival")) >= 0))
    return out


def _backlog(path: Path, win: Window, end_time_s: float):
    """Waiting-to-enter delays of measured vehicles that never entered, and the backlog
    count per approach at the start and end of the window."""
    data = json.loads(path.read_text(encoding="utf-8"))
    waiting: dict[str, list[float]] = defaultdict(list)
    for v in data["never_inserted"]:
        if not v["id"].startswith(RUNOUT_PREFIX) and v["first_seen_s"] in win:
            waiting[_arm_of_vehicle(v["id"])].append(end_time_s - v["first_seen_s"])
    times = data["times_s"]

    def at(t: float, arm: str) -> int:
        idx = max((i for i, x in enumerate(times) if x <= t), default=None)
        return data["counts_by_arm"].get(arm, [])[idx] if idx is not None else 0

    ends = {arm: (at(win.start_s, arm), at(win.end_s, arm)) for arm in data["counts_by_arm"]}
    return waiting, ends


def _detector_intervals(path: Path, win: Window):
    for interval in ET.parse(path).getroot().iter("interval"):
        if float(interval.get("begin")) in win and float(interval.get("end")) <= win.end_s:
            yield interval


def seed_metrics(
    scenario: Scenario, seed_dir: Path, cfg: Config | None = None
) -> dict[str, SeedApproach]:
    cfg = cfg or get_config()
    win = measurement_window(cfg)
    run = json.loads((seed_dir / "run.json").read_text(encoding="utf-8"))
    trips = _delays_from_tripinfo(seed_dir / "tripinfo.xml", win)
    waiting, backlog_ends = _backlog(seed_dir / "backlog.json", win, run["end_time_s"])

    crossed: dict[str, int] = defaultdict(int)
    for iv in _detector_intervals(seed_dir / "e1.xml", win):
        crossed[_arm_of_lane(iv.get("id").removeprefix("e1_"))] += int(iv.get("nVehContrib"))
    queue_m: dict[str, float] = defaultdict(float)
    queue_veh: dict[str, float] = defaultdict(float)
    for iv in _detector_intervals(seed_dir / "e2.xml", win):
        arm = _arm_of_lane(iv.get("id").removeprefix("e2_"))
        queue_m[arm] = max(queue_m[arm], float(iv.get("maxJamLengthInMeters")))
        queue_veh[arm] = max(queue_veh[arm], float(iv.get("maxJamLengthInVehicles")))

    v_c = signal_v_c(scenario, cfg) if isinstance(scenario.control, FixedTimeSignal) else {}
    result = {}
    for arm in scenario.approaches:
        delays = [d for d, _ in trips[arm]] + waiting[arm]
        start, end = backlog_ends.get(arm, (0, 0))
        m = SeedApproach(
            arm=arm,
            vehicles=len(delays),
            finished=sum(f for _, f in trips[arm]),
            unfinished=sum(not f for _, f in trips[arm]),
            never_inserted=len(waiting[arm]),
            delay_s=statistics.fmean(delays) if delays else None,
            throughput_vph=crossed[arm] / win.hours,
            demand_vph=len(delays) / win.hours,
            queue_m=queue_m[arm],
            queue_veh=queue_veh[arm],
            backlog_start=start,
            backlog_end=end,
            v_c=v_c.get(arm),
        )
        m.los, m.f_reasons = grade(
            scenario, cfg, m.delay_s, m.throughput_vph, m.demand_vph, m.backlog_growth, m.v_c
        )
        result[arm] = m
    return result


# --------------------------------------------------------------------------- aggregation


@dataclass(frozen=True)
class Stat:
    mean: float
    min: float
    max: float

    @classmethod
    def of(cls, values: list[float]) -> Stat | None:
        values = [v for v in values if v is not None]
        if not values:
            return None
        return cls(statistics.fmean(values), min(values), max(values))


@dataclass
class ApproachSummary:
    arm: str
    delay_s: Stat | None
    queue_m: Stat
    queue_veh: Stat
    throughput_vph: Stat
    demand_vph: Stat
    los: str
    los_by_seed: list[str]
    f_reasons: list[str]
    v_c: float | None
    delay_lower_bound: bool
    spillback_seeds: list[int]

    @property
    def los_range(self) -> str:
        grades = [g for g in self.los_by_seed if g != "-"]
        if not grades:
            return "-"
        lo, hi = min(grades), max(grades)  # "A" < "F"
        return lo if lo == hi else f"{lo}-{hi}"


@dataclass
class JunctionSummary:
    delay_s: Stat | None
    throughput_vph: Stat
    los: str  # signalised and all-way stop only; "-" for a priority rule


@dataclass
class Summary:
    approaches: dict[str, ApproachSummary]
    junction: JunctionSummary
    seeds: list[int]
    warnings: list[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def summarize(scenario: Scenario, seed_dirs: dict[int, Path], cfg: Config | None = None) -> Summary:
    """Aggregate per-seed measures as mean and range. LOS is graded from the means."""
    cfg = cfg or get_config()
    per_seed = {seed: seed_metrics(scenario, d, cfg) for seed, d in seed_dirs.items()}
    seeds = list(per_seed)
    limit = cfg.geometry.queue_warning_share * scenario.approach_length_m
    warnings = []

    approaches = {}
    for arm in scenario.approaches:
        ms = [per_seed[s][arm] for s in seeds]
        delay = Stat.of([m.delay_s for m in ms])
        throughput = Stat.of([m.throughput_vph for m in ms])
        demand = Stat.of([m.demand_vph for m in ms])
        growth = statistics.fmean(m.backlog_growth for m in ms)
        los, reasons = grade(
            scenario,
            cfg,
            delay.mean if delay else None,
            throughput.mean,
            demand.mean,
            growth,
            ms[0].v_c,
        )
        spill = [s for s, m in zip(seeds, ms, strict=True) if m.queue_m >= limit]
        if spill:
            warnings.append(
                f"{arm}: queue reached {cfg.geometry.queue_warning_share:.0%} of the approach "
                f"length ({limit:.0f} m) in seed(s) {', '.join(map(str, spill))}"
            )
        lower = any(m.delay_lower_bound for m in ms)
        if lower:
            warnings.append(
                f"{arm}: some measured vehicles had not finished at the time cap; "
                "delay is a lower bound"
            )
        approaches[arm] = ApproachSummary(
            arm=arm,
            delay_s=delay,
            queue_m=Stat.of([m.queue_m for m in ms]),
            queue_veh=Stat.of([m.queue_veh for m in ms]),
            throughput_vph=throughput,
            demand_vph=demand,
            los=los,
            los_by_seed=[m.los for m in ms],
            f_reasons=reasons,
            v_c=ms[0].v_c,
            delay_lower_bound=lower,
            spillback_seeds=spill,
        )

    def junction_delay(seed: int) -> float | None:
        ms = per_seed[seed].values()
        n = sum(m.vehicles for m in ms)
        return sum((m.delay_s or 0) * m.vehicles for m in ms) / n if n else None

    j_delay = Stat.of([junction_delay(s) for s in seeds])
    j_los = "-"
    if j_delay and not isinstance(scenario.control, PriorityRule):
        table = (
            cfg.los.signalised
            if isinstance(scenario.control, FixedTimeSignal)
            else cfg.los.unsignalised
        )
        j_los = los_grade(j_delay.mean, table)
    junction = JunctionSummary(
        delay_s=j_delay,
        throughput_vph=Stat.of(
            [sum(m.throughput_vph for m in per_seed[s].values()) for s in seeds]
        ),
        los=j_los,
    )
    return Summary(approaches=approaches, junction=junction, seeds=seeds, warnings=warnings)
