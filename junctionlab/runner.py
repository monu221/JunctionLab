"""Run a scenario in SUMO: build files once, then one headless run per seed (method §2).

Each run is driven through TraCI in steps of CHECK_INTERVAL_S so that it can
- warn about gridlock: any vehicle stopped longer than `gridlock_stop_s`;
- stop as soon as every vehicle from the warm-up and measurement blocks has left (run-out
  vehicles, ids starting with "ro_", don't count), or at the hard time cap at the latest.
Teleports are disabled. Collisions are logged (`--collision.action warn`) and counted.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import traci

from junctionlab import build, sumo_env
from junctionlab.baseline import measure_baseline
from junctionlab.build import RUNOUT_PREFIX
from junctionlab.config import REPO_ROOT, Config, get_config
from junctionlab.metrics import Summary, summarize
from junctionlab.scenario import FixedTimeSignal, Scenario

CHECK_INTERVAL_S = 10
_TRACI_START_LOCK = threading.Lock()  # free-port selection in traci.start is not thread-safe


@dataclass(frozen=True)
class GridlockEvent:
    vehicle: str
    lane: str
    detected_at_s: float
    stopped_for_s: float


@dataclass
class SeedResult:
    seed: int
    dir: Path
    ended_by: str  # "measured_vehicles_left" or "time_cap"
    end_time_s: float
    measured_vehicles_remaining: int
    gridlock: list[GridlockEvent] = field(default_factory=list)
    collisions: int = 0
    teleports: int = 0
    vehicles_inserted: int = 0
    vehicles_arrived: int = 0
    wall_time_s: float = 0.0

    @property
    def tripinfo(self) -> Path:
        return self.dir / "tripinfo.xml"

    @property
    def e2(self) -> Path:
        return self.dir / "e2.xml"

    @property
    def e1(self) -> Path:
        return self.dir / "e1.xml"

    @property
    def backlog(self) -> Path:
        return self.dir / "backlog.json"

    @property
    def warnings(self) -> list[str]:
        w = []
        if self.gridlock:
            w.append(f"gridlock: {len(self.gridlock)} vehicle(s) stopped too long")
        if self.ended_by == "time_cap":
            w.append(
                f"time cap reached with {self.measured_vehicles_remaining} measured vehicle(s) "
                "still in or waiting to enter the network"
            )
        if self.collisions:
            w.append(f"{self.collisions} collision(s)")
        if self.teleports:
            w.append(f"{self.teleports} teleport(s)")
        return w


@dataclass
class RunResult:
    scenario: Scenario
    dir: Path
    net: Path
    seeds: list[SeedResult]
    summary: Summary


# --------------------------------------------------------------------------- preparation


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "scenario"


def default_run_dir(scenario: Scenario) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return REPO_ROOT / "runs" / f"{stamp}_{_slug(scenario.name)}"


def prepare(scenario: Scenario, run_dir: Path, config: Config | None = None) -> dict[str, Path]:
    """Write all seed-independent SUMO files into run_dir."""
    cfg = config or get_config()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "scenario.json").write_text(scenario.model_dump_json(indent=2), encoding="utf-8")
    files = {"net": build.build_network(scenario, run_dir)}
    files["routes"] = build.write_routes(scenario, run_dir, cfg)
    if isinstance(scenario.control, FixedTimeSignal):
        files["signal"] = build.write_signal_program(scenario, files["net"], run_dir, cfg)
    return files


# --------------------------------------------------------------------------- one seed


@dataclass
class Backlog:
    """Vehicles waiting to be inserted (they are not in tripinfo until they enter).

    Records the count per approach at every check, and when each waiting vehicle was first
    seen, which approximates its intended departure (to within one check interval).
    """

    interval_s: float
    arms: list[str]
    times: list[float] = field(default_factory=list)
    counts: dict[str, list[int]] = field(default_factory=dict)
    first_seen: dict[str, float] = field(default_factory=dict)

    def record(self, t: float, pending: tuple[str, ...] | list[str]) -> None:
        self.times.append(t)
        per_arm = dict.fromkeys(self.arms, 0)
        for vid in pending:
            self.first_seen.setdefault(vid, t)
            arm = vid.removeprefix(RUNOUT_PREFIX).split("_", 1)[0]
            per_arm[arm] = per_arm.get(arm, 0) + 1
        for arm, n in per_arm.items():
            self.counts.setdefault(arm, []).append(n)

    def save(self, path: Path, still_pending: tuple[str, ...] | list[str]) -> None:
        data = {
            "interval_s": self.interval_s,
            "times_s": self.times,
            "counts_by_arm": self.counts,
            "never_inserted": [
                {"id": vid, "first_seen_s": self.first_seen.get(vid, self.times[-1])}
                for vid in still_pending
            ],
        }
        path.write_text(json.dumps(data), encoding="utf-8")


def _sumo_command(files: dict[str, Path], seed_dir: Path, seed: int, cfg: Config) -> list[str]:
    sim = cfg.simulation
    additional = [str(seed_dir / "detectors.add.xml")]
    if "signal" in files:
        additional.insert(0, str(files["signal"]))
    return [
        str(sumo_env.binary("sumo")),
        "--net-file", str(files["net"]),
        "--route-files", str(files["routes"]),
        "--additional-files", ",".join(additional),
        "--seed", str(seed),
        "--begin", "0",
        "--end", str(sim.end_s),
        "--step-length", f"{sim.step_length_s:g}",
        "--time-to-teleport", str(sim.time_to_teleport_s),
        "--collision.action", "warn",
        "--tripinfo-output", str(seed_dir / "tripinfo.xml"),
        "--tripinfo-output.write-unfinished", "true",
        "--statistic-output", str(seed_dir / "statistics.xml"),
        "--log", str(seed_dir / "sumo.log"),
        "--no-step-log", "true",
        "--duration-log.disable", "true",
    ]  # fmt: skip


def _is_measured(vehicle_id: str) -> bool:
    return not vehicle_id.startswith(RUNOUT_PREFIX)


def run_seed(
    scenario: Scenario, files: dict[str, Path], run_dir: Path, seed: int, cfg: Config
) -> SeedResult:
    sim = cfg.simulation
    seed_dir = run_dir / f"seed_{seed}"
    seed_dir.mkdir(exist_ok=True)
    build.write_detectors(scenario, files["net"], seed_dir)

    label = f"jl-{seed}-{uuid.uuid4().hex[:8]}"
    started = time.perf_counter()
    with _TRACI_START_LOCK:
        traci.start(_sumo_command(files, seed_dir, seed, cfg), label=label)
    conn = traci.getConnection(label)

    window_end = sim.warmup_s + sim.measurement_s
    flagged: dict[str, GridlockEvent] = {}
    backlog = Backlog(interval_s=CHECK_INTERVAL_S, arms=list(scenario.approaches))
    t = 0.0
    ended_by = "time_cap"
    remaining = 0
    try:
        while True:
            t = min(t + CHECK_INTERVAL_S, sim.end_s)
            conn.simulationStep(t)
            running = conn.vehicle.getIDList()
            pending = conn.simulation.getPendingVehicles()
            backlog.record(t, pending)
            for vid in running:
                if vid in flagged:
                    continue
                stopped = conn.vehicle.getWaitingTime(vid)
                if stopped > sim.gridlock_stop_s:
                    flagged[vid] = GridlockEvent(vid, conn.vehicle.getLaneID(vid), t, stopped)
            if t >= window_end:
                remaining = sum(map(_is_measured, running)) + sum(map(_is_measured, pending))
                if remaining == 0:
                    ended_by = "measured_vehicles_left"
                    break
            if t >= sim.end_s:
                break
    finally:
        conn.close()

    result = SeedResult(
        seed=seed,
        dir=seed_dir,
        ended_by=ended_by,
        end_time_s=t,
        measured_vehicles_remaining=remaining,
        gridlock=list(flagged.values()),
        wall_time_s=round(time.perf_counter() - started, 2),
    )
    _read_statistics(result)
    backlog.save(seed_dir / "backlog.json", still_pending=pending)
    summary = asdict(result) | {"dir": str(seed_dir), "warnings": result.warnings}
    (seed_dir / "run.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return result


def _read_statistics(result: SeedResult) -> None:
    root = ET.parse(result.dir / "statistics.xml").getroot()
    if (safety := root.find("safety")) is not None:
        result.collisions = int(safety.get("collisions", 0))
    if (teleports := root.find("teleports")) is not None:
        result.teleports = int(teleports.get("total", 0))
    if (vehicles := root.find("vehicles")) is not None:
        result.vehicles_inserted = int(vehicles.get("inserted", 0))
    trips = ET.parse(result.tripinfo).getroot().iter("tripinfo")
    result.vehicles_arrived = sum(float(t.get("arrival", -1)) >= 0 for t in trips)


# --------------------------------------------------------------------------- scenario


def run_scenario(
    scenario: Scenario,
    run_dir: Path | None = None,
    seeds: list[int] | None = None,
    config: Config | None = None,
    parallel: bool = True,
    on_progress: Callable[[str, int | None], None] | None = None,
    max_workers: int | None = None,
) -> RunResult:
    """Build the SUMO files and run every seed. Results land in run_dir/seed_<n>/.

    `on_progress(step, seed)` is called with step "baseline", "seed_started", "seed_done" or
    "summary". Seeds run in worker threads, so the callback must be thread-safe. `max_workers`
    limits how many SUMO processes run at once (default: one per seed); results do not depend
    on it, because each seed is its own run.

    Raises UnconfirmedDemand if the model's reading of the demand is not confirmed yet.
    """
    report = on_progress or (lambda _step, _seed: None)
    scenario.require_confirmed()
    cfg = config or get_config()
    run_dir = Path(run_dir or default_run_dir(scenario))
    seeds = list(seeds or cfg.simulation.seeds)
    report("baseline", None)
    files = prepare(scenario, run_dir, cfg)
    workers = min(max_workers or len(seeds), len(seeds))
    baseline = measure_baseline(scenario, run_dir / "free_flow", cfg, seeds, workers)

    def one(seed: int) -> SeedResult:
        report("seed_started", seed)
        result = run_seed(scenario, files, run_dir, seed, cfg)
        report("seed_done", seed)
        return result

    if parallel and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(one, seeds))
    else:
        results = [one(s) for s in seeds]
    report("summary", None)
    summary = summarize(scenario, {r.seed: r.dir for r in results}, cfg, baseline)
    (run_dir / "summary.json").write_text(summary.to_json(), encoding="utf-8")
    return RunResult(
        scenario=scenario, dir=run_dir, net=files["net"], seeds=results, summary=summary
    )
