"""Plausibility check against Webster's delay formula (docs/method.md, section 4).

Network: one straight approach (one lane, cars only) into a fixed-time signal, then a short
exit. The signal shows green, amber, all-red, then red for the rest of the cycle (the
"other phase"). No SUMO parameter is tuned here.

1. Saturation flow and lost time come from a saturated run: demand far above capacity, so
   a queue always waits at the stop line. An instant loop 1 m before the stop line logs each
   vehicle's crossing time.
   - Saturation headway: mean headway from the 5th queued vehicle of each green onwards
     (the first four carry the start-up loss), up to the end of the displayed green.
   - Saturation flow s = 3600 / saturation headway.
   - Vehicles per cycle N: all crossings from green start until the next red phase.
   - Lost time per cycle l = (green + amber) - N x headway.
     Effective green = green + amber - l, green ratio lambda = effective green / cycle.
2. Test cases: Poisson demand q = x * s * lambda; simulated delay (timeLoss + departDelay,
   measured vehicles as in section 3) is compared with Webster's formula.
3. A free-flow run (no signal) shows the background timeLoss, which should be close to zero.
"""

from __future__ import annotations

import json
import statistics
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from junctionlab import sumo_env
from junctionlab.config import REPO_ROOT, Config, get_config
from junctionlab.metrics import Stat, measurement_window

EXIT_LENGTH_M = 100.0
LOOP_OFFSET_M = 1.0
FIRST_SATURATED_POSITION = 5  # headways from the 5th queued vehicle on (HCM practice)
SATURATED_DEMAND_VPH = 3600  # far above capacity: the queue never clears


# --------------------------------------------------------------------------- formula


def webster_delay(cycle_s: float, green_ratio: float, q_vph: float, x: float) -> float:
    """Webster's average delay per vehicle (s) for an undersaturated fixed-time approach."""
    c, lam, q = cycle_s, green_ratio, q_vph / 3600
    uniform = c * (1 - lam) ** 2 / (2 * (1 - lam * x))
    random = x**2 / (2 * q * (1 - x))
    correction = 0.65 * (c / q**2) ** (1 / 3) * x ** (2 + 5 * lam)
    return uniform + random - correction


# --------------------------------------------------------------------------- SUMO files


def _write(root: ET.Element, path: Path) -> Path:
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def build_corridor(out_dir: Path, cfg: Config, signal: bool) -> Path:
    """One-lane approach into node C (signal or plain priority node) and a short exit."""
    out_dir.mkdir(parents=True, exist_ok=True)
    length = cfg.geometry.approach_length_m
    speed = cfg.geometry.speed_limit_kmh / 3.6
    nodes = ET.Element("nodes")
    ET.SubElement(nodes, "node", id="W", x=f"{-length:.2f}", y="0")
    ET.SubElement(
        nodes, "node", id="C", x="0", y="0", type="traffic_light" if signal else "priority"
    )
    ET.SubElement(nodes, "node", id="E", x=f"{EXIT_LENGTH_M:.2f}", y="0")
    edges = ET.Element("edges")
    for eid, a, b in (("in", "W", "C"), ("out", "C", "E")):
        e = ET.SubElement(edges, "edge", id=eid, to=b, numLanes="1", speed=f"{speed:.4f}")
        e.set("from", a)
    _write(nodes, out_dir / "corridor.nod.xml")
    _write(edges, out_dir / "corridor.edg.xml")
    net = out_dir / "corridor.net.xml"
    sumo_env.run(
        "netconvert",
        ["-n", "corridor.nod.xml", "-e", "corridor.edg.xml", "-o", net.name,
         "--no-turnarounds", "true"],
        cwd=out_dir,
    )  # fmt: skip
    return net


def write_signal(out_dir: Path, cfg: Config) -> Path:
    w = cfg.webster
    c = cfg.control
    red = w.cycle_s - w.green_s - c.amber_s - c.all_red_s
    root = ET.Element("additional")
    tl = ET.SubElement(root, "tlLogic", id="C", type="static", programID="webster", offset="0")
    for duration, state in ((w.green_s, "G"), (c.amber_s, "y"), (c.all_red_s, "r"), (red, "r")):
        ET.SubElement(tl, "phase", duration=f"{duration:g}", state=state)
    return _write(root, out_dir / "signal.add.xml")


def write_demand(out_dir: Path, cfg: Config, q_vph: float, poisson: bool) -> Path:
    root = ET.Element("routes")
    ET.SubElement(
        root,
        "vType",
        id="car",
        vClass=cfg.vehicles.car_vclass,
        carFollowModel=cfg.driver.car_follow_model,
        sigma=f"{cfg.driver.sigma:g}",
    )
    period = f"exp({q_vph / 3600:.12g})" if poisson else f"{3600 / q_vph:.12g}"
    flow = ET.SubElement(
        root,
        "flow",
        id="f",
        type="car",
        begin="0",
        end=str(cfg.simulation.end_s),
        period=period,
        departLane="best",
        departSpeed="desired",
    )
    flow.set("from", "in")
    flow.set("to", "out")
    return _write(root, out_dir / "demand.rou.xml")


def write_stop_line_loop(out_dir: Path, net: Path) -> Path:
    import sumolib

    lane = sumolib.net.readNet(str(net)).getEdge("in").getLanes()[0]
    root = ET.Element("additional")
    ET.SubElement(
        root,
        "instantInductionLoop",
        id="stop",
        lane=lane.getID(),
        pos=f"{lane.getLength() - LOOP_OFFSET_M:.2f}",
        file="stopline.xml",
    )
    return _write(root, out_dir / "loop.add.xml")


def run_sumo(
    net: Path,
    routes: Path,
    additional: list[Path],
    seed_dir: Path,
    seed: int,
    cfg: Config,
    end_s: float | None = None,
    extra_args: list[str] | None = None,
) -> Path:
    """Plain headless run (no TraCI) up to end_s (default: the configured time cap)."""
    seed_dir.mkdir(parents=True, exist_ok=True)
    sim = cfg.simulation
    args = [
        "--net-file", str(net),
        "--route-files", str(routes),
        "--seed", str(seed),
        "--begin", "0",
        "--end", f"{end_s if end_s is not None else sim.end_s:g}",
        "--step-length", f"{sim.step_length_s:g}",
        "--time-to-teleport", str(sim.time_to_teleport_s),
        "--collision.action", "warn",
        "--tripinfo-output", str(seed_dir / "tripinfo.xml"),
        "--tripinfo-output.write-unfinished", "true",
        "--no-step-log", "true",
        "--duration-log.disable", "true",
        "--log", str(seed_dir / "sumo.log"),
    ]  # fmt: skip
    if additional:
        args += ["--additional-files", ",".join(str(a) for a in additional)]
    args += extra_args or []
    # Detector output paths are relative to the additional file, so copy it per seed.
    sumo_env.run("sumo", args, cwd=seed_dir)
    return seed_dir


# --------------------------------------------------------------------------- saturation flow


@dataclass(frozen=True)
class Saturation:
    saturation_flow_vph: float
    headway_s: float
    lost_time_s: float
    vehicles_per_cycle: float
    cycles: int
    headways: int
    by_seed_vph: list[float]


def saturation_from_crossings(crossings: list[float], cfg: Config) -> tuple[list[float], list[int]]:
    """Saturation headways and vehicles per cycle from stop-line crossing times."""
    w, c = cfg.webster, cfg.control
    win = measurement_window(cfg)
    headways: list[float] = []
    per_cycle: list[int] = []
    k = 0
    while (start := k * w.cycle_s) + w.cycle_s <= win.end_s:
        k += 1
        if start < win.start_s:
            continue
        green_end = start + w.green_s
        released = [t for t in crossings if start <= t < green_end + c.amber_s + c.all_red_s]
        per_cycle.append(len(released))
        in_green = [t for t in released if t < green_end]
        for i in range(FIRST_SATURATED_POSITION - 1, len(in_green)):
            headways.append(in_green[i] - in_green[i - 1])
    return headways, per_cycle


def _crossings(path: Path) -> list[float]:
    return sorted(
        float(e.get("time"))
        for e in ET.parse(path).getroot().iter("instantOut")
        if e.get("state") == "enter"
    )


def measure_saturation(out_dir: Path, cfg: Config, seeds: list[int]) -> Saturation:
    net = build_corridor(out_dir, cfg, signal=True)
    signal = write_signal(out_dir, cfg)
    routes = write_demand(out_dir, cfg, SATURATED_DEMAND_VPH, poisson=False)

    def one(seed: int) -> tuple[list[float], list[int]]:
        seed_dir = out_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        loop = write_stop_line_loop(seed_dir, net)
        run_sumo(net, routes, [signal, loop], seed_dir, seed, cfg)
        return saturation_from_crossings(_crossings(seed_dir / "stopline.xml"), cfg)

    with ThreadPoolExecutor(max_workers=len(seeds)) as pool:
        results = list(pool.map(one, seeds))
    headways = [h for hs, _ in results for h in hs]
    per_cycle = [n for _, ns in results for n in ns]
    h = statistics.fmean(headways)
    n = statistics.fmean(per_cycle)
    w, c = cfg.webster, cfg.control
    return Saturation(
        saturation_flow_vph=3600 / h,
        headway_s=h,
        lost_time_s=w.green_s + c.amber_s - n * h,
        vehicles_per_cycle=n,
        cycles=len(per_cycle),
        headways=len(headways),
        by_seed_vph=[3600 / statistics.fmean(hs) for hs, _ in results],
    )


# --------------------------------------------------------------------------- test cases


def measured_delays(tripinfo: Path, cfg: Config) -> list[float]:
    """timeLoss + departDelay of vehicles whose intended departure is in the window."""
    win = measurement_window(cfg)
    out = []
    for t in ET.parse(tripinfo).getroot().iter("tripinfo"):
        dd = float(t.get("departDelay"))
        if float(t.get("depart")) - dd in win:
            out.append(float(t.get("timeLoss")) + dd)
    return out


def run_demand(
    out_dir: Path, cfg: Config, seeds: list[int], q_vph: float, signal: bool
) -> list[float]:
    """Mean measured delay per seed for Poisson demand q_vph."""
    net = build_corridor(out_dir, cfg, signal=signal)
    additional = [write_signal(out_dir, cfg)] if signal else []
    routes = write_demand(out_dir, cfg, q_vph, poisson=True)

    def one(seed: int) -> float:
        d = run_sumo(net, routes, additional, out_dir / f"seed_{seed}", seed, cfg)
        return statistics.fmean(measured_delays(d / "tripinfo.xml", cfg))

    with ThreadPoolExecutor(max_workers=len(seeds)) as pool:
        return list(pool.map(one, seeds))


@dataclass(frozen=True)
class Case:
    x: float
    q_vph: float
    green_ratio: float
    webster_s: float
    simulated_s: Stat  # control delay = total - free-flow baseline (compared with Webster)
    simulated_total_s: Stat
    diff_s: float
    diff_pct: float
    tested: bool
    passed: bool | None  # None for information-only cases


@dataclass(frozen=True)
class WebsterReport:
    saturation: Saturation
    cycle_s: float
    green_s: float
    effective_green_s: float
    green_ratio: float
    free_flow_q_vph: float
    free_flow_delay_s: Stat
    cases: list[Case]
    margin_relative: float
    margin_absolute_s: float

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.cases if c.tested)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def run_check(out_dir: Path | None = None, config: Config | None = None) -> WebsterReport:
    cfg = config or get_config()
    w, c = cfg.webster, cfg.control
    seeds = list(cfg.simulation.seeds)
    out_dir = Path(out_dir or REPO_ROOT / "runs" / f"{datetime.now():%Y%m%d-%H%M%S}_webster")

    sat = measure_saturation(out_dir / "saturated", cfg, seeds)
    effective_green = w.green_s + c.amber_s - sat.lost_time_s
    lam = effective_green / w.cycle_s
    capacity = sat.saturation_flow_vph * lam

    # Free-flow baseline first: same corridor, no signal, low demand.
    free_q = cfg.baseline.demand_vph_per_movement
    free = Stat.of(run_demand(out_dir / "free_flow", cfg, seeds, free_q, signal=False))

    cases = []
    for x, tested in [(x, True) for x in w.x_tested] + [(x, False) for x in w.x_info_only]:
        q = x * capacity
        total = run_demand(out_dir / f"x_{x:g}", cfg, seeds, q, signal=True)
        sim = Stat.of([d - free.mean for d in total])
        ref = webster_delay(w.cycle_s, lam, q, x)
        diff = sim.mean - ref
        margin = max(w.margin_relative * ref, w.margin_absolute_s)
        cases.append(
            Case(
                x=x,
                q_vph=q,
                green_ratio=lam,
                webster_s=ref,
                simulated_s=sim,
                simulated_total_s=Stat.of(total),
                diff_s=diff,
                diff_pct=100 * diff / ref,
                tested=tested,
                passed=abs(diff) <= margin if tested else None,
            )
        )

    report = WebsterReport(
        saturation=sat,
        cycle_s=w.cycle_s,
        green_s=w.green_s,
        effective_green_s=effective_green,
        green_ratio=lam,
        free_flow_q_vph=free_q,
        free_flow_delay_s=free,
        cases=cases,
        margin_relative=w.margin_relative,
        margin_absolute_s=w.margin_absolute_s,
    )
    (out_dir / "webster.json").write_text(report.to_json(), encoding="utf-8")
    return report
