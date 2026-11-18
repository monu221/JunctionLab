"""Free-flow delay baseline per approach and movement (docs/method.md, section 3).

`timeLoss` is measured against each driver's own desired speed, so even without any control
a vehicle loses some time: Krauss dawdling, and slowing down for the turn. That background
loss is measured on the scenario's own network with the junction made unregulated (no
signal, no right of way) and low, cars-only Poisson demand. Control delay is then total
delay minus the baseline of the vehicle's approach and movement.
"""

from __future__ import annotations

import json
import statistics
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from junctionlab import build
from junctionlab.config import Config, get_config
from junctionlab.metrics import Baseline
from junctionlab.scenario import MOVEMENTS, Scenario, destination
from junctionlab.webster import run_sumo

BASELINE_FILE = "baseline.json"


def baseline_movements(scenario: Scenario) -> dict[str, list[str]]:
    """Movements that carry traffic: general traffic with a share > 0, plus bus movements."""
    result = {}
    for arm, a in scenario.approaches.items():
        moves = {m for m, share in a.turning_shares_pct.items() if share > 0}
        if a.bus_demand_bph > 0:
            if a.bus_right_share_pct < 100:
                moves.add("through")
            if a.bus_right_share_pct > 0:
                moves.add("right")
        result[arm] = [m for m in MOVEMENTS if m in moves]
    return result


def write_baseline_routes(scenario: Scenario, out_dir: Path, cfg: Config) -> Path:
    root = ET.Element("routes")
    ET.SubElement(
        root,
        "vType",
        id="car",
        vClass=cfg.vehicles.car_vclass,
        carFollowModel=cfg.driver.car_follow_model,
        sigma=f"{cfg.driver.sigma:g}",
    )
    rate = cfg.baseline.demand_vph_per_movement / 3600
    for arm, moves in baseline_movements(scenario).items():
        for movement in moves:
            flow = ET.SubElement(
                root,
                "flow",
                id=f"{arm}_{movement}_car",
                type="car",
                begin="0",
                end=str(cfg.baseline.duration_s),
                period=f"exp({rate:.12g})",
                to=build.out_edge(destination(arm, movement)),
                departLane="best",
                departSpeed="desired",
            )
            flow.set("from", build.in_edge(arm))
    ET.indent(root)
    path = out_dir / "baseline.rou.xml"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def measure_baseline(
    scenario: Scenario, out_dir: Path, config: Config | None = None, seeds: list[int] | None = None
) -> Baseline:
    """Mean timeLoss + departDelay per approach and movement over all seeds."""
    cfg = config or get_config()
    seeds = list(seeds or cfg.simulation.seeds)
    out_dir.mkdir(parents=True, exist_ok=True)
    net = build.build_network(scenario, out_dir, free_flow=True)
    routes = write_baseline_routes(scenario, out_dir, cfg)
    end = cfg.baseline.duration_s + 600  # enough for the last vehicles to leave

    def one(seed: int) -> list[tuple[str, str, float]]:
        seed_dir = out_dir / f"seed_{seed}"
        collisions = seed_dir / "collisions.xml"
        d = run_sumo(
            net, routes, [], seed_dir, seed, cfg, end_s=end,
            extra_args=["--collision-output", str(collisions)],
        )  # fmt: skip
        # Streams can touch where they merge on the unregulated junction; vehicles involved
        # may brake hard, so they are left out of the baseline.
        involved = {
            v
            for c in ET.parse(collisions).getroot().iter("collision")
            for v in (c.get("collider"), c.get("victim"))
        }
        out = []
        for t in ET.parse(d / "tripinfo.xml").getroot().iter("tripinfo"):
            if float(t.get("arrival")) < 0 or t.get("id") in involved:
                continue
            arm, movement, _ = t.get("id").split(".")[0].split("_")
            out.append((arm, movement, float(t.get("timeLoss")) + float(t.get("departDelay"))))
        return out

    with ThreadPoolExecutor(max_workers=len(seeds)) as pool:
        rows = [row for result in pool.map(one, seeds) for row in result]

    baseline: Baseline = {}
    for arm, moves in baseline_movements(scenario).items():
        baseline[arm] = {}
        for movement in moves:
            values = [d for a, m, d in rows if a == arm and m == movement]
            baseline[arm][movement] = statistics.fmean(values) if values else 0.0
    (out_dir / BASELINE_FILE).write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    return baseline


def by_movement(baseline: Baseline) -> dict[str, float]:
    """Mean baseline per movement over all approaches (for reporting)."""
    groups: dict[str, list[float]] = {}
    for moves in baseline.values():
        for movement, value in moves.items():
            groups.setdefault(movement, []).append(value)
    return {m: statistics.fmean(groups[m]) for m in MOVEMENTS if m in groups}
