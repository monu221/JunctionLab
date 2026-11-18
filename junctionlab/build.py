"""Scenario -> SUMO files (docs/method.md, section 2).

Layout: the junction node "C" sits at the origin, each arm ends at a node `approach_length_m`
away. Each arm has an inbound edge `<arm>_in` and an exit edge `<arm>_out` with the same lane
count. SUMO numbers lanes from the right (index 0 = rightmost); a bus lane is index 0.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import sumolib

from junctionlab import sumo_env
from junctionlab.config import Config, get_config
from junctionlab.scenario import (
    ARMS,
    MOVEMENTS,
    AllWayStop,
    Arm,
    FixedTimeSignal,
    Movement,
    PriorityRule,
    Scenario,
    destination,
    opposite,
    road_of,
)
from junctionlab.signal import phase_arms

JUNCTION_ID = "C"
TLS_PROGRAM_ID = "junctionlab"
_DIRECTION = {"north": (0, 1), "east": (1, 0), "south": (0, -1), "west": (-1, 0)}
_SUMO_MOVEMENT = {"l": "left", "s": "through", "r": "right"}


def in_edge(arm: Arm) -> str:
    return f"{arm}_in"


def out_edge(arm: Arm) -> str:
    return f"{arm}_out"


def arm_of(edge_id: str) -> Arm:
    return edge_id.rsplit("_", 1)[0]  # type: ignore[return-value]


def sumo_lane_index(scenario: Scenario, arm: Arm, general_lane: int) -> int:
    """SUMO index of a general lane given in left-to-right order (0 = leftmost)."""
    a = scenario.approaches[arm]
    return (len(a.lanes) - 1 - general_lane) + int(a.bus_lane)


@dataclass(frozen=True)
class Connection:
    from_arm: Arm
    from_lane: int  # SUMO index
    to_arm: Arm
    to_lane: int  # SUMO index
    movement: Movement
    bus_lane: bool = False


def connections(scenario: Scenario) -> list[Connection]:
    """Lane-to-lane connections. Turning lanes keep to the matching side of the exit leg."""
    result: list[Connection] = []
    for arm, a in scenario.approaches.items():
        for movement in MOVEMENTS:
            serving = [
                sumo_lane_index(scenario, arm, i)
                for i, lane in enumerate(a.lanes)
                if movement in lane
            ]
            if not serving:
                continue
            to_arm = destination(arm, movement)
            n_out = scenario.approaches[to_arm].lane_count
            if movement == "left":  # leftmost serving lane -> leftmost exit lane
                ordered = sorted(serving, reverse=True)
                targets = [max(n_out - 1 - k, 0) for k in range(len(ordered))]
            elif movement == "right":  # rightmost serving lane -> rightmost exit lane
                ordered = sorted(serving)
                targets = [min(k, n_out - 1) for k in range(len(ordered))]
            else:  # through: keep the lane position among the general lanes
                ordered = sorted(serving)
                targets = [min(i - int(a.bus_lane), n_out - 1) for i in ordered]
            result += [
                Connection(arm, f, to_arm, t, movement)
                for f, t in zip(ordered, targets, strict=True)
            ]
        if a.bus_lane:
            for movement in ("through", "right"):
                to_arm = destination(arm, movement)
                if to_arm in scenario.approaches:
                    result.append(Connection(arm, 0, to_arm, 0, movement, bus_lane=True))
    return result


# --------------------------------------------------------------------------- plain XML files


def _write(root: ET.Element, path: Path) -> Path:
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def _node_type(scenario: Scenario) -> str:
    c = scenario.control
    if isinstance(c, FixedTimeSignal):
        return "traffic_light"
    if isinstance(c, AllWayStop):
        return "allway_stop"
    return "priority"


def _edge_priority(scenario: Scenario, arm: Arm) -> int:
    c = scenario.control
    if isinstance(c, PriorityRule):
        return 2 if road_of(arm) == c.major_road else 1
    return 1


def write_plain_network(
    scenario: Scenario, out_dir: Path, extra_m: dict[str, float] | None = None
) -> tuple[Path, Path, Path]:
    """Plain XML files. `extra_m` lengthens arms so lanes outside the junction area
    reach the full approach length."""
    speed = scenario.speed_limit_kmh / 3.6

    nodes = ET.Element("nodes")
    ET.SubElement(nodes, "node", id=JUNCTION_ID, x="0", y="0", type=_node_type(scenario))
    for arm in scenario.approaches:
        dx, dy = _DIRECTION[arm]
        length = scenario.approach_length_m + (extra_m or {}).get(arm, 0.0)
        ET.SubElement(nodes, "node", id=arm, x=f"{dx * length:.2f}", y=f"{dy * length:.2f}")

    edges = ET.Element("edges")
    for arm, a in scenario.approaches.items():
        common = {
            "numLanes": str(a.lane_count),
            "speed": f"{speed:.4f}",
            "priority": str(_edge_priority(scenario, arm)),
        }
        inbound = ET.SubElement(edges, "edge", id=in_edge(arm), to=JUNCTION_ID, **common)
        inbound.set("from", arm)
        if a.bus_lane:
            ET.SubElement(inbound, "lane", index="0", allow="bus")
        outbound = ET.SubElement(edges, "edge", id=out_edge(arm), to=arm, **common)
        outbound.set("from", JUNCTION_ID)

    cons = ET.Element("connections")
    for c in connections(scenario):
        el = ET.SubElement(cons, "connection", to=out_edge(c.to_arm))
        el.set("from", in_edge(c.from_arm))
        el.set("fromLane", str(c.from_lane))
        el.set("toLane", str(c.to_lane))

    return (
        _write(nodes, out_dir / "junction.nod.xml"),
        _write(edges, out_dir / "junction.edg.xml"),
        _write(cons, out_dir / "junction.con.xml"),
    )


def build_network(scenario: Scenario, out_dir: Path) -> Path:
    """Write plain XML and run netconvert. Returns the .net.xml path.

    Built twice: the junction area shortens each edge, so the second pass moves the arm end
    nodes out until every inbound edge is `approach_length_m` long.
    """
    net = out_dir / "junction.net.xml"
    extra: dict[str, float] = {}
    for _ in range(2):
        nod, edg, con = write_plain_network(scenario, out_dir, extra)
        sumo_env.run(
            "netconvert",
            [
                "--node-files", nod.name,
                "--edge-files", edg.name,
                "--connection-files", con.name,
                "--output-file", net.name,
                "--no-turnarounds", "true",
            ],
            cwd=out_dir,
        )  # fmt: skip
        built = sumolib.net.readNet(str(net), withPrograms=False)
        extra = {
            arm: extra.get(arm, 0.0)
            + scenario.approach_length_m
            - built.getEdge(in_edge(arm)).getLength()
            for arm in scenario.approaches
        }
    return net


# --------------------------------------------------------------------------- signal program


def signal_program(
    scenario: Scenario, net_path: Path, config: Config | None = None
) -> list[tuple[float, str]]:
    """(duration, state) per phase step: green, amber, all-red for each phase in order.

    In a green phase, links from served approaches get 'G', except left turns facing a served
    opposite approach, which are permitted ('g', must yield). Protected lefts are not in v1.
    """
    cfg = config or get_config()
    c = scenario.control
    assert isinstance(c, FixedTimeSignal) and c.green_s is not None
    links = tls_links(net_path)
    n_links = len(links)

    steps: list[tuple[float, str]] = []
    for arms, green in zip(phase_arms(scenario), c.green_s, strict=True):
        state = []
        for i in range(n_links):
            link = links[i]
            if link.from_arm not in arms:
                state.append("r")
            elif link.movement == "left" and opposite(link.from_arm) in arms:
                state.append("g")
            else:
                state.append("G")
        # Two 'G' links into the same exit lane would be unsafe (e.g. a bus lane merging
        # with the general lanes): the lower-ranked link yields.
        by_target: dict[str, list[int]] = {}
        for i in range(n_links):
            if state[i] == "G":
                by_target.setdefault(links[i].to_lane, []).append(i)
        for indices in by_target.values():
            for i in sorted(indices, key=lambda i: links[i].rank)[1:]:
                state[i] = "g"
        green_state = "".join(state)
        amber_state = "".join("y" if s in "Gg" else "r" for s in green_state)
        steps.append((green, green_state))
        steps.append((cfg.control.amber_s, amber_state))
        if cfg.control.all_red_s > 0:
            steps.append((cfg.control.all_red_s, "r" * n_links))
    return steps


@dataclass(frozen=True)
class TlsLink:
    from_arm: Arm
    movement: Movement
    to_lane: str
    bus_lane: bool

    @property
    def rank(self) -> int:
        """Which link keeps 'G' when several share an exit lane (lowest wins)."""
        if self.bus_lane:
            return 2
        return {"through": 0, "right": 1, "left": 3}[self.movement]


def tls_links(net_path: Path) -> dict[int, TlsLink]:
    """Signal link index -> link description, read from the built network."""
    net = sumolib.net.readNet(str(net_path), withPrograms=False)
    links: dict[int, TlsLink] = {}
    for arm in ARMS:
        if not net.hasEdge(in_edge(arm)):
            continue
        for conns in net.getEdge(in_edge(arm)).getOutgoing().values():
            for con in conns:
                to_arm = arm_of(con.getTo().getID())
                movement = next(m for m in MOVEMENTS if destination(arm, m) == to_arm)
                bus = con.getFromLane().allows("bus") and not con.getFromLane().allows("passenger")
                links[con.getTLLinkIndex()] = TlsLink(arm, movement, con.getToLane().getID(), bus)
    return dict(sorted(links.items()))


def write_signal_program(
    scenario: Scenario, net_path: Path, out_dir: Path, config: Config | None = None
) -> Path:
    root = ET.Element("additional")
    tl = ET.SubElement(
        root, "tlLogic", id=JUNCTION_ID, type="static", programID=TLS_PROGRAM_ID, offset="0"
    )
    for duration, state in signal_program(scenario, net_path, config):
        ET.SubElement(tl, "phase", duration=f"{duration:g}", state=state)
    return _write(root, out_dir / "signal.add.xml")


# --------------------------------------------------------------------------- demand


def flows(scenario: Scenario, config: Config | None = None) -> list[dict[str, str]]:
    """Poisson flows per approach, movement and vehicle type.

    Two blocks per stream: the main block covers warm-up + measurement (these vehicles may be
    measured); the run-out block ("ro_") keeps demand going while measured vehicles leave.
    """
    cfg = config or get_config()
    sim = cfg.simulation
    window_end = sim.warmup_s + sim.measurement_s
    blocks = [("", 0, window_end), ("ro_", window_end, sim.end_s)]

    streams: list[tuple[str, Arm, Movement, str, float, str]] = []
    for arm, a in scenario.approaches.items():
        hv = a.heavy_vehicle_share_pct / 100
        for movement, share in a.turning_shares_pct.items():
            q = a.demand_vph * share / 100
            streams.append((f"{arm}_{movement}_car", arm, movement, "car", q * (1 - hv), "best"))
            streams.append((f"{arm}_{movement}_hgv", arm, movement, "hgv", q * hv, "best"))
        bus_right = a.bus_demand_bph * a.bus_right_share_pct / 100
        bus_lane = "0" if a.bus_lane else "best"
        for movement, q in (("through", a.bus_demand_bph - bus_right), ("right", bus_right)):
            streams.append((f"{arm}_{movement}_bus", arm, movement, "bus", q, bus_lane))

    result = []
    for prefix, begin, end in blocks:
        if end <= begin:
            continue
        for name, arm, movement, vtype, q_vph, depart_lane in streams:
            if q_vph <= 0:
                continue
            result.append(
                {
                    "id": prefix + name,
                    "type": vtype,
                    "begin": str(begin),
                    "end": str(end),
                    "period": f"exp({q_vph / 3600:.12g})",
                    "from": in_edge(arm),
                    "to": out_edge(destination(arm, movement)),
                    "departLane": depart_lane,
                    "departSpeed": "desired",
                }
            )
    return result


def write_routes(scenario: Scenario, out_dir: Path, config: Config | None = None) -> Path:
    cfg = config or get_config()
    root = ET.Element("routes")
    for vtype_id, vclass in (
        ("car", cfg.vehicles.car_vclass),
        ("hgv", cfg.vehicles.heavy_vclass),
        ("bus", cfg.vehicles.bus_vclass),
    ):
        ET.SubElement(
            root,
            "vType",
            id=vtype_id,
            vClass=vclass,
            carFollowModel=cfg.driver.car_follow_model,
            sigma=f"{cfg.driver.sigma:g}",
        )
    for f in sorted(flows(scenario, cfg), key=lambda f: (int(f["begin"]), f["id"])):
        ET.SubElement(root, "flow", f)
    return _write(root, out_dir / "demand.rou.xml")


# --------------------------------------------------------------------------- detectors


DETECTOR_PERIOD_S = 60


def write_detectors(
    scenario: Scenario, net_path: Path, out_dir: Path, output_file: str = "e2.xml"
) -> Path:
    """E2 lane-area detector over the full length of every approach lane."""
    net = sumolib.net.readNet(str(net_path), withPrograms=False)
    root = ET.Element("additional")
    for arm in scenario.approaches:
        for lane in net.getEdge(in_edge(arm)).getLanes():
            ET.SubElement(
                root,
                "laneAreaDetector",
                id=f"e2_{lane.getID()}",
                lane=lane.getID(),
                pos="0",
                endPos=f"{lane.getLength():.2f}",
                period=str(DETECTOR_PERIOD_S),
                file=output_file,
            )
    return _write(root, out_dir / "detectors.add.xml")
