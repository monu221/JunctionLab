import xml.etree.ElementTree as ET
from collections import defaultdict

import pytest
import sumolib

from junctionlab import build
from junctionlab.config import get_config
from junctionlab.examples import list_examples
from junctionlab.scenario import ScenarioDraft, resolve

EXAMPLES = {e.key: e for e in list_examples()}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Each bundled example built once: key -> (scenario, net path, sumolib net)."""
    out = {}
    for key, example in EXAMPLES.items():
        d = tmp_path_factory.mktemp(key)
        s = example.scenario()
        net = build.build_network(s, d)
        out[key] = (s, net, sumolib.net.readNet(str(net), withPrograms=False))
    return out


def _foes(net_path):
    """Link index -> set of foe link indices, from the junction's request table."""
    junction = next(
        j for j in ET.parse(net_path).getroot().iter("junction") if j.get("id") == build.JUNCTION_ID
    )
    foes = {}
    for req in junction.iter("request"):
        bits = req.get("foes")[::-1]
        foes[int(req.get("index"))] = {i for i, b in enumerate(bits) if b == "1"}
    return foes


# ------------------------------------------------------------------ network


@pytest.mark.parametrize("key", EXAMPLES)
def test_network_has_exactly_the_specified_connections(built, key):
    s, _, net = built[key]
    expected = {
        (build.in_edge(c.from_arm), c.from_lane, build.out_edge(c.to_arm), c.to_lane)
        for c in build.connections(s)
    }
    got = {
        (
            c.getFrom().getID(),
            c.getFromLane().getIndex(),
            c.getTo().getID(),
            c.getToLane().getIndex(),
        )
        for e in net.getEdges()
        for cons in e.getOutgoing().values()
        for c in cons
    }
    assert got == expected


@pytest.mark.parametrize("key", EXAMPLES)
def test_lane_counts_and_lengths(built, key):
    s, _, net = built[key]
    for arm, a in s.approaches.items():
        inbound, outbound = net.getEdge(build.in_edge(arm)), net.getEdge(build.out_edge(arm))
        assert inbound.getLaneNumber() == a.lane_count
        assert outbound.getLaneNumber() == a.lane_count  # exit leg mirrors the inbound side
        assert inbound.getLength() == pytest.approx(s.approach_length_m, abs=0.5)
        assert inbound.getSpeed() == pytest.approx(s.speed_limit_kmh / 3.6, abs=0.01)


def test_bus_lane_is_rightmost_and_bus_only(built):
    _, _, net = built["four_arm_bus_lane"]
    lanes = net.getEdge("west_in").getLanes()
    assert lanes[0].allows("bus") and not lanes[0].allows("passenger")
    assert all(lane.allows("passenger") for lane in lanes[1:])


def test_node_types(built):
    types = {key: net.getNode(build.JUNCTION_ID).getType() for key, (_, _, net) in built.items()}
    assert types["four_arm_signal"] == "traffic_light"
    assert types["t_junction_priority"] == "priority"


def test_priority_rule_major_road_has_right_of_way(built):
    s, net_path, _ = built["t_junction_priority"]
    edges = {e.get("id"): e for e in ET.parse(net_path).getroot().iter("edge")}
    assert int(edges["east_in"].get("priority")) > int(edges["south_in"].get("priority"))


def test_all_way_stop_node_type(tmp_path):
    draft = EXAMPLES["t_junction_priority"].draft.model_dump()
    draft["control"] = {"kind": "all_way_stop"}
    s = resolve(ScenarioDraft.model_validate(draft))
    net = sumolib.net.readNet(str(build.build_network(s, tmp_path)))
    assert net.getNode(build.JUNCTION_ID).getType() == "allway_stop"


# ------------------------------------------------------------------ signal program


@pytest.mark.parametrize("key", ["four_arm_signal", "four_arm_bus_lane"])
def test_signal_program(built, key):
    s, net_path, _ = built[key]
    cfg = get_config()
    steps = build.signal_program(s, net_path)
    links = build.tls_links(net_path)
    foes = _foes(net_path)

    assert sum(d for d, _ in steps) == pytest.approx(s.control.cycle_s)
    assert [d for d, _ in steps][::3] == list(s.control.green_s)
    assert all(len(state) == len(links) for _, state in steps)
    for i in links:
        assert any(state[i] in "Gg" for _, state in steps), f"link {i} never green"
    for (_, green), (amber_s, amber) in zip(steps[::3], steps[1::3], strict=True):
        assert amber_s == cfg.control.amber_s
        assert all((g in "Gg") == (a == "y") for g, a in zip(green, amber, strict=True))
        major = [i for i, ch in enumerate(green) if ch == "G"]
        for i in major:  # 'G' links never conflict with each other ...
            assert not (foes[i] & set(major)), f"G links {i} and {foes[i] & set(major)} conflict"
        targets = defaultdict(list)  # ... nor share an exit lane
        for i in major:
            targets[links[i].to_lane].append(i)
        assert all(len(v) == 1 for v in targets.values()), targets


def test_permitted_left_yields(built):
    s, net_path, _ = built["four_arm_signal"]
    links = build.tls_links(net_path)
    green_ns = build.signal_program(s, net_path)[0][1]
    for i, link in links.items():
        if link.from_arm == "north":
            assert green_ns[i] == ("g" if link.movement == "left" else "G")
        if link.from_arm in ("east", "west"):
            assert green_ns[i] == "r"


def test_four_phase_split_serves_one_approach_at_a_time(tmp_path):
    draft = EXAMPLES["four_arm_signal"].draft.model_dump()
    draft["control"] = {"kind": "signal", "cycle_s": 100, "phase_plan": "four_phase_split"}
    s = resolve(ScenarioDraft.model_validate(draft))
    net = build.build_network(s, tmp_path)
    links = build.tls_links(net)
    greens = build.signal_program(s, net)[::3]
    assert len(greens) == 4
    for _, state in greens:
        served = {links[i].from_arm for i, ch in enumerate(state) if ch != "r"}
        assert len(served) == 1
        assert "g" not in state  # no opposing traffic, so lefts are protected here


# ------------------------------------------------------------------ demand


def test_flows_match_demand():
    s = EXAMPLES["four_arm_bus_lane"].scenario()
    cfg = get_config()
    main = [f for f in build.flows(s, cfg) if not f["id"].startswith("ro_")]
    rate = defaultdict(float)
    for f in main:
        assert f["period"].startswith("exp(")
        rate[(f["from"], f["type"])] += float(f["period"][4:-1]) * 3600
    for arm, a in s.approaches.items():
        edge = build.in_edge(arm)
        hv = a.heavy_vehicle_share_pct / 100
        assert rate[(edge, "car")] == pytest.approx(a.demand_vph * (1 - hv), rel=1e-6)
        assert rate[(edge, "hgv")] == pytest.approx(a.demand_vph * hv, rel=1e-6)
        assert rate[(edge, "bus")] == pytest.approx(a.bus_demand_bph, rel=1e-6)
    bus = next(f for f in main if f["type"] == "bus")
    assert bus["departLane"] == "0"


def test_flow_blocks_cover_window_then_runout():
    cfg = get_config()
    sim = cfg.simulation
    fl = build.flows(EXAMPLES["four_arm_signal"].scenario(), cfg)
    main = {(f["begin"], f["end"]) for f in fl if not f["id"].startswith("ro_")}
    runout = {(f["begin"], f["end"]) for f in fl if f["id"].startswith("ro_")}
    window_end = sim.warmup_s + sim.measurement_s
    assert main == {("0", str(window_end))}
    assert runout == {(str(window_end), str(sim.end_s))}


def test_routes_file_sets_driver_model(tmp_path):
    path = build.write_routes(EXAMPLES["four_arm_signal"].scenario(), tmp_path)
    vtypes = {v.get("id"): v for v in ET.parse(path).getroot().iter("vType")}
    assert vtypes["hgv"].get("vClass") == "truck"
    assert all(
        v.get("carFollowModel") == "Krauss" and v.get("sigma") == "0.5" for v in vtypes.values()
    )


# ------------------------------------------------------------------ detectors


def test_e2_detector_on_every_approach_lane(built, tmp_path):
    s, net_path, net = built["four_arm_bus_lane"]
    path = build.write_detectors(s, net_path, tmp_path)
    dets = {d.get("lane"): d for d in ET.parse(path).getroot().iter("laneAreaDetector")}
    lanes = [lane for arm in s.approaches for lane in net.getEdge(build.in_edge(arm)).getLanes()]
    assert set(dets) == {lane.getID() for lane in lanes}
    for lane in lanes:
        assert float(dets[lane.getID()].get("endPos")) == pytest.approx(lane.getLength(), abs=0.01)
