import sumolib

from junctionlab import build
from junctionlab.baseline import BASELINE_FILE, baseline_movements, by_movement, measure_baseline
from junctionlab.examples import list_examples

EXAMPLES = {e.key: e for e in list_examples()}


def test_movements_include_bus_movements():
    s = EXAMPLES["four_arm_bus_lane"].scenario()
    assert baseline_movements(s)["west"] == ["left", "through", "right"]
    t = EXAMPLES["t_junction_priority"].scenario()
    assert baseline_movements(t) == {
        "east": ["left", "through"],
        "south": ["left", "right"],
        "west": ["through", "right"],
    }


def test_free_flow_network_is_unregulated(tmp_path):
    s = EXAMPLES["four_arm_signal"].scenario()
    net = sumolib.net.readNet(str(build.build_network(s, tmp_path, free_flow=True)))
    assert net.getNode(build.JUNCTION_ID).getType() == "unregulated"
    assert net.getEdge("north_in").getLength() == build_length(s, tmp_path)


def build_length(scenario, tmp_path):
    regulated = tmp_path / "regulated"
    regulated.mkdir()
    net = sumolib.net.readNet(str(build.build_network(scenario, regulated)))
    return net.getEdge("north_in").getLength()


def test_baseline_values(tmp_path):
    s = EXAMPLES["four_arm_signal"].scenario()
    baseline = measure_baseline(s, tmp_path)
    assert (tmp_path / BASELINE_FILE).is_file()
    assert set(baseline) == set(s.approaches)
    for moves in baseline.values():
        assert set(moves) == {"left", "through", "right"}
        assert all(0 < v < 10 for v in moves.values())
    per_movement = by_movement(baseline)
    # Turning costs extra time against the desired speed; the tight right turn the most.
    assert per_movement["through"] < per_movement["left"] < per_movement["right"]
