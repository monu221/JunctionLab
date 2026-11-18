"""Checks that the SUMO binaries from the eclipse-sumo pip package actually run."""

import xml.etree.ElementTree as ET

from junctionlab import sumo_env

NODES = """<nodes>
  <node id="a" x="0" y="0"/>
  <node id="b" x="300" y="0"/>
</nodes>"""
EDGES = """<edges>
  <edge id="ab" from="a" to="b" numLanes="1" speed="13.89"/>
</edges>"""
ROUTES = """<routes>
  <vType id="car" vClass="passenger"/>
  <flow id="f" type="car" begin="0" end="60" period="exp(0.2)" from="ab" to="ab"/>
</routes>"""


def test_pinned_sumo_version_runs():
    assert "1.28" in sumo_env.version()


def test_netconvert_and_sumo_run_headless(tmp_path):
    (tmp_path / "n.nod.xml").write_text(NODES, encoding="utf-8")
    (tmp_path / "n.edg.xml").write_text(EDGES, encoding="utf-8")
    (tmp_path / "r.rou.xml").write_text(ROUTES, encoding="utf-8")

    sumo_env.run(
        "netconvert",
        ["-n", "n.nod.xml", "-e", "n.edg.xml", "-o", "n.net.xml"],
        cwd=tmp_path,
    )
    assert (tmp_path / "n.net.xml").is_file()

    sumo_env.run(
        "sumo",
        [
            "-n",
            "n.net.xml",
            "-r",
            "r.rou.xml",
            "--seed",
            "1",
            "--end",
            "300",
            "--time-to-teleport",
            "-1",
            "--tripinfo-output",
            "trips.xml",
            "--no-step-log",
        ],
        cwd=tmp_path,
    )

    trips = ET.parse(tmp_path / "trips.xml").getroot().findall("tripinfo")
    assert trips, "no vehicle completed its trip"
    for trip in trips:
        assert float(trip.get("timeLoss")) >= 0
        assert float(trip.get("departDelay")) >= 0
