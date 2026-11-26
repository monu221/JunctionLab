"""App smoke tests (Streamlit AppTest, real SUMO, no API calls) and the logic behind the form."""

from pathlib import Path

import pytest
from pydantic import ValidationError
from streamlit.testing.v1 import AppTest
from test_field_states import ALL_DEMAND, NOTE, interpretation, scenario

from junctionlab import llm, ui
from junctionlab.config import get_config
from junctionlab.examples import list_examples
from junctionlab.runner import run_scenario
from junctionlab.scenario import resolve

APP = str(Path(__file__).parents[1] / "app" / "streamlit_app.py")
EXAMPLES = {e.key: e for e in list_examples()}
TIMEOUT_S = 240


def app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=TIMEOUT_S).run()


def button(at: AppTest, key: str):
    return next(b for b in at.button if b.key == key)


def text_of(at: AppTest) -> str:
    parts = [*at.markdown, *at.caption, *at.warning, *at.error, *at.info]
    return "\n".join(str(x.value) for x in parts)


# ------------------------------------------------------------------ app


def test_app_starts_without_a_scenario():
    at = app()
    assert not at.exception
    assert len([b for b in at.button if b.key and b.key.startswith("example_")]) == 3


@pytest.mark.parametrize("key", list(EXAMPLES))
def test_each_example_opens_with_run_enabled(key):
    at = app()
    button(at, f"example_{key}").click().run()
    assert not at.exception
    assert button(at, "run").disabled is False
    assert "Not simulated" in text_of(at)
    assert any("assumed" in n.label for n in at.number_input)


def test_interpreted_demand_blocks_run_until_confirmed():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT_S)
    at.session_state["scenario"] = scenario([ALL_DEMAND])
    at.session_state["not_simulated"] = ["pedestrian crossing on the west arm"]
    at.session_state["source"] = "text"
    at.run()
    assert not at.exception
    assert NOTE in text_of(at)
    assert "pedestrian crossing on the west arm" in text_of(at)
    assert button(at, "run").disabled is True
    button(at, "confirm_demand").click().run()
    assert button(at, "run").disabled is False


def test_not_simulated_lists_only_ignored_details():
    conversion = "15-minute counts multiplied by 4."
    at = AppTest.from_file(APP, default_timeout=TIMEOUT_S)
    at.session_state["scenario"] = scenario([ALL_DEMAND])
    at.session_state["notes"] = [NOTE, conversion]
    at.session_state["not_simulated"] = [NOTE, "pedestrian crossing on the west arm"]
    at.session_state["source"] = "text"
    at.run()
    assert not at.exception
    assert text_of(at).count(NOTE) == 1  # only in the interpreted-demand box
    assert any(c.value == f"Conversions: {conversion}" for c in at.caption)
    assert any(m.value == "- pedestrian crossing on the west arm" for m in at.markdown)


def test_scenario_without_a_name_gets_one_that_follows_edits():
    s = scenario()
    assert s.name == "4-arm signal, 600 veh/h each"
    edited = s.with_values({"approaches.north.demand_vph": 800})
    assert edited.name == "4-arm signal, 600–800 veh/h"


def test_stated_name_is_kept_after_an_edit():
    draft, _ = llm.to_draft(interpretation(name="Station Road"))
    edited = resolve(draft).with_values({"approaches.north.demand_vph": 800})
    assert edited.name == "Station Road"


def test_placeholder_name_is_replaced():
    draft, _ = llm.to_draft(interpretation(name="Scenario"))
    assert resolve(draft).name == "4-arm signal, 600 veh/h each"


def test_form_edit_is_applied_and_invalid_edit_is_reported():
    at = app()
    button(at, "example_four_arm_signal").click().run()
    version = at.session_state["form_version"]
    at.number_input(key=f"v{version}.approaches.north.demand_vph").set_value(650.0)
    button(at, "apply_edits").click().run()
    s = at.session_state["scenario"]
    assert s.approaches["north"].demand_vph == 650
    assert s.field_state("approaches.north.demand_vph") == "stated"

    version = at.session_state["form_version"]
    at.number_input(key=f"v{version}.approaches.north.share.left").set_value(90.0)
    button(at, "apply_edits").click().run()
    assert not at.exception
    assert "sum to 100" in text_of(at)
    assert at.session_state["scenario"].approaches["north"].demand_vph == 650


def test_free_limit_message_points_to_the_examples(monkeypatch):
    def limited(*_args, **_kwargs):
        raise llm.RateLimited("The free Groq tier's rate limit is reached. Try again in about 7 s.")

    monkeypatch.setattr(llm, "interpret", limited)
    at = app()
    at.text_area(key="description").set_value("A crossroads")
    next(b for b in at.button if b.label == "Read description").click().run()
    assert not at.exception
    message = "\n".join(w.value for w in at.warning)
    assert "Free limit reached" in message and "examples" in message and "7 s" in message


@pytest.mark.parametrize("key", list(EXAMPLES))
def test_each_example_runs_with_no_api_call(key, monkeypatch):
    def no_call(*_args, **_kwargs):
        raise AssertionError("an example must not call the API")

    for name in ("api_key", "make_client", "interpret"):
        monkeypatch.setattr(llm, name, no_call)
    at = app()
    button(at, f"example_{key}").click().run()
    button(at, "run").click().run()
    assert not at.exception
    assert at.session_state["result"] is not None
    assert [m.label for m in at.metric][:2] == ["Junction LOS", "Control delay"]
    assert len(at.dataframe) == 1
    approaches = at.session_state["scenario"].approaches
    assert len(at.dataframe[0].value) == len(approaches)


def test_successful_run_clears_an_earlier_failure():
    at = app()
    button(at, "example_t_junction_priority").click().run()
    at.session_state["message"] = ("error", "The simulation failed: earlier failure")
    button(at, "run").click().run()
    assert not at.exception
    assert not any("earlier failure" in e.value for e in at.error)


# ------------------------------------------------------------------ form logic


def test_lane_text_round_trip():
    lanes = [["left", "through"], ["through"], ["through", "right"]]
    assert ui.lanes_text(lanes) == "L+T | T | T+R"
    assert ui.parse_lanes("l+t | T | t + r") == lanes
    with pytest.raises(ValueError, match="L, T or R"):
        ui.parse_lanes("L+X")


def test_unchanged_form_changes_nothing():
    s = EXAMPLES["four_arm_signal"].scenario()
    values = {
        "speed_limit_kmh": s.speed_limit_kmh,
        "control.kind": "signal",
        "control.green_s": list(s.control.green_s),
        "approaches.north.turning_shares_pct": dict(s.approaches["north"].turning_shares_pct),
        "approaches.north.lanes": [list(x) for x in s.approaches["north"].lanes],
    }
    assert ui.apply_form(s, values) == s


def test_shares_are_edited_together():
    s = EXAMPLES["four_arm_signal"].scenario()
    path = "approaches.north.turning_shares_pct"
    out = ui.apply_form(s, {path: {"left": 20, "through": 60, "right": 20}})
    assert out.approaches["north"].turning_shares_pct["through"] == 60
    assert out.field_state(path) == "stated"
    with pytest.raises(ValidationError, match="sum to 100"):
        ui.apply_form(s, {path: {"left": 90, "through": 60, "right": 20}})


def test_new_phase_plan_rederives_greens_as_assumed():
    s = EXAMPLES["four_arm_signal"].scenario()
    out = ui.apply_form(s, {"control.phase_plan": "four_phase_split"})
    assert len(out.control.green_s) == 4
    assert out.field_state("control.green_s") == "assumed"


@pytest.mark.parametrize("key", list(EXAMPLES))
@pytest.mark.parametrize("kind", ["signal", "priority", "all_way_stop"])
def test_switching_control_gives_a_valid_scenario(key, kind):
    s = EXAMPLES[key].scenario()
    out = ui.apply_form(s, {"control.kind": kind})
    assert out.control.kind == kind
    if kind == "signal" and s.control.kind != "signal":
        assert out.field_state("control.cycle_s") == "assumed"
        assert out.control.green_s is not None


def test_editing_demand_keeps_other_arms_unconfirmed():
    s = scenario([ALL_DEMAND])
    out = ui.apply_form(s, {"approaches.north.demand_vph": 300.0})
    assert len(out.needs_confirmation) == 3


# ------------------------------------------------------------------ results and drawing


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    s = EXAMPLES["four_arm_bus_lane"].scenario()
    return run_scenario(s, run_dir=tmp_path_factory.mktemp("app") / "run", seeds=[1, 2])


def test_results_table_has_means_ranges_and_los(run):
    table = ui.results_table(run.summary)
    assert list(table["Approach"]) == ["North", "East", "South", "West"]
    assert table["Control delay (s/veh)"].str.fullmatch(r"(≥ )?[\d.]+ \([\d.]+–[\d.]+\)").all()
    assert table["LOS"].isin(list("ABCDEF")).all()


def test_los_cell_uses_the_worse_grade_of_a_range():
    assert ui.LOS_COLOURS["C"] in ui.los_cell_style("B-C")
    assert ui.los_cell_style("-") == ""


def test_drawing_comes_from_the_sumo_network(run):
    drawing = ui.read_network(run.net)
    expected = sum(2 * a.lane_count for a in run.scenario.approaches.values())
    assert len(drawing.lanes) == expected
    assert sum(lane.bus_only and lane.inbound for lane in drawing.lanes) == 1
    fig = ui.network_figure(drawing, run.scenario, run.summary)
    assert len(fig.data) == expected + 1
    labels = [a.text for a in fig.layout.annotations]
    west = next(x for x in labels if x.startswith("West<br>LOS"))
    assert west.endswith("L+T | T+R | BUS")


def test_delay_figure_has_one_point_per_approach(run):
    fig = ui.delay_figure(run.summary)
    assert list(fig.data[0].y) == ["North", "East", "South", "West"]


def test_comparison_of_two_runs(run):
    table = ui.comparison_table(run.summary, run.summary)
    assert (table["Change (s/veh)"] == "+0.0").all()
    assert ui.scenario_changes(run.scenario, run.scenario) == []
    other = run.scenario.with_values({"approaches.north.demand_vph": 100})
    assert any("north.demand_vph" in x for x in ui.scenario_changes(run.scenario, other))


# ------------------------------------------------------------------ text checks


@pytest.mark.parametrize(
    "text",
    [
        "Signalised crossroads. Counts N 800, S 600, E 500, W 450.",
        "Signalised crossroads. Counts N 800 S 600 E 500 W 450",
        "Signalised crossroads, 90 s cycle. Counts for north, south, east, west: 800, 600, 500, "
        "450.",
        "Crossroads with traffic lights. We counted 800 north, 600 south, 500 east and 450 west.",
    ],
)
def test_counts_without_units_get_their_own_message(text):
    i = interpretation(unclear=["Demand counts given without a time period"])
    for arm in ("north", "east", "south", "west"):
        getattr(i.approaches, arm).demand_vph = None
    with pytest.raises(llm.InterpretationError) as e:
        llm.to_draft(i, text)
    assert e.value.reasons == [llm.COUNTS_WITHOUT_UNITS]


@pytest.mark.parametrize(
    "text",
    [
        "A busy signalised crossroads near the station.",
        "A 4-arm junction, 2 lanes each, 90 s cycle, 50 km/h.",
        "Crossroads, north 800 veh/h, south 600 veh/h.",
    ],
)
def test_truly_missing_traffic_is_still_reported_as_missing(text):
    i = interpretation()
    for arm in ("north", "east", "south", "west"):
        getattr(i.approaches, arm).demand_vph = None
    with pytest.raises(llm.InterpretationError) as e:
        llm.to_draft(i, text)
    assert "is not stated" in e.value.reasons[0]


@pytest.mark.parametrize(
    "text",
    [
        "Traffic lights at a crossroads, about 600 cars an hour on each road.",
        "Crossroads, 600 veh/h per street.",
        "Crossroads with 600 veh/h in both directions.",
    ],
)
def test_volume_per_road_needs_confirmation(text):
    draft, _ = llm.to_draft(interpretation())
    s = resolve(llm._flag_per_road_volume(text, draft))
    assert len(s.needs_confirmation) == 4
    assert s.interpreted["approaches.west.demand_vph"] == llm.PER_ROAD_NOTE


@pytest.mark.parametrize("text", ["600 veh/h on every approach", "600 veh/h from each arm"])
def test_volume_per_approach_is_not_flagged(text):
    draft, _ = llm.to_draft(interpretation())
    assert llm._flag_per_road_volume(text, draft).interpreted == {}


def test_t_junction_missing_arm_written_as_nulls_is_dropped():
    i = interpretation(junction_type="t_junction")
    empty = {name: None for name in type(i.approaches.south).model_fields}
    i.approaches.south = llm.ApproachText(**empty)
    i.control.kind = "priority"
    draft, _ = llm.to_draft(i)
    assert set(draft.approaches) == {"north", "east", "west"}


def test_four_arm_approach_of_nulls_is_still_missing_traffic():
    i = interpretation()
    empty = {name: None for name in type(i.approaches.south).model_fields}
    i.approaches.south = llm.ApproachText(**empty)
    with pytest.raises(llm.InterpretationError, match="south approach is not stated"):
        llm.to_draft(i)


def test_not_simulated_keeps_the_model_list_and_adds_what_it_missed():
    text = "Signals, 90 s cycle, a zebra crossing on the west arm and a bike lane going north."
    items = llm.not_simulated_items(text, ["zebra crossing on the west arm"])
    assert items == ["zebra crossing on the west arm", "Cyclists or cycle lanes"]


def test_signal_cycle_is_not_mistaken_for_cyclists():
    assert llm.not_simulated_items("Signal with a 90 s cycle and a cycle time of 90", []) == []


def test_config_has_five_seeds_for_the_run_label():
    assert len(get_config().simulation.seeds) == 5
