"""JunctionLab Streamlit app. Start with:  streamlit run app/streamlit_app.py

Layout: examples and the description in the sidebar. In the main area: results (when there
are any), then the checks before a run (interpreted demand, what is not simulated), then the
editable scenario form with the Run button. Each part is one render_* function.
"""

from __future__ import annotations

import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st
from pydantic import ValidationError

from junctionlab import llm, ui
from junctionlab.config import get_config
from junctionlab.examples import list_examples
from junctionlab.metrics import Summary
from junctionlab.runner import run_scenario
from junctionlab.scenario import FixedTimeSignal, PriorityRule, Scenario
from junctionlab.signal import phase_arms

PAGE_MAX_WIDTH_PX = 1300
TICK_S = 0.25
EXAMPLE_LABELS = {
    "four_arm_signal": "4-arm fixed-time signal",
    "t_junction_priority": "T-junction, priority rule",
    "four_arm_bus_lane": "4-arm with west bus lane",
}
CONTROL_LABELS = {
    "signal": "Fixed-time signal",
    "priority": "Priority rule",
    "all_way_stop": "All-way stop",
}
PLAN_LABELS = {"two_phase": "Two phases", "four_phase_split": "Four phases (one approach each)"}
STATE_SUFFIX = {"stated": "", "interpreted": " · interpreted", "assumed": " · assumed"}
PAGE_CSS = f"""<style>
.block-container, [data-testid='stMainBlockContainer'] {{max-width:{PAGE_MAX_WIDTH_PX}px;}}
div[class*="st-key-fs-assumed"] {{background:rgba(237,161,0,.15);border-left:4px solid #eda100;
  border-radius:4px;padding:4px 8px;}}
div[class*="st-key-fs-interpreted"] {{background:rgba(98,80,214,.16);border-left:4px solid #6250d6;
  border-radius:4px;padding:4px 8px;}}
div[class*="st-key-fs-stated"] {{border-left:4px solid transparent;padding:4px 8px;}}
.jl-legend {{display:flex;flex-wrap:wrap;gap:8px 18px;font-size:.875rem;margin:0 0 8px;}}
.jl-chip {{padding:2px 8px;border-radius:4px;}}
.jl-chip.stated {{border-left:4px solid rgba(128,128,128,.5);}}
.jl-chip.assumed {{background:rgba(237,161,0,.15);border-left:4px solid #eda100;}}
.jl-chip.interpreted {{background:rgba(98,80,214,.16);border-left:4px solid #6250d6;}}
</style>"""
LEGEND = (
    '<div class="jl-legend">'
    '<span class="jl-chip stated"><b>Stated</b>: from your text or edits</span>'
    '<span class="jl-chip interpreted"><b>Interpreted</b>: the model chose one reading</span>'
    '<span class="jl-chip assumed"><b>Assumed</b>: a default, please check</span>'
    "</div>"
)


@dataclass
class RunOutput:
    scenario: Scenario
    summary: Summary
    drawing: ui.NetworkDrawing
    warnings: list[str]


# --- State ---------------------------------------------------------------------------


def set_scenario(
    scenario: Scenario, notes: list[str], not_simulated: list[str], source: str
) -> None:
    st.session_state.update(
        scenario=scenario,
        notes=notes,
        not_simulated=not_simulated,
        source=source,
        form_version=st.session_state.get("form_version", 0) + 1,
        result=None,
        message=None,
        form_errors=[],
    )


def load_example(key: str) -> None:
    example = next(e for e in list_examples() if e.key == key)
    st.session_state["description"] = example.description
    set_scenario(example.scenario(), [], [], "example")


def confirm_demand() -> None:
    st.session_state["scenario"] = st.session_state["scenario"].confirm()


def keep_for_comparison() -> None:
    result: RunOutput = st.session_state["result"]
    st.session_state["comparison"] = result


def clear_comparison() -> None:
    st.session_state["comparison"] = None


def read_description(text: str) -> None:
    """One LLM call. A failure leaves the current scenario as it is."""
    session = st.session_state.setdefault("llm_session", llm.LlmSession())
    try:
        with st.spinner("Reading the description…"):
            out = llm.interpret(text, session)
    except llm.InterpretationError as e:
        st.session_state["message"] = ("error", f"{e}\n\n{llm.REPHRASE_HINT}")
    except (llm.RateLimited, llm.LimitReached) as e:
        hint = "Pick one of the **examples** above: they run without any API call."
        st.session_state["message"] = ("limit", f"**Free limit reached.** {e}\n\n{hint}")
    except llm.LlmError as e:
        st.session_state["message"] = ("error", str(e))
    else:
        set_scenario(out.scenario, out.notes, out.not_simulated, "text")


# --- Sidebar -------------------------------------------------------------------------


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Junction")
        st.caption("Examples (no API call)")
        for example in list_examples():
            st.button(
                EXAMPLE_LABELS.get(example.key, example.key),
                key=f"example_{example.key}",
                on_click=load_example,
                args=(example.key,),
                width="stretch",
            )
        st.divider()
        limit = get_config().llm.max_input_chars
        with st.form("description_form", border=False):
            text = st.text_area(
                "Describe a junction",
                key="description",
                height=170,
                max_chars=limit,
                placeholder="Four-arm junction, two lanes per approach, 800 veh/h from the north, "
                "600 from the south, 500 each from east and west, fixed-time signal, 90 s cycle.",
            )
            read = st.form_submit_button("Read description", type="primary", width="stretch")
        st.caption("Give the junction type, traffic per approach in veh/h, and the control.")
        if read:
            if text.strip():
                read_description(text)
            else:
                st.session_state["message"] = ("error", "Write a description first.")


def render_message() -> None:
    message = st.session_state.get("message")
    if not message:
        return
    kind, text = message
    if kind == "limit":
        st.warning(text, icon=":material/hourglass_top:")
    else:
        st.error(text, icon=":material/error:")


# --- Checks before the run -----------------------------------------------------------


def field_label(path: str) -> str:
    parts = path.split(".")
    names = {"demand_vph": "demand", "bus_demand_bph": "bus demand"}
    if parts[0] == "approaches":
        return f"{parts[1].capitalize()} {names.get(parts[2], parts[2].replace('_', ' '))}"
    return parts[-1].replace("_", " ")


def render_interpreted(s: Scenario) -> None:
    pending = s.needs_confirmation
    demand_paths = [
        p for p in s.interpreted if p.rsplit(".", 1)[-1] in ("demand_vph", "bus_demand_bph")
    ]
    if demand_paths:
        by_note: dict[str, list[str]] = {}
        for path in demand_paths:
            arm, name = path.split(".")[1:]
            value = getattr(s.approaches[arm], name)
            unit = "veh/h" if name == "demand_vph" else "buses/h"
            by_note.setdefault(s.interpreted[path], []).append(
                f"{arm.capitalize()} {value:g} {unit}"
            )
        with st.container(border=True):
            st.markdown(
                "**Interpreted demand**"
                + (" · needs your confirmation" if pending else " · confirmed")
            )
            for note, values in by_note.items():
                st.markdown(f"{note}\n\nRead as: {', '.join(values)}.")
            if pending:
                st.button(
                    "Confirm this reading",
                    key="confirm_demand",
                    type="primary",
                    on_click=confirm_demand,
                )
                st.caption("Or correct the demand in the form and press **Apply edits**.")
    others = {p: n for p, n in s.interpreted.items() if p not in demand_paths}
    if others:
        with st.expander(f"Other interpreted fields ({len(others)})"):
            for path, note in others.items():
                st.markdown(f"**{field_label(path)}:** {note}")


def render_conversions(s: Scenario) -> None:
    shown = set(s.interpreted.values())
    notes = [n for n in st.session_state.get("notes") or [] if n not in shown]
    if notes:
        st.caption("Conversions: " + " ".join(notes))


def render_not_simulated(s: Scenario) -> None:
    shown = set(s.interpreted.values())
    items = [i for i in st.session_state.get("not_simulated") or [] if i not in shown]
    source = st.session_state.get("source")
    with st.container(border=True, key="not_simulated"):
        st.markdown("**Not simulated**")
        if items:
            st.caption("Your description mentions these, but the simulation leaves them out:")
            st.markdown("\n".join(f"- {item}" for item in items))
        elif source == "text":
            st.caption("The model reported no details that the simulation leaves out.")
        else:
            st.caption("This example has no details that the simulation leaves out.")


# --- Form ------------------------------------------------------------------------------


def _label(s: Scenario, path: str, text: str) -> str:
    return text + STATE_SUFFIX[s.field_state(path)]


def _box(s: Scenario, path: str, version: int):
    """A container highlighted by the field's state (see PAGE_CSS)."""
    state = s.field_state(path)
    # Unused features (no bus lane) stay plain, as method.md says.
    if state == "assumed" and path in s.implicit:
        state = "stated"
    return st.container(key=f"fs-{state}-{version}-{path.replace('.', '-')}")


def _number(s, path, text, version, value, **kw) -> float:
    with _box(s, path, version):
        return st.number_input(
            _label(s, path, text), value=float(value), key=f"v{version}.{path}", format="%.2f", **kw
        )


def render_approach(s: Scenario, arm: str, version: int, values: dict[str, Any]) -> None:
    a = s.approaches[arm]
    p = f"approaches.{arm}"
    st.markdown(f"**{arm.capitalize()}**")
    values[f"{p}.demand_vph"] = _number(
        s, f"{p}.demand_vph", "Demand (veh/h)", version, a.demand_vph, min_value=0.0, step=10.0
    )
    with _box(s, f"{p}.lanes", version):
        lanes = st.text_input(
            _label(s, f"{p}.lanes", "Lanes, left to right"),
            value=ui.lanes_text(a.lanes),
            key=f"v{version}.{p}.lanes",
            help="L = left, T = through, R = right. Example: L+T | T+R is two lanes.",
        )
    values[f"{p}.lanes"] = lanes
    shares = {}
    with _box(s, f"{p}.turning_shares_pct", version):
        st.caption("Turning shares (%)" + STATE_SUFFIX[s.field_state(f"{p}.turning_shares_pct")])
        for movement in s.allowed_movements(arm):
            shares[movement] = st.number_input(
                movement.capitalize(),
                value=float(a.turning_shares_pct.get(movement, 0)),
                min_value=0.0,
                max_value=100.0,
                step=5.0,
                format="%.2f",
                key=f"v{version}.{p}.share.{movement}",
            )
    values[f"{p}.turning_shares_pct"] = shares
    values[f"{p}.heavy_vehicle_share_pct"] = _number(
        s, f"{p}.heavy_vehicle_share_pct", "Heavy vehicles (%)", version, a.heavy_vehicle_share_pct,
        min_value=0.0, max_value=30.0, step=1.0,
    )  # fmt: skip
    with _box(s, f"{p}.bus_lane", version):
        values[f"{p}.bus_lane"] = st.checkbox(
            "Bus lane", value=a.bus_lane, key=f"v{version}.{p}.bus_lane"
        )
    values[f"{p}.bus_demand_bph"] = _number(
        s,
        f"{p}.bus_demand_bph",
        "Buses per hour",
        version,
        a.bus_demand_bph,
        min_value=0.0,
        step=1.0,
    )
    values[f"{p}.bus_right_share_pct"] = _number(
        s, f"{p}.bus_right_share_pct", "Buses turning right (%)", version, a.bus_right_share_pct,
        min_value=0.0, max_value=100.0, step=5.0,
    )  # fmt: skip


def render_control(s: Scenario, version: int, values: dict[str, Any]) -> None:
    c = s.control
    cols = st.columns(4)
    with cols[0], _box(s, "control.kind", version):
        values["control.kind"] = st.selectbox(
            _label(s, "control.kind", "Control"),
            list(CONTROL_LABELS),
            index=list(CONTROL_LABELS).index(c.kind),
            format_func=CONTROL_LABELS.get,
            key=f"v{version}.control.kind",
            help="After changing this, press Apply edits to see its settings.",
        )
    if isinstance(c, FixedTimeSignal):
        with cols[1]:
            values["control.cycle_s"] = _number(
                s, "control.cycle_s", "Cycle (s)", version, c.cycle_s, min_value=1.0, step=5.0
            )
        if s.junction_type == "four_arm":
            with cols[2], _box(s, "control.phase_plan", version):
                values["control.phase_plan"] = st.selectbox(
                    _label(s, "control.phase_plan", "Phase plan"),
                    list(PLAN_LABELS),
                    index=list(PLAN_LABELS).index(c.phase_plan),
                    format_func=PLAN_LABELS.get,
                    key=f"v{version}.control.phase_plan",
                )
        phases = phase_arms(s)
        with _box(s, "control.green_s", version):
            st.caption("Green time per phase (s)" + STATE_SUFFIX[s.field_state("control.green_s")])
            green_cols = st.columns(len(phases))
            values["control.green_s"] = [
                col.number_input(
                    " + ".join(arm.capitalize() for arm in arms),
                    value=float(green),
                    min_value=1.0,
                    step=1.0,
                    format="%.2f",
                    key=f"v{version}.control.green.{i}",
                )
                for i, (col, arms, green) in enumerate(
                    zip(green_cols, phases, c.green_s, strict=True)
                )
            ]
    elif isinstance(c, PriorityRule) and s.stem is None:
        with cols[1], _box(s, "control.major_road", version):
            options = ["north-south", "east-west"]
            values["control.major_road"] = st.selectbox(
                _label(s, "control.major_road", "Major road"),
                options,
                index=options.index(c.major_road),
                key=f"v{version}.control.major_road",
            )


def render_form(s: Scenario, progress_slot) -> None:
    version = st.session_state.get("form_version", 0)
    st.html(LEGEND)
    values: dict[str, Any] = {}
    with st.form(f"scenario_form_{version}", border=False, enter_to_submit=False):
        cols = st.columns(3)
        with cols[0]:
            kind = "4-arm" if s.junction_type == "four_arm" else "T-junction"
            with _box(s, "junction_type", version):
                st.text_input(
                    _label(s, "junction_type", "Junction type"), value=kind, disabled=True,
                    key=f"v{version}.junction_type",
                    help="To change it, change the description or pick another example.",
                )  # fmt: skip
        with cols[1]:
            values["speed_limit_kmh"] = _number(
                s,
                "speed_limit_kmh",
                "Speed limit (km/h)",
                version,
                s.speed_limit_kmh,
                min_value=5.0,
                max_value=130.0,
                step=5.0,
            )
        with cols[2]:
            values["approach_length_m"] = _number(
                s,
                "approach_length_m",
                "Approach length (m)",
                version,
                s.approach_length_m,
                min_value=50.0,
                max_value=2000.0,
                step=10.0,
            )
        render_control(s, version, values)
        st.divider()
        arm_cols = st.columns(len(s.approaches))
        for col, arm in zip(arm_cols, s.approaches, strict=True):
            with col:
                render_approach(s, arm, version, values)
        for line in st.session_state.get("form_errors", []):
            st.error(line, icon=":material/error:")
        blocked = bool(s.needs_confirmation)
        if blocked:
            st.info(
                "Confirm the interpreted demand above to enable the run.", icon=":material/lock:"
            )
        apply_col, run_col = st.columns(2)
        apply_pressed = apply_col.form_submit_button(
            "Apply edits", key="apply_edits", width="stretch"
        )
        run_pressed = run_col.form_submit_button(
            "Run simulation (5 seeds)"
            if len(get_config().simulation.seeds) == 5
            else "Run simulation",
            key="run",
            type="primary",
            disabled=blocked,
            width="stretch",
        )
    if apply_pressed or run_pressed:
        handle_submit(s, values, run_pressed, progress_slot)


def handle_submit(s: Scenario, values: dict[str, Any], run: bool, progress_slot) -> None:
    try:
        parsed = {p: ui.parse_lanes(v) if p.endswith(".lanes") else v for p, v in values.items()}
        new = ui.apply_form(s, parsed)
    except (ValidationError, ValueError) as e:
        st.session_state["form_errors"] = ui.error_lines(e)
        st.rerun()
    st.session_state["form_errors"] = []
    if new != s:
        st.session_state["scenario"] = new
        st.session_state["form_version"] = st.session_state.get("form_version", 0) + 1
    if run and not new.needs_confirmation:
        execute_run(new, progress_slot)
    st.rerun()


# --- Run -------------------------------------------------------------------------------


def _simulate(scenario: Scenario, events: list) -> RunOutput:
    """Runs in a worker thread: no Streamlit calls here. The run files are temporary."""
    with tempfile.TemporaryDirectory(prefix="junctionlab-", ignore_cleanup_errors=True) as tmp:
        result = run_scenario(
            scenario,
            run_dir=Path(tmp) / "run",
            on_progress=lambda step, seed: events.append((step, seed)),
        )
        drawing = ui.read_network(result.net)
        warnings = list(result.summary.warnings)
        for r in result.seeds:
            warnings += [f"Seed {r.seed}: {w}" for w in r.warnings]
    return RunOutput(scenario, result.summary, drawing, warnings)


def _progress_text(seeds: list[int], events: list) -> tuple[float, str]:
    started = {seed for step, seed in events if step == "seed_started"}
    done = {seed for step, seed in events if step == "seed_done"}
    steps = {step for step, _ in events}
    total = len(seeds) + 2  # baseline, seeds, summary
    finished = len(done) + (1 if started or "summary" in steps else 0) + ("summary" in steps)
    marks = [
        f"Seed {s} ✓"
        if s in done
        else f"Seed {s} running…"
        if s in started
        else f"Seed {s} waiting"
        for s in seeds
    ]
    if "summary" in steps:
        head = "Reading the results"
    elif started:
        head = f"{len(done)} of {len(seeds)} seeds done"
    else:
        head = "Building the network and measuring free flow"
    return min(finished / total, 1.0), f"{head}  \n" + " · ".join(marks)


def execute_run(scenario: Scenario, slot) -> None:
    st.session_state["message"] = None  # an earlier failure no longer applies
    seeds = list(get_config().simulation.seeds)
    events: list = []
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="junctionlab-run")
    future = pool.submit(_simulate, scenario, events)
    pool.shutdown(wait=False)
    with slot.container():
        bar = st.progress(0.0, "Starting…")
        while not future.done():
            share, text = _progress_text(seeds, list(events))
            bar.progress(share, text)
            time.sleep(TICK_S)
    slot.empty()
    try:
        st.session_state["result"] = future.result()
    except Exception as e:  # SUMO failed: tell the user and keep the form
        st.session_state["result"] = None
        st.session_state["message"] = ("error", f"The simulation failed: {e}")


# --- Results ---------------------------------------------------------------------------


def render_summary_metrics(summary: Summary) -> None:
    j = summary.junction
    worst = max(
        summary.approaches.values(),
        key=lambda a: a.control_delay_s.mean if a.control_delay_s else -1,
    )
    queue = max(summary.approaches.values(), key=lambda a: a.queue_m.mean)
    c = st.columns(4)
    c[0].metric(
        "Junction LOS",
        j.los if j.los != "-" else "n/a",
        help="HCM grade from mean control delay. A priority junction has no overall grade: "
        "read its approaches.",
    )
    c[1].metric(
        "Control delay",
        f"{j.control_delay_s.mean:.1f} s/veh" if j.control_delay_s else "–",
        help=f"Seed range {ui.fmt_stat(j.control_delay_s)}",
    )
    c[2].metric(
        "Worst approach",
        f"{worst.arm.capitalize()} · LOS {worst.los}",
        help=f"Control delay {ui.fmt_stat(worst.control_delay_s)} s/veh",
    )
    c[3].metric(
        "Longest queue",
        f"{queue.queue_m.mean:.0f} m",
        help=f"{queue.arm.capitalize()} approach, seed range {ui.fmt_stat(queue.queue_m, 0)} m",
    )


def render_results(result: RunOutput, current: Scenario) -> None:
    st.subheader(f"Results · {result.scenario.name}")
    if result.scenario.model_dump(exclude={"confirmed"}) != current.model_dump(
        exclude={"confirmed"}
    ):
        st.warning(
            "The scenario was edited after this run. These results are for the earlier version.",
            icon=":material/history:",
        )
    render_summary_metrics(result.summary)
    for line in result.warnings:
        st.warning(line, icon=":material/warning:")
    for a in result.summary.approaches.values():
        if a.f_reasons:
            st.caption(f"{a.arm.capitalize()}: LOS F because {', '.join(a.f_reasons)}.")
    left, right = st.columns([5, 6], gap="medium")
    with left:
        st.markdown("**Junction** · drawn from the SUMO network, lanes coloured by LOS")
        st.plotly_chart(
            ui.network_figure(result.drawing, result.scenario, result.summary),
            width="stretch",
            key="network",
            config={"displayModeBar": False},
        )
    with right:
        st.markdown("**Control delay per approach**")
        st.plotly_chart(
            ui.delay_figure(result.summary),
            width="stretch",
            key="delay",
            config={"displayModeBar": False},
        )
    st.markdown(f"**Per approach** · mean (min–max) over {len(result.summary.seeds)} seeds")
    st.dataframe(ui.style_los(ui.results_table(result.summary)), hide_index=True, width="stretch")
    st.caption(
        "LOS uses control delay: total delay minus the free-flow time loss. ≥ marks a lower bound."
    )
    st.button(
        "Keep this run as A, then edit and compare", key="keep_run", on_click=keep_for_comparison
    )


def render_comparison(result: RunOutput | None) -> None:
    kept: RunOutput | None = st.session_state.get("comparison")
    if kept is None:
        return
    st.subheader("Compare")
    if result is None or result.scenario == kept.scenario:
        st.info(f"**A** is kept: {kept.scenario.name}. Edit the scenario and run it to compare.")
    else:
        st.caption(f"A: {kept.scenario.name} · B: {result.scenario.name}")
        table = ui.comparison_table(kept.summary, result.summary)
        st.dataframe(ui.style_los(table, ("LOS A", "LOS B")), hide_index=True, width="stretch")
        changes = ui.scenario_changes(kept.scenario, result.scenario)
        if changes:
            with st.expander(f"What changed from A to B ({len(changes)})"):
                st.markdown("\n".join(f"- `{line}`" for line in changes))
    st.button("Clear comparison", key="clear_comparison", on_click=clear_comparison)


# --- Page ------------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="JunctionLab", page_icon="🚦", layout="wide", initial_sidebar_state="expanded"
    )
    st.html(PAGE_CSS)
    render_sidebar()
    st.title("JunctionLab")
    st.caption("Describe a road junction, check how it was read, and simulate it in SUMO.")
    render_message()
    scenario: Scenario | None = st.session_state.get("scenario")
    if scenario is None:
        st.info(
            "Pick an example in the sidebar, or describe a junction and press **Read description**."
        )
        return
    progress_slot = st.empty()
    result: RunOutput | None = st.session_state.get("result")
    if result is not None:
        render_results(result, scenario)
    render_comparison(result)
    st.subheader(f"Scenario · {scenario.name}")
    render_interpreted(scenario)
    render_conversions(scenario)
    render_not_simulated(scenario)
    with st.expander("Edit the scenario", expanded=result is None):
        render_form(scenario, progress_slot)


main()
