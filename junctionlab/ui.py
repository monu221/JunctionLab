"""Logic behind the Streamlit app, kept free of Streamlit so it can be unit-tested.

- Form: lane text, applying a batch of form values to a scenario, switching control type.
- Results: tables with mean and seed range, LOS colours, comparison of two runs.
- Figures: the junction drawn from the SUMO network, and delay per approach.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import sumolib
from pydantic import ValidationError

from junctionlab.build import JUNCTION_ID, arm_of
from junctionlab.config import Config, get_config
from junctionlab.metrics import Stat, Summary
from junctionlab.scenario import Scenario, destination, road_of

# Ordered green to dark red. Always shown with the letter, never by colour alone.
LOS_COLOURS = {
    "A": "#0ca30c",
    "B": "#7fb61f",
    "C": "#eda100",
    "D": "#ec835a",
    "E": "#d03b3b",
    "F": "#8f1f3f",
}
LOS_INK = {"A": "#0b0b0b", "B": "#0b0b0b", "C": "#0b0b0b", "D": "#0b0b0b"}  # white on E and F
NO_LOS_COLOUR = "#898781"
EXIT_LANE_COLOUR = "#c3c2b7"
JUNCTION_COLOUR = "#898781"
DRAWING_HALF_WIDTH_M = 70.0  # the drawing shows the junction, not the full approach length
_ARM_LABEL_M = 48.0
_ARM_VECTOR = {"north": (0, 1), "east": (1, 0), "south": (0, -1), "west": (-1, 0)}
_LETTER = {"left": "L", "through": "T", "right": "R"}
_MOVEMENT = {v: k for k, v in _LETTER.items()}
_SAME_TOLERANCE = 0.006  # number inputs show two decimals


# --------------------------------------------------------------------------- form


def lanes_text(lanes: Any) -> str:
    """[["left", "through"], ["through", "right"]] -> 'L+T | T+R' (leftmost lane first)."""
    return " | ".join("+".join(_LETTER[m] for m in lane) for lane in lanes)


def parse_lanes(text: str) -> list[list[str]]:
    """'L+T | T+R' -> lane lists. Raises ValueError with a message for the user."""
    lanes = []
    for part in text.split("|"):
        letters = [x.strip().upper() for x in part.split("+")]
        if any(x not in _MOVEMENT for x in letters):
            raise ValueError(f'lanes "{text}": write each lane as L, T or R joined by +, e.g. L+T')
        lanes.append([_MOVEMENT[x] for x in letters])
    return lanes


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, int | float) and isinstance(b, int | float):
        return abs(a - b) < _SAME_TOLERANCE
    if isinstance(a, dict) and isinstance(b, dict):
        return all(_same(a.get(k, 0), b.get(k, 0)) for k in {*a, *b})
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def _value(data: dict[str, Any], path: str) -> Any:
    node: Any = data
    for key in path.split("."):
        node = node[key]
    return node


def _with_marks(s: Scenario, assumed: tuple[str, ...] = (), implicit: tuple[str, ...] = ()):
    return s.model_copy(
        update={
            "assumed": tuple(sorted({*s.assumed, *assumed})),
            "implicit": tuple(sorted({*s.implicit, *implicit})),
        }
    )


def switch_control(s: Scenario, kind: str, config: Config | None = None) -> Scenario:
    """Change the control type. Its new settings are defaults, so they are marked assumed."""
    cfg = config or get_config()
    assumed: list[str] = []
    implicit: list[str] = []
    control: dict[str, Any] = {"kind": kind}
    if kind == "signal":
        t_junction = s.junction_type == "t_junction"
        plan = "two_phase" if t_junction else cfg.control.four_arm_phase_plan_default
        control |= {"cycle_s": cfg.control.cycle_default_s, "phase_plan": plan, "green_s": None}
        assumed += ["control.cycle_s", "control.green_s"]
        (implicit if t_junction else assumed).append("control.phase_plan")
    elif kind == "priority":
        if s.stem:
            control["major_road"] = road_of(destination(s.stem, "left"))
            implicit.append("control.major_road")
        else:
            control["major_road"] = cfg.control.major_road_default
            assumed.append("control.major_road")
    return _with_marks(s.with_values({"control": control}, cfg), tuple(assumed), tuple(implicit))


def apply_form(s: Scenario, values: dict[str, Any], config: Config | None = None) -> Scenario:
    """Apply the fields that really changed. Raises ValidationError or ValueError if invalid."""
    cfg = config or get_config()
    kind = values.get("control.kind", s.control.kind)
    if kind != s.control.kind:
        s = switch_control(s, kind, cfg)
        values = {p: v for p, v in values.items() if not p.startswith("control.")}
    data = s.model_dump(mode="json")
    updates = {
        p: v for p, v in values.items() if p != "control.kind" and not _same(_value(data, p), v)
    }
    if not updates:
        return s
    new_plan = "control.phase_plan" in updates
    if new_plan:  # the old greens do not fit the new number of phases
        updates["control.green_s"] = None
    out = s.with_values(updates, cfg)
    return _with_marks(out, ("control.green_s",)) if new_plan else out


def error_lines(error: Exception) -> list[str]:
    """Readable messages from a validation error."""
    if not isinstance(error, ValidationError):
        return [str(error)]
    lines = []
    for e in error.errors():
        message = e["msg"].removeprefix("Value error, ")
        where = ".".join(str(x) for x in e["loc"])
        lines.append(f"{where}: {message}" if where else message)
    return list(dict.fromkeys(lines))


def scenario_changes(a: Scenario, b: Scenario) -> list[str]:
    """What differs between two scenarios, one line per field."""
    da, db = a.model_dump(mode="json"), b.model_dump(mode="json")
    out = []
    for path in dict.fromkeys([*a.field_paths(), *b.field_paths()]):
        try:
            old, new = _value(da, path), _value(db, path)
        except KeyError:
            out.append(f"{path}: only in one scenario")
            continue
        if not _same(old, new):
            out.append(f"{path}: {old} → {new}")
    return out


# --------------------------------------------------------------------------- results


def fmt_stat(stat: Stat | None, digits: int = 1, lower_bound: bool = False) -> str:
    """'34.1 (31.0–37.2)': mean and range over seeds. '≥' marks a lower bound."""
    if stat is None:
        return "–"
    mark = "≥ " if lower_bound else ""
    return f"{mark}{stat.mean:.{digits}f} ({stat.min:.{digits}f}–{stat.max:.{digits}f})"


def results_table(summary: Summary) -> pd.DataFrame:
    rows = []
    for a in summary.approaches.values():
        rows.append(
            {
                "Approach": a.arm.capitalize(),
                "LOS": a.los,
                "LOS by seed": a.los_range,
                "Control delay (s/veh)": fmt_stat(a.control_delay_s, 1, a.delay_lower_bound),
                "Total delay (s/veh)": fmt_stat(a.delay_s, 1, a.delay_lower_bound),
                "Max queue (m)": fmt_stat(a.queue_m, 0),
                "Max queue (veh)": fmt_stat(a.queue_veh, 0),
                "Throughput (veh/h)": fmt_stat(a.throughput_vph, 0),
                "v/c": "–" if a.v_c is None else f"{a.v_c:.2f}",
            }
        )
    return pd.DataFrame(rows)


def los_cell_style(value: Any) -> str:
    """CSS for a LOS cell. A range such as 'B-C' takes the colour of its worse grade."""
    grade = str(value)[-1:]
    if grade not in LOS_COLOURS:
        return ""
    ink = LOS_INK.get(grade, "#ffffff")
    return f"background-color: {LOS_COLOURS[grade]}; color: {ink}; font-weight: 600"


def style_los(table: pd.DataFrame, columns: tuple[str, ...] = ("LOS", "LOS by seed")):
    return table.style.map(los_cell_style, subset=[c for c in columns if c in table.columns])


def comparison_table(a: Summary, b: Summary) -> pd.DataFrame:
    """Control delay, LOS and queue per approach, for run A and run B."""
    rows = []
    for arm in [x for x in a.approaches if x in b.approaches]:
        x, y = a.approaches[arm], b.approaches[arm]
        change = "–"
        if x.control_delay_s and y.control_delay_s:
            change = f"{y.control_delay_s.mean - x.control_delay_s.mean:+.1f}"
        rows.append(
            {
                "Approach": arm.capitalize(),
                "LOS A": x.los,
                "LOS B": y.los,
                "Delay A (s/veh)": fmt_stat(x.control_delay_s, 1, x.delay_lower_bound),
                "Delay B (s/veh)": fmt_stat(y.control_delay_s, 1, y.delay_lower_bound),
                "Change (s/veh)": change,
                "Queue A (m)": fmt_stat(x.queue_m, 0),
                "Queue B (m)": fmt_stat(y.queue_m, 0),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- figures


@dataclass(frozen=True)
class LaneShape:
    arm: str
    inbound: bool
    index: int  # SUMO index, 0 = rightmost
    width: float
    points: tuple[tuple[float, float], ...]
    bus_only: bool


@dataclass(frozen=True)
class NetworkDrawing:
    """Geometry read from the SUMO network, centred on the junction. Holds no file handles,
    so the run directory can be deleted afterwards."""

    lanes: tuple[LaneShape, ...]
    junction: tuple[tuple[float, float], ...]


def read_network(net_file: Path) -> NetworkDrawing:
    net = sumolib.net.readNet(str(net_file))
    node = net.getNode(JUNCTION_ID)
    cx, cy = node.getCoord()
    lanes = []
    for edge in net.getEdges():
        edge_id = edge.getID()
        for lane in edge.getLanes():
            lanes.append(
                LaneShape(
                    arm=arm_of(edge_id),
                    inbound=edge_id.endswith("_in"),
                    index=lane.getIndex(),
                    width=lane.getWidth(),
                    points=tuple((x - cx, y - cy) for x, y in lane.getShape()),
                    bus_only=lane.allows("bus") and not lane.allows("passenger"),
                )
            )
    return NetworkDrawing(tuple(lanes), tuple((x - cx, y - cy) for x, y in node.getShape()))


def _lane_outline(lane: LaneShape) -> tuple[list[float], list[float]]:
    """Closed polygon around the lane's centre line."""
    half = lane.width / 2 - 0.15  # a thin gap shows the lane markings
    left, right = [], []
    points = lane.points
    for i, (x, y) in enumerate(points):
        x0, y0 = points[max(i - 1, 0)]
        x1, y1 = points[min(i + 1, len(points) - 1)]
        dx, dy = x1 - x0, y1 - y0
        length = (dx * dx + dy * dy) ** 0.5 or 1.0
        nx, ny = -dy / length * half, dx / length * half
        left.append((x + nx, y + ny))
        right.append((x - nx, y - ny))
    ring = left + right[::-1] + left[:1]
    return [p[0] for p in ring], [p[1] for p in ring]


def _lane_use(scenario: Scenario, lane: LaneShape) -> str:
    if lane.bus_only:
        return "BUS"
    a = scenario.approaches[lane.arm]
    general = len(a.lanes) - 1 - (lane.index - int(a.bus_lane))
    return "+".join(_LETTER[m] for m in a.lanes[general])


def network_figure(
    drawing: NetworkDrawing, scenario: Scenario, summary: Summary | None = None
) -> go.Figure:
    """The real SUMO geometry. Inbound lanes take their approach's LOS colour."""
    fig = go.Figure()
    grades = {a.arm: a.los for a in summary.approaches.values()} if summary else {}
    jx, jy = zip(*[*drawing.junction, drawing.junction[0]], strict=True)
    fig.add_trace(
        go.Scatter(
            x=jx, y=jy, fill="toself", mode="lines", fillcolor=JUNCTION_COLOUR,
            line={"color": JUNCTION_COLOUR, "width": 1}, hoverinfo="skip", showlegend=False,
        )
    )  # fmt: skip
    for lane in drawing.lanes:
        xs, ys = _lane_outline(lane)
        grade = grades.get(lane.arm, "-") if lane.inbound else "-"
        colour = LOS_COLOURS.get(grade, NO_LOS_COLOUR) if lane.inbound else EXIT_LANE_COLOUR
        if lane.inbound:
            use = _lane_use(scenario, lane)
            hover = f"{lane.arm.capitalize()} approach, lane {use}, LOS {grade}"
        else:
            hover = f"{lane.arm.capitalize()} exit"
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, fill="toself", mode="lines", fillcolor=colour,
                line={"color": colour, "width": 0.5}, text=hover, hoveron="fills",
                hoverinfo="text", showlegend=False,
            )
        )  # fmt: skip
    for arm, approach in scenario.approaches.items():
        ux, uy = _ARM_VECTOR[arm]
        label = arm.capitalize()
        if arm in grades:
            label += f"<br>LOS {grades[arm]}"
        # Lanes are too narrow to label one by one, so the arm's label lists them.
        label += "<br>" + lanes_text(approach.lanes) + (" | BUS" if approach.bus_lane else "")
        fig.add_annotation(
            x=ux * _ARM_LABEL_M + uy * 22,
            y=uy * _ARM_LABEL_M + ux * 22,
            text=label,
            showarrow=False,
            font={"size": 13},
        )
    r = DRAWING_HALF_WIDTH_M
    fig.update_xaxes(visible=False, range=[-r, r], fixedrange=True)
    fig.update_yaxes(visible=False, range=[-r, r], scaleanchor="x", fixedrange=True)
    fig.update_layout(
        height=430,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def delay_figure(summary: Summary) -> go.Figure:
    """Mean control delay per approach, with the range over seeds and the LOS letter."""
    rows = [a for a in summary.approaches.values() if a.control_delay_s]
    arms = [a.arm.capitalize() for a in rows]
    means = [a.control_delay_s.mean for a in rows]
    fig = go.Figure(
        go.Scatter(
            x=means,
            y=arms,
            mode="markers+text",
            text=[f"LOS {a.los}" for a in rows],
            textposition="top center",
            marker={
                "size": 13,
                "color": [LOS_COLOURS.get(a.los, NO_LOS_COLOUR) for a in rows],
                "line": {"width": 2, "color": "rgba(255,255,255,0.9)"},
            },
            error_x={
                "type": "data",
                "symmetric": False,
                "array": [a.control_delay_s.max - a.control_delay_s.mean for a in rows],
                "arrayminus": [a.control_delay_s.mean - a.control_delay_s.min for a in rows],
                "thickness": 2,
                "width": 5,
                "color": "#898781",
            },
            customdata=[[a.control_delay_s.min, a.control_delay_s.max, a.los] for a in rows],
            hovertemplate=(
                "%{y}: %{x:.1f} s/veh<br>Seeds: %{customdata[0]:.1f}–%{customdata[1]:.1f}"
                "<br>LOS %{customdata[2]}<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(title="Control delay (s/veh), mean and seed range", rangemode="tozero")
    fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=arms)
    fig.update_layout(
        height=120 + 52 * len(rows),
        margin={"l": 0, "r": 16, "t": 16, "b": 0},
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
