"""Signal phase plans and Webster green splits (docs/method.md, section 1)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from junctionlab.config import Config, get_config

if TYPE_CHECKING:
    from junctionlab.scenario import Arm, Movement, Scenario


def phase_arms(scenario: Scenario) -> list[tuple[Arm, ...]]:
    """Approaches served by each phase, in phase order."""
    c = scenario.control
    present = [a for a in ("north", "east", "south", "west") if a in scenario.approaches]
    if scenario.junction_type == "t_junction":
        stem = scenario.stem
        return [tuple(a for a in present if a != stem), (stem,)]
    if c.phase_plan == "four_phase_split":
        return [(a,) for a in present]
    return [("north", "south"), ("east", "west")]


def lane_flows_vph(scenario: Scenario, arm: Arm) -> list[float]:
    """Flow per inbound lane, leftmost general lane first, bus lane (if any) last.

    Each movement's flow is split equally over the general lanes that serve it. Buses use
    the bus lane if there is one, otherwise they are split over the lanes serving their
    movement.
    """
    a = scenario.approaches[arm]
    flows = [0.0] * a.lane_count

    def spread(movement: Movement, q: float) -> None:
        serving = [i for i, lane in enumerate(a.lanes) if movement in lane]
        for i in serving:
            flows[i] += q / len(serving)

    for movement, share in a.turning_shares_pct.items():
        spread(movement, a.demand_vph * share / 100)
    bus_right = a.bus_demand_bph * a.bus_right_share_pct / 100
    if a.bus_lane:
        flows[-1] += a.bus_demand_bph
    elif a.bus_demand_bph > 0:
        spread("through", a.bus_demand_bph - bus_right)
        if bus_right:
            spread("right", bus_right)
    return flows


def critical_flow_ratios(scenario: Scenario, config: Config | None = None) -> list[float]:
    """Per phase: the highest lane flow / saturation flow among the approaches it serves."""
    cfg = config or get_config()
    s = cfg.control.saturation_flow_vphpl
    return [
        max(max(lane_flows_vph(scenario, arm)) for arm in arms) / s for arms in phase_arms(scenario)
    ]


def webster_greens(scenario: Scenario, config: Config | None = None) -> tuple[float, ...]:
    """Green time per phase (whole seconds) in proportion to the critical flow ratios.

    Available green = cycle - phases x intergreen. Phases below the minimum green are raised
    to it and the rest is shared out again. Uses the configured saturation flow (a placeholder
    until M4 measures it).
    """
    cfg = config or get_config()
    cycle = scenario.control.cycle_s
    y = critical_flow_ratios(scenario, cfg)
    n = len(y)
    available = cycle - n * cfg.control.intergreen_s
    gmin = cfg.control.min_green_s

    fixed: dict[int, float] = {}
    while True:
        free = [i for i in range(n) if i not in fixed]
        rest = available - sum(fixed.values())
        y_free = sum(y[i] for i in free)
        greens = {i: rest * (y[i] / y_free if y_free > 0 else 1 / len(free)) for i in free}
        low = [i for i, g in greens.items() if g < gmin]
        if not low:
            break
        for i in low:
            fixed[i] = gmin
    raw = [fixed.get(i, greens.get(i, 0.0)) for i in range(n)]
    return _round_preserving_sum(raw, available)


def _round_preserving_sum(values: list[float], total: float) -> tuple[float, ...]:
    """Round to whole seconds so that the sum stays equal to the (whole) total."""
    floors = [int(v) for v in values]
    remainder = round(total) - sum(floors)
    order = sorted(range(len(values)), key=lambda i: values[i] - floors[i], reverse=True)
    for i in order[: max(remainder, 0)]:
        floors[i] += 1
    return tuple(float(v) for v in floors)
