"""Scenario model (docs/method.md, section 1).

Two layers:

- `ScenarioDraft` is what the LLM or a bundled example supplies. Required: junction type,
  demand per approach, control type. Everything else may be missing.
- `Scenario` is complete and validated. `resolve()` turns a draft into a scenario, filling
  gaps from config.yaml and recording each filled field's dotted path in `Scenario.assumed`,
  so the form can highlight it. `Scenario.with_values()` applies form edits and clears the
  assumed mark of every edited field.

Every field has one of three states (`Scenario.field_state`):
- stated: the text gave it, or the user set it in the form.
- interpreted: the text allowed several readings and the model chose one. The model's note is
  kept in `Scenario.interpreted`. Interpreted demand must be confirmed before a run.
- assumed: a default. `assumed` lists the ones the form highlights; `implicit` lists defaults
  that are not highlighted (an unused bus lane, a plan fixed by the geometry).

Conventions: right-hand traffic; an approach is named after the arm vehicles come from;
lanes are listed from the leftmost to the rightmost; shares are in percent.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

from junctionlab.config import Config, get_config

Arm = Literal["north", "east", "south", "west"]
Movement = Literal["left", "through", "right"]
JunctionType = Literal["four_arm", "t_junction"]
MajorRoad = Literal["north-south", "east-west"]
PhasePlan = Literal["two_phase", "four_phase_split"]

ARMS: tuple[Arm, ...] = ("north", "east", "south", "west")  # clockwise
MOVEMENTS: tuple[Movement, ...] = ("left", "through", "right")
_TURN_OFFSET = {"left": 1, "through": 2, "right": 3}  # right-hand traffic, clockwise arms
MAX_LANES = 4
SHARE_TOLERANCE_PCT = 0.1
CYCLE_TOLERANCE_S = 0.5
DEMAND_FIELDS = ("demand_vph", "bus_demand_bph")
FieldState = Literal["stated", "interpreted", "assumed"]


class UnconfirmedDemand(ValueError):
    """A run was requested while interpreted demand is still unconfirmed."""

    def __init__(self, paths: tuple[str, ...]):
        self.paths = paths
        super().__init__("Confirm the interpreted demand before running: " + ", ".join(paths))


def destination(arm: Arm, movement: Movement) -> Arm:
    """Arm a vehicle leaves by. From the north approach, a left turn exits east."""
    return ARMS[(ARMS.index(arm) + _TURN_OFFSET[movement]) % 4]


def opposite(arm: Arm) -> Arm:
    return destination(arm, "through")


def road_of(arm: Arm) -> MajorRoad:
    return "north-south" if arm in ("north", "south") else "east-west"


def _check_arms(junction_type: JunctionType, arms: set[str]) -> None:
    if junction_type == "four_arm" and arms != set(ARMS):
        raise ValueError("a 4-arm junction needs approaches north, east, south and west")
    if junction_type == "t_junction" and len(arms) != 3:
        raise ValueError("a T-junction needs exactly three approaches")


def _config(info: ValidationInfo | None) -> Config:
    ctx = info.context if info is not None else None
    return (ctx or {}).get("config") or get_config()


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------- complete scenario

LaneUse = Annotated[tuple[Movement, ...], Field(min_length=1, max_length=3)]


class Approach(_Strict):
    demand_vph: float = Field(ge=0, description="Cars and heavy vehicles, veh/h. Buses separate.")
    lanes: tuple[LaneUse, ...] = Field(min_length=1, max_length=MAX_LANES)
    turning_shares_pct: dict[Movement, float]
    heavy_vehicle_share_pct: float = Field(ge=0, le=30)
    bus_lane: bool = False
    bus_demand_bph: float = Field(default=0, ge=0)
    bus_right_share_pct: float = Field(default=0, ge=0, le=100)

    @model_validator(mode="after")
    def _check(self) -> Approach:
        if len(self.lanes) + self.bus_lane > MAX_LANES:
            raise ValueError(f"at most {MAX_LANES} lanes including the bus lane")
        for lane in self.lanes:
            order = [MOVEMENTS.index(m) for m in lane]
            if order != sorted(set(order)):
                raise ValueError(f"lane {list(lane)}: list movements once, left to right")
        for left_lane, right_lane in zip(self.lanes, self.lanes[1:], strict=False):
            if MOVEMENTS.index(left_lane[-1]) > MOVEMENTS.index(right_lane[0]):
                raise ValueError(
                    f"lanes {list(left_lane)} and {list(right_lane)} cross: "
                    "movements must not cross between lanes"
                )
        shares = self.turning_shares_pct
        if any(v < 0 for v in shares.values()):
            raise ValueError("turning shares must be >= 0")
        if abs(sum(shares.values()) - 100) > SHARE_TOLERANCE_PCT:
            raise ValueError(f"turning shares must sum to 100%, got {sum(shares.values()):g}%")
        served = {m for lane in self.lanes for m in lane}
        unserved = [m for m, v in shares.items() if v > 0 and m not in served]
        if unserved:
            raise ValueError(f"no lane serves movement(s) {unserved}")
        return self

    @property
    def lane_count(self) -> int:
        """Total inbound lanes, including the bus lane. The exit leg mirrors this."""
        return len(self.lanes) + self.bus_lane


class FixedTimeSignal(_Strict):
    kind: Literal["signal"] = "signal"
    cycle_s: float = Field(gt=0)
    phase_plan: PhasePlan
    green_s: tuple[float, ...] | None = Field(
        default=None,
        description="Green time per phase in phase order. Filled with Webster splits if not given.",
    )


class PriorityRule(_Strict):
    kind: Literal["priority"] = "priority"
    major_road: MajorRoad


class AllWayStop(_Strict):
    kind: Literal["all_way_stop"] = "all_way_stop"


Control = Annotated[FixedTimeSignal | PriorityRule | AllWayStop, Field(discriminator="kind")]


def phase_count(junction_type: JunctionType, plan: PhasePlan) -> int:
    return 4 if (junction_type == "four_arm" and plan == "four_phase_split") else 2


class Scenario(_Strict):
    name: str = "Scenario"
    junction_type: JunctionType
    speed_limit_kmh: float = Field(gt=0, le=130)
    approach_length_m: float = Field(ge=50, le=2000)
    approaches: dict[Arm, Approach]
    control: Control
    assumed: tuple[str, ...] = Field(
        default=(), description="Dotted paths of fields filled from defaults."
    )
    implicit: tuple[str, ...] = Field(
        default=(), description="Defaults that the form does not highlight."
    )
    interpreted: dict[str, str] = Field(
        default_factory=dict, description="Dotted path -> the model's note on its reading."
    )
    confirmed: tuple[str, ...] = Field(
        default=(), description="Interpreted fields the user has accepted."
    )

    # ---- derived geometry

    @property
    def missing_arm(self) -> Arm | None:
        missing = [a for a in ARMS if a not in self.approaches]
        return missing[0] if missing else None

    @property
    def stem(self) -> Arm | None:
        """Minor arm of a T-junction: the one opposite the missing arm."""
        return opposite(self.missing_arm) if self.missing_arm else None

    def allowed_movements(self, arm: Arm) -> tuple[Movement, ...]:
        return tuple(m for m in MOVEMENTS if destination(arm, m) in self.approaches)

    # ---- validation

    @model_validator(mode="after")
    def _check(self, info: ValidationInfo) -> Scenario:
        cfg = _config(info)
        _check_arms(self.junction_type, set(self.approaches))

        for arm, approach in self.approaches.items():
            allowed = set(self.allowed_movements(arm))
            used = {m for lane in approach.lanes for m in lane}
            used |= {m for m, v in approach.turning_shares_pct.items() if v > 0}
            bad = used - allowed
            if bad:
                raise ValueError(f"{arm} approach: movement(s) {sorted(bad)} lead to no arm")
            if approach.bus_demand_bph > 0:
                bus_moves = set()
                if approach.bus_right_share_pct < 100:
                    bus_moves.add("through")
                if approach.bus_right_share_pct > 0:
                    bus_moves.add("right")
                if bus_moves - allowed:
                    raise ValueError(
                        f"{arm} approach: buses go straight or right; "
                        f"{sorted(bus_moves - allowed)} is not possible here"
                    )

        c = self.control
        if isinstance(c, FixedTimeSignal):
            if self.junction_type == "t_junction" and c.phase_plan != "two_phase":
                raise ValueError("a T-junction signal uses the two-phase plan")
            n = phase_count(self.junction_type, c.phase_plan)
            ig, gmin = cfg.control.intergreen_s, cfg.control.min_green_s
            if c.cycle_s < n * (gmin + ig):
                raise ValueError(
                    f"cycle {c.cycle_s:g} s is too short for {n} phases "
                    f"(min green {gmin:g} s + intergreen {ig:g} s each)"
                )
            if c.green_s is not None:
                if len(c.green_s) != n:
                    raise ValueError(f"{c.phase_plan} needs {n} green times, got {len(c.green_s)}")
                if min(c.green_s) < gmin:
                    raise ValueError(f"every green must be at least {gmin:g} s")
                total = sum(c.green_s) + n * ig
                if abs(total - c.cycle_s) > CYCLE_TOLERANCE_S:
                    raise ValueError(
                        f"greens + intergreens = {total:g} s, but the cycle is {c.cycle_s:g} s"
                    )
        if isinstance(c, PriorityRule) and self.stem and c.major_road == road_of(self.stem):
            raise ValueError("on a T-junction the stem is always the minor road")
        return self

    # ---- naming

    def generated_name(self) -> str:
        """A short name from the scenario, such as '4-arm signal, 600 veh/h each'."""
        junction = "4-arm" if self.junction_type == "four_arm" else "T-junction"
        control = {"signal": "signal", "priority": "priority", "all_way_stop": "all-way stop"}
        demands = [round(a.demand_vph) for a in self.approaches.values()]
        low, high = min(demands), max(demands)
        volume = f"{low} veh/h each" if low == high else f"{low}–{high} veh/h"
        return f"{junction} {control[self.control.kind]}, {volume}"

    # ---- form edits

    def is_assumed(self, path: str) -> bool:
        return path in self.assumed

    def field_paths(self) -> list[str]:
        """Dotted path of every field that has a state."""
        paths = ["junction_type", "speed_limit_kmh", "approach_length_m"]
        for arm in self.approaches:
            paths += [f"approaches.{arm}.{name}" for name in Approach.model_fields]
        paths += [f"control.{name}" for name in type(self.control).model_fields]
        return paths

    def field_state(self, path: str) -> FieldState:
        if path in self.interpreted:
            return "interpreted"
        return "assumed" if path in self.assumed or path in self.implicit else "stated"

    def field_states(self) -> dict[str, FieldState]:
        return {path: self.field_state(path) for path in self.field_paths()}

    @property
    def needs_confirmation(self) -> tuple[str, ...]:
        """Interpreted demand fields the user has not confirmed yet."""
        return tuple(
            path
            for path in self.interpreted
            if path.rsplit(".", 1)[-1] in DEMAND_FIELDS and path not in self.confirmed
        )

    def require_confirmed(self) -> None:
        if self.needs_confirmation:
            raise UnconfirmedDemand(self.needs_confirmation)

    def confirm(self, paths: Iterable[str] | None = None) -> Scenario:
        """Accept interpreted fields as read (all that need confirmation if none are named)."""
        chosen = tuple(self.needs_confirmation if paths is None else paths)
        unknown = [path for path in chosen if path not in self.interpreted]
        if unknown:
            raise ValueError(f"not interpreted fields: {unknown}")
        return self.model_copy(update={"confirmed": tuple(dict.fromkeys(self.confirmed + chosen))})

    def with_values(self, updates: dict[str, Any], config: Config | None = None) -> Scenario:
        """Apply edits by dotted path (e.g. 'control.cycle_s') and mark them as confirmed.

        Editing a parent clears assumed marks below it. Raises ValidationError if the result
        is invalid.
        """
        data = self.model_dump(mode="json")
        for path, value in updates.items():
            *parents, leaf = path.split(".")
            node = data
            for key in parents:
                node = node[key]
            node[leaf] = value

        def untouched(a: str) -> bool:
            return not any(
                a == p or a.startswith(p + ".") or p.startswith(a + ".") for p in updates
            )

        data["assumed"] = [a for a in self.assumed if untouched(a)]
        data["implicit"] = [a for a in self.implicit if untouched(a)]
        data["interpreted"] = {a: n for a, n in self.interpreted.items() if untouched(a)}
        data["confirmed"] = [a for a in self.confirmed if untouched(a)]
        cfg = config or get_config()
        if "control.green_s" in data["assumed"]:
            data["control"]["green_s"] = None  # re-derived below from the edited values
        edited = _validate_with_greens(data, cfg)
        if self.name == self.generated_name():  # a generated name follows the edited values
            edited = edited.model_copy(update={"name": edited.generated_name()})
        return edited


# --------------------------------------------------------------------------- draft


class ApproachDraft(_Strict):
    demand_vph: float = Field(ge=0)
    lane_count: int | None = Field(
        default=None, ge=1, le=MAX_LANES, description="General-traffic lanes, if use not given."
    )
    lanes: list[list[Movement]] | None = None
    turning_shares_pct: dict[Movement, float] | None = None
    heavy_vehicle_share_pct: float | None = None
    bus_lane: bool | None = None
    bus_demand_bph: float | None = None
    bus_right_share_pct: float | None = None


class ControlDraft(_Strict):
    kind: Literal["signal", "priority", "all_way_stop"]
    cycle_s: float | None = None
    phase_plan: PhasePlan | None = None
    green_s: list[float] | None = None
    major_road: MajorRoad | None = None


class ScenarioDraft(_Strict):
    """What the text states. Missing optional fields are filled by resolve()."""

    name: str | None = None
    junction_type: JunctionType
    speed_limit_kmh: float | None = None
    approach_length_m: float | None = None
    approaches: dict[Arm, ApproachDraft]
    control: ControlDraft
    interpreted: dict[str, str] = Field(
        default_factory=dict, description="Dotted path -> the model's note on its reading."
    )

    @model_validator(mode="after")
    def _arm_count(self) -> ScenarioDraft:
        _check_arms(self.junction_type, set(self.approaches))
        return self


def standard_lanes(count: int) -> list[list[Movement]]:
    """Lane use for a lane count: leftmost left+through, rightmost through+right, middle through."""
    if count == 1:
        return [["left", "through", "right"]]
    return [["left", "through"], *[["through"]] * (count - 2), ["through", "right"]]


def _strip(lanes: list[list[Movement]], allowed: tuple[Movement, ...]) -> list[list[Movement]]:
    """Drop movements that lead to no arm (T-junction); drop lanes left empty."""
    kept = [[m for m in lane if m in allowed] for lane in lanes]
    return [lane for lane in kept if lane]


def _default_shares(
    cfg: Config, junction_type: JunctionType, arm: Arm, stem: Arm | None
) -> dict[Movement, float]:
    d = cfg.scenario_defaults.turning_shares_pct
    if junction_type == "four_arm":
        return dict(d.four_arm)
    if arm == stem:
        return dict(d.t_stem)
    into_stem: Movement = "left" if destination(arm, "left") == stem else "right"
    return {"through": d.t_major["through"], into_stem: d.t_major["into_stem"]}


def resolve(draft: ScenarioDraft, config: Config | None = None) -> Scenario:
    """Fill a draft's gaps from config and record each filled field as assumed."""
    cfg = config or get_config()
    assumed: list[str] = []
    implicit: list[str] = []

    def pick(value: Any, default: Any, path: str) -> Any:
        if value is None:
            assumed.append(path)
            return default
        return value

    missing = [a for a in ARMS if a not in draft.approaches]
    stem = opposite(missing[0]) if draft.junction_type == "t_junction" and missing else None
    arms_present = set(draft.approaches)

    approaches: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        a = draft.approaches.get(arm)
        if a is None:
            continue
        p = f"approaches.{arm}"
        allowed = tuple(m for m in MOVEMENTS if destination(arm, m) in arms_present)
        if a.lanes is not None:
            lanes = a.lanes
        else:
            assumed.append(f"{p}.lanes")
            base = (
                standard_lanes(a.lane_count)
                if a.lane_count is not None
                else [list(lane) for lane in cfg.scenario_defaults.lanes]
            )
            lanes = _strip(base, allowed)
        bus_demand = a.bus_demand_bph if a.bus_demand_bph is not None else 0
        if a.bus_lane is None:
            implicit.append(f"{p}.bus_lane")
        if a.bus_demand_bph is None:
            implicit.append(f"{p}.bus_demand_bph")
        if bus_demand <= 0 and a.bus_right_share_pct is None:
            implicit.append(f"{p}.bus_right_share_pct")
        approaches[arm] = {
            "demand_vph": a.demand_vph,
            "lanes": lanes,
            "turning_shares_pct": pick(
                a.turning_shares_pct,
                _default_shares(cfg, draft.junction_type, arm, stem),
                f"{p}.turning_shares_pct",
            ),
            "heavy_vehicle_share_pct": pick(
                a.heavy_vehicle_share_pct,
                cfg.scenario_defaults.heavy_vehicle_share_pct,
                f"{p}.heavy_vehicle_share_pct",
            ),
            # No bus lane and no buses are the absence of a feature, not an assumption.
            "bus_lane": bool(a.bus_lane),
            "bus_demand_bph": bus_demand,
            "bus_right_share_pct": (
                pick(a.bus_right_share_pct, 0, f"{p}.bus_right_share_pct")
                if bus_demand > 0
                else (a.bus_right_share_pct or 0)
            ),
        }

    c = draft.control
    control: dict[str, Any] = {"kind": c.kind}
    if c.kind == "signal":
        control["cycle_s"] = pick(c.cycle_s, cfg.control.cycle_default_s, "control.cycle_s")
        if draft.junction_type == "t_junction":
            control["phase_plan"] = c.phase_plan or "two_phase"
            if c.phase_plan is None:
                implicit.append("control.phase_plan")
        else:
            control["phase_plan"] = pick(
                c.phase_plan, cfg.control.four_arm_phase_plan_default, "control.phase_plan"
            )
        # None is filled with Webster splits from critical flow ratios after validation.
        control["green_s"] = pick(c.green_s, None, "control.green_s")
    elif c.kind == "priority":
        if stem is not None:
            # On a T-junction the major road is fixed by geometry: the arms other than the stem.
            control["major_road"] = c.major_road or road_of(_other(stem))
            if c.major_road is None:
                implicit.append("control.major_road")
        else:
            control["major_road"] = pick(
                c.major_road, cfg.control.major_road_default, "control.major_road"
            )

    name = (draft.name or "").strip()
    if name.casefold() == "scenario":  # a placeholder, not a real name
        name = ""
    data = {
        "name": name or "Scenario",
        "junction_type": draft.junction_type,
        "speed_limit_kmh": pick(
            draft.speed_limit_kmh, cfg.geometry.speed_limit_kmh, "speed_limit_kmh"
        ),
        "approach_length_m": pick(
            draft.approach_length_m, cfg.geometry.approach_length_m, "approach_length_m"
        ),
        "approaches": approaches,
        "control": control,
        "assumed": sorted(a for a in assumed if a not in draft.interpreted),
        "implicit": sorted(a for a in implicit if a not in draft.interpreted),
        "interpreted": dict(draft.interpreted),
    }
    scenario = _validate_with_greens(data, cfg)
    return scenario if name else scenario.model_copy(update={"name": scenario.generated_name()})


def _validate_with_greens(data: dict[str, Any], cfg: Config) -> Scenario:
    """Validate; for a signal without green times, derive them with Webster's split method."""
    scenario = Scenario.model_validate(data, context={"config": cfg})
    c = scenario.control
    if isinstance(c, FixedTimeSignal) and c.green_s is None:
        from junctionlab.signal import webster_greens

        data = scenario.model_dump(mode="json")
        data["control"]["green_s"] = list(webster_greens(scenario, cfg))
        scenario = Scenario.model_validate(data, context={"config": cfg})
    return scenario


def _other(stem: Arm) -> Arm:
    """An arm on the major road of a T-junction (perpendicular to the stem)."""
    return destination(stem, "left")
