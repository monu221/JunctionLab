"""Text to scenario with a configurable LLM provider (docs/method.md, section 1).

The provider is `llm.provider` in config.yaml: Groq (default, free tier) or Anthropic.

The LLM only fills `Interpretation`, a structured-output schema. Python then turns it into a
`ScenarioDraft` and `resolve()` fills the gaps from config.yaml. The LLM never writes SUMO
files.

Structured outputs need `additionalProperties: false` on every object, so the LLM schema has
one field per arm and per movement instead of the dicts that `ScenarioDraft` uses.

Failures fall into three kinds, each with its own exception:
- `InterpretationError`: the text could not be turned into a valid scenario. Ask the user to
  rephrase; the reasons say what was missing or wrong.
- `LimitReached`: the per-session or daily call limit is used up.
- `LlmUnavailable`: no API key, or the API could not be reached or refused the request.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import anthropic
import httpx2
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from junctionlab.config import REPO_ROOT, Config, get_config
from junctionlab.scenario import (
    ARMS,
    MOVEMENTS,
    ApproachDraft,
    Arm,
    ControlDraft,
    JunctionType,
    MajorRoad,
    Movement,
    PhasePlan,
    Scenario,
    ScenarioDraft,
    destination,
    resolve,
)

API_KEY_NAMES = {"groq": "GROQ_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_TIMEOUT_S = 60.0
REPHRASE_HINT = (
    "Please rephrase. Say what kind of junction it is (4-arm or T-junction), the traffic "
    "on each approach in vehicles per hour, and how it is controlled (signal, priority or "
    "all-way stop)."
)


# --------------------------------------------------------------------------- exceptions


class LlmError(Exception):
    """Base class; the message is shown to the user."""


class InterpretationError(LlmError):
    def __init__(self, reasons: list[str]):
        self.reasons = reasons
        super().__init__("Could not read the junction description: " + "; ".join(reasons))


class LimitReached(LlmError):
    pass


class LlmUnavailable(LlmError):
    pass


class RateLimited(LlmUnavailable):
    """The provider's free-tier limit is reached; the bundled examples still work."""


# --------------------------------------------------------------------------- LLM schema


class _Llm(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TurningShares(_Llm):
    left: float | None = Field(description="Percent of the approach's vehicles turning left.")
    through: float | None = Field(description="Percent going straight on.")
    right: float | None = Field(description="Percent turning right.")


class ApproachText(_Llm):
    demand_vph: float | None = Field(
        description="Vehicles per hour arriving on this approach (cars and lorries, not buses)."
    )
    lane_count: int | None = Field(
        description="Number of general-traffic lanes, if stated without saying what each is for."
    )
    lanes: list[list[Movement]] | None = Field(
        description="Movements allowed in each lane, listed from the leftmost lane to the "
        "rightmost. Only if the text says what each lane is for."
    )
    turning_shares_pct: TurningShares | None
    heavy_vehicle_share_pct: float | None = Field(description="Percent lorries/heavy vehicles.")
    bus_lane: bool | None = Field(description="True if the approach has a bus lane.")
    bus_demand_bph: float | None = Field(description="Buses per hour on this approach.")
    bus_right_share_pct: float | None = Field(description="Percent of buses turning right.")


class Approaches(_Llm):
    north: ApproachText | None = Field(description="Traffic coming from the north.")
    east: ApproachText | None = Field(description="Traffic coming from the east.")
    south: ApproachText | None = Field(description="Traffic coming from the south.")
    west: ApproachText | None = Field(description="Traffic coming from the west.")


class ControlText(_Llm):
    kind: Literal["signal", "priority", "all_way_stop"] | None
    cycle_s: float | None = Field(description="Signal cycle time in seconds.")
    phase_plan: PhasePlan | None = Field(
        description="two_phase: north-south then east-west. four_phase_split: one approach at "
        "a time, in the order north, east, south, west."
    )
    green_s: list[float] | None = Field(description="Green time per phase, in phase order.")
    major_road: MajorRoad | None = Field(description="Road with right of way (priority rule).")


ApproachField = Literal[
    "demand_vph",
    "lane_count",
    "lanes",
    "turning_shares_pct",
    "heavy_vehicle_share_pct",
    "bus_lane",
    "bus_demand_bph",
    "bus_right_share_pct",
]
ScenarioField = Literal[
    "junction_type",
    "speed_limit_kmh",
    "approach_length_m",
    "control.kind",
    "control.cycle_s",
    "control.phase_plan",
    "control.green_s",
    "control.major_road",
]
APPROACH_FIELDS = frozenset(ApproachField.__args__)


class InterpretedField(_Llm):
    arm: Arm | None = Field(
        description="The approach this applies to. Null if it applies to every approach, or "
        "if the field is not an approach field."
    )
    field_name: Literal[ApproachField, ScenarioField] = Field(
        description="The field you chose a value for."
    )
    note: str = Field(
        description="One sentence: the words in the text, the reading you chose, and the other "
        "possible reading."
    )


class Interpretation(_Llm):
    understood: bool = Field(
        description="False if the text does not describe a road junction this tool can model."
    )
    unclear: list[str] = Field(
        description="Problems with the required facts only (junction type, demand on every "
        "approach, control type): missing or contradictory. Empty if none."
    )
    notes: list[str] = Field(
        description="Conversions you made to fill the schema, such as scaling counts to an hour."
    )
    not_simulated: list[str] = Field(
        description="Each detail in the text that the schema or simulation cannot represent, as "
        "a short phrase, such as 'pedestrian crossing on the west arm'. Empty if none."
    )
    interpreted: list[InterpretedField] = Field(
        description="Every field where the text supports more than one reasonable value and "
        "you chose one. Empty if none."
    )
    name: str | None = Field(description="Short name for the scenario.")
    junction_type: JunctionType | None
    speed_limit_kmh: float | None
    approach_length_m: float | None
    approaches: Approaches
    control: ControlText


SYSTEM_PROMPT = """\
You turn a plain-language description of a road junction into a structured scenario for a \
traffic simulation. Fill the schema from what the text states. Use null for anything the \
text does not state: the tool fills gaps with documented defaults and shows the user which \
values were assumed, so a guess from you would hide an assumption.

Conventions:
- Traffic drives on the right.
- An approach is named after the direction its traffic comes from: "traffic from the north", \
"northbound traffic entering from the south" is the south approach.
- junction_type is four_arm (a crossroads) or t_junction (three arms). On a T-junction leave \
the missing arm null itself, not an object of nulls.
- Lanes are listed from the leftmost to the rightmost. Each lane lists its movements (left, \
through, right) in that order. Give lanes only when the text says what each lane is for; \
otherwise give lane_count if the number of lanes is stated.
- A bus lane is extra to the general-traffic lanes. Buses go straight or turn right.
- Demand is vehicles per hour per approach. If the text gives counts for another period, \
convert them to an hourly rate. If it gives counts per movement, add them up for demand_vph \
and give the turning shares in percent. Say what you converted in notes.
- Turning shares are percentages of the approach's vehicles. Give them only if the text \
states them or the counts that determine them; if two of the three are stated, the third is \
what is left to 100. Movements that lead to a missing arm are 0. Leave turning_shares_pct null \
for every approach whose shares the text does not give; never fill a gap with 100% through.
- Signal: cycle_s, phase_plan and green_s only if stated. Priority rule: major_road only if \
stated. "Stop signs on all arms" or "4-way stop" is all_way_stop.

Required: the junction type, the demand on every approach, and the control type. If any of \
these is missing or contradictory, set understood to true but list each problem in unclear. \
If the text is not about a road junction, or describes something this tool cannot model \
(a roundabout, more than four arms, a junction with no traffic control information that \
can be mapped), set understood to false and explain in unclear. Never list optional details \
in unclear (lanes, turning shares, signal timing, phase plan, major road, speed, length, \
heavy vehicles): leave them null and the tool fills them.

Report your readings in interpreted. Add an entry whenever the text supports more than one \
reasonable value for a field and you chose one. Two common cases for demand: a volume given \
"per street" or "per direction pair" may be for one direction or for both together, and \
counts with no time period may be hourly or for another period. List every approach affected, \
or use a null arm if it is all of them. Do not list exact arithmetic on stated facts (scaling \
counts whose period is stated, adding movements, the remainder to 100), and do not list a value \
just because it is rounded or says "about". Facts that are simply missing are not interpreted.

List every stated detail you cannot represent in not_simulated, so the user sees it before \
the run: pedestrians and crossings, cyclists and cycle lanes, parking, trams, nearby land use, \
time of day, protected left-turn phases, turn bays, slip lanes, actuated or adaptive signals, \
gradients, weather. Use notes only for conversions. Keep unclear, notes and not_simulated \
short, one item each."""


# --------------------------------------------------------------------------- conversion

NO_PERIOD_NOTE = (
    "The text gives traffic counts with no time period; they were read as vehicles per hour, "
    "but they could cover another period."
)
PER_ROAD_NOTE = (
    "The text gives the volume per road; it was read as the volume on each approach, but it "
    "could be both directions together."
)
_PER_ROAD = re.compile(
    r"\b(?:each|every|per|both)\s+(?:of\s+the\s+)?(?:roads?|streets?)\b"
    r"|\bper\s+direction\s+pair\b|\b(?:in\s+)?both\s+directions\b|\bboth\s+ways\b"
    r"|\btwo[\s-]way\s+(?:flow|volume|traffic)\b",
    re.IGNORECASE,
)
COUNTS_WITHOUT_UNITS = (
    "the text lists traffic counts but gives no unit or time period; say what they measure, "
    'for example "800 veh/h from the north"'
)
_RUSH_HOUR = re.compile(r"\brush[\s-]*hours?\b", re.I)
_CYCLE_TIME = re.compile(r"\d+(\.\d+)?[\s-]*(minutes?|mins?)\b[\s-]*(signal\s+)?cycle", re.I)
_PERIOD_TEXT = re.compile(
    r"\bhours?\b|\bhourly\b|\bhrs?\b|/\s*h\b|\bvph\b|\bminutes?\b|\bmins?\b|\bper day\b|\bdaily\b",
    re.IGNORECASE,
)
_DEMAND_TEXT = re.compile(r"demand|volume|traffic|veh", re.IGNORECASE)
_JUNCTION_TEXT = re.compile(r"junction.?type|t-junction|four.arm|\barms?\b", re.IGNORECASE)
# A number with no unit after it: "N 800", "800, 600", but not "90 s", "2 lanes" or "4-arm".
# Case-sensitive, so the compass letters in "800 S 600 E" are not read as seconds.
_BARE_NUMBER = re.compile(
    r"(?<![\d.])\d+(?:\.\d+)?(?!\d|\.\d)(?!\s*-?\s*(?:%|percent|s\b|secs?\b|seconds?\b|m\b|"
    r"met(?:er|re)s?\b|km|kph|mph|lanes?\b|arms?\b|way\b|legs?\b|phases?\b|cycle\b|bus))"
)
# Backstop for details the model may leave out of not_simulated: (pattern, label).
_NOT_SIMULATED_TEXT = (
    (r"pedestrian|zebra|crosswalk|foot(way|path)|sidewalk", "Pedestrians or pedestrian crossings"),
    (r"cyclists?\b|bicycle|\bbikes?\b|cycle[\s-]*(lane|path|track)", "Cyclists or cycle lanes"),
    (r"\bpark(ing|ed)\b", "Parking"),
    (r"\btrams?\b|light rail|level crossing", "Trams or rail"),
    (r"protected left|left[\s-]*turn (arrow|phase)", "Protected left-turn phases"),
    (r"\bactuated\b|\badaptive\b|demand[\s-]responsive", "Actuated or adaptive signals"),
    (r"turn(ing)?[\s-]*(bay|pocket)|slip[\s-]*(lane|road)", "Turn bays or slip lanes"),
)


def _shares(
    s: TurningShares | None, allowed: tuple[Movement, ...], arm: str, notes: list[str]
) -> dict[Movement, float] | None:
    if s is None:
        return None
    values: dict[Movement, float | None] = {m: getattr(s, m) for m in MOVEMENTS}
    if all(v is None for v in values.values()):
        return None
    for m in MOVEMENTS:
        if m not in allowed and values[m] is None:
            values[m] = 0.0
    missing = [m for m, v in values.items() if v is None]
    if len(missing) == 1:
        rest = 100 - sum(v for v in values.values() if v is not None)
        if rest < 0:
            return {m: v or 0.0 for m, v in values.items()}  # resolve() reports the bad sum
        values[missing[0]] = rest
    elif missing:
        notes.append(f"{arm}: turning shares only partly stated; defaults used for all of them.")
        return None
    return {m: float(v) for m, v in values.items() if v is not None}


def _states_a_period(text: str) -> bool:
    return bool(_PERIOD_TEXT.search(_CYCLE_TIME.sub(" ", _RUSH_HOUR.sub(" ", text))))


def _lists_counts_without_units(text: str, approaches: int) -> bool:
    """True if the text has a unit-less number for every approach and no time period."""
    return not _states_a_period(text) and len(_BARE_NUMBER.findall(text)) >= approaches


def not_simulated_items(text: str, reported: list[str]) -> list[str]:
    """The model's list, plus known details in the text that it left out."""
    items = _dedupe([r.strip() for r in reported if r.strip()])
    for pattern, label in _NOT_SIMULATED_TEXT:
        if re.search(pattern, text, re.I) and not any(re.search(pattern, i, re.I) for i in items):
            items.append(label)
    return items


def to_draft(interp: Interpretation, text: str = "") -> tuple[ScenarioDraft, list[str]]:
    """Turn the LLM output into a draft. Raises InterpretationError if something required is
    missing. Returns the draft and notes for the user. `text` lets the message tell counts
    without units apart from traffic that is really missing."""
    notes = list(interp.notes)
    if not interp.understood:
        raise InterpretationError(interp.unclear or ["the text does not describe a junction"])
    given = {arm: getattr(interp.approaches, arm) for arm in ARMS}
    given = {arm: a for arm, a in given.items() if a is not None}
    if interp.junction_type == "t_junction" and len(given) == 4:
        # The model sometimes writes the missing arm as an object of nulls instead of null.
        empty = [arm for arm, a in given.items() if all(v is None for v in a.model_dump().values())]
        if len(empty) == 1:
            del given[empty[0]]
    no_demand = [arm for arm, a in given.items() if a.demand_vph is None]
    # Python reports missing demand once, so drop the model's own lines about it.
    problems = [u for u in interp.unclear if not (no_demand and _DEMAND_TEXT.search(u))]
    if interp.junction_type is None and not any(_JUNCTION_TEXT.search(u) for u in problems):
        problems.append("the junction type is not stated")
    if interp.control.kind is None:
        problems.append("the traffic control (signal, priority or all-way stop) is not stated")
    if not given:
        problems.append("no approaches with traffic are described")
    if no_demand and _lists_counts_without_units(text, len(given)):
        problems.append(COUNTS_WITHOUT_UNITS)
    elif no_demand:
        arms = (
            ", ".join(no_demand[:-1]) + " and " + no_demand[-1] if no_demand[1:] else no_demand[0]
        )
        plural = "approaches" if no_demand[1:] else "approach"
        problems.append(f"the traffic on the {arms} {plural} is not stated")
    if problems:
        raise InterpretationError(_dedupe(problems))

    approaches: dict[str, ApproachDraft] = {}
    for arm, a in given.items():
        allowed = tuple(m for m in MOVEMENTS if destination(arm, m) in given)
        approaches[arm] = ApproachDraft(
            demand_vph=a.demand_vph,
            lane_count=a.lane_count if a.lanes is None else None,
            lanes=a.lanes,
            turning_shares_pct=_shares(a.turning_shares_pct, allowed, arm, notes),
            heavy_vehicle_share_pct=a.heavy_vehicle_share_pct,
            bus_lane=a.bus_lane,
            bus_demand_bph=a.bus_demand_bph,
            bus_right_share_pct=a.bus_right_share_pct,
        )
    c = interp.control
    try:
        draft = ScenarioDraft(
            name=interp.name,
            junction_type=interp.junction_type,
            speed_limit_kmh=interp.speed_limit_kmh,
            approach_length_m=interp.approach_length_m,
            approaches=approaches,
            interpreted=_interpreted_paths(interp, approaches),
            control=ControlDraft(
                kind=c.kind,
                cycle_s=c.cycle_s,
                phase_plan=c.phase_plan,
                green_s=c.green_s,
                major_road=c.major_road,
            ),
        )
    except ValidationError as e:
        raise InterpretationError(_messages(e)) from e
    return draft, notes


def _interpreted_paths(
    interp: Interpretation, approaches: dict[str, ApproachDraft]
) -> dict[str, str]:
    """Dotted path -> note, for fields that hold a value the model actually read."""
    out: dict[str, str] = {}
    for item in interp.interpreted:
        name = item.field_name
        if name not in APPROACH_FIELDS:
            value: Any = interp
            for key in name.split("."):
                value = getattr(value, key)
            if value is not None:
                out.setdefault(name, item.note)
            continue
        for arm in [item.arm] if item.arm else list(approaches):
            draft = approaches.get(arm)
            if draft is None:
                continue
            if name == "lane_count":  # the scenario holds lanes, not a count
                if draft.lane_count is not None or draft.lanes is not None:
                    out.setdefault(f"approaches.{arm}.lanes", item.note)
            elif getattr(draft, name) is not None:
                out.setdefault(f"approaches.{arm}.{name}", item.note)
    return out


def _flag_per_road_volume(text: str, draft: ScenarioDraft) -> ScenarioDraft:
    """Mark demand as interpreted when a volume is given per road rather than per approach.

    "600 an hour on each road" may be per direction or both together. A check in code, because
    the model does not report this case reliably.
    """
    if not _PER_ROAD.search(text):
        return draft
    interpreted = dict(draft.interpreted)
    for arm in draft.approaches:
        interpreted.setdefault(f"approaches.{arm}.demand_vph", PER_ROAD_NOTE)
    return draft.model_copy(update={"interpreted": interpreted})


def _flag_missing_period(text: str, draft: ScenarioDraft) -> ScenarioDraft:
    """Mark demand as interpreted when the text states no time period at all.

    A check in code, because the model does not report this case reliably.
    """
    if _states_a_period(text):
        return draft
    interpreted = dict(draft.interpreted)
    for arm, a in draft.approaches.items():
        interpreted.setdefault(f"approaches.{arm}.demand_vph", NO_PERIOD_NOTE)
        if a.bus_demand_bph:
            interpreted.setdefault(f"approaches.{arm}.bus_demand_bph", NO_PERIOD_NOTE)
    return draft.model_copy(update={"interpreted": interpreted})


def _messages(e: ValidationError) -> list[str]:
    out = []
    for err in e.errors():
        msg = err["msg"].removeprefix("Value error, ")
        where = ".".join(str(x) for x in err["loc"])
        out.append(f"{where}: {msg}" if where else msg)
    return _dedupe(out)


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


# --------------------------------------------------------------------------- limits


@dataclass
class LlmSession:
    """Per-user state; the app keeps one per browser session."""

    calls: int = 0


class CallLimiter:
    """Per-session and daily call limits. The daily count lives in a JSON file keyed by the
    UTC date; it is not shared between servers, and the API spending limit is the hard cap."""

    _lock = threading.Lock()

    def __init__(self, cfg: Config, counter_file: Path | None = None, now=None):
        self.per_session = cfg.llm.calls_per_session
        self.per_day = cfg.llm.calls_per_day
        path = counter_file or cfg.llm.counter_file
        self.path = path if path.is_absolute() else REPO_ROOT / path
        self._now = now or (lambda: datetime.now(UTC))

    def _today(self) -> str:
        return self._now().date().isoformat()

    def used_today(self) -> int:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return 0
        return int(data.get("calls", 0)) if data.get("date") == self._today() else 0

    def reserve(self, session: LlmSession) -> None:
        """Count one call before it is made, or raise LimitReached."""
        if session.calls >= self.per_session:
            raise LimitReached(
                f"This session has used its {self.per_session} descriptions. Reload the page "
                "for a new session, or edit the scenario in the form."
            )
        with self._lock:
            used = self.used_today()
            if used >= self.per_day:
                raise LimitReached(
                    f"The daily limit of {self.per_day} descriptions is used up. Try again "
                    "tomorrow, or start from a bundled example."
                )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"date": self._today(), "calls": used + 1}), "utf-8")
            tmp.replace(self.path)
        session.calls += 1


# --------------------------------------------------------------------------- client


def api_key(provider: str | None = None) -> str | None:
    """Streamlit secrets first (deployment), then the environment and .env (local)."""
    name = API_KEY_NAMES[provider or get_config().llm.provider]
    try:
        import streamlit as st

        key = st.secrets.get(name)
        if key:
            return str(key)
    except Exception:  # streamlit missing, no secrets file, or not running in an app
        pass
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env", override=False)
    return os.environ.get(name) or None


class GroqClient:
    """Minimal client for Groq's OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        key: str,
        transport: httpx2.BaseTransport | None = None,
        max_retries: int = 2,
    ):
        self.max_retries = max_retries
        self._http = httpx2.Client(
            base_url=GROQ_BASE_URL,
            headers={"Authorization": f"Bearer {key}"},
            timeout=GROQ_TIMEOUT_S,
            transport=transport,
        )

    def post_chat(self, body: dict[str, Any]) -> httpx2.Response:
        """POST with retries on transient server errors. A 429 is never retried here: the
        wait can be long, so the user is told instead."""
        for attempt in range(self.max_retries + 1):
            last = attempt == self.max_retries
            try:
                response = self._http.post("/chat/completions", json=body)
            except httpx2.TransportError as e:
                if last:
                    raise LlmUnavailable("Could not reach Groq. Check the connection.") from e
            else:
                if response.status_code not in (500, 502, 503) or last:
                    return response
            time.sleep(0.5 * 2**attempt)
        raise AssertionError("unreachable")


LlmClient = anthropic.Anthropic | GroqClient


def make_client(
    key: str | None = None,
    transport: httpx2.BaseTransport | None = None,
    max_retries: int = 2,
    provider: str | None = None,
) -> LlmClient:
    """A client for the provider. `transport` replaces the network (tests, recording)."""
    provider = provider or get_config().llm.provider
    key = key or api_key(provider)
    if not key:
        raise LlmUnavailable(
            f"No API key: set {API_KEY_NAMES[provider]} in .env or the Streamlit secrets. The "
            "bundled examples work without one."
        )
    if provider == "groq":
        return GroqClient(key, transport=transport, max_retries=max_retries)
    http_client = anthropic.DefaultHttpxClient(transport=transport) if transport else None
    return anthropic.Anthropic(api_key=key, http_client=http_client, max_retries=max_retries)


@dataclass
class Interpreted:
    scenario: Scenario
    draft: ScenarioDraft
    notes: list[str] = field(default_factory=list)
    interpretation: Interpretation | None = None
    not_simulated: list[str] = field(default_factory=list)


OUTPUT_FORMAT = {"type": "json_schema", "schema": anthropic.transform_schema(Interpretation)}


def _parse_answer(text_out: str | None) -> Interpretation:
    if text_out is None:
        raise InterpretationError(["the model gave no structured answer"])
    try:
        return Interpretation.model_validate_json(text_out)
    except ValidationError as e:
        raise InterpretationError(["the model's answer did not fit the scenario form"]) from e


def call_llm(text: str, client: LlmClient, cfg: Config) -> Interpretation:
    """One structured-output call to the client's provider. No limits are checked here."""
    if isinstance(client, GroqClient):
        return _call_groq(text, client, cfg)
    return _call_anthropic(text, client, cfg)


def groq_request(text: str, cfg: Config) -> dict[str, Any]:
    """Chat completion in strict structured-output mode, using the same schema as Anthropic."""
    return {
        "model": cfg.llm.groq.model,
        "max_completion_tokens": cfg.llm.groq.max_tokens,
        "temperature": 0,
        "reasoning_effort": cfg.llm.groq.reasoning_effort,
        "include_reasoning": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "junction_scenario",
                "strict": True,
                "schema": OUTPUT_FORMAT["schema"],
            },
        },
    }


def _wait_text(retry_after: str | None) -> str:
    try:
        seconds = float(retry_after)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "Try again in a minute."
    if seconds < 90:
        return f"Try again in about {max(1, round(seconds))} s."
    if seconds < 5400:
        return f"Try again in about {round(seconds / 60)} min."
    return f"Try again in about {round(seconds / 3600)} h."


def _groq_error(response: httpx2.Response) -> LlmError:
    status = response.status_code
    try:
        error = response.json().get("error") or {}
    except (ValueError, AttributeError):
        error = {}
    if status == 429:
        return RateLimited(
            "The free Groq tier's rate limit is reached. "
            + _wait_text(response.headers.get("retry-after"))
            + " You can still edit the scenario in the form or start from a bundled example, "
            "which needs no API call."
        )
    if status == 401:
        return LlmUnavailable("The API key was rejected.")
    if status == 403:
        return LlmUnavailable("The API key may not use this model.")
    if status == 400 and error.get("code") == "json_validate_failed":
        return InterpretationError(["the model's answer did not fit the scenario form"])
    if status in (400, 404, 413, 422):
        return LlmUnavailable(f"The API rejected the request ({status}).")
    return LlmUnavailable(f"The API returned an error ({status}). Try again.")


def _call_groq(text: str, client: GroqClient, cfg: Config) -> Interpretation:
    response = client.post_chat(groq_request(text, cfg))
    if response.status_code != 200:
        raise _groq_error(response)
    try:
        choice = response.json()["choices"][0]
        message = choice["message"]
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise LlmUnavailable("The API returned an unreadable response. Try again.") from e
    if message.get("refusal"):
        raise InterpretationError(["the model declined to process this text"])
    if choice.get("finish_reason") == "length":
        raise InterpretationError(["the model's answer was cut off"])
    return _parse_answer(message.get("content"))


def _call_anthropic(text: str, client: anthropic.Anthropic, cfg: Config) -> Interpretation:
    """Uses `messages.create` with `output_config.format` rather than `messages.parse`, so the
    stop reason can be checked before the JSON is parsed: a refused or cut-off answer is not
    reported as a schema mismatch.
    """
    try:
        response = client.messages.create(
            model=cfg.llm.anthropic.model,
            max_tokens=cfg.llm.anthropic.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": text}],
            output_config={"format": OUTPUT_FORMAT, "effort": cfg.llm.anthropic.effort},
        )
    except anthropic.AuthenticationError as e:
        raise LlmUnavailable("The API key was rejected.") from e
    except anthropic.PermissionDeniedError as e:
        raise LlmUnavailable("The API key may not use this model.") from e
    except anthropic.BadRequestError as e:
        raise LlmUnavailable(f"The API rejected the request: {e.message}") from e
    except anthropic.RateLimitError as e:
        raise LlmUnavailable("The API is rate-limited right now. Try again in a minute.") from e
    except anthropic.APIStatusError as e:
        raise LlmUnavailable(f"The API returned an error ({e.status_code}). Try again.") from e
    except anthropic.APIConnectionError as e:
        raise LlmUnavailable("Could not reach the API. Check the connection.") from e

    if response.stop_reason == "refusal":
        raise InterpretationError(["the model declined to process this text"])
    if response.stop_reason == "max_tokens":
        raise InterpretationError(["the model's answer was cut off"])
    return _parse_answer(next((b.text for b in response.content if b.type == "text"), None))


def interpret(
    text: str,
    session: LlmSession,
    client: LlmClient | None = None,
    config: Config | None = None,
    limiter: CallLimiter | None = None,
) -> Interpreted:
    """Text to a resolved scenario. Raises InterpretationError (rephrase), LimitReached or
    LlmUnavailable."""
    cfg = config or get_config()
    text = text.strip()
    if not text:
        raise InterpretationError(["the description is empty"])
    if len(text) > cfg.llm.max_input_chars:
        raise InterpretationError(
            [f"the description is longer than {cfg.llm.max_input_chars} characters"]
        )
    client = client or make_client(provider=cfg.llm.provider)
    (limiter or CallLimiter(cfg)).reserve(session)
    interp = call_llm(text, client, cfg)
    draft, notes = to_draft(interp, text)
    draft = _flag_per_road_volume(text, _flag_missing_period(text, draft))
    try:
        scenario = resolve(draft, cfg)
    except ValidationError as e:
        raise InterpretationError(_messages(e)) from e
    return Interpreted(
        scenario=scenario,
        draft=draft,
        notes=notes,
        interpretation=interp,
        not_simulated=not_simulated_items(text, interp.not_simulated),
    )


# --------------------------------------------------------------------------- recordings


class ReplayTransport(httpx2.BaseTransport):
    """Answers every request with a recorded response. Keeps the requests it saw."""

    def __init__(self, recording: dict[str, Any]):
        self.recording = recording
        self.requests: list[dict[str, Any]] = []

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.read()))
        r = self.recording["response"]
        headers = {"request-id": "req_replay", **r.get("headers", {})}
        return httpx2.Response(r["status"], json=r["body"], headers=headers)


class RecordingTransport(httpx2.HTTPTransport):
    """Passes requests to the network and saves the response for replay in tests."""

    def __init__(self, path: Path, text: str):
        super().__init__()
        self.path, self.text = path, text

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        response = super().handle_request(request)
        body = response.read()
        recording = {
            "source": "live",
            "recorded_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "text": self.text,
            "response": {"status": response.status_code, "body": json.loads(body)},
        }
        if "retry-after" in response.headers:  # the only response header replay needs
            recording["response"]["headers"] = {"retry-after": response.headers["retry-after"]}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(recording, indent=2) + "\n", encoding="utf-8")
        # The body is already decoded, so drop the headers that describe the raw stream.
        headers = {
            k: v
            for k, v in response.headers.items()
            if k.lower() not in ("content-encoding", "content-length", "transfer-encoding")
        }
        return httpx2.Response(response.status_code, headers=headers, content=body)


def load_recording(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
