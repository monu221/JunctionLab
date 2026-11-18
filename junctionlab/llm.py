"""Text to scenario with Claude (docs/method.md, section 1).

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
import threading
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

API_KEY_NAME = "ANTHROPIC_API_KEY"
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


class Interpretation(_Llm):
    understood: bool = Field(
        description="False if the text does not describe a road junction this tool can model."
    )
    unclear: list[str] = Field(
        description="Required facts that are missing or contradictory. Empty if none."
    )
    notes: list[str] = Field(
        description="Conversions you made and stated details the schema cannot hold."
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
the missing arm null.
- Lanes are listed from the leftmost to the rightmost. Each lane lists its movements (left, \
through, right) in that order. Give lanes only when the text says what each lane is for; \
otherwise give lane_count if the number of lanes is stated.
- A bus lane is extra to the general-traffic lanes. Buses go straight or turn right.
- Demand is vehicles per hour per approach. If the text gives counts for another period, \
convert them to an hourly rate. If it gives counts per movement, add them up for demand_vph \
and give the turning shares in percent. Say what you converted in notes.
- Turning shares are percentages of the approach's vehicles. Give them only if the text \
states them or the counts that determine them; if two of the three are stated, the third is \
what is left to 100. Movements that lead to a missing arm are 0.
- Signal: cycle_s, phase_plan and green_s only if stated. Priority rule: major_road only if \
stated. "Stop signs on all arms" or "4-way stop" is all_way_stop.

Required: the junction type, the demand on every approach, and the control type. If any of \
these is missing or contradictory, set understood to true but list each problem in unclear. \
If the text is not about a road junction, or describes something this tool cannot model \
(a roundabout, more than four arms, a junction with no traffic control information that \
can be mapped), set understood to false and explain in unclear.

Put details you cannot represent (pedestrians, cycle lanes, parking, time of day) in notes \
rather than dropping them silently. Keep unclear and notes short, one sentence per item."""


# --------------------------------------------------------------------------- conversion


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


def to_draft(interp: Interpretation) -> tuple[ScenarioDraft, list[str]]:
    """Turn the LLM output into a draft. Raises InterpretationError if something required is
    missing. Returns the draft and notes for the user."""
    notes = list(interp.notes)
    if not interp.understood:
        raise InterpretationError(interp.unclear or ["the text does not describe a junction"])
    problems = list(interp.unclear)
    if interp.junction_type is None:
        problems.append("the junction type is not stated")
    if interp.control.kind is None:
        problems.append("the traffic control (signal, priority or all-way stop) is not stated")
    given = {arm: getattr(interp.approaches, arm) for arm in ARMS}
    given = {arm: a for arm, a in given.items() if a is not None}
    if not given:
        problems.append("no approaches with traffic are described")
    for arm, a in given.items():
        if a.demand_vph is None:
            problems.append(f"the traffic on the {arm} approach is not stated")
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


def api_key() -> str | None:
    """Streamlit secrets first (deployment), then the environment and .env (local)."""
    try:
        import streamlit as st

        key = st.secrets.get(API_KEY_NAME)
        if key:
            return str(key)
    except Exception:  # streamlit missing, no secrets file, or not running in an app
        pass
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env", override=False)
    return os.environ.get(API_KEY_NAME) or None


def make_client(
    key: str | None = None,
    transport: httpx2.BaseTransport | None = None,
    max_retries: int = 2,
) -> anthropic.Anthropic:
    """A client for the Messages API. `transport` replaces the network (tests, recording)."""
    key = key or api_key()
    if not key:
        raise LlmUnavailable(
            f"No API key: set {API_KEY_NAME} in .env or the Streamlit secrets. The bundled "
            "examples work without one."
        )
    http_client = anthropic.DefaultHttpxClient(transport=transport) if transport else None
    return anthropic.Anthropic(api_key=key, http_client=http_client, max_retries=max_retries)


@dataclass
class Interpreted:
    scenario: Scenario
    draft: ScenarioDraft
    notes: list[str] = field(default_factory=list)
    interpretation: Interpretation | None = None


OUTPUT_FORMAT = {"type": "json_schema", "schema": anthropic.transform_schema(Interpretation)}


def call_llm(text: str, client: anthropic.Anthropic, cfg: Config) -> Interpretation:
    """One Messages API call with structured output. No limits are checked here.

    Uses `messages.create` with `output_config.format` rather than `messages.parse`, so the
    stop reason can be checked before the JSON is parsed: a refused or cut-off answer is not
    reported as a schema mismatch.
    """
    try:
        response = client.messages.create(
            model=cfg.llm.model,
            max_tokens=cfg.llm.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": text}],
            output_config={"format": OUTPUT_FORMAT, "effort": cfg.llm.effort},
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
    text_out = next((b.text for b in response.content if b.type == "text"), None)
    if text_out is None:
        raise InterpretationError(["the model gave no structured answer"])
    try:
        return Interpretation.model_validate_json(text_out)
    except ValidationError as e:
        raise InterpretationError(["the model's answer did not fit the scenario form"]) from e


def interpret(
    text: str,
    session: LlmSession,
    client: anthropic.Anthropic | None = None,
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
    client = client or make_client()
    (limiter or CallLimiter(cfg)).reserve(session)
    interp = call_llm(text, client, cfg)
    draft, notes = to_draft(interp)
    try:
        scenario = resolve(draft, cfg)
    except ValidationError as e:
        raise InterpretationError(_messages(e)) from e
    return Interpreted(scenario=scenario, draft=draft, notes=notes, interpretation=interp)


# --------------------------------------------------------------------------- recordings


class ReplayTransport(httpx2.BaseTransport):
    """Answers every request with a recorded response. Keeps the requests it saw."""

    def __init__(self, recording: dict[str, Any]):
        self.recording = recording
        self.requests: list[dict[str, Any]] = []

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.read()))
        r = self.recording["response"]
        return httpx2.Response(r["status"], json=r["body"], headers={"request-id": "req_replay"})


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
