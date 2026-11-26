"""Load and validate config.yaml."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"
LOS_GRADES = ("A", "B", "C", "D", "E")
# Deployment setting (Streamlit secret or environment variable) that overrides
# simulation.max_parallel_runs.
PARALLEL_RUNS_SETTING = "JUNCTIONLAB_MAX_PARALLEL_RUNS"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Geometry(_Frozen):
    approach_length_m: float = Field(gt=0)
    queue_warning_share: float = Field(gt=0, le=1)
    speed_limit_kmh: float = Field(gt=0)


class Simulation(_Frozen):
    seeds: tuple[int, ...] = Field(min_length=1)
    warmup_s: int = Field(ge=0)
    measurement_s: int = Field(gt=0)
    runout_max_s: int = Field(ge=0)
    time_to_teleport_s: int
    step_length_s: float = Field(gt=0)
    gridlock_stop_s: float = Field(gt=0)
    max_parallel_runs: int = Field(gt=0)

    @property
    def end_s(self) -> int:
        """Hard time cap for every run."""
        return self.warmup_s + self.measurement_s + self.runout_max_s


class Driver(_Frozen):
    car_follow_model: Literal["Krauss"]
    sigma: float = Field(ge=0, le=1)


class Vehicles(_Frozen):
    car_vclass: str
    heavy_vclass: str
    bus_vclass: str
    bus_demand_default_bph: float = Field(ge=0)


Movement = Literal["left", "through", "right"]


def _check_100(shares: dict[str, float]) -> dict[str, float]:
    if abs(sum(shares.values()) - 100) > 0.1 or any(v < 0 for v in shares.values()):
        raise ValueError(f"shares must be >= 0 and sum to 100, got {shares}")
    return shares


class TurningShareDefaults(_Frozen):
    four_arm: dict[Movement, float]
    t_major: dict[Literal["through", "into_stem"], float]
    t_stem: dict[Literal["left", "right"], float]

    _sum_100 = field_validator("four_arm", "t_major", "t_stem")(_check_100)


class ScenarioDefaults(_Frozen):
    lanes: tuple[tuple[Movement, ...], ...] = Field(min_length=1, max_length=4)
    heavy_vehicle_share_pct: float = Field(ge=0, le=30)
    turning_shares_pct: TurningShareDefaults


class Control(_Frozen):
    major_road_default: Literal["north-south", "east-west"]
    four_arm_phase_plan_default: Literal["two_phase", "four_phase_split"]
    amber_s: float = Field(ge=0)
    all_red_s: float = Field(ge=0)
    cycle_default_s: float = Field(gt=0)
    min_green_s: float = Field(gt=0)
    saturation_flow_vphpl: float = Field(gt=0)
    lost_time_s: float = Field(ge=0)

    def effective_green_s(self, green_s: float) -> float:
        return green_s + self.amber_s - self.lost_time_s

    @property
    def intergreen_s(self) -> float:
        return self.amber_s + self.all_red_s


class Los(_Frozen):
    signalised: dict[str, float]
    unsignalised: dict[str, float]
    overload_throughput_share: float = Field(gt=0, le=1)
    backlog_growth_min_veh: int = Field(ge=1)

    @field_validator("signalised", "unsignalised")
    @classmethod
    def _increasing_a_to_e(cls, table: dict[str, float]) -> dict[str, float]:
        if tuple(table) != LOS_GRADES:
            raise ValueError(f"LOS table must list grades {LOS_GRADES} in order")
        values = list(table.values())
        if any(b <= a for a, b in zip(values, values[1:], strict=False)):
            raise ValueError("LOS thresholds must increase from A to E")
        return table


class Baseline(_Frozen):
    demand_vph_per_movement: float = Field(gt=0)
    duration_s: int = Field(gt=0)


class Webster(_Frozen):
    cycle_s: float = Field(gt=0)
    green_s: float = Field(gt=0)
    x_tested: tuple[float, ...]
    x_info_only: tuple[float, ...]
    margin_relative: float = Field(ge=0)
    margin_absolute_s: float = Field(ge=0)


class GroqLlm(_Frozen):
    model: str
    max_tokens: int = Field(gt=0)
    reasoning_effort: Literal["low", "medium", "high"]


class AnthropicLlm(_Frozen):
    model: str
    max_tokens: int = Field(gt=0)
    effort: Literal["low", "medium", "high"]


class Llm(_Frozen):
    provider: Literal["groq", "anthropic"]
    groq: GroqLlm
    anthropic: AnthropicLlm
    calls_per_session: int = Field(ge=0)
    calls_per_day: int = Field(ge=0)
    counter_file: Path
    max_input_chars: int = Field(gt=0)


class AgentEvaluation(_Frozen):
    cycle_step_s: int = Field(gt=0)
    split_offsets_pct: tuple[int, ...]
    holdout_seeds: tuple[int, ...]
    repeats: int = Field(gt=0)
    random_designs: int = Field(gt=0)
    results_dir: Path
    max_parallel_runs: int = Field(gt=0)
    min_free_memory_gb: float = Field(ge=0)
    rate_limit_waits: int = Field(ge=0)
    rate_limit_default_wait_s: float = Field(ge=0)
    pause_between_runs_s: float = Field(ge=0)


class Agent(_Frozen):
    max_simulations_per_goal: int = Field(gt=0)
    max_goals_per_session: int = Field(gt=0)
    max_llm_calls_per_goal: int = Field(gt=1)
    llm_calls_per_day: int = Field(ge=0)
    counter_file: Path
    max_goal_chars: int = Field(gt=0)
    pause_between_calls_s: float = Field(ge=0)
    cycle_min_s: int = Field(gt=0)
    cycle_max_s: int = Field(gt=0)
    evaluation: AgentEvaluation


class Config(_Frozen):
    geometry: Geometry
    simulation: Simulation
    driver: Driver
    vehicles: Vehicles
    scenario_defaults: ScenarioDefaults
    control: Control
    los: Los
    baseline: Baseline
    webster: Webster
    llm: Llm
    agent: Agent


def load_config(path: Path | str | None = None) -> Config:
    """Load a config file. Defaults to $JUNCTIONLAB_CONFIG, then the repo's config.yaml."""
    path = Path(path or os.environ.get("JUNCTIONLAB_CONFIG") or DEFAULT_CONFIG_PATH)
    with path.open(encoding="utf-8") as f:
        return Config.model_validate(yaml.safe_load(f))


def setting(name: str) -> str | None:
    """A deployment setting: a Streamlit secret when the app is running, else the environment.

    Streamlit is only asked when it is already imported, so the command line never loads it.
    """
    st = sys.modules.get("streamlit")
    if st is not None:
        try:
            value = st.secrets.get(name)
            if value is not None and str(value) != "":
                return str(value)
        except Exception:  # no secrets file, or not running in an app
            pass
    return os.environ.get(name) or None


def with_settings(cfg: Config, lookup: Callable[[str], str | None] = setting) -> Config:
    """The config with deployment settings applied (only the number of parallel SUMO runs)."""
    raw = lookup(PARALLEL_RUNS_SETTING)
    if raw is None:
        return cfg
    try:
        runs = int(raw)
    except ValueError:
        runs = 0
    if runs < 1:
        raise ValueError(
            f"{PARALLEL_RUNS_SETTING} must be a whole number of at least 1, not {raw!r}"
        )
    simulation = cfg.simulation.model_copy(update={"max_parallel_runs": runs})
    return cfg.model_copy(update={"simulation": simulation})


@cache
def get_config() -> Config:
    """The default config with deployment settings applied, loaded once."""
    return with_settings(load_config())
