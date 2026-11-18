"""Load and validate config.yaml."""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"
LOS_GRADES = ("A", "B", "C", "D", "E")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Geometry(_Frozen):
    approach_length_m: float = Field(gt=0)
    queue_warning_share: float = Field(gt=0, le=1)


class Simulation(_Frozen):
    seeds: tuple[int, ...] = Field(min_length=1)
    warmup_s: int = Field(ge=0)
    measurement_s: int = Field(gt=0)
    runout_max_s: int = Field(ge=0)
    time_to_teleport_s: int

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


class Control(_Frozen):
    major_road_default: Literal["north-south", "east-west"]
    four_arm_phase_plan_default: Literal["two_phase", "four_phase_split"]
    amber_s: float = Field(ge=0)
    all_red_s: float = Field(ge=0)


class Los(_Frozen):
    signalised: dict[str, float]
    unsignalised: dict[str, float]

    @field_validator("signalised", "unsignalised")
    @classmethod
    def _increasing_a_to_e(cls, table: dict[str, float]) -> dict[str, float]:
        if tuple(table) != LOS_GRADES:
            raise ValueError(f"LOS table must list grades {LOS_GRADES} in order")
        values = list(table.values())
        if any(b <= a for a, b in zip(values, values[1:], strict=False)):
            raise ValueError("LOS thresholds must increase from A to E")
        return table


class Webster(_Frozen):
    x_tested: tuple[float, ...]
    x_info_only: tuple[float, ...]
    margin_relative: float = Field(ge=0)
    margin_absolute_s: float = Field(ge=0)


class Llm(_Frozen):
    model: str
    calls_per_session: int = Field(ge=0)
    calls_per_day: int = Field(ge=0)
    counter_file: Path


class Config(_Frozen):
    geometry: Geometry
    simulation: Simulation
    driver: Driver
    vehicles: Vehicles
    control: Control
    los: Los
    webster: Webster
    llm: Llm


def load_config(path: Path | str | None = None) -> Config:
    """Load a config file. Defaults to $JUNCTIONLAB_CONFIG, then the repo's config.yaml."""
    path = Path(path or os.environ.get("JUNCTIONLAB_CONFIG") or DEFAULT_CONFIG_PATH)
    with path.open(encoding="utf-8") as f:
        return Config.model_validate(yaml.safe_load(f))


@cache
def get_config() -> Config:
    """The default config, loaded once."""
    return load_config()
