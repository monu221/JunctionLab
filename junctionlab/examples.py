"""Bundled example scenarios. They need no API key: each file holds the draft an LLM
would return for its description, and goes through the same resolve() as LLM output."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from junctionlab.config import REPO_ROOT, Config
from junctionlab.scenario import Scenario, ScenarioDraft, resolve

EXAMPLES_DIR = REPO_ROOT / "examples"


@dataclass(frozen=True)
class Example:
    key: str
    description: str
    draft: ScenarioDraft

    def scenario(self, config: Config | None = None) -> Scenario:
        return resolve(self.draft, config)


def load_example(path: Path) -> Example:
    data = json.loads(path.read_text(encoding="utf-8"))
    return Example(
        key=path.stem,
        description=data["description"],
        draft=ScenarioDraft.model_validate(data["draft"]),
    )


def list_examples() -> list[Example]:
    return [load_example(p) for p in sorted(EXAMPLES_DIR.glob("*.json"))]
