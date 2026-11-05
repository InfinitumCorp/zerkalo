"""Scenario configuration (loaded from YAML)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class Premise(BaseModel):
    """The message under test, released into the synthetic public."""

    label: str = "A"
    source: str = "Owned outlet"
    text: str
    stance: float = Field(0.8, description="Direction the premise pushes: -1 oppose … +1 support")
    reach: float = Field(0.15, description="Share of the public that follows the source at injection")
    inject_round: int = 2

    @field_validator("stance")
    @classmethod
    def _clamp_stance(cls, v: float) -> float:
        return max(-1.0, min(1.0, v))

    @field_validator("reach")
    @classmethod
    def _clamp_reach(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class SegmentSpec(BaseModel):
    name: str
    share: float
    description: str = ""
    baseline_stance: float = 0.0
    susceptibility: float = 0.5


class Models(BaseModel):
    agent: str = "claude-opus-5-5"
    analyst: str = "claude-opus-5-5"
    agent_effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    analyst_effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    concurrency: int = 8


class Scenario(BaseModel):
    name: str
    proposition: str = Field(..., description="The outcome being forecast, phrased so +1 = support")
    goal: Literal["increase", "decrease"] = "increase"
    context: str = ""
    seed_files: list[str] = []
    segments: list[SegmentSpec] = []
    premise: Premise | None = None
    variants: list[Premise] = []
    n_agents: int = 300
    n_llm_agents: int = 24
    rounds: int = 10
    active_fraction: float = 0.35
    language: str = "English"
    random_seed: int = 7
    models: Models = Models()

    base_dir: Path = Path(".")

    def seed_text(self) -> str:
        root = self.base_dir.resolve()
        parts = []
        for f in self.seed_files:
            p = (root / f).resolve()
            if not p.is_relative_to(root):
                raise ValueError(f"seed file escapes the scenario directory: {f}")
            parts.append(f"### {p.name}\n{p.read_text(encoding='utf-8')}")
        return "\n\n".join(parts)

    def all_premises(self) -> list[Premise]:
        out = list(self.variants)
        if self.premise and all(v.text != self.premise.text for v in out):
            out.insert(0, self.premise)
        return out


def load_scenario(path: str | Path) -> Scenario:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    sc = Scenario(**data)
    sc.base_dir = path.parent
    return sc
