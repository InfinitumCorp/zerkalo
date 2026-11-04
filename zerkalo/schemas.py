"""Structured-output schemas exchanged with the LLM.

Kept to plain types so they translate cleanly to JSON Schema for structured outputs;
numeric ranges are enforced in code (see ``clamp``), not in the schema.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


def clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


class Stakeholder(BaseModel):
    name: str
    role: str
    likely_position: str


class Segment(BaseModel):
    name: str
    share: float
    description: str
    baseline_stance: float
    susceptibility: float


class WorldModel(BaseModel):
    summary: str
    key_facts: list[str]
    stakeholders: list[Stakeholder]
    fault_lines: list[str]
    segments: list[Segment]


class PersonaDraft(BaseModel):
    name: str
    age: int
    occupation: str
    city: str
    bio: str
    values: list[str]
    media_diet: list[str]
    stance: float
    certainty: float
    voice: str


class PersonaBatch(BaseModel):
    personas: list[PersonaDraft]


class AgentAction(BaseModel):
    action: Literal["post", "repost", "comment", "like", "ignore"]
    target_post_id: int | None
    text: str | None
    new_stance: float
    reasoning: str


class ForecastReport(BaseModel):
    headline: str
    predicted_outcome: str
    confidence: Literal["low", "medium", "high"]
    key_drivers: list[str]
    resistant_groups: list[str]
    tipping_points: list[str]
    risks: list[str]
    recommended_adjustments: list[str]
    narrative: str
