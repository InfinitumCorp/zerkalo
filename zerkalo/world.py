"""Step 1 — mirror the world: turn seed material into a structured world model."""

from __future__ import annotations

from .config import Scenario
from .llm import LLM
from .schemas import Segment, WorldModel, clamp

SYSTEM = (
    "You are the world-modelling analyst inside Zerkalo, a public-opinion simulation engine used "
    "for forecasting and message testing. You read seed material and describe the social world "
    "a simulated public will live in. Be concrete, neutral and evidence-based; do not invent "
    "statistics that the material does not support."
)


def _prompt(sc: Scenario, seed_text: str) -> str:
    segs = ""
    if sc.segments:
        segs = "\nThe client has defined these audience segments; keep their names and shares:\n" + "\n".join(
            f"- {s.name} (share {s.share}): {s.description}" for s in sc.segments
        )
    return f"""Build a world model for this simulation.

Proposition being forecast (+1 = full support, -1 = full opposition):
{sc.proposition}

Context from the client:
{sc.context or "(none)"}

Seed material:
{seed_text or "(none supplied)"}
{segs}

Return:
- summary: 3-5 sentences on the situation as the public currently sees it
- key_facts: the facts most likely to shape opinion
- stakeholders: who has a voice in this debate and where they likely stand
- fault_lines: the disagreements opinion will split along
- segments: 3-6 audience segments; share values sum to 1; baseline_stance in [-1, 1] toward the
  proposition before any campaign; susceptibility in [0, 1] = how easily the segment's view moves."""


async def build_world(sc: Scenario, llm: LLM) -> WorldModel:
    hint = {"proposition": sc.proposition, "segments": [s.model_dump() for s in sc.segments] or None}
    world = await llm.structured(SYSTEM, _prompt(sc, sc.seed_text()), WorldModel, role="analyst", hint=hint)
    return normalise(world)


def normalise(world: WorldModel) -> WorldModel:
    segs = world.segments or [Segment(name="General public", share=1, description="", baseline_stance=0, susceptibility=0.5)]
    total = sum(max(s.share, 0.0) for s in segs) or 1.0
    world.segments = [
        Segment(
            name=s.name,
            share=max(s.share, 0.0) / total,
            description=s.description,
            baseline_stance=clamp(s.baseline_stance),
            susceptibility=clamp(s.susceptibility, 0, 1),
        )
        for s in segs
    ]
    return world
