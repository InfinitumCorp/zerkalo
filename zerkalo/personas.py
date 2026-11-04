"""Step 2 — populate the public.

A small core of agents gets rich LLM-written personas and reasons with the LLM each round.
The rest of the crowd is procedural: cheap, fast agents that follow an opinion-dynamics rule.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import asdict, dataclass, field

from .config import Scenario
from .llm import CITIES, FIRST, JOBS, LAST, LLM
from .schemas import PersonaBatch, WorldModel, clamp

SYSTEM = (
    "You write realistic, varied personas for a public-opinion simulation. Each persona is a "
    "plausible ordinary member of the given audience segment, with their own life, values and "
    "media habits. Avoid caricature and avoid making everyone articulate or extreme."
)


@dataclass
class Agent:
    id: int
    name: str
    segment: str
    age: int
    occupation: str
    city: str
    bio: str
    values: list[str]
    media_diet: list[str]
    stance: float
    initial_stance: float
    certainty: float
    susceptibility: float
    activity: float
    voice: str = "plain"
    is_llm: bool = False
    influence: float = 0.0
    memory: list[str] = field(default_factory=list)

    def remember(self, line: str, keep: int = 10) -> None:
        self.memory.append(line)
        del self.memory[:-keep]

    def to_dict(self) -> dict:
        return asdict(self)


def _allocate(total: int, shares: list[float]) -> list[int]:
    raw = [s * total for s in shares]
    counts = [int(x) for x in raw]
    for i in sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)[: total - sum(counts)]:
        counts[i] += 1
    return counts


async def build_population(sc: Scenario, world: WorldModel, llm: LLM) -> list[Agent]:
    rng = random.Random(sc.random_seed)
    segs = world.segments
    n_llm = min(sc.n_llm_agents, sc.n_agents)
    llm_counts = _allocate(n_llm, [s.share for s in segs])
    crowd_counts = _allocate(sc.n_agents - n_llm, [s.share for s in segs])

    async def llm_batch(seg, n):
        if n == 0:
            return []
        prompt = (
            f"Proposition under debate (+1 support, -1 oppose): {sc.proposition}\n"
            f"World summary: {world.summary}\nFault lines: {'; '.join(world.fault_lines)}\n\n"
            f"Segment: {seg.name} — {seg.description}\n"
            f"Segment baseline stance: {seg.baseline_stance:+.2f}; susceptibility {seg.susceptibility:.2f}.\n"
            f"Write {n} personas from this segment. Language/culture: {sc.language}. "
            "stance in [-1,1] scattered around the baseline; certainty in [0,1]; "
            "media_diet = channels they actually use; voice = how they write online (a few words)."
        )
        out: list = []
        while len(out) < n:  # the model may return fewer than asked; top up
            k = min(10, n - len(out))
            batch = await llm.structured(SYSTEM, prompt.replace(f"Write {n}", f"Write {k}"), PersonaBatch, role="analyst",
                                         hint={"n": k, "baseline_stance": seg.baseline_stance})
            out.extend(batch.personas[:k])
        return [(seg, p) for p in out]

    drafts = [d for chunk in await asyncio.gather(*(llm_batch(s, c) for s, c in zip(segs, llm_counts))) for d in chunk]

    agents: list[Agent] = []
    for seg, p in drafts:
        st = clamp(p.stance)
        agents.append(Agent(
            id=len(agents), name=p.name, segment=seg.name, age=p.age, occupation=p.occupation, city=p.city,
            bio=p.bio, values=p.values, media_diet=p.media_diet, stance=st, initial_stance=st,
            certainty=clamp(p.certainty, 0, 1), susceptibility=seg.susceptibility,
            activity=rng.uniform(0.6, 1.0), voice=p.voice, is_llm=True,
        ))
    for seg, c in zip(segs, crowd_counts):
        for _ in range(c):
            st = clamp(rng.gauss(seg.baseline_stance, 0.35))
            agents.append(Agent(
                id=len(agents), name=f"{rng.choice(FIRST)} {rng.choice(LAST)}", segment=seg.name,
                age=rng.randint(18, 80), occupation=rng.choice(JOBS), city=rng.choice(CITIES), bio="",
                values=[], media_diet=[], stance=st, initial_stance=st, certainty=rng.uniform(0.15, 0.9),
                susceptibility=seg.susceptibility, activity=rng.uniform(0.15, 0.8),
            ))
    return agents
