"""LLM backends.

``ClaudeLLM`` calls Claude through the official Anthropic SDK with structured outputs.
``MockLLM`` is a deterministic offline stand-in so the whole pipeline runs (and is tested)
without an API key or spend. Its numbers are synthetic; never treat a mock forecast as real.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel

from .config import Models
from .schemas import (
    AgentAction,
    ForecastReport,
    PersonaBatch,
    PersonaDraft,
    Segment,
    Stakeholder,
    WorldModel,
    clamp,
)

T = TypeVar("T", bound=BaseModel)
Role = Literal["agent", "analyst"]


class ZerkaloLLMError(RuntimeError):
    pass


class LLM(Protocol):
    async def structured(self, system: str, prompt: str, schema: type[T], *, role: Role, hint: dict | None = None) -> T: ...
    async def text(self, system: str, messages: list[dict], *, role: Role, hint: dict | None = None) -> str: ...


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, u: Any) -> None:
        self.calls += 1
        self.input_tokens += getattr(u, "input_tokens", 0) or 0
        self.output_tokens += getattr(u, "output_tokens", 0) or 0


# ---------------------------------------------------------------------------- Claude

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ClaudeLLM:
    """Claude via the Anthropic SDK. Credentials resolve from the environment
    (ANTHROPIC_API_KEY, or an `ant auth login` profile)."""

    def __init__(self, models: Models, client: Any | None = None):
        import anthropic

        self._anthropic = anthropic
        self.models = models
        self.client = client or anthropic.AsyncAnthropic(max_retries=4)
        self.sem = asyncio.Semaphore(models.concurrency)
        self.usage = Usage()

    def _cfg(self, role: Role) -> tuple[str, str]:
        if role == "agent":
            return self.models.agent, self.models.agent_effort
        return self.models.analyst, self.models.analyst_effort

    async def structured(self, system, prompt, schema, *, role, hint=None):
        model, effort = self._cfg(role)
        a = self._anthropic
        async with self.sem:
            try:
                resp = await self.client.beta.messages.parse(
                    model=model,
                    max_tokens=16000,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                    output_format=schema,
                    output_config={"effort": effort},
                    betas=[FALLBACK_BETA],
                    fallbacks="default",
                )
            except a.RateLimitError as e:
                raise ZerkaloLLMError(f"rate limited after retries: {e}") from e
            except a.BadRequestError as e:
                raise ZerkaloLLMError(f"bad request ({model}): {e}") from e
            except a.APIStatusError as e:
                raise ZerkaloLLMError(f"API error {e.status_code}: {e}") from e
            except a.APIConnectionError as e:
                raise ZerkaloLLMError(f"connection error: {e}") from e
        self.usage.add(resp.usage)
        if resp.stop_reason == "refusal":
            raise ZerkaloLLMError(f"model declined ({getattr(resp.stop_details, 'category', None)})")
        if resp.parsed_output is None:
            raise ZerkaloLLMError(f"no structured output (stop_reason={resp.stop_reason})")
        return resp.parsed_output

    async def text(self, system, messages, *, role, hint=None):
        model, effort = self._cfg(role)
        a = self._anthropic
        async with self.sem:
            try:
                resp = await self.client.beta.messages.create(
                    model=model,
                    max_tokens=16000,
                    system=system,
                    messages=messages,
                    output_config={"effort": effort},
                    betas=[FALLBACK_BETA],
                    fallbacks="default",
                )
            except a.APIStatusError as e:
                raise ZerkaloLLMError(f"API error {e.status_code}: {e}") from e
            except a.APIConnectionError as e:
                raise ZerkaloLLMError(f"connection error: {e}") from e
        self.usage.add(resp.usage)
        if resp.stop_reason == "refusal":
            raise ZerkaloLLMError("model declined")
        return "".join(b.text for b in resp.content if b.type == "text")


# ---------------------------------------------------------------------------- Mock

FIRST = ["Anna", "Ivan", "Olga", "Dmitri", "Elena", "Sergei", "Maria", "Pavel", "Irina", "Alexei",
         "Natalia", "Mikhail", "Sofia", "Artem", "Daria", "Nikolai", "Ksenia", "Yuri", "Vera", "Oleg"]
LAST = ["Ivanova", "Petrov", "Smirnova", "Kuznetsov", "Popova", "Volkov", "Sokolova", "Lebedev",
        "Kozlova", "Novikov", "Morozova", "Orlov", "Fedorova", "Zaitsev", "Belova", "Gusev"]
JOBS = ["teacher", "engineer", "student", "nurse", "small-business owner", "retiree", "IT developer",
        "driver", "accountant", "designer", "journalist", "civil servant"]
CITIES = ["Moscow", "Saint Petersburg", "Kazan", "Novosibirsk", "Yekaterinburg", "Nizhny Novgorod"]


def _rng(*parts: Any) -> random.Random:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def premise_strength(text: str) -> float:
    """Mock-only: a stable pseudo 'persuasiveness' for a premise text, in [0.25, 0.85]."""
    return 0.25 + 0.6 * _rng("premise", text).random()


@dataclass
class MockLLM:
    seed: int = 7
    usage: Usage = field(default_factory=Usage)

    async def structured(self, system, prompt, schema, *, role, hint=None):
        self.usage.calls += 1
        hint = hint or {}
        r = _rng(self.seed, schema.__name__, prompt)
        if schema is WorldModel:
            segs = hint.get("segments") or [
                {"name": "Urban professionals", "share": 0.35, "description": "", "baseline_stance": 0.1, "susceptibility": 0.5},
                {"name": "Pensioners", "share": 0.25, "description": "", "baseline_stance": -0.2, "susceptibility": 0.35},
                {"name": "Students", "share": 0.2, "description": "", "baseline_stance": 0.0, "susceptibility": 0.7},
                {"name": "Regional workers", "share": 0.2, "description": "", "baseline_stance": -0.1, "susceptibility": 0.45},
            ]
            return WorldModel(
                summary=f"[mock] Synthetic world for: {hint.get('proposition', 'scenario')}",
                key_facts=["[mock] fact derived from seed material"],
                stakeholders=[Stakeholder(name="Public", role="audience", likely_position="mixed")],
                fault_lines=["[mock] cost vs. benefit", "[mock] trust in the messenger"],
                segments=[Segment(**s) for s in segs],
            )
        if schema is PersonaBatch:
            n = hint.get("n", 5)
            base = hint.get("baseline_stance", 0.0)
            return PersonaBatch(personas=[
                PersonaDraft(
                    name=f"{r.choice(FIRST)} {r.choice(LAST)}", age=r.randint(18, 78),
                    occupation=r.choice(JOBS), city=r.choice(CITIES),
                    bio="[mock] persona", values=["family", "stability"], media_diet=["Telegram"],
                    stance=clamp(base + r.gauss(0, 0.35)), certainty=round(r.uniform(0.2, 0.9), 2), voice="plain",
                ) for _ in range(n)
            ])
        if schema is AgentAction:
            s, fm, cert = hint.get("stance", 0.0), hint.get("feed_mean", 0.0), hint.get("certainty", 0.5)
            pull = 0.35 * (fm - s) * (1 - cert)
            if hint.get("premise_in_feed"):
                pull += 0.25 * hint.get("premise_strength", 0.5) * (hint.get("premise_stance", 0.8) - s) * (1 - cert * 0.6)
            new = clamp(s + pull + r.gauss(0, 0.03))
            ids = hint.get("feed_ids") or []
            roll = r.random()
            if roll < 0.25:
                act, tgt, txt = "post", None, f"[mock] My view on this: {'for' if new > 0 else 'against'}."
            elif roll < 0.45 and ids:
                act, tgt, txt = "repost", r.choice(ids), None
            elif roll < 0.6 and ids:
                act, tgt, txt = "comment", r.choice(ids), "[mock] Not convinced." if new < 0 else "[mock] Agreed."
            elif roll < 0.8 and ids:
                act, tgt, txt = "like", r.choice(ids), None
            else:
                act, tgt, txt = "ignore", None, None
            return AgentAction(action=act, target_post_id=tgt, text=txt, new_stance=new, reasoning="[mock]")
        if schema is ForecastReport:
            m = hint.get("metrics", {})
            return ForecastReport(
                headline=f"[mock] Support moves to {m.get('final_support', 0):.0%}",
                predicted_outcome=f"[mock] {m.get('final_support', 0):.0%} support, {m.get('final_oppose', 0):.0%} oppose.",
                confidence="low",
                key_drivers=["[mock] reach of the source", "[mock] high-influence early adopters"],
                resistant_groups=["[mock] most certain segment"],
                tipping_points=[f"[mock] round {m.get('tipping_round', '?')}"],
                risks=["[mock] backlash among opponents"],
                recommended_adjustments=["[mock] seed through trusted local voices"],
                narrative="This is a **mock** report generated offline. Run without `--mock` for a real forecast.",
            )
        raise ZerkaloLLMError(f"MockLLM has no generator for {schema.__name__}")

    async def text(self, system, messages, *, role, hint=None):
        self.usage.calls += 1
        last = messages[-1]["content"] if messages else ""
        name = (hint or {}).get("name", "Agent")
        stance = (hint or {}).get("stance", 0.0)
        view = "I'm broadly in favour" if stance > 0.2 else "I'm against it" if stance < -0.2 else "I haven't made up my mind"
        return f"[mock {name}] You asked: “{last[:80]}”. {view}."


def make_llm(models: Models, mock: bool, seed: int = 7) -> LLM:
    return MockLLM(seed=seed) if mock else ClaudeLLM(models)
