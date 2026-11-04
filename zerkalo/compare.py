"""Design the outcome: A/B test premise variants against the same synthetic public.

Every variant runs in the same world, with the same agents, graph and random stream
(common random numbers), so differences in outcome come from the premise alone.
"""

from __future__ import annotations

from pathlib import Path

from .config import Scenario
from .llm import LLM
from .simulation import RunResult, Simulation


async def compare(sc: Scenario, llm: LLM, out_dir: Path, include_baseline: bool = True, progress=None) -> list[RunResult]:
    base = Simulation(sc, llm)
    await base.setup()
    premises = ([None] if include_baseline else []) + sc.all_premises()
    results = []
    for p in premises:
        label = p.label if p else "baseline"
        if progress:
            progress(label)
        res = await base.fork(p).run()
        res.save(out_dir / label)
        results.append(res)
    (out_dir / "comparison.md").write_text(table(sc, results), encoding="utf-8")
    return results


def rank(sc: Scenario, results: list[RunResult]) -> list[RunResult]:
    sign = 1 if sc.goal == "increase" else -1
    return sorted(results, key=lambda r: sign * r.summary()["final_support"], reverse=True)


def table(sc: Scenario, results: list[RunResult]) -> str:
    rows = []
    for i, r in enumerate(rank(sc, results), 1):
        s = r.summary()
        label = r.premise.label if r.premise else "baseline"
        text = (r.premise.text[:70] + "…") if r.premise and len(r.premise.text) > 70 else (r.premise.text if r.premise else "—")
        rows.append(f"| {i} | {label} | {s['final_support']:.0%} | {s['support_delta']:+.0%} | {s['final_oppose']:.0%} "
                    f"| {s['exposure']:.0%} | {s['final_polarization']:.2f} | {text} |")
    return (f"# Zerkalo™ comparison — {sc.name}\n\nGoal: **{sc.goal}** support for: {sc.proposition}\n\n"
            "| Rank | Variant | Support | Δ | Oppose | Exposure | Polarization | Premise |\n|---|---|---|---|---|---|---|---|\n"
            + "\n".join(rows) + "\n")
