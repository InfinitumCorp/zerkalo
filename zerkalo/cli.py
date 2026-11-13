"""Command line: zerkalo run | compare | report | interview | init | settings"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from . import __version__
from .compare import compare, rank
from .config import load_scenario
from .geo import announce, annotate_saved_run
from .interview import ask, load_agent
from .llm import ZerkaloLLMError, make_llm
from .report import generate
from .settings import (
    GeoSettingsError,
    geo_routing_enabled,
    load_settings,
    save_settings,
    set_cli_geo_override,
    settings_path,
)
from .simulation import Simulation, run_dir

BANNER = "ZERKALO™ — reflexive simulation engine"

SCENARIO_TEMPLATE = """# Zerkalo scenario
name: {name}
proposition: "Residents support <the thing being decided>"   # +1 = support
goal: increase            # or: decrease
context: |
  One paragraph of background the public already knows.
seed_files:
  - seed/brief.md         # news, research, transcripts: anything that shapes the world
language: Russian
n_agents: 300             # whole public
n_llm_agents: 24          # how many of them reason with Claude each round
rounds: 10
active_fraction: 0.35
random_seed: 7
# geo_routing: false   # cyber geo-routing annotations; leave unset to keep the default off
premise:
  label: A
  source: "Owned outlet"
  text: "The message under test."
  stance: 0.8             # direction it pushes
  reach: 0.15             # share of the public the source reaches directly
  inject_round: 2
variants: []              # more premises for `zerkalo compare`
models:
  agent: claude-opus-5-5
  analyst: claude-opus-5-5
  agent_effort: low
  analyst_effort: high
  concurrency: 8
"""


def _bar(v: float, w: int = 24) -> str:
    k = int(round(v * w))
    return "█" * k + "·" * (w - k)


def _progress(snap: dict) -> None:
    print(f"  round {snap['round']:>2}  support {_bar(snap['support'])} {snap['support']:>5.0%}  "
          f"oppose {snap['oppose']:>5.0%}  exposed {snap['exposed']:>5.0%}", flush=True)


def _add_geo_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--geo-routing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="annotate with cyber geo-routing for this run (IP country and domain category). Default off.",
    )


async def cmd_run(a) -> int:
    sc = load_scenario(a.scenario)
    if a.rounds:
        sc.rounds = a.rounds
    geo_routing_enabled(sc)
    llm = make_llm(sc.models, a.mock, sc.random_seed)
    print(f"{BANNER}\n{'[MOCK MODE — synthetic numbers] ' if a.mock else ''}Scenario: {sc.name}")
    sim = Simulation(sc, llm)
    print("· mirroring the world …", flush=True)
    await sim.setup()
    print(f"· populated {len(sim.agents)} agents ({sum(x.is_llm for x in sim.agents)} LLM-driven), "
          f"{sim.graph.number_of_edges()} follow ties\n· rehearsing:")
    res = await sim.run(progress=_progress)
    out = res.save(run_dir(a.out, sc))
    announce(out, sc)
    print("· reading the forecast …", flush=True)
    fc = await generate(out, llm)
    s = res.summary()
    print(f"\n{fc.headline}\nSupport {s['baseline_support']:.0%} → {s['final_support']:.0%} "
          f"({s['support_delta']:+.0%}) · confidence {fc.confidence}\nReport: {out / 'report.html'}")
    if res.failures:
        print(f"note: {res.failures} agent decisions fell back to the rule model after LLM errors", file=sys.stderr)
    return 0


async def cmd_compare(a) -> int:
    sc = load_scenario(a.scenario)
    if not sc.all_premises():
        print("scenario has no premise or variants to compare", file=sys.stderr)
        return 2
    geo_routing_enabled(sc)
    llm = make_llm(sc.models, a.mock, sc.random_seed)
    out = run_dir(a.out, sc, "compare")
    print(f"{BANNER}\n{'[MOCK MODE — synthetic numbers] ' if a.mock else ''}Comparing {len(sc.all_premises())} premise(s) + baseline")
    results = await compare(sc, llm, out, progress=lambda l: print(f"· running {l} …", flush=True))
    best = rank(sc, results)[0]
    print((out / "comparison.md").read_text(encoding="utf-8"))
    if a.report:
        for r in results:
            await generate(out / (r.premise.label if r.premise else "baseline"), llm)
    announce(out, sc)
    print(f"Best: {best.premise.label if best.premise else 'baseline'}  ·  {out}")
    return 0


async def cmd_report(a) -> int:
    path = Path(a.run)
    sc = load_scenario(a.scenario) if a.scenario else None
    from .config import Models
    geo_routing_enabled(sc)
    llm = make_llm(sc.models if sc else Models(), a.mock)
    fc = await generate(path, llm)
    annotate_saved_run(path, sc)
    announce(path, sc)
    print(f"{fc.headline}\n{path / 'report.html'}")
    return 0


async def cmd_interview(a) -> int:
    path = Path(a.run)
    from .config import Models
    llm = make_llm(Models(), a.mock)
    agent = load_agent(path, a.agent)
    print(f"Interviewing {agent['name']} ({agent['segment']}, stance {agent['stance']:+.2f}). Empty line to quit.")
    history: list[dict] = []
    questions = [a.ask] if a.ask else None
    while True:
        q = questions.pop(0) if questions else (None if a.ask else input("you › ").strip())
        if not q:
            break
        history.append({"role": "user", "content": q})
        reply = await ask(path, a.agent, history, llm)
        history.append({"role": "assistant", "content": reply})
        print(f"{agent['name'].split()[0].lower()} › {reply}\n")
        if a.ask and not questions:
            break
    return 0


def cmd_settings(a) -> int:
    current = load_settings(strict=True)
    path = settings_path()
    if a.geo_routing is not None:
        current.geo_routing = a.geo_routing
        path = save_settings(current)
    print(f"cyber geo-routing: {'on' if current.geo_routing else 'off'}")
    env = os.environ.get("INFINITUM_GEO_ENABLED")
    if env is not None and env.strip() != "":
        print(f"INFINITUM_GEO_ENABLED: {env.strip()}")
    print(f"file: {path}")
    return 0


def cmd_init(a) -> int:
    d = Path(a.dir)
    (d / "seed").mkdir(parents=True, exist_ok=True)
    f = d / "scenario.yaml"
    if f.exists():
        print(f"{f} already exists", file=sys.stderr)
        return 1
    f.write_text(SCENARIO_TEMPLATE.format(name=d.name), encoding="utf-8")
    (d / "seed" / "brief.md").write_text("# Seed material\n\nPaste news, research and context here.\n", encoding="utf-8")
    print(f"created {f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles (e.g. cp1251) can't print the bars/Cyrillic otherwise
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(prog="zerkalo", description=BANNER)
    p.add_argument("--version", action="version", version=f"zerkalo {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="simulate one scenario and write a forecast")
    r.add_argument("scenario")
    r.add_argument("--out", default="runs")
    r.add_argument("--rounds", type=int)
    r.add_argument("--mock", action="store_true", help="offline deterministic LLM (no API calls, synthetic numbers)")
    _add_geo_flag(r)

    c = sub.add_parser("compare", help="A/B test premise variants on the same public")
    c.add_argument("scenario")
    c.add_argument("--out", default="runs")
    c.add_argument("--report", action="store_true", help="also write a full report per variant")
    c.add_argument("--mock", action="store_true")
    _add_geo_flag(c)

    rp = sub.add_parser("report", help="(re)write the forecast for a finished run")
    rp.add_argument("run")
    rp.add_argument("--scenario")
    rp.add_argument("--mock", action="store_true")
    _add_geo_flag(rp)

    st = sub.add_parser("settings", help="show or save options; cyber geo-routing defaults to off")
    st.add_argument(
        "--geo-routing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="persist cyber geo-routing on or off",
    )

    i = sub.add_parser("interview", help="question an agent from a finished run")
    i.add_argument("run")
    i.add_argument("--agent", type=int, required=True)
    i.add_argument("--ask", help="ask a single question and exit")
    i.add_argument("--mock", action="store_true")

    n = sub.add_parser("init", help="scaffold a scenario folder")
    n.add_argument("dir")

    a = p.parse_args(argv)
    if a.cmd != "settings":
        set_cli_geo_override(getattr(a, "geo_routing", None))
    try:
        if a.cmd == "init":
            return cmd_init(a)
        if a.cmd == "settings":
            return cmd_settings(a)
        return asyncio.run({"run": cmd_run, "compare": cmd_compare, "report": cmd_report, "interview": cmd_interview}[a.cmd](a))
    except ZerkaloLLMError as e:
        print(f"LLM error: {e}", file=sys.stderr)
        return 1
    except GeoSettingsError as e:
        print(str(e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
