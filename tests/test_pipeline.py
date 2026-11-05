import asyncio
import json
from pathlib import Path

import pytest

from zerkalo.cli import main
from zerkalo.compare import compare, rank
from zerkalo.config import Premise, load_scenario
from zerkalo.llm import MockLLM
from zerkalo.network import build_graph
from zerkalo.personas import _allocate
from zerkalo.report import generate
from zerkalo.simulation import Simulation

EXAMPLE = Path(__file__).parents[1] / "examples" / "weekend-pedestrian" / "scenario.yaml"


def small():
    sc = load_scenario(EXAMPLE)
    sc.n_agents, sc.n_llm_agents, sc.rounds = 120, 12, 8
    return sc


def run(coro):
    return asyncio.run(coro)


def test_allocate_sums_exactly():
    assert sum(_allocate(97, [0.32, 0.22, 0.2, 0.16, 0.1])) == 97
    assert _allocate(10, [0.5, 0.5]) == [5, 5]


def test_population_and_graph():
    sc = small()
    sim = Simulation(sc, MockLLM())
    run(sim.setup())
    assert len(sim.agents) == 120
    assert sum(a.is_llm for a in sim.agents) == 12
    assert {a.segment for a in sim.agents} == {s.name for s in sc.segments}
    assert all(-1 <= a.stance <= 1 for a in sim.agents)
    g = sim.graph
    assert g.number_of_edges() > 120
    assert not any(u == v for u, v in g.edges)
    assert max(a.influence for a in sim.agents) == 1.0


def test_run_is_deterministic():
    a = run(Simulation(small(), MockLLM()).run())
    b = run(Simulation(small(), MockLLM()).run())
    assert a.timeline == b.timeline
    assert len(a.timeline) == 9


def test_premise_beats_baseline_and_is_seen():
    sc = small()
    base = Simulation(sc, MockLLM())
    run(base.setup())
    with_premise = run(base.fork(sc.premise).run()).summary()
    without = run(base.fork(None).run()).summary()
    assert with_premise["exposure"] > 0.1
    assert without["exposure"] == 0
    assert with_premise["final_support"] > without["final_support"]


def test_opposing_premise_lowers_support():
    sc = small()
    base = Simulation(sc, MockLLM())
    run(base.setup())
    pro = run(base.fork(sc.premise).run()).summary()
    anti = run(base.fork(Premise(label="X", text="It will be chaos for everyone.", stance=-0.9)).run()).summary()
    assert anti["final_support"] < pro["final_support"]


def test_save_report_and_compare(tmp_path):
    sc = small()
    res = run(Simulation(sc, MockLLM()).run())
    out = res.save(tmp_path / "r1")
    fc = run(generate(out, MockLLM()))
    assert fc.headline
    for f in ("summary.json", "agents.json", "timeline.json", "posts.jsonl", "report.md", "report.html"):
        assert (out / f).exists(), f
    assert "<svg" in (out / "report.html").read_text(encoding="utf-8")
    assert json.loads((out / "summary.json").read_text(encoding="utf-8"))["agents"] == 120

    results = run(compare(sc, MockLLM(), tmp_path / "cmp"))
    assert len(results) == 4  # baseline + A, B, C
    ranked = rank(sc, results)
    s = [r.summary()["final_support"] for r in ranked]
    assert s == sorted(s, reverse=True)
    assert (tmp_path / "cmp" / "comparison.md").exists()


def test_cli_end_to_end(tmp_path, capsys):
    assert main(["run", str(EXAMPLE), "--mock", "--rounds", "4", "--out", str(tmp_path)]) == 0
    run_path = next(tmp_path.iterdir())
    assert (run_path / "report.html").exists()
    assert main(["interview", str(run_path), "--agent", "0", "--ask", "Why do you feel that way?", "--mock"]) == 0
    assert "mock" in capsys.readouterr().out
    assert main(["init", str(tmp_path / "new")]) == 0
    assert load_scenario(tmp_path / "new" / "scenario.yaml").premise is not None


def test_seed_file_cannot_escape(tmp_path):
    scenario = tmp_path / "scenario"
    scenario.mkdir()
    (tmp_path / "secret.txt").write_text("nope", encoding="utf-8")
    (scenario / "ok.txt").write_text("visible", encoding="utf-8")
    (scenario / "scenario.yaml").write_text("name: x\nproposition: y\nseed_files:\n  - ../secret.txt\n", encoding="utf-8")
    sc = load_scenario(scenario / "scenario.yaml")
    with pytest.raises(ValueError, match="escapes"):
        sc.seed_text()
    sc.seed_files = ["ok.txt"]
    assert "visible" in sc.seed_text()


def test_compare_label_cannot_escape(tmp_path):
    sc = small()
    sc.n_agents, sc.rounds = 20, 1
    sc.variants = []
    sc.premise = Premise(label="../escape", text="hello", stance=0.2)
    with pytest.raises(ValueError, match="folder name"):
        run(compare(sc, MockLLM(), tmp_path / "cmp", include_baseline=False))
    assert not (tmp_path / "escape").exists()
