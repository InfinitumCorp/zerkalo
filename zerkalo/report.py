"""Step 4 — read the forecast: analyst write-up + Markdown and HTML reports."""

from __future__ import annotations

import html
import json
from pathlib import Path

from .llm import LLM
from .schemas import ForecastReport

SYSTEM = (
    "You are the lead analyst for Zerkalo, a public-opinion simulation engine. You turn simulation "
    "output into a forecast a communications team can act on. Ground every claim in the metrics "
    "and posts provided, say plainly how uncertain the forecast is, and remember that a simulation "
    "of a synthetic public is evidence, not proof."
)


async def write_forecast(summary: dict, timeline: list[dict], world: dict, llm: LLM) -> ForecastReport:
    curve = ", ".join(f"r{t['round']}: {t['support']:.0%}/{t['oppose']:.0%}" for t in timeline)
    prompt = f"""Write the forecast for this run.

World: {world.get('summary', '')}
Fault lines: {'; '.join(world.get('fault_lines', []))}

Proposition: {summary['proposition']}  (client goal: {summary['goal']} support)
Premise tested: {json.dumps(summary['premise'], ensure_ascii=False)}

Support/oppose by round: {curve}
Summary metrics: {json.dumps({k: v for k, v in summary.items() if k not in ('top_posts', 'biggest_movers', 'premise')}, ensure_ascii=False)}
Most engaged posts: {json.dumps(summary['top_posts'], ensure_ascii=False)}
Biggest movers: {json.dumps(summary['biggest_movers'], ensure_ascii=False)}

Fill every field. narrative: 3-6 short paragraphs of Markdown explaining how opinion moved and why."""
    return await llm.structured(SYSTEM, prompt, ForecastReport, role="analyst", hint={"metrics": summary})


def to_markdown(summary: dict, fc: ForecastReport) -> str:
    bl = lambda xs: "\n".join(f"- {x}" for x in xs) or "- —"
    segs = "\n".join(f"| {k} | {v['n']} | {v['initial']:+.2f} | {v['final']:+.2f} | {v['final'] - v['initial']:+.2f} |"
                     for k, v in summary["segments"].items())
    pr = summary["premise"] or {}
    return f"""# Zerkalo™ forecast — {summary['scenario']}

**{fc.headline}**

| | |
|---|---|
| Proposition | {summary['proposition']} |
| Premise ({pr.get('label', '—')}) | {pr.get('text', '— (baseline, no premise)')} |
| Support | {summary['baseline_support']:.0%} → **{summary['final_support']:.0%}** ({summary['support_delta']:+.0%}) |
| Oppose (final) | {summary['final_oppose']:.0%} |
| Exposure to premise | {summary['exposure']:.0%} |
| Tipping round | {summary['tipping_round'] or '—'} |
| Confidence | {fc.confidence} |
| Public | {summary['agents']} agents ({summary['llm_agents']} LLM-driven), {summary['rounds']} rounds |

## Predicted outcome
{fc.predicted_outcome}

## Key drivers
{bl(fc.key_drivers)}

## Resistant groups
{bl(fc.resistant_groups)}

## Tipping points
{bl(fc.tipping_points)}

## Risks
{bl(fc.risks)}

## Recommended adjustments
{bl(fc.recommended_adjustments)}

## Segments
| Segment | Agents | Initial | Final | Shift |
|---|---|---|---|---|
{segs}

## Analysis
{fc.narrative}

---
*Simulated forecast of a synthetic public. Treat as directional evidence, not a poll.*
"""


def _chart(timeline: list[dict], w: int = 720, h: int = 260) -> str:
    pad = 34
    n = max(len(timeline) - 1, 1)
    xy = lambda i, v: (pad + (w - 2 * pad) * i / n, h - pad - (h - 2 * pad) * v)
    def path(key):
        return " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, t in enumerate(timeline) for x, y in [xy(i, t[key])])
    grid = "".join(
        f'<line x1="{pad}" x2="{w - pad}" y1="{xy(0, g)[1]:.1f}" y2="{xy(0, g)[1]:.1f}" class="g"/>'
        f'<text x="4" y="{xy(0, g)[1] + 4:.1f}" class="t">{int(g * 100)}%</text>' for g in (0, .25, .5, .75, 1)
    )
    xs = "".join(f'<text x="{xy(i, 0)[0]:.1f}" y="{h - 10}" class="t" text-anchor="middle">{t["round"]}</text>'
                 for i, t in enumerate(timeline))
    return (f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="Support and opposition by round">{grid}{xs}'
            f'<path d="{path("exposed")}" class="ex"/><path d="{path("oppose")}" class="op"/>'
            f'<path d="{path("support")}" class="sp"/></svg>')


def to_html(summary: dict, fc: ForecastReport, timeline: list[dict]) -> str:
    e = html.escape
    li = lambda xs: "".join(f"<li>{e(x)}</li>" for x in xs)
    segs = "".join(
        f"<tr><td>{e(k)}</td><td>{v['n']}</td><td>{v['initial']:+.2f}</td><td>{v['final']:+.2f}</td>"
        f"<td class='{'up' if v['final'] >= v['initial'] else 'dn'}'>{v['final'] - v['initial']:+.2f}</td></tr>"
        for k, v in summary["segments"].items())
    posts = "".join(f"<li><b>{e(p['author'])}</b> <span class='m'>r{p['round']} · {p['engagement']} eng · {p['stance']:+.2f}</span><br>{e(p['text'])}</li>"
                    for p in summary["top_posts"])
    pr = summary["premise"] or {}
    narrative = "".join(f"<p>{e(par)}</p>" for par in fc.narrative.split("\n\n") if par.strip())
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Zerkalo forecast</title><style>
:root{{--p:#f7f7f2;--i:#0b0c0a;--n:#2bff6a;--m:#7d7e77;--l:#d9d9d1}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--p);color:var(--i);font:16px/1.5 system-ui,sans-serif}}
main{{max-width:960px;margin:0 auto;padding:40px 16px}}.k{{font:12px ui-monospace,monospace;letter-spacing:.08em;text-transform:uppercase;color:var(--m)}}
h1{{font-size:clamp(30px,5vw,54px);line-height:1;letter-spacing:-.03em;margin:.3em 0}}h2{{font-size:15px;text-transform:uppercase;letter-spacing:.06em;border-top:1px solid var(--i);padding-top:10px;margin-top:40px}}
.hl{{background:linear-gradient(var(--n),var(--n)) 0 85%/100% 38% no-repeat}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:1px;background:var(--l);border:1px solid var(--l);margin-top:24px}}
.stats div{{background:var(--p);padding:16px}}.stats b{{display:block;font-size:34px;letter-spacing:-.03em}}
.chart{{width:100%;height:auto;background:var(--i);margin-top:16px}}.g{{stroke:#2a2b27}}.t{{fill:#8b8c85;font:10px ui-monospace,monospace}}
.sp,.op,.ex{{fill:none;stroke-width:2.5}}.sp{{stroke:var(--n)}}.op{{stroke:#f7f7f2;stroke-dasharray:5 5}}.ex{{stroke:#555;stroke-width:1.5}}
table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;padding:8px 6px;border-bottom:1px solid var(--l)}}.up{{color:#0a7a33}}.dn{{color:#b3261e}}
ul{{padding-left:18px}}.posts li{{margin-bottom:12px}}.m{{color:var(--m);font-size:13px}}.leg span{{margin-right:16px}}
</style></head><body><main>
<div class="k">Zerkalo™ forecast · {e(summary['scenario'])}</div>
<h1><span class="hl">{e(fc.headline)}</span></h1>
<p>{e(summary['proposition'])}</p>
<p class="k">Premise {e(str(pr.get('label', '—')))}: {e(pr.get('text', 'baseline, no premise'))}</p>
<div class="stats"><div><span class="k">Support</span><b>{summary['final_support']:.0%}</b><span class="m">from {summary['baseline_support']:.0%} ({summary['support_delta']:+.0%})</span></div>
<div><span class="k">Oppose</span><b>{summary['final_oppose']:.0%}</b></div><div><span class="k">Exposure</span><b>{summary['exposure']:.0%}</b></div>
<div><span class="k">Confidence</span><b>{e(fc.confidence)}</b></div></div>
{_chart(timeline)}
<p class="leg k"><span style="color:#0a7a33">━ support</span><span>┅ oppose</span><span>━ exposure</span></p>
<h2>Predicted outcome</h2><p>{e(fc.predicted_outcome)}</p>
<h2>Key drivers</h2><ul>{li(fc.key_drivers)}</ul><h2>Resistant groups</h2><ul>{li(fc.resistant_groups)}</ul>
<h2>Tipping points</h2><ul>{li(fc.tipping_points)}</ul><h2>Risks</h2><ul>{li(fc.risks)}</ul>
<h2>Recommended adjustments</h2><ul>{li(fc.recommended_adjustments)}</ul>
<h2>Segments</h2><table><tr><th>Segment</th><th>Agents</th><th>Initial</th><th>Final</th><th>Shift</th></tr>{segs}</table>
<h2>Analysis</h2>{narrative}
<h2>Most engaged posts</h2><ul class="posts">{posts}</ul>
<p class="k" style="margin-top:48px">Simulated forecast of a synthetic public ({summary['agents']} agents, {summary['rounds']} rounds). Directional evidence, not a poll.</p>
</main></body></html>"""


async def generate(run_path: Path, llm: LLM) -> ForecastReport:
    summary = json.loads((run_path / "summary.json").read_text(encoding="utf-8"))
    timeline = json.loads((run_path / "timeline.json").read_text(encoding="utf-8"))
    world = json.loads((run_path / "world.json").read_text(encoding="utf-8"))
    fc = await write_forecast(summary, timeline, world, llm)
    (run_path / "forecast.json").write_text(fc.model_dump_json(indent=2), encoding="utf-8")
    (run_path / "report.md").write_text(to_markdown(summary, fc), encoding="utf-8")
    (run_path / "report.html").write_text(to_html(summary, fc, timeline), encoding="utf-8")
    return fc
