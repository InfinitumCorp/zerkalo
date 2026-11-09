# ZERKALO™

**Зеркало, "mirror". A reflexive simulation engine.**
*Forecast the outcome. Then design it.*

Zerkalo builds a synthetic public: hundreds of agents modelled on real audience segments, each with
its own persona, memory, certainty and place in a social graph. You release a message into that
public and watch it spread. It shows who amplifies the message, who resists it, and where opinion
settles. Then you change the premise and run it again against the same public, until the forecast
matches the outcome you need.

Built in-house by Reflexive Control, Moscow.

```
# illustrative console output
  round  0  support ████████················   34%  oppose   29%  exposed    0%
  round  5  support █████████···············   38%  oppose   27%  exposed   23%
  round 10  support ███████████·············   45%  oppose   24%  exposed   31%
```

## How it works

| Stage | Module | What happens |
|---|---|---|
| **01 Mirror the world** | `world.py` | Seed material (news, research, transcripts) becomes a structured world model with facts, stakeholders, fault lines and audience segments. |
| **02 Populate the public** | `personas.py`, `network.py` | A core of LLM-driven agents gets rich personas; a larger procedural crowd fills out the public. A homophilous, preferential-attachment follow graph gives a realistic, heavy-tailed spread of influence. |
| **03 Rehearse the scenario** | `simulation.py` | Each round, agents come online, read a feed (accounts they follow plus trending posts), then post, repost, comment, like or ignore, and their stance shifts. The premise enters through a source account that reaches a set share of the public. LLM agents reason in character; the crowd follows a bounded-confidence opinion model. |
| **04 Read the forecast** | `report.py` | Metrics (support, opposition, exposure, polarization, tipping round, segment shifts, top posts, biggest movers) are turned by an analyst model into a forecast with drivers, risks and recommended adjustments, in Markdown and HTML. |
| **Design the outcome** | `compare.py` | A/B tests premise variants against the *same* world, agents, graph and random stream, so any difference in outcome comes from the message alone. |
| **Interview** | `interview.py` | After a run, question any simulated person about why they think what they think. |

## Install

```bash
pip install -e ".[dev]"
```

Python 3.10+. To use Claude, set `ANTHROPIC_API_KEY` or sign in with `ant auth login`.

## Use

```bash
# scaffold a scenario
zerkalo init scenarios/my-campaign

# offline dry run: no API calls, deterministic synthetic numbers
zerkalo run examples/weekend-pedestrian/scenario.yaml --mock

# real run with Claude
zerkalo run examples/weekend-pedestrian/scenario.yaml

# A/B test every premise in the scenario, plus a no-premise baseline
zerkalo compare examples/weekend-pedestrian/scenario.yaml --report

# question an agent from a finished run
zerkalo interview runs/<run-dir> --agent 3
```

Each run writes a folder containing `report.html`, `report.md`, `summary.json`, `timeline.json`,
`agents.json` (personas, final stance, memory), `posts.jsonl` (every post, repost and comment) and
`world.json`.

## Scenario file

```yaml
name: Weekend pedestrian zone
proposition: "Moscow residents support closing the boulevard ring to cars on summer weekends"
goal: increase                 # or decrease
seed_files: [seed/brief.md]
segments:                      # optional; otherwise derived from the seed material
  - { name: Drivers & commuters, share: 0.22, baseline_stance: -0.45, susceptibility: 0.3 }
n_agents: 300                  # whole public
n_llm_agents: 24               # agents that reason with Claude every round
rounds: 10
premise:
  label: A
  source: "City news outlet"
  text: "The message under test."
  stance: 0.8                  # direction it pushes (-1 … +1)
  reach: 0.15                  # share of the public the source reaches directly
  inject_round: 2
variants: [ … ]                # more premises for `zerkalo compare`
models:
  agent: claude-opus-5-5
  analyst: claude-opus-5-5
  agent_effort: low            # effort is the main cost lever for the agent swarm
  analyst_effort: high
  concurrency: 8
```

See [`examples/weekend-pedestrian`](examples/weekend-pedestrian) for a complete, fictional example.

## Cost and scale

Each round, roughly `n_llm_agents × active_fraction` agents call the model, plus a few analyst
calls per run for the world, personas and report. The procedural crowd costs nothing, so you can
scale the public with `n_agents` and the depth of reasoning with `n_llm_agents`. Lowering
`agent_effort`, or pointing `models.agent` at a smaller model such as `claude-haiku-5-5`, cuts the
cost of large swarms. Calls go through the Anthropic SDK with structured outputs, retries and
server-side refusal fallbacks. Any agent decision that fails falls back to the rule model and is
counted in `summary.json → llm_failures`.

## Reading the numbers

Stance runs from −1 (strongly against) to +1 (strongly for). *Support* is the share of agents above
+0.2 and *oppose* the share below −0.2. *Exposure* is the share of agents who saw the premise or a
repost of it. A forecast describes a synthetic public: use it as directional evidence and to compare
messages, not as a poll. `--mock` output is synthetic and only exercises the pipeline.

## Cyber geo-routing

Optional annotations label IP addresses and domain names in a run with country and category
data from the v2fly geoip and geosite lists. The switch is off unless you turn it on, and a
run with it off is unchanged.

```bash
zerkalo settings                   # cyber geo-routing: off
zerkalo settings --geo-routing     # persist on
zerkalo settings --no-geo-routing  # persist off
zerkalo run scenario.yaml --geo-routing
```

The saved switch is `~/.config/zerkalo/settings.json` (override the path with
`ZERKALO_SETTINGS_PATH`). For one process, `INFINITUM_GEO_ENABLED=1` or `0` overrides
that file, and `--geo-routing` / `--no-geo-routing` overrides both. A scenario can set
`geo_routing: true` or `false`; omit the key to keep the saved switch. Lists are not
in git. From a checkout:

```bash
python scripts/fetch_geo_dat.py
```

That downloads the pinned v2fly release assets into `data/geo/` and checks them against
`data/geo.lock.json` and the official `.sha256sum` files. See `THIRD_PARTY_NOTICES`.

## Develop

```bash
pytest -q
```

---
Proprietary. © Reflexive Control. All rights reserved.
