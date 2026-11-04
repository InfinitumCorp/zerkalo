"""Step 3 — rehearse the scenario.

Each round, a share of the public comes online, reads a feed built from the accounts it
follows plus what is trending, and acts (post / repost / comment / like / ignore) while its
stance toward the proposition shifts. Decisions within a round are made on the same snapshot
and applied together, so agent order never matters.
"""

from __future__ import annotations

import asyncio
import copy
import json
import random
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import networkx as nx

from .config import Premise, Scenario
from .llm import LLM, ZerkaloLLMError, premise_strength
from .network import build_graph
from .personas import Agent, build_population
from .schemas import AgentAction, WorldModel, clamp
from .world import build_world

SOURCE_ID = -1
SUPPORT, OPPOSE = 0.2, -0.2

AGENT_SYSTEM = (
    "You are role-playing one ordinary member of the public inside Zerkalo, a social simulation used "
    "to forecast how opinion forms. Stay in character: react the way this specific person plausibly "
    "would, given their background, values, certainty and what they have just seen. Most people "
    "change their minds slowly, ignore most content and rarely post. Do not be persuaded just "
    "because something is repeated, and do push back when a post conflicts with your values."
)


@dataclass
class Post:
    id: int
    round: int
    author: int
    author_name: str
    kind: str  # post | repost | comment | premise
    text: str
    stance: float
    parent: int | None = None
    root: int | None = None
    premise: bool = False
    likes: int = 0
    reposts: int = 0
    comments: int = 0

    @property
    def engagement(self) -> int:
        return self.likes + 2 * self.reposts + 2 * self.comments


@dataclass
class RunResult:
    scenario: Scenario
    premise: Premise | None
    world: WorldModel
    agents: list[Agent]
    posts: list[Post]
    timeline: list[dict]
    usage: dict = field(default_factory=dict)
    failures: int = 0

    # ---- analysis -------------------------------------------------------------
    def summary(self) -> dict:
        tl = self.timeline
        base, final = tl[0], tl[-1]
        sign = 1 if self.scenario.goal == "increase" else -1
        deltas = [(tl[i]["support"] - tl[i - 1]["support"]) * sign for i in range(1, len(tl))]
        tip = (deltas.index(max(deltas)) + 1) if deltas and max(deltas) > 0 else None
        segs = {}
        for a in self.agents:
            s = segs.setdefault(a.segment, {"n": 0, "initial": 0.0, "final": 0.0})
            s["n"] += 1
            s["initial"] += a.initial_stance
            s["final"] += a.stance
        for s in segs.values():
            s["initial"] = round(s["initial"] / s["n"], 3)
            s["final"] = round(s["final"] / s["n"], 3)
        movers = sorted(self.agents, key=lambda a: abs(a.stance - a.initial_stance), reverse=True)[:5]
        top = sorted((p for p in self.posts if p.kind != "repost"), key=lambda p: p.engagement, reverse=True)[:8]
        return {
            "scenario": self.scenario.name,
            "proposition": self.scenario.proposition,
            "goal": self.scenario.goal,
            "premise": self.premise.model_dump() if self.premise else None,
            "agents": len(self.agents),
            "llm_agents": sum(a.is_llm for a in self.agents),
            "rounds": len(tl) - 1,
            "baseline_support": base["support"],
            "final_support": final["support"],
            "final_oppose": final["oppose"],
            "support_delta": round(final["support"] - base["support"], 4),
            "mean_stance_delta": round(final["mean"] - base["mean"], 4),
            "final_polarization": final["polarization"],
            "exposure": final["exposed"],
            "tipping_round": tip,
            "segments": segs,
            "top_posts": [{"id": p.id, "round": p.round, "author": p.author_name, "kind": p.kind, "text": p.text,
                           "stance": round(p.stance, 2), "engagement": p.engagement} for p in top],
            "biggest_movers": [{"id": a.id, "name": a.name, "segment": a.segment,
                                "from": round(a.initial_stance, 2), "to": round(a.stance, 2)} for a in movers],
            "llm_failures": self.failures,
            "usage": self.usage,
        }

    # ---- persistence ----------------------------------------------------------
    def save(self, out_dir: str | Path) -> Path:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        dump = lambda name, obj: (d / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        dump("scenario.json", json.loads(self.scenario.model_dump_json(exclude={"base_dir"})))
        dump("world.json", self.world.model_dump())
        dump("agents.json", [a.to_dict() for a in self.agents])
        dump("timeline.json", self.timeline)
        dump("summary.json", self.summary())
        with (d / "posts.jsonl").open("w", encoding="utf-8") as f:
            for p in self.posts:
                f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
        return d


class Simulation:
    _DEFAULT = object()

    def __init__(self, scenario: Scenario, llm: LLM, premise: "Premise | None | object" = _DEFAULT):
        """premise: omit to use the scenario's premise; pass None for a no-premise baseline."""
        self.sc = scenario
        self.llm = llm
        self.premise = scenario.premise if premise is Simulation._DEFAULT else premise
        self.world: WorldModel | None = None
        self.agents: list[Agent] = []
        self.graph: nx.DiGraph | None = None

    async def setup(self) -> None:
        self.world = await build_world(self.sc, self.llm)
        self.agents = await build_population(self.sc, self.world, self.llm)
        self.graph = build_graph(self.agents, self.sc.random_seed)

    def fork(self, premise: Premise | None) -> "Simulation":
        """Same world, same public, same graph, different premise (for A/B tests)."""
        s = Simulation(self.sc, self.llm, premise)
        s.world, s.graph, s.agents = self.world, self.graph, copy.deepcopy(self.agents)
        return s

    # ---------------------------------------------------------------------------
    async def run(self, progress: Callable[[dict], None] | None = None) -> RunResult:
        if self.world is None:
            await self.setup()
        sc, g = self.sc, self.graph
        rng = random.Random(sc.random_seed * 7919 + 1)
        posts: list[Post] = []
        by_id = {a.id: a for a in self.agents}
        source_followers: set[int] = set()
        exposed: set[int] = set()
        failures = 0
        timeline = [self._snapshot(0, posts, exposed)]
        if progress:
            progress(timeline[-1])

        pr = self.premise
        if pr:
            n = round(pr.reach * len(self.agents))
            ranked = sorted(self.agents, key=lambda a: a.influence, reverse=True)
            top = [a.id for a in ranked[: n // 2]]
            rest = [a.id for a in ranked[n // 2:]]
            # separate stream, so variants keep the same activity/decision randomness (common random numbers)
            frng = random.Random(sc.random_seed * 31 + 5)
            source_followers = set(top) | set(frng.sample(rest, min(len(rest), n - len(top))))
        strength = premise_strength(pr.text) if pr else 0.0

        for r in range(1, sc.rounds + 1):
            if pr and r == pr.inject_round:
                posts.append(Post(id=len(posts), round=r, author=SOURCE_ID, author_name=pr.source, kind="premise",
                                  text=pr.text, stance=pr.stance, root=len(posts), premise=True))
                posts[-1].root = posts[-1].id

            active = [a for a in self.agents if rng.random() < sc.active_fraction * a.activity]
            feeds = {a.id: self._feed(a, posts, r, source_followers, g, rng) for a in active}
            for a in active:
                if any(p.premise for p in feeds[a.id]):
                    exposed.add(a.id)

            async def decide(a: Agent):
                feed = feeds[a.id]
                if a.is_llm:
                    try:
                        return a, await self._llm_decide(a, feed, pr, strength)
                    except ZerkaloLLMError:
                        nonlocal failures
                        failures += 1
                return a, self._rule_decide(a, feed, rng)

            decisions = await asyncio.gather(*(decide(a) for a in active))
            for a, act in decisions:
                self._apply(a, act, posts, r, by_id)

            snap = self._snapshot(r, posts, exposed)
            timeline.append(snap)
            if progress:
                progress(snap)

        usage = getattr(self.llm, "usage", None)
        return RunResult(self.sc, pr, self.world, self.agents, posts, timeline,
                         usage=asdict(usage) if usage else {}, failures=failures)

    # ---------------------------------------------------------------------------
    def _feed(self, a: Agent, posts: list[Post], r: int, src: set[int], g: nx.DiGraph, rng: random.Random) -> list[Post]:
        follows = set(g.successors(a.id))
        recent = [p for p in posts if p.round >= r - 3 and p.author != a.id]
        mine = [p for p in recent if p.author in follows or (p.author == SOURCE_ID and a.id in src)]
        trending = sorted(recent, key=lambda p: p.engagement, reverse=True)[:2]
        seen, feed = set(), []
        for p in sorted(mine, key=lambda p: (p.round, p.engagement), reverse=True)[:6] + trending:
            if p.id not in seen:
                seen.add(p.id)
                feed.append(p)
        return feed

    def _feed_mean(self, feed: list[Post], by_author: dict[int, float] | None = None) -> float:
        if not feed:
            return 0.0
        w = [1.0 + (2.0 if p.premise else 0.0) + 0.1 * p.engagement for p in feed]
        return sum(p.stance * wi for p, wi in zip(feed, w)) / sum(w)

    async def _llm_decide(self, a: Agent, feed: list[Post], pr: Premise | None, strength: float) -> AgentAction:
        lines = "\n".join(
            f"[{p.id}] {p.author_name}{' (reposted)' if p.kind == 'repost' else ''}: {p.text}  "
            f"({p.likes} likes, {p.reposts} reposts)" for p in feed
        ) or "(your feed is quiet)"
        memory = "\n".join(f"- {m}" for m in a.memory[-6:]) or "- (nothing yet)"
        prompt = f"""You are {a.name}, {a.age}, {a.occupation} from {a.city}. Segment: {a.segment}.
About you: {a.bio}
Values: {', '.join(a.values) or 'n/a'}. You get news from: {', '.join(a.media_diet) or 'n/a'}. You write like: {a.voice}.

The public debate is about: "{self.sc.proposition}"
Your current stance is {a.stance:+.2f} on a scale from -1 (strongly against) to +1 (strongly for); your certainty is {a.certainty:.2f} (0-1).

What you remember recently:
{memory}

Your feed right now:
{lines}

Decide what you do next. action is one of post, repost, comment, like, ignore.
- target_post_id: the [id] you repost/comment/like, else null.
- text: your post or comment written in {self.sc.language} in your own voice (max 280 chars), else null.
- new_stance: your stance after this moment, in [-1, 1]. Small moves are normal; big swings need a strong reason.
- reasoning: one short sentence on why."""
        hint = {"stance": a.stance, "certainty": a.certainty, "feed_mean": self._feed_mean(feed),
                "feed_ids": [p.id for p in feed], "premise_in_feed": any(p.premise for p in feed),
                "premise_strength": strength, "premise_stance": pr.stance if pr else 0.0}
        return await self.llm.structured(AGENT_SYSTEM, prompt, AgentAction, role="agent", hint=hint)

    def _rule_decide(self, a: Agent, feed: list[Post], rng: random.Random) -> AgentAction:
        """Bounded-confidence update for the procedural crowd."""
        s = a.stance
        if feed:
            fm = self._feed_mean(feed)
            d = fm - s
            eps = 0.45 + 0.9 * (1 - a.certainty)
            if abs(d) < eps:
                s += 0.3 * a.susceptibility * (1 - 0.5 * a.certainty) * d
            else:  # too far from my view: mild backfire
                s -= 0.04 * a.certainty * (1 if d > 0 else -1)
        s = clamp(s + rng.gauss(0, 0.01))
        aligned = [p for p in feed if abs(p.stance - s) < 0.35]
        roll = rng.random()
        if aligned and roll < 0.35:
            return AgentAction(action="like", target_post_id=rng.choice(aligned).id, text=None, new_stance=s, reasoning="")
        if aligned and roll < 0.35 + 0.12 * (1 + a.influence):
            return AgentAction(action="repost", target_post_id=max(aligned, key=lambda p: p.engagement).id,
                               text=None, new_stance=s, reasoning="")
        if roll > 0.93:
            word = "support" if s > SUPPORT else "oppose" if s < OPPOSE else "am unsure about"
            return AgentAction(action="post", target_post_id=None, text=f"I {word} this.", new_stance=s, reasoning="")
        return AgentAction(action="ignore", target_post_id=None, text=None, new_stance=s, reasoning="")

    def _apply(self, a: Agent, act: AgentAction, posts: list[Post], r: int, by_id: dict[int, Agent]) -> None:
        old = a.stance
        a.stance = clamp(act.new_stance)
        target = posts[act.target_post_id] if act.target_post_id is not None and 0 <= act.target_post_id < len(posts) else None
        if act.action == "post" and act.text:
            posts.append(Post(len(posts), r, a.id, a.name, "post", act.text, a.stance))
            posts[-1].root = posts[-1].id
        elif act.action == "repost" and target:
            target.reposts += 1
            root = posts[target.root] if target.root is not None else target
            posts.append(Post(len(posts), r, a.id, a.name, "repost", target.text, target.stance,
                              parent=target.id, root=root.id, premise=target.premise))
        elif act.action == "comment" and target and act.text:
            target.comments += 1
            posts.append(Post(len(posts), r, a.id, a.name, "comment", act.text, a.stance, parent=target.id,
                              root=target.root))
        elif act.action == "like" and target:
            target.likes += 1
        if act.action != "ignore" or abs(a.stance - old) > 0.05:
            what = f"you {act.action}" + (f" post [{target.id}] by {target.author_name}" if target else "")
            a.remember(f"round {r}: {what}; stance {old:+.2f} → {a.stance:+.2f}" + (f" ({act.reasoning})" if act.reasoning else ""))

    def _snapshot(self, r: int, posts: list[Post], exposed: set[int]) -> dict:
        st = [a.stance for a in self.agents]
        n = len(st)
        segs: dict[str, list[float]] = {}
        for a in self.agents:
            segs.setdefault(a.segment, []).append(a.stance)
        this = [p for p in posts if p.round == r]
        return {
            "round": r,
            "mean": round(statistics.fmean(st), 4),
            "support": round(sum(s > SUPPORT for s in st) / n, 4),
            "oppose": round(sum(s < OPPOSE for s in st) / n, 4),
            "neutral": round(sum(OPPOSE <= s <= SUPPORT for s in st) / n, 4),
            "polarization": round(statistics.pstdev(st), 4),
            "posts": sum(p.kind in ("post", "comment") for p in this),
            "reposts": sum(p.kind == "repost" for p in this),
            "exposed": round(len(exposed) / n, 4),
            "segments": {k: round(statistics.fmean(v), 4) for k, v in segs.items()},
        }


def run_dir(base: str | Path, scenario: Scenario, label: str = "") -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = "".join(c if c.isalnum() else "-" for c in scenario.name.lower()).strip("-")
    return Path(base) / f"{stamp}-{slug}{'-' + label if label else ''}"
