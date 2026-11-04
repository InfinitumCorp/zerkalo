"""Social graph: who follows whom.

Homophilous preferential attachment: people mostly follow people from their own segment, and
already-popular accounts attract more followers, giving the heavy-tailed influence real feeds have.
Edge u -> v means u follows v (u sees v's posts).
"""

from __future__ import annotations

import random

import networkx as nx

from .personas import Agent


def build_graph(agents: list[Agent], seed: int, follows: int = 8, homophily: float = 0.75) -> nx.DiGraph:
    rng = random.Random(seed)
    g = nx.DiGraph()
    g.add_nodes_from(a.id for a in agents)
    by_seg: dict[str, list[int]] = {}
    for a in agents:
        by_seg.setdefault(a.segment, []).append(a.id)
    indeg = {a.id: 1.0 for a in agents}
    order = list(agents)
    rng.shuffle(order)
    for a in order:
        k = max(2, int(rng.gauss(follows, follows / 3)))
        targets: set[int] = set()
        tries = 0
        while len(targets) < k and tries < k * 20:
            tries += 1
            pool = by_seg[a.segment] if rng.random() < homophily else [x.id for x in agents]
            # preferential attachment within the chosen pool (sample a few, keep the most followed)
            cand = max(rng.sample(pool, min(3, len(pool))), key=lambda i: indeg[i] * rng.random())
            if cand != a.id:
                targets.add(cand)
        for t in targets:
            g.add_edge(a.id, t)
            indeg[t] += 1
    top = max(indeg.values())
    for a in agents:
        a.influence = round((indeg[a.id] - 1) / max(top - 1, 1), 4)
    return g
