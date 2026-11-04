"""Talk to any agent after a run — ask a simulated person why they think what they think."""

from __future__ import annotations

import json
from pathlib import Path

from .llm import LLM


def load_agent(run_path: Path, agent_id: int) -> dict:
    agents = json.loads((run_path / "agents.json").read_text(encoding="utf-8"))
    for a in agents:
        if a["id"] == agent_id:
            return a
    raise KeyError(f"no agent {agent_id} in {run_path}")


def system_for(agent: dict, proposition: str) -> str:
    memory = "\n".join(f"- {m}" for m in agent.get("memory", [])) or "- (no notable activity)"
    return (
        f"You are {agent['name']}, {agent['age']}, {agent['occupation']} from {agent['city']} "
        f"({agent['segment']}). {agent.get('bio', '')}\n"
        f"You have just lived through a public debate about: \"{proposition}\".\n"
        f"Your stance moved from {agent['initial_stance']:+.2f} to {agent['stance']:+.2f} (-1 against … +1 for).\n"
        f"What happened to you during the debate:\n{memory}\n\n"
        "Answer an interviewer in character, briefly and honestly, as this person would. If asked "
        "what would change your mind, give a concrete answer. You are a simulated persona; if asked "
        "directly whether you are real, say you are a simulated respondent."
    )


async def ask(run_path: Path, agent_id: int, history: list[dict], llm: LLM) -> str:
    agent = load_agent(run_path, agent_id)
    sc = json.loads((run_path / "scenario.json").read_text(encoding="utf-8"))
    return await llm.text(system_for(agent, sc["proposition"]), history, role="analyst",
                          hint={"name": agent["name"], "stance": agent["stance"]})
