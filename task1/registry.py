"""Wire the agents together. One place to swap in test doubles."""
from __future__ import annotations

from shared.agent_base import Agent
from shared.llm import LLMClient

from .agents.inventory import InventoryAgent
from .agents.production import ProductionAgent
from .agents.report import ReportAgent
from .agents.supervisor import SupervisorAgent


def build_system(llm: LLMClient | None = None) -> SupervisorAgent:
    llm = llm or LLMClient()
    registry: dict[str, Agent] = {
        "inventory": InventoryAgent(llm=llm),
        "production": ProductionAgent(llm=llm),
        "report": ReportAgent(llm=llm),
    }
    return SupervisorAgent(llm=llm, registry=registry)
