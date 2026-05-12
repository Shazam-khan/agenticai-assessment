"""Five orchestration scenarios.

Chosen to exercise:
  1. Routing precision (no spurious agent calls).
  2. Multi-agent fan-out + report synthesis.
  3. A second routing case to confirm the planner is not biased to one agent.
  4. Graceful degradation when an agent hard-fails (escalate, continue).
  5. Self-correction when an agent returns a schema-violating payload (retry-with-modified-instruction succeeds).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from shared.agent_base import Agent

from ..agents.inventory import InventoryAgent
from ..agents.production import ProductionAgent
from ..agents.report import ReportAgent
from .doubles import AlwaysFailingProduction, HallucinatingInventoryOnce


def _default_registry(llm) -> dict[str, Agent]:
    return {
        "inventory": InventoryAgent(llm=llm),
        "production": ProductionAgent(llm=llm),
        "report": ReportAgent(llm=llm),
    }


def _failing_production_registry(llm) -> dict[str, Agent]:
    return {
        "inventory": InventoryAgent(llm=llm),
        "production": AlwaysFailingProduction(llm=llm),
        "report": ReportAgent(llm=llm),
    }


def _hallucinating_inventory_registry(llm) -> dict[str, Agent]:
    return {
        "inventory": HallucinatingInventoryOnce(llm=llm),
        "production": ProductionAgent(llm=llm),
        "report": ReportAgent(llm=llm),
    }


@dataclass
class Scenario:
    id: str
    objective: str
    registry_factory: Callable[[object], dict[str, Agent]]
    expected_agents_called: set[str]
    forbidden_agents_called: set[str] = field(default_factory=set)
    expected_subtask_statuses: dict[str, str] = field(default_factory=dict)
    must_produce_final_answer: bool = True
    notes: str = ""


SCENARIOS: list[Scenario] = [
    Scenario(
        id="s1_stock_only",
        objective="Are we running low on glycerin or vitamin E?",
        registry_factory=_default_registry,
        expected_agents_called={"inventory"},
        forbidden_agents_called={"production", "report"},
        expected_subtask_statuses={"inventory": "success"},
        notes="Pure routing test: should NOT pull in production or report.",
    ),
    Scenario(
        id="s2_schedule_only",
        objective="What's blocking production today?",
        registry_factory=_default_registry,
        expected_agents_called={"production"},
        forbidden_agents_called={"inventory", "report"},
        expected_subtask_statuses={"production": "success"},
        notes="Routing test in the other direction.",
    ),
    Scenario(
        id="s3_full_briefing",
        objective=(
            "Give me the morning operations briefing covering both stock levels "
            "for our key materials and any production bottlenecks."
        ),
        registry_factory=_default_registry,
        expected_agents_called={"inventory", "production", "report"},
        expected_subtask_statuses={"inventory": "success", "production": "success"},
        notes="Fan-out + synthesis. report is wired by the supervisor, not the planner.",
    ),
    Scenario(
        id="s4_graceful_degradation",
        objective=(
            "Give me the morning operations briefing covering both stock levels "
            "for our key materials and any production bottlenecks."
        ),
        registry_factory=_failing_production_registry,
        expected_agents_called={"inventory", "production"},
        expected_subtask_statuses={"inventory": "success", "production": "escalated"},
        notes="Production hard-fails; supervisor must escalate it but still answer.",
    ),
    Scenario(
        id="s5_retry_corrects_hallucination",
        objective="Are we running low on glycerin or vitamin E?",
        registry_factory=_hallucinating_inventory_registry,
        expected_agents_called={"inventory"},
        expected_subtask_statuses={"inventory": "success"},
        notes=(
            "Inventory returns invalid output on call 1, valid on call 2. "
            "Supervisor must retry-with-modified-instruction and recover."
        ),
    ),
]
