"""
Structured Output with Agno
============================
Demonstrates how to enforce type-safe, schema-validated responses using Pydantic models.
This is critical for agent outputs that feed into downstream systems.

Run: python 3_structured_output.py
"""

import os
from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
for var in ['OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_API_BASE']:
    if os.getenv(var):
        del os.environ[var]

os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_ROUTER_KEY")
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

from agno.agent import Agent
from agno.models.openai import OpenAIChat


# ─────────────────────────────────────────────
# 1. Simple Structured Output
# ─────────────────────────────────────────────

class AgenticPattern(BaseModel):
    """Schema for describing an agentic AI design pattern."""
    name: str = Field(..., description="Pattern name (e.g. 'Sequential Pattern')")
    category: str = Field(..., description="single-agent | multi-agent | iterative | special")
    google_cloud_reference: str = Field(..., description="Section name from Google Cloud architecture guide")
    use_case: str = Field(..., description="Primary use case in one sentence")
    latency: str = Field(..., description="Low / Medium / High")
    cost: str = Field(..., description="Low / Medium / High")
    complexity: str = Field(..., description="Low / Medium / High")
    when_to_use: List[str] = Field(..., description="3 bullet points on when to choose this pattern")
    when_to_avoid: List[str] = Field(..., description="2 bullet points on when NOT to use this pattern")


pattern_classifier = Agent(
    model=OpenAIChat(id="google/gemini-2.5-flash"),
    description="You are an expert in agentic AI architecture patterns.",
    output_schema=AgenticPattern,
    structured_outputs=True,
)

print("=" * 60)
print("1. STRUCTURED PATTERN DESCRIPTION")
print("=" * 60)
result = pattern_classifier.run(
    "Describe the Coordinator pattern (multi-agent) from Google Cloud's agentic AI architecture guide."
)
pattern = result.content
print(f"\nPattern: {pattern.name}")
print(f"Category: {pattern.category}")
print(f"GCP Reference: {pattern.google_cloud_reference}")
print(f"Use Case: {pattern.use_case}")
print(f"Latency/Cost/Complexity: {pattern.latency} / {pattern.cost} / {pattern.complexity}")
print(f"\nWhen to use:")
for bullet in pattern.when_to_use:
    print(f"  • {bullet}")
print(f"\nWhen to avoid:")
for bullet in pattern.when_to_avoid:
    print(f"  • {bullet}")


# ─────────────────────────────────────────────
# 2. Nested Structured Output
# ─────────────────────────────────────────────

class AgentSpec(BaseModel):
    name: str
    role: str
    tools: List[str]
    instructions: List[str]

class MultiAgentSystemDesign(BaseModel):
    """Complete multi-agent system design blueprint."""
    system_name: str = Field(..., description="Name of the agentic system")
    pattern: str = Field(..., description="Which GCP pattern this implements")
    problem_statement: str = Field(..., description="The problem being solved")
    agents: List[AgentSpec] = Field(..., description="List of agents in the system")
    orchestration_logic: str = Field(..., description="How agents are coordinated")
    expected_output: str = Field(..., description="What the system produces")
    success_criteria: List[str] = Field(..., description="How to measure success")


system_designer = Agent(
    model=OpenAIChat(id="google/gemini-2.5-flash"),
    description="You are an AI systems architect specializing in multi-agent design.",
    output_schema=MultiAgentSystemDesign,
    structured_outputs=True,
)

print("\n" + "=" * 60)
print("2. MULTI-AGENT SYSTEM DESIGN BLUEPRINT")
print("=" * 60)

design_result = system_designer.run(
    """Design a multi-agent system using the Sequential Pattern that:
    - Fetches latest news about a company
    - Analyzes sentiment and financial impact
    - Generates a structured investment report

    Follow the Google Cloud sequential pattern (predefined linear order, no model orchestration)."""
)
design = design_result.content
print(f"\nSystem: {design.system_name}")
print(f"Pattern: {design.pattern}")
print(f"Problem: {design.problem_statement}")
print(f"\nAgents ({len(design.agents)} total):")
for agent in design.agents:
    print(f"\n  Agent: {agent.name} — {agent.role}")
    print(f"  Tools: {', '.join(agent.tools)}")
    print(f"  Instructions: {agent.instructions[0]}...")
print(f"\nOrchestration: {design.orchestration_logic}")
print(f"Output: {design.expected_output}")
print(f"\nSuccess Criteria:")
for criterion in design.success_criteria:
    print(f"  ✓ {criterion}")


# ─────────────────────────────────────────────
# 3. Enum-Constrained Output
# ─────────────────────────────────────────────

from enum import Enum

class PatternCategory(str, Enum):
    SINGLE_AGENT = "single-agent"
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    LOOP = "loop"
    COORDINATOR = "coordinator"
    HIERARCHICAL = "hierarchical"
    SWARM = "swarm"
    REACT = "react"
    HUMAN_IN_LOOP = "human-in-the-loop"
    CUSTOM = "custom"

class PatternRecommendation(BaseModel):
    recommended_pattern: PatternCategory
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence 0-1")
    reasoning: str = Field(..., description="Why this pattern was chosen")
    alternative_pattern: Optional[PatternCategory] = None
    alternative_reasoning: Optional[str] = None

advisor = Agent(
    model=OpenAIChat(id="google/gemini-2.5-flash"),
    description="You are a Google Cloud agentic AI architecture advisor.",
    output_schema=PatternRecommendation,
    structured_outputs=True,
)

print("\n" + "=" * 60)
print("3. PATTERN RECOMMENDATION ENGINE")
print("=" * 60)

scenarios = [
    "I need to analyze customer feedback: simultaneously check sentiment, extract keywords, categorize topic, and detect urgency.",
    "I need to generate a blog post and have it reviewed by a quality checker before publishing.",
    "I need a system that decomposes complex research questions into sub-tasks and delegates them to specialist agents.",
]

for scenario in scenarios:
    rec_result = advisor.run(f"Which GCP agentic pattern fits this scenario? {scenario}")
    rec = rec_result.content
    print(f"\nScenario: {scenario[:70]}...")
    print(f"→ Recommended: {rec.recommended_pattern.value} (confidence: {rec.confidence:.0%})")
    print(f"  Reason: {rec.reasoning}")
    if rec.alternative_pattern:
        print(f"  Alternative: {rec.alternative_pattern.value} — {rec.alternative_reasoning}")

print("\n✅ Done! All structured outputs validated against Pydantic schemas.")
