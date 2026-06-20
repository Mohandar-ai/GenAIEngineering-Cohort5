"""
Agno OS — Multi-Agent Playground (Day 2)
==========================================
Advanced Agno OS playground showcasing all Day 2 agentic patterns:

1. Coordinator Team    — routes customer requests to specialists
2. Research Director   — hierarchical task decomposition
3. Code Review Team    — review-critique (generator + critic loop)
4. Pattern Advisor     — recommends GCP agentic patterns

Run:
    python 6_agno_os_multiagent.py

Then visit: http://localhost:7777

Tip: In Agno OS, you can switch between agents and see tool calls,
memory, and reasoning in real-time.
"""

import os
from dotenv import load_dotenv

load_dotenv()
for var in ['OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_API_BASE']:
    if os.getenv(var):
        del os.environ[var]

os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_ROUTER_KEY")
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

from agno.agent import Agent
from agno.team import Team
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.storage.sqlite import SqliteStorage
from agno.app.playground import Playground, serve_playground_app
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

model_id = "google/gemini-2.5-flash"
storage_db = "agno_day2.db"


# ─────────────────────────────────────────────────────────────
# 1. Coordinator Pattern Team
# ─────────────────────────────────────────────────────────────

order_agent = Agent(
    name="Order Specialist",
    role="Handle order tracking, shipping, and delivery questions",
    model=OpenAIChat(id=model_id),
    instructions=["Be specific about delivery timelines. Orders take 5-7 business days."],
    markdown=True,
)

returns_agent = Agent(
    name="Returns Specialist",
    role="Handle returns, exchanges, and product issues",
    model=OpenAIChat(id=model_id),
    instructions=["30-day return policy for most items. Always provide return label."],
    markdown=True,
)

billing_agent = Agent(
    name="Billing Specialist",
    role="Handle refunds, billing disputes, and payment issues",
    model=OpenAIChat(id=model_id),
    instructions=["Refunds in 5-7 days. Offer 10% bonus store credit as alternative."],
    markdown=True,
)

tech_agent_support = Agent(
    name="Tech Support Specialist",
    role="Handle technical issues and product troubleshooting",
    model=OpenAIChat(id=model_id),
    instructions=["Start with basic troubleshooting. Escalate if unresolved after 3 steps."],
    markdown=True,
)

coordinator_team = Team(
    name="Customer Service Coordinator (Pattern 4)",
    team_id="coordinator-team",
    mode="coordinate",
    model=OpenAIChat(id=model_id),
    members=[order_agent, returns_agent, billing_agent, tech_agent_support],
    storage=SqliteStorage(table_name="coordinator_team", db_file=storage_db),
    description=(
        "I demonstrate the COORDINATOR pattern from Google Cloud's Agentic AI guide. "
        "I analyze your request and dynamically route it to the right specialist. "
        "Try: order status, return request, refund inquiry, or technical issue."
    ),
    instructions=[
        "Analyze the request type and route to the appropriate specialist.",
        "Route based on: order/shipping → Order Specialist, returns/exchanges → Returns Specialist, "
        "refunds/billing → Billing Specialist, technical problems → Tech Support.",
        "Greet warmly and summarize the resolution.",
    ],
    show_tool_calls=True,
    markdown=True,
)


# ─────────────────────────────────────────────────────────────
# 2. Hierarchical Research Director
# ─────────────────────────────────────────────────────────────

web_searcher = Agent(
    name="Web Searcher",
    role="Search for current news and information",
    model=OpenAIChat(id=model_id),
    tools=[DuckDuckGoTools()],
    instructions=["Search for 3-5 recent sources. Extract key facts and dates."],
    markdown=True,
)

data_analyst = Agent(
    name="Data Analyst",
    role="Analyze quantitative data and market statistics",
    model=OpenAIChat(id=model_id),
    tools=[YFinanceTools(stock_price=True, company_info=True)],
    instructions=["Focus on numbers: market size, growth rates, financial metrics."],
    markdown=True,
)

trend_analyst_team = Team(
    name="Research Gathering Team",
    mode="parallel",
    model=OpenAIChat(id=model_id),
    members=[web_searcher, data_analyst],
    instructions=["Gather information from all available sources simultaneously."],
    markdown=True,
)

report_synthesis_agent = Agent(
    name="Report Writer",
    role="Synthesize research into executive reports",
    model=OpenAIChat(id=model_id),
    instructions=[
        "Write comprehensive executive reports.",
        "Structure: Executive Summary, Key Findings, Market Data, Trends, Recommendations.",
        "Use markdown headers, tables, and bullet points.",
    ],
    markdown=True,
)

research_director_team = Team(
    name="Research Director (Hierarchical Pattern 5)",
    team_id="research-director",
    mode="coordinate",
    model=OpenAIChat(id=model_id),
    members=[trend_analyst_team, report_synthesis_agent],
    storage=SqliteStorage(table_name="research_director", db_file=storage_db),
    description=(
        "I demonstrate the HIERARCHICAL TASK DECOMPOSITION pattern. "
        "I'm the root agent — I decompose your research question, coordinate a research team "
        "(parallel data gathering), then synthesize everything into a comprehensive report. "
        "Ask me to research any market, technology, or company."
    ),
    instructions=[
        "You are the root research director.",
        "Phase 1: Send research request to the Research Gathering Team (parallel).",
        "Phase 2: Pass gathered data to the Report Writer for synthesis.",
        "Ensure the final report is comprehensive and actionable.",
    ],
    show_tool_calls=True,
    markdown=True,
)


# ─────────────────────────────────────────────────────────────
# 3. Review-Critique Code Review Team
# ─────────────────────────────────────────────────────────────

code_reviewer = Agent(
    name="Code Review & Critique Team (Pattern 7-8)",
    agent_id="code-reviewer",
    model=OpenAIChat(id=model_id),
    storage=SqliteStorage(table_name="code_reviewer", db_file=storage_db),
    description=(
        "I demonstrate the REVIEW AND CRITIQUE pattern. "
        "Give me any code snippet and I'll perform a comprehensive review covering: "
        "security vulnerabilities, performance issues, code quality, and best practices. "
        "I'll also suggest an improved version."
    ),
    instructions=[
        "Perform thorough code review covering: Security, Performance, Readability, Best Practices.",
        "Rate each dimension: 🟢 Good / 🟡 Needs Improvement / 🔴 Critical Issue.",
        "Provide a revised version of the code with all issues fixed.",
        "Use this structure: ## Security | ## Performance | ## Code Quality | ## Improved Code",
    ],
    markdown=True,
)


# ─────────────────────────────────────────────────────────────
# 4. Agentic Pattern Advisor
# ─────────────────────────────────────────────────────────────

class GCPPattern(str, Enum):
    SINGLE = "single-agent"
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    LOOP = "loop"
    REVIEW_CRITIQUE = "review-and-critique"
    ITERATIVE_REFINEMENT = "iterative-refinement"
    COORDINATOR = "coordinator"
    HIERARCHICAL = "hierarchical-task-decomposition"
    SWARM = "swarm"
    REACT = "react"
    HUMAN_IN_LOOP = "human-in-the-loop"
    CUSTOM = "custom-logic"

class PatternRecommendation(BaseModel):
    primary_pattern: GCPPattern
    secondary_pattern: Optional[GCPPattern] = None
    confidence: int = Field(..., ge=0, le=100, description="Confidence percentage")
    reasoning: str
    implementation_complexity: str = Field(..., description="Low / Medium / High")
    estimated_latency: str = Field(..., description="Low / Medium / High")
    estimated_cost: str = Field(..., description="Low / Medium / High")
    code_sketch: str = Field(..., description="Brief Python pseudo-code showing the pattern structure")
    gcp_reference_url: str = Field(..., description="URL to GCP documentation section")

pattern_advisor = Agent(
    name="GCP Agentic Pattern Advisor",
    agent_id="pattern-advisor",
    model=OpenAIChat(id=model_id),
    storage=SqliteStorage(table_name="pattern_advisor", db_file=storage_db),
    response_model=PatternRecommendation,
    structured_outputs=True,
    description=(
        "I'm your Google Cloud Agentic AI architecture advisor. "
        "Describe your use case and I'll recommend the optimal pattern from the GCP guide, "
        "explain the trade-offs, and give you a code sketch to get started. "
        "Reference: https://docs.cloud.google.com/architecture/choose-design-pattern-agentic-ai-system"
    ),
    instructions=[
        "You are an expert in Google Cloud's 12 agentic AI patterns.",
        "Patterns: single-agent, sequential, parallel, loop, review-and-critique, iterative-refinement, "
        "coordinator, hierarchical-task-decomposition, swarm, react, human-in-the-loop, custom-logic.",
        "Always recommend the SIMPLEST pattern that solves the problem (don't over-engineer).",
        "Provide accurate GCP documentation URLs for the recommended pattern.",
        "Code sketches should use agno framework syntax.",
    ],
    markdown=True,
)


# ─────────────────────────────────────────────────────────────
# Agno OS — launch all agents/teams
# ─────────────────────────────────────────────────────────────

app = Playground(
    agents=[code_reviewer, pattern_advisor],
    teams=[coordinator_team, research_director_team],
).get_app()

if __name__ == "__main__":
    print("🚀 Starting Agno OS — Multi-Agent Playground (Day 2)")
    print("=" * 65)
    print("📱 Visit: http://localhost:7777")
    print("\nAgents & Teams available:")
    print("\n  TEAMS (multi-agent):")
    print("  1. Customer Service Coordinator — routes to specialists (Pattern 4)")
    print("  2. Research Director — hierarchical decomposition (Pattern 5)")
    print("\n  AGENTS:")
    print("  3. Code Review & Critique — generator-critic loop (Patterns 7-8)")
    print("  4. GCP Pattern Advisor — recommends the right pattern for your use case")
    print("\nAll sessions are persisted in agno_day2.db")
    print("\nPress Ctrl+C to stop.\n")
    serve_playground_app("6_agno_os_multiagent:app", reload=True)
