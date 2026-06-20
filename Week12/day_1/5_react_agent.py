"""
Pattern: ReAct (Reason and Act)
================================
> Google Cloud Reference: https://docs.cloud.google.com/architecture/choose-design-pattern-agentic-ai-system#react-pattern

The ReAct pattern is an iterative loop of:
  1. THOUGHT  — model reasons about what to do next
  2. ACTION   — model calls a tool or produces final answer
  3. OBSERVATION — model receives tool output and saves to memory

The loop continues until:
  - A conclusive answer is found
  - Max iterations reached
  - An error occurs

Best for: complex, dynamic tasks requiring continuous planning and adaptation.

Architecture:
    User Query
        │
        ▼
    ┌─────────────────────────────────┐
    │  THOUGHT: What should I do?     │◄──── Observation (tool output)
    │  ACTION: Call tool / Answer     │
    │  OBSERVATION: Save result       │
    └────────────────┬────────────────┘
                     │ (loop)
                     ▼
              Final Answer to User

Run: python 5_react_agent.py
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
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools


# ─────────────────────────────────────────────────────────────
# 1. Basic ReAct Agent
#    Agno agents are ReAct by default — the model reasons about
#    which tools to call, observes the result, and loops.
# ─────────────────────────────────────────────────────────────

print("=" * 65)
print("REACT PATTERN — REASON, ACT, OBSERVE LOOP")
print("=" * 65)

react_agent = Agent(
    name="ReAct Research Agent",
    model=OpenAIChat(id="google/gemini-2.5-flash"),
    tools=[
        DuckDuckGoTools(),
        YFinanceTools(
            enable_stock_price=True,
            enable_analyst_recommendations=True,
            enable_company_news=True,
        ),
    ],
    instructions=[
        "You operate in a ReAct loop: Think, Act, Observe, Repeat.",
        "Before answering, explicitly state your reasoning step.",
        "Search for information, observe the results, then refine your answer.",
        "Always verify facts with at least 2 tool calls before concluding.",
        "Present final answer with clear reasoning trail.",
    ],
    markdown=True,
)

print("\n📋 Query: Comprehensive analysis of Tesla's current position")
print("-" * 65)
react_agent.print_response(
    """
    Perform a comprehensive analysis of Tesla (TSLA):
    1. Current stock price and recent trend
    2. Latest news that might affect the stock
    3. Analyst sentiment
    4. Synthesize: is this a good time to research Tesla further?

    Show your reasoning at each step.
    """,
    stream=True,
)


# ─────────────────────────────────────────────────────────────
# 2. ReAct with Custom Tools and Explicit Thought Display
#    Demonstrates the full Thought-Action-Observation cycle
# ─────────────────────────────────────────────────────────────

import json
from datetime import datetime, timedelta

def search_patent_database(company: str, technology_area: str) -> str:
    """Search for recent patents filed by a company in a specific technology area.

    Args:
        company: Company name to search patents for.
        technology_area: Technology domain to search (e.g., 'AI', 'battery', 'robotics').

    Returns:
        JSON string with patent count and sample titles.
    """
    # Simulated patent data
    patent_data = {
        "Tesla": {
            "AI": {"count": 47, "titles": ["Neural network for autopilot", "Vision-based parking"]},
            "battery": {"count": 123, "titles": ["4680 cell chemistry", "Thermal management"]},
        },
        "NVIDIA": {
            "AI": {"count": 312, "titles": ["GPU attention mechanism", "DLSS algorithm"]},
        },
    }

    company_patents = patent_data.get(company, {})
    area_patents = company_patents.get(technology_area, {"count": 0, "titles": []})

    return json.dumps({
        "company": company,
        "technology_area": technology_area,
        "patent_count_last_2_years": area_patents["count"],
        "sample_patents": area_patents["titles"],
        "data_date": datetime.now().strftime("%Y-%m-%d"),
    })

def calculate_pe_ratio(stock_price: float, earnings_per_share: float) -> str:
    """Calculate the P/E ratio and interpret it.

    Args:
        stock_price: Current stock price in USD.
        earnings_per_share: Earnings per share (EPS) in USD.

    Returns:
        P/E ratio with market context interpretation.
    """
    if earnings_per_share <= 0:
        return "Cannot calculate P/E — negative or zero earnings (growth stock)"

    pe = stock_price / earnings_per_share

    if pe < 15:
        interpretation = "potentially undervalued"
    elif pe < 25:
        interpretation = "fairly valued"
    elif pe < 40:
        interpretation = "premium valuation"
    else:
        interpretation = "highly speculative/growth premium"

    return f"P/E Ratio: {pe:.1f}x ({interpretation}). S&P 500 average P/E is ~22x."

def get_competitor_comparison(company: str) -> str:
    """Get a quick competitive landscape summary for a company.

    Args:
        company: Company to analyze.

    Returns:
        Competitor comparison data as a formatted string.
    """
    comps = {
        "Tesla": "Key competitors: BYD (EV market share ~35% global), GM EV, Ford F-150 Lightning. Tesla leads in software/autonomy but faces margin pressure.",
        "NVIDIA": "Key competitors: AMD (GPU), Intel Arc, Google TPU. NVIDIA has ~80% AI chip market share.",
    }
    return comps.get(company, f"Competitive analysis for {company}: data not available in local DB.")


advanced_react_agent = Agent(
    name="Advanced ReAct Analyst",
    model=OpenAIChat(id="google/gemini-2.5-flash"),
    tools=[
        DuckDuckGoTools(),
        YFinanceTools(enable_stock_price=True, enable_company_info=True),
        search_patent_database,
        calculate_pe_ratio,
        get_competitor_comparison,
    ],
    instructions=[
        "You are a thorough investment research analyst.",
        "For each analysis step, state what you're reasoning about before calling a tool.",
        "Always use multiple tools to triangulate your conclusions.",
        "Never give a final answer without checking at least 3 data sources.",
        "Format your final output as a structured research brief.",
    ],

    markdown=True,
)

print("\n\n" + "=" * 65)
print("ADVANCED REACT — MULTI-TOOL RESEARCH LOOP")
print("=" * 65)
print("\n📋 Query: Deep-dive research on NVIDIA")
print("-" * 65)

advanced_react_agent.print_response(
    """
    Conduct a deep-dive research brief on NVIDIA (NVDA):

    Step through your reasoning for each data point you need:
    - Get current stock data and calculate P/E if possible
    - Check their AI patent activity
    - Review competitive position
    - Search for latest news
    - Provide a final structured brief with: Bull Case, Bear Case, and Key Risk
    """,
    stream=True,
)


# ─────────────────────────────────────────────────────────────
# 3. ReAct with Max Iterations Guard
#    Prevents runaway loops — critical for production agents
# ─────────────────────────────────────────────────────────────

print("\n\n" + "=" * 65)
print("REACT WITH ITERATION GUARDRAILS")
print("=" * 65)

guarded_agent = Agent(
    name="Guarded ReAct Agent",
    model=OpenAIChat(id="google/gemini-2.5-flash"),
    tools=[DuckDuckGoTools()],
    instructions=[
        "You have a maximum of 5 reasoning iterations.",
        "After each tool call, explicitly note which iteration you are on (e.g., 'Iteration 2/5').",
        "If you reach iteration 5 without a complete answer, summarize what you know.",
    ],

    markdown=True,
)

guarded_agent.print_response(
    "Trace the history of the ReAct paper: who wrote it, when, and how it influenced modern AI agents.",
    stream=True,
)

print("\n✅ ReAct pattern demo complete.")
print("Key takeaway: ReAct = Thought + Action + Observation, repeated until exit condition.")
