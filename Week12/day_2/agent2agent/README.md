# Agent-to-Agent (A2A) Communication Demo

Cross-framework agent communication using the **A2A protocol** (JSON-RPC 2.0 over HTTP).

## What is A2A?

A2A lets agents built on *different frameworks* talk to each other as peers via a common standard — no framework lock-in.

```
CrewAI Agent  ──A2A──▶  Client  ──A2A──▶  Agno Agent
(port 9321)                                (port 9331)
```

## Files

| File | Purpose |
|------|---------|
| `agno_server.py` | Agno Financial Analyst agent as A2A server (port 9331) |
| `crewai_server.py` | CrewAI Market Researcher agent as A2A server (port 9321) |
| `client.py` | A2A client — 3 demos: independent, collaboration, reverse delegation |
| `service_patterns_illustration.html` | Visual guide: REST → MCP → A2A evolution |

## Quick Start

```bash
# Install
pip install agno openai crewai fasta2a uvicorn python-dotenv yfinance

# Terminal 1 — start Agno server
python agno_server.py

# Terminal 2 — start CrewAI server
python crewai_server.py

# Terminal 3 — run the demo
python client.py
```

## Demo Scenarios

1. **Independent queries** — each agent answers in its own specialty
2. **Collaboration** — CrewAI researches → Agno turns research into investment advice
3. **Reverse delegation** — Agno fetches live data → CrewAI writes the market narrative

## Install extra dep

```bash
pip install fasta2a
```
