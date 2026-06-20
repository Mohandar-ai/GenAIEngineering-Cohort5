"""
client.py — Agent-to-Agent Communication Demo
==============================================
Demonstrates cross-framework A2A communication:
  • CrewAI  agent (Market Researcher) on port 9321
  • Agno    agent (Financial Analyst) on port 9331

The A2A protocol (JSON-RPC 2.0 over HTTP) lets agents built on completely
different frameworks talk to each other via a common standard.

Run servers first (in separate terminals):
    python crewai_server.py
    python agno_server.py

Then run this client:
    python client.py
"""

import requests
import uuid
import time
import sys


class A2AClient:
    """Generic A2A client that works with any A2A-compatible agent server."""

    def __init__(self, url: str, name: str):
        self.url = url
        self.name = name

    def send(self, text: str) -> str:
        """Send a message and return the task ID."""
        payload = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": text}],
                    "kind": "message",
                    "messageId": str(uuid.uuid4()),
                }
            },
            "id": 1,
        }
        response = requests.post(self.url, json=payload, timeout=30)
        return response.json()['result']['id']

    def get_result(self, task_id: str, timeout: int = 60) -> str:
        """Poll for task completion and return the agent's response."""
        for i in range(timeout):
            payload = {
                "jsonrpc": "2.0",
                "method": "tasks/get",
                "params": {"id": task_id},
                "id": 2,
            }
            result = requests.post(self.url, json=payload, timeout=10).json()['result']
            state = result['status']['state']

            if state == 'completed':
                for msg in reversed(result['history']):
                    if msg['role'] == 'agent':
                        return msg['parts'][0]['text']
                return "No response found"
            elif state == 'failed':
                return "Task failed"

            if i % 5 == 0 and i > 0:
                print(f"  ⏳ [{state}] waiting...")
            time.sleep(1)

        return "Timeout — agent took too long"

    def ask(self, question: str) -> str:
        """Send a question and wait for the answer. Returns the response text."""
        print(f"\n{'='*65}")
        print(f"📤 → {self.name}")
        print(f"   {question}")
        print('─' * 65)
        task_id = self.send(question)
        answer = self.get_result(task_id)
        print(f"\n{answer}")
        print('=' * 65)
        return answer


# ── Preflight checks ──────────────────────────────────────────────────────────

print("🔍 Checking agent servers...")

crewai_ok = agno_ok = False

try:
    requests.get("http://localhost:9321/.well-known/agent.json", timeout=2)
    print("  ✅ CrewAI  (port 9321)")
    crewai_ok = True
except Exception:
    print("  ❌ CrewAI  — start it: python crewai_server.py")

try:
    requests.get("http://localhost:9331/.well-known/agent.json", timeout=2)
    print("  ✅ Agno    (port 9331)")
    agno_ok = True
except Exception:
    print("  ❌ Agno    — start it: python agno_server.py")

if not crewai_ok or not agno_ok:
    print("\n⚠️  Start both servers before running this client.")
    sys.exit(1)

# ── A2A clients ───────────────────────────────────────────────────────────────

crewai = A2AClient("http://localhost:9321", "CrewAI Market Researcher")
agno   = A2AClient("http://localhost:9331", "Agno Financial Analyst")

# ── Demo 1: Independent queries (each agent answers its specialty) ─────────────

print("\n\n" + "█" * 65)
print("  DEMO 1 — Independent Queries")
print("  Each agent answers using its own tools and framework")
print("█" * 65)

crewai.ask("What are the top 3 AI market trends in 2025?")
agno.ask("What is the current stock price and analyst recommendation for NVDA?")

# ── Demo 2: Agent collaboration (CrewAI research → feeds Agno analysis) ────────

print("\n\n" + "█" * 65)
print("  DEMO 2 — Agent Collaboration (Cross-Framework)")
print("  CrewAI does research → Agno turns it into investment advice")
print("█" * 65)

research = crewai.ask("Give a 3-paragraph overview of the AI semiconductor market in 2025")

if research and len(research) > 100:
    agno.ask(
        f"Based on this market research about AI semiconductors:\n\n"
        f"{research[:800]}\n\n"
        f"Which stocks in this sector look most attractive? "
        f"Pull live data for NVDA and AMD to support your analysis."
    )

# ── Demo 3: Reverse delegation (Agno data → CrewAI narrative) ──────────────────

print("\n\n" + "█" * 65)
print("  DEMO 3 — Reverse Delegation")
print("  Agno fetches financial data → CrewAI writes the market narrative")
print("█" * 65)

financial_data = agno.ask("Get the current stock price and 52-week performance for MSFT and GOOGL")

if financial_data and len(financial_data) > 50:
    crewai.ask(
        f"Write a 2-paragraph market commentary based on this financial data:\n\n"
        f"{financial_data[:600]}\n\n"
        f"Focus on what this tells us about Big Tech's AI investment thesis."
    )

print("\n\n✅ All A2A demos completed!")
print("\nKey takeaway: The A2A protocol lets agents from completely different")
print("frameworks (Agno, CrewAI, LangChain, etc.) collaborate as peers —")
print("none of them needs to know how the other is built.\n")
