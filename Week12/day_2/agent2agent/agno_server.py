"""
agno_server.py — Agno Agent as an A2A Server
=============================================
Exposes an Agno Financial Analyst agent via the Agent-to-Agent (A2A) protocol
using fastA2A. Replaces the old phidata_Server.py.

Protocol: A2A (Agent-to-Agent) — JSON-RPC 2.0 over HTTP
Port    : 9331

Run:
    python agno_server.py

Then send requests via client.py or with curl:
    curl -X POST http://localhost:9331 \\
         -H "Content-Type: application/json" \\
         -d '{"jsonrpc":"2.0","method":"message/send","params":{...},"id":1}'
"""

import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import uvicorn

load_dotenv()
for var in ['OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_API_BASE']:
    if os.getenv(var):
        del os.environ[var]

os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_ROUTER_KEY")
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.yfinance import YFinanceTools
from fasta2a import FastA2A, Worker
from fasta2a.broker import InMemoryBroker
from fasta2a.storage import InMemoryStorage
from fasta2a.schema import Message, TextPart, TaskSendParams, TaskIdParams

Context = list[Message]


class AgnoWorker(Worker[Context]):
    """Wraps an Agno agent as an A2A-compatible worker."""

    def __init__(self, storage, broker):
        super().__init__(storage=storage, broker=broker)
        self.agent = Agent(
            name="Financial Analyst",
            model=OpenAIChat(id="google/gemini-2.5-flash"),
            tools=[YFinanceTools(enable_stock_price=True, enable_company_info=True, enable_analyst_recommendations=True)],
            description="Expert financial analyst. Provide data-backed investment insights.",
            instructions=[
                "Always retrieve live stock data before making recommendations.",
                "Be concise — lead with the key number or recommendation.",
                "Include risk warnings for any investment advice.",
            ],
            markdown=False,
        )

    async def run_task(self, params: TaskSendParams) -> None:
        task = await self.storage.load_task(params['id'])
        if not task:
            return
        await self.storage.update_task(task['id'], state='working')

        # Extract user message text from A2A task history
        user_message = ""
        for msg in task.get('history', []):
            if msg.get('role') == 'user':
                for part in msg.get('parts', []):
                    if part.get('kind') == 'text':
                        user_message += part.get('text', '')

        try:
            response_obj = self.agent.run(user_message)
            result = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)

            response = Message(
                role='agent',
                parts=[TextPart(text=f"[Agno]\n{result}", kind='text')],
                kind='message',
                message_id=str(uuid.uuid4()),
            )
            context = await self.storage.load_context(task['context_id']) or []
            context.append(response)
            await self.storage.update_context(task['context_id'], context)
            await self.storage.update_task(task['id'], state='completed', new_messages=[response])

        except Exception as e:
            error_msg = Message(
                role='agent',
                parts=[TextPart(text=f"Error: {e}", kind='text')],
                kind='message',
                message_id=str(uuid.uuid4()),
            )
            await self.storage.update_task(task['id'], state='failed', new_messages=[error_msg])

    async def cancel_task(self, params: TaskIdParams) -> None:
        await self.storage.update_task(params['id'], state='cancelled')

    def build_message_history(self, history):
        return history

    def build_artifacts(self, result):
        return []


# ── FastA2A app ───────────────────────────────────────────────────────────────

storage = InMemoryStorage()
broker  = InMemoryBroker()
worker  = AgnoWorker(storage=storage, broker=broker)


@asynccontextmanager
async def lifespan(app: FastA2A) -> AsyncIterator[None]:
    async with app.task_manager:
        async with worker.run():
            yield


app = FastA2A(storage=storage, broker=broker, lifespan=lifespan)

if __name__ == "__main__":
    print("🟢 Agno Financial Analyst on http://localhost:9331")
    print("   Protocol : A2A (JSON-RPC 2.0)")
    print("   Model    : google/gemini-2.5-flash via OpenRouter")
    print("   Tools    : YFinanceTools (live stock data)")
    uvicorn.run(app, host="0.0.0.0", port=9331, log_level="warning")
