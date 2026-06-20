"""
crewai_server.py — CrewAI Agent as an A2A Server
=================================================
Exposes a CrewAI Market Researcher agent via the Agent-to-Agent (A2A) protocol
using fastA2A.

Protocol: A2A (JSON-RPC 2.0) over HTTP
Port    : 9321

Run:
    python crewai_server.py

Requires:
    pip install crewai fasta2a uvicorn
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
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"  # CrewAI uses this key
os.environ["OPENROUTER_API_KEY"] = os.getenv("OPEN_ROUTER_KEY")  # litellm uses this for the "openrouter/" model prefix

from crewai import Agent, Task, Crew
from fasta2a import FastA2A, Worker
from fasta2a.broker import InMemoryBroker
from fasta2a.storage import InMemoryStorage
from fasta2a.schema import Message, TextPart, TaskSendParams, TaskIdParams

Context = list[Message]


class CrewAIWorker(Worker[Context]):
    """Wraps a CrewAI agent as an A2A-compatible worker."""

    def __init__(self, storage, broker):
        super().__init__(storage=storage, broker=broker)
        self.agent = Agent(
            role="Market Researcher",
            goal="Research AI and technology markets and provide concise, factual summaries",
            backstory="You are an expert market researcher with 10 years of experience in AI industry analysis.",
            verbose=False,
            llm="openrouter/google/gemini-2.5-flash",
        )

    async def run_task(self, params: TaskSendParams) -> None:
        task = await self.storage.load_task(params['id'])
        if not task:
            return
        await self.storage.update_task(task['id'], state='working')

        user_message = ""
        for msg in task.get('history', []):
            if msg.get('role') == 'user':
                for part in msg.get('parts', []):
                    if part.get('kind') == 'text':
                        user_message += part.get('text', '')

        try:
            crew_task = Task(
                description=f"Research and summarize: {user_message}",
                agent=self.agent,
                expected_output="A concise, factual 3-5 paragraph market research summary.",
            )
            crew = Crew(agents=[self.agent], tasks=[crew_task], verbose=False)
            result = str(crew.kickoff())

            response = Message(
                role='agent',
                parts=[TextPart(text=f"[CrewAI]\n{result}", kind='text')],
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
worker  = CrewAIWorker(storage=storage, broker=broker)


@asynccontextmanager
async def lifespan(app: FastA2A) -> AsyncIterator[None]:
    async with app.task_manager:
        async with worker.run():
            yield


app = FastA2A(storage=storage, broker=broker, lifespan=lifespan)

if __name__ == "__main__":
    print("🔵 CrewAI Market Researcher on http://localhost:9321")
    print("   Protocol : A2A (JSON-RPC 2.0)")
    print("   Model    : google/gemini-2.5-flash via OpenRouter")
    uvicorn.run(app, host="0.0.0.0", port=9321, log_level="warning")
