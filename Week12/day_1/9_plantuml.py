"""
PlantUML Diagram Generator — Agno port of old_Week12/Day_2/1_agent2agent/7_plantUML.py

Given any Python file, the agent reads it and generates a PlantUML diagram (.puml).
Supports: class diagrams, sequence diagrams, activity diagrams.

Usage:
    python 9_plantuml.py
"""

import os
from dotenv import load_dotenv

load_dotenv()
for var in ['OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_API_BASE']:
    if os.getenv(var):
        print(f"⚠️  Removing conflicting {var}")
        del os.environ[var]

os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_ROUTER_KEY")
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.file import FileTools


# ── Custom tools ─────────────────────────────────────────────────────────────

def read_code_file(file_path: str) -> str:
    """Read a source code file and return its contents for analysis."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        return f"=== {file_path} ===\n{content}"
    except Exception as e:
        return f"Error reading {file_path}: {e}"


def save_plantuml(content: str, filename: str) -> str:
    """Save PlantUML diagram markup to a .puml file."""
    try:
        # Strip markdown code fences if the model wrapped the output
        lines = content.strip().splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines)
        with open(filename, 'w') as f:
            f.write(cleaned)
        return f"✅ PlantUML saved to {filename}"
    except Exception as e:
        return f"Error saving {filename}: {e}"


def list_python_files(directory: str = ".") -> str:
    """List all Python files in the given directory."""
    import glob
    files = glob.glob(f"{directory}/**/*.py", recursive=True)
    return "\n".join(files) if files else "No Python files found."


# ── Agent ────────────────────────────────────────────────────────────────────

plantuml_agent = Agent(
    name="PlantUML Diagram Agent",
    model=OpenAIChat(id="google/gemini-2.5-flash"),
    tools=[
        FileTools(),
        read_code_file,
        save_plantuml,
        list_python_files,
    ],
    instructions=[
        "You are a PlantUML expert and software architect.",
        "When given a code file, read it fully before generating any diagram.",
        "Always wrap PlantUML output between @startuml and @enduml.",
        "For Python scripts: prefer sequence or activity diagrams showing control flow.",
        "For Python classes/modules: use class diagrams showing inheritance and composition.",
        "Include method signatures, relationships, and key attributes.",
        "After generating the diagram, save it to a .puml file using save_plantuml.",
        "Then print the full PlantUML content so the user can preview it.",
    ],

    markdown=True,
)


# ── Demo runs ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("🎨 PlantUML Diagram Agent — Agno")
    print("=" * 65)

    # Example 1: sequence/activity diagram for the ReAct agent script
    print("\n📐 Generating sequence diagram for 5_react_agent.py...")
    plantuml_agent.print_response(
        "Read the file '5_react_agent.py' and create a sequence diagram "
        "showing how the ReAct agent loops through Thought → Action → Observation. "
        "Save it as 'react_agent_sequence.puml'.",
        stream=True,
    )

    print("\n" + "─" * 65)

    # Example 2: class diagram for the structured output script
    print("\n📐 Generating class diagram for 3_structured_output.py...")
    plantuml_agent.print_response(
        "Read the file '3_structured_output.py' and create a class diagram "
        "showing all Pydantic models, their fields, and relationships. "
        "Save it as 'structured_output_classes.puml'.",
        stream=True,
    )

    print("\n" + "─" * 65)

    # Example 3: interactive — diagram any file
    # Uncomment to use interactively:
    # target = input("\nEnter a Python file path to diagram: ")
    # diagram_type = input("Diagram type (sequence / class / activity): ")
    # plantuml_agent.print_response(
    #     f"Read '{target}' and create a {diagram_type} diagram. Save as 'output.puml'.",
    #     stream=True,
    # )
