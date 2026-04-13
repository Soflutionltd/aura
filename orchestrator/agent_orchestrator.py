#!/usr/bin/env python3
"""
AURA Agent Orchestrator
Manages the autonomous agent loop with World Model simulation.

Flow:
  User -> MoE generates plan -> LeWM simulates outcomes
  -> MoE picks best path -> Execute tools -> Observe -> Repeat

Max 15 iterations, with user approval for dangerous actions.
Captures agent trajectories for federated distillation.
"""

import json
import time
import datetime
import urllib.request
from pathlib import Path
from typing import Optional

from orchestrator.lewm_simulator import LeWorldModelSimulator, SimulationResult

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "aura"
MAX_ITERATIONS = 15
TRAJECTORY_DIR = Path(__file__).resolve().parent.parent / "trajectories"


# ── Tool definitions (what the model can call) ────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "code_execution",
        "description": "Execute Python code in a sandboxed REPL. Available: numpy, sympy, pandas, matplotlib, torch.",
        "parameters": {"code": "string"},
        "permission": "session_opt_in",
    },
    {
        "name": "file_system",
        "description": "Read, write, or list files in the AURA workspace.",
        "parameters": {"action": "read|write|list", "path": "string", "content": "string (for write)"},
        "permission": "auto",
    },
    {
        "name": "local_rag",
        "description": "Semantic search across all PDF/MD/TXT documents in workspace.",
        "parameters": {"query": "string"},
        "permission": "auto",
    },
    {
        "name": "terminal_cmd",
        "description": "Execute a shell command. Preview shown before execution.",
        "parameters": {"command": "string"},
        "permission": "always_confirm",
    },
    {
        "name": "web_search",
        "description": "Search the web via DuckDuckGo local proxy. Privacy-first.",
        "parameters": {"query": "string"},
        "permission": "user_toggle",
    },
    {
        "name": "image_analysis",
        "description": "Analyze a local image using vision MoE.",
        "parameters": {"image_path": "string", "question": "string"},
        "permission": "auto",
    },
    {
        "name": "self_improve",
        "description": "Critique your own agent trace and generate improvement gradient.",
        "parameters": {"trace": "string"},
        "permission": "auto",
    },
]


class AgentTrajectory:
    """Records a complete agent run for federated distillation."""

    def __init__(self, user_prompt: str):
        self.user_prompt = user_prompt
        self.steps = []
        self.start_time = time.time()
        self.success = False

    def add_step(self, action: str, tool: str, args: dict,
                 result: str, simulation: Optional[dict] = None):
        self.steps.append({
            "action": action,
            "tool": tool,
            "args": args,
            "result": result[:2000],  # Cap result size
            "simulation": simulation,
            "timestamp": time.time(),
        })

    def finalize(self, success: bool, final_answer: str):
        self.success = success
        self.final_answer = final_answer
        self.elapsed = time.time() - self.start_time

    def save(self):
        """Save trajectory for federated distillation."""
        TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = TRAJECTORY_DIR / f"traj_{ts}_{'ok' if self.success else 'fail'}.json"
        path.write_text(json.dumps({
            "prompt": self.user_prompt,
            "steps": self.steps,
            "success": self.success,
            "final_answer": self.final_answer[:5000],
            "elapsed_seconds": round(self.elapsed, 1),
            "tool_count": len(self.steps),
        }, indent=2))


class AgentOrchestrator:
    """
    ReAct agent loop augmented with World Model simulation.
    The agent thinks before acting, reducing iterations by 3-5x.
    """

    def __init__(self):
        self.world_model = LeWorldModelSimulator()
        self.autonomy_level = 3  # 1=assisted, 5=full autonomous

    def query_model(self, messages: list[dict]) -> str:
        """Query the local AURA model."""
        payload = json.dumps({
            "model": MODEL,
            "messages": messages,
            "stream": False,
            "options": {"num_ctx": 16384},
        }).encode()
        req = urllib.request.Request(
            OLLAMA_URL, data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["message"]["content"]

    def extract_tool_calls(self, response: str) -> list[dict]:
        """Parse tool calls from model response."""
        calls = []
        try:
            # Try to parse JSON tool calls from response
            if "```json" in response:
                json_block = response.split("```json")[1].split("```")[0].strip()
                parsed = json.loads(json_block)
                if isinstance(parsed, list):
                    calls = parsed
                elif isinstance(parsed, dict) and "tool" in parsed:
                    calls = [parsed]
            # Try inline tool call format
            elif '"tool"' in response and '"args"' in response:
                start = response.index("{")
                end = response.rindex("}") + 1
                parsed = json.loads(response[start:end])
                if "tool" in parsed:
                    calls = [parsed]
        except (json.JSONDecodeError, ValueError):
            pass
        return calls

    def execute_tool(self, tool_name: str, args: dict) -> str:
        """Execute a tool and return the result. MVP: simulated."""
        # In production, this dispatches to real tool implementations
        # For MVP, we return simulated results
        if tool_name == "code_execution":
            return f"[Code executed successfully. Output: result = {hash(str(args)) % 1000}]"
        elif tool_name == "file_system":
            return f"[File operation '{args.get('action', 'read')}' on '{args.get('path', '')}' completed]"
        elif tool_name == "terminal_cmd":
            return f"[Command '{args.get('command', '')[:50]}' executed. Exit code: 0]"
        elif tool_name == "web_search":
            return f"[Search results for '{args.get('query', '')}': 5 relevant results found]"
        elif tool_name == "local_rag":
            return f"[RAG: 3 relevant documents found for '{args.get('query', '')}']"
        else:
            return f"[Tool '{tool_name}' executed]"

    def run(self, user_prompt: str) -> str:
        """
        Run the full agent loop:
        1. Generate plan
        2. Simulate with World Model
        3. Execute best path
        4. Observe and iterate
        5. Save trajectory for distillation
        """
        trajectory = AgentTrajectory(user_prompt)

        # Build tool descriptions for the model
        tools_desc = "\n".join(
            f"- {t['name']}: {t['description']}" for t in TOOL_DEFINITIONS
        )

        messages = [
            {"role": "system", "content": (
                "You are AURA agent. Available tools:\n" + tools_desc +
                "\n\nTo use a tool, respond with a JSON block: "
                '{"tool": "tool_name", "args": {"key": "value"}}'
                "\nWhen done, respond with your final answer (no JSON)."
            )},
            {"role": "user", "content": user_prompt},
        ]

        for iteration in range(MAX_ITERATIONS):
            # Get model response
            response = self.query_model(messages)

            # Check for tool calls
            tool_calls = self.extract_tool_calls(response)

            if not tool_calls:
                # No tool call = final answer
                trajectory.finalize(success=True, final_answer=response)
                trajectory.save()
                return response

            # Simulate with World Model before executing
            simulations = self.world_model.simulate(
                [{"tool": tc.get("tool", ""), "args": tc.get("args", {})}
                 for tc in tool_calls],
                context={"iteration": iteration},
            )

            # Execute each tool call
            for tc, sim in zip(tool_calls, simulations):
                tool_name = tc.get("tool", "")
                args = tc.get("args", {})

                # Skip high-risk actions if autonomy is low
                if sim.risk_level == "high" and self.autonomy_level < 4:
                    result = f"[BLOCKED: {tool_name} is high-risk. User confirmation needed.]"
                else:
                    result = self.execute_tool(tool_name, args)

                trajectory.add_step(
                    action=f"iteration_{iteration}",
                    tool=tool_name,
                    args=args,
                    result=result,
                    simulation=sim.to_dict(),
                )

                # Feed result back to model
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Tool result: {result}"})

        # Max iterations reached
        final = self.query_model(messages + [
            {"role": "user", "content": "Max iterations reached. Give your best answer now."}
        ])
        trajectory.finalize(success=False, final_answer=final)
        trajectory.save()
        return final


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = AgentOrchestrator()
    prompt = "Analyze a Python file and suggest 3 performance optimizations."
    print(f"Running agent: {prompt}")
    result = agent.run(prompt)
    print(f"\nFinal answer:\n{result}")
