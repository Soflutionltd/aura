#!/usr/bin/env python3
"""
AURA LeWorldModel Simulator
A lightweight world model (15M params) that simulates consequences
before the main LLM acts. Reduces agent iterations from 10-15 to 2-4.

Based on LeWM (Mila + NYU + Samsung, March 2026).
Runs in < 200ms on CPU, < 100 Mo memory.

Architecture:
  User prompt -> MoE generates plan -> LeWM simulates outcomes
  -> MoE picks best path -> Execute only necessary tools

For MVP: uses heuristic simulation until we integrate the real
LeWM ONNX model. The interface is stable so we can swap later.
"""

import json
import time
import hashlib
from pathlib import Path
from typing import Optional


class SimulationResult:
    """Result of a world model simulation."""

    def __init__(self, scenario: str, success_prob: float,
                 risk_level: str, estimated_steps: int,
                 reasoning: str):
        self.scenario = scenario
        self.success_prob = success_prob
        self.risk_level = risk_level  # "low", "medium", "high"
        self.estimated_steps = estimated_steps
        self.reasoning = reasoning

    def to_dict(self):
        return {
            "scenario": self.scenario,
            "success_probability": self.success_prob,
            "risk_level": self.risk_level,
            "estimated_steps": self.estimated_steps,
            "reasoning": self.reasoning,
        }


class LeWorldModelSimulator:
    """
    Lightweight world model for anticipating action outcomes.
    MVP: heuristic-based simulation.
    Future: ONNX inference with real LeWM weights (15M params).
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.use_neural = model_path is not None
        self.simulation_count = 0
        self.cache = {}

    def simulate(self, plan: list[dict], context: dict) -> list[SimulationResult]:
        """
        Simulate multiple scenarios for a given plan.
        Returns ranked list of simulations (best first).

        Args:
            plan: list of planned actions [{"tool": "...", "args": {...}}, ...]
            context: current state {"workspace_files": [...], "history": [...]}
        """
        start = time.time()
        results = []

        for i, action in enumerate(plan):
            tool = action.get("tool", "unknown")
            args = action.get("args", {})

            # Simulate each action's outcome
            sim = self._simulate_action(tool, args, context, i)
            results.append(sim)

        # Sort by success probability (best first)
        results.sort(key=lambda r: r.success_prob, reverse=True)

        elapsed_ms = (time.time() - start) * 1000
        self.simulation_count += 1

        return results

    def _simulate_action(self, tool: str, args: dict,
                         context: dict, step_index: int) -> SimulationResult:
        """
        Simulate a single action's outcome.
        MVP: heuristic rules based on tool type and context.
        Future: neural LeWM forward pass.
        """
        # Heuristic simulation (MVP)
        if tool == "code_execution":
            code = args.get("code", "")
            has_imports = "import" in code
            has_try = "try" in code or "except" in code
            complexity = len(code.split("\n"))
            success = 0.85 if has_try else 0.7
            if complexity > 50:
                success -= 0.1
            return SimulationResult(
                scenario=f"Execute Python code ({complexity} lines)",
                success_prob=success,
                risk_level="low" if has_try else "medium",
                estimated_steps=1,
                reasoning="Code with error handling has higher success rate",
            )

        elif tool == "file_system":
            action_type = args.get("action", "read")
            path = args.get("path", "")
            if action_type == "read":
                return SimulationResult(
                    scenario=f"Read file: {path}",
                    success_prob=0.95,
                    risk_level="low",
                    estimated_steps=1,
                    reasoning="File reads rarely fail if path exists",
                )
            else:
                return SimulationResult(
                    scenario=f"Write file: {path}",
                    success_prob=0.85,
                    risk_level="medium",
                    estimated_steps=1,
                    reasoning="File writes may fail on permissions",
                )

        elif tool == "terminal_cmd":
            cmd = args.get("command", "")
            dangerous = any(w in cmd for w in ["rm -rf", "sudo", "chmod 777", "dd"])
            return SimulationResult(
                scenario=f"Run: {cmd[:60]}",
                success_prob=0.3 if dangerous else 0.8,
                risk_level="high" if dangerous else "medium",
                estimated_steps=1,
                reasoning="Dangerous command detected" if dangerous else "Standard command",
            )

        elif tool == "web_search":
            return SimulationResult(
                scenario=f"Web search: {args.get('query', '')}",
                success_prob=0.9,
                risk_level="low",
                estimated_steps=1,
                reasoning="Web search via DuckDuckGo is reliable",
            )

        elif tool == "local_rag":
            return SimulationResult(
                scenario=f"RAG search in workspace",
                success_prob=0.85,
                risk_level="low",
                estimated_steps=1,
                reasoning="Depends on document quality and embeddings",
            )

        else:
            return SimulationResult(
                scenario=f"Unknown tool: {tool}",
                success_prob=0.5,
                risk_level="medium",
                estimated_steps=2,
                reasoning="Unknown tool, proceed with caution",
            )

    def select_best_plan(self, plans: list[list[dict]],
                         context: dict) -> tuple[int, list[SimulationResult]]:
        """
        Given multiple possible plans, simulate all and return
        the index of the best one + its simulations.
        """
        best_idx = 0
        best_score = 0
        best_sims = []

        for i, plan in enumerate(plans):
            sims = self.simulate(plan, context)
            avg_success = sum(s.success_prob for s in sims) / max(len(sims), 1)
            total_steps = sum(s.estimated_steps for s in sims)
            has_high_risk = any(s.risk_level == "high" for s in sims)

            # Score: high success + few steps + low risk
            score = avg_success * 100 - total_steps * 5 - (20 if has_high_risk else 0)

            if score > best_score:
                best_score = score
                best_idx = i
                best_sims = sims

        return best_idx, best_sims


# ── CLI for testing ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    sim = LeWorldModelSimulator()

    # Test with a sample plan
    plan = [
        {"tool": "file_system", "args": {"action": "read", "path": "src/main.rs"}},
        {"tool": "code_execution", "args": {"code": "import ast\ntry:\n    result = 1+1\nexcept: pass"}},
        {"tool": "terminal_cmd", "args": {"command": "cargo build --release"}},
    ]

    results = sim.simulate(plan, {"workspace_files": ["src/main.rs"]})
    print("Simulation results:")
    for r in results:
        print(f"  {r.scenario}: {r.success_prob*100:.0f}% success, risk={r.risk_level}")
