#!/usr/bin/env python3
"""
AURA Expert Specialization Engine
Tracks which MoE experts are best at which tasks and guides
the federated distillation to specialize them further.

Goal: after 2-3 months of community contributions,
certain experts become specialists:
  - Expert 12,45,67: coding experts
  - Expert 3,28,91: reasoning/math experts
  - Expert 7,55,102: creative/writing experts
  - Expert 19,33,88: agent/tool-use experts

The routing then becomes deterministic and ultra-efficient.
Based on Multi-Head LatentMoE paper (Feb 2026).
"""

import json
import datetime
from pathlib import Path
from collections import defaultdict

SPECIALIZATION_DIR = Path(__file__).resolve().parent.parent / "expert-stats"


class ExpertTracker:
    """Track performance of individual MoE experts by task type."""

    TASK_TYPES = ["coding", "reasoning", "math", "creative", "agent", "analysis", "multilingual"]

    def __init__(self):
        SPECIALIZATION_DIR.mkdir(parents=True, exist_ok=True)
        self.stats_path = SPECIALIZATION_DIR / "expert_performance.json"
        self.stats = self._load()

    def _load(self) -> dict:
        if self.stats_path.exists():
            return json.loads(self.stats_path.read_text())
        return {"experts": {}, "task_counts": defaultdict(int), "last_updated": ""}

    def _save(self):
        self.stats["last_updated"] = datetime.datetime.now().isoformat()
        self.stats_path.write_text(json.dumps(self.stats, indent=2, default=str))

    def record_performance(self, task_type: str, expert_ids: list[int],
                           success: bool, quality_score: float):
        """
        Record which experts were active during a task and how well it went.
        Over time, this builds a map of expert specializations.
        """
        if task_type not in self.TASK_TYPES:
            task_type = "reasoning"  # default

        self.stats["task_counts"][task_type] = self.stats["task_counts"].get(task_type, 0) + 1

        for eid in expert_ids:
            key = str(eid)
            if key not in self.stats["experts"]:
                self.stats["experts"][key] = {t: {"count": 0, "success": 0, "avg_quality": 0.0}
                                               for t in self.TASK_TYPES}

            entry = self.stats["experts"][key][task_type]
            entry["count"] += 1
            if success:
                entry["success"] += 1
            # Running average quality
            n = entry["count"]
            entry["avg_quality"] = ((entry["avg_quality"] * (n - 1)) + quality_score) / n

        self._save()

    def get_best_experts(self, task_type: str, top_k: int = 8) -> list[dict]:
        """Get the top-K experts for a given task type."""
        scores = []
        for eid, data in self.stats.get("experts", {}).items():
            if task_type in data:
                entry = data[task_type]
                if entry["count"] > 0:
                    success_rate = entry["success"] / entry["count"]
                    score = success_rate * 0.5 + entry["avg_quality"] * 0.5
                    scores.append({"expert_id": int(eid), "score": round(score, 3),
                                      "count": entry["count"],
                                      "success_rate": round(success_rate, 3)})
        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:top_k]

    def get_specialization_map(self) -> dict:
        """
        Generate a full specialization map: which experts are best at what.
        Used by the federated distillation to target specific experts.
        """
        spec_map = {}
        for task in self.TASK_TYPES:
            best = self.get_best_experts(task, top_k=3)
            if best:
                spec_map[task] = {
                    "primary_experts": [b["expert_id"] for b in best],
                    "avg_score": round(sum(b["score"] for b in best) / len(best), 3),
                    "total_samples": sum(b["count"] for b in best),
                }
        return spec_map

    def suggest_distillation_targets(self) -> list[dict]:
        """
        Suggest which experts need the most improvement via distillation.
        Returns experts with low scores in high-demand task types.
        """
        suggestions = []
        for task in self.TASK_TYPES:
            count = self.stats.get("task_counts", {}).get(task, 0)
            if count < 10:
                continue  # Not enough data
            best = self.get_best_experts(task, top_k=1)
            if best and best[0]["score"] < 0.7:
                suggestions.append({
                    "task_type": task,
                    "current_best_score": best[0]["score"],
                    "samples": count,
                    "recommendation": f"Distill more {task} traces into expert {best[0]['expert_id']}",
                })
        return suggestions


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tracker = ExpertTracker()

    # Simulate some data
    import random
    for _ in range(100):
        task = random.choice(ExpertTracker.TASK_TYPES)
        experts = random.sample(range(128), 8)  # 8 active out of 128
        success = random.random() > 0.3
        quality = random.uniform(0.4, 1.0)
        tracker.record_performance(task, experts, success, quality)

    print("Specialization map:")
    for task, info in tracker.get_specialization_map().items():
        print(f"  {task}: experts {info['primary_experts']} (score: {info['avg_score']})")

    print("\nDistillation suggestions:")
    for s in tracker.suggest_distillation_targets():
        print(f"  {s['recommendation']}")
