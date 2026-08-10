import json
import os

class SelfModel2:
    """
    Evidence-based metacognition.
    Does not trust its own claims; updates confidence based on benchmark evaluation and regression tests.
    """
    def __init__(self, state_path="data/self_state.json"):
        self.state_path = state_path
        self.state = {
            "capabilities": {
                "math": 0.0,
                "factual": 0.0,
                "tool_use": 0.0,
                "hinglish": 0.0
            },
            "weaknesses": [],
            "recent_errors": [],
            "learned_skills": [],
            "goals": [],
            "current_state": "idle"
        }
        self.load()

    def load(self):
        if os.path.exists(self.state_path):
            with open(self.state_path, 'r') as f:
                self.state.update(json.load(f))

    def save(self):
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path, 'w') as f:
            json.dump(self.state, f, indent=4)

    def update_capability_from_benchmark(self, domain: str, new_score: float):
        """
        Updates self-confidence based on hard empirical evidence.
        """
        if domain in self.state["capabilities"]:
            old_score = self.state["capabilities"][domain]
            # EMA style update
            self.state["capabilities"][domain] = 0.7 * old_score + 0.3 * new_score
            self.save()

    def log_error(self, error_description: str):
        self.state["recent_errors"].append(error_description)
        if len(self.state["recent_errors"]) > 50:
            self.state["recent_errors"].pop(0)
        self.save()

    def get_confidence(self, domain: str) -> float:
        return self.state["capabilities"].get(domain, 0.0)
