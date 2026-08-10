import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class WorkingMemoryState:
    """
    Represents the structured workspace for the Executive Controller.
    """
    goal: str = ""
    plan: List[str] = field(default_factory=list)
    facts: List[str] = field(default_factory=list)
    unknowns: List[str] = field(default_factory=list)
    hypotheses: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    tool_results: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    
    def to_dict(self):
        return {
            "goal": self.goal,
            "plan": self.plan,
            "facts": self.facts,
            "unknowns": self.unknowns,
            "hypotheses": self.hypotheses,
            "evidence": self.evidence,
            "constraints": self.constraints,
            "tool_results": self.tool_results,
            "confidence": self.confidence
        }
        
    def to_prompt_context(self) -> str:
        """
        Formats the current working memory into a prompt for the language model.
        """
        context = "=== WORKING MEMORY ===\n"
        context += f"Goal: {self.goal}\n" if self.goal else ""
        if self.plan:
            context += "Plan:\n" + "\n".join(f"- {step}" for step in self.plan) + "\n"
        if self.facts:
            context += "Known Facts:\n" + "\n".join(f"- {fact}" for fact in self.facts) + "\n"
        if self.unknowns:
            context += "Unknowns:\n" + "\n".join(f"- {u}" for u in self.unknowns) + "\n"
        if self.hypotheses:
            context += "Hypotheses:\n" + "\n".join(f"- {h}" for h in self.hypotheses) + "\n"
        if self.evidence:
            context += "Evidence:\n" + "\n".join(f"- {e}" for e in self.evidence) + "\n"
        if self.constraints:
            context += "Constraints:\n" + "\n".join(f"- {c}" for c in self.constraints) + "\n"
        if self.tool_results:
            context += "Tool Results:\n" + json.dumps(self.tool_results, indent=2) + "\n"
        context += f"Overall Confidence: {self.confidence:.2f}\n"
        context += "======================\n"
        return context

class WorkingMemory:
    """
    Manages the active state of a thought process or interaction.
    """
    def __init__(self):
        self.state = WorkingMemoryState()
        
    def update_goal(self, goal: str):
        self.state.goal = goal
        
    def add_fact(self, fact: str):
        if fact not in self.state.facts:
            self.state.facts.append(fact)
            
    def add_unknown(self, unknown: str):
        if unknown not in self.state.unknowns:
            self.state.unknowns.append(unknown)
            
    def resolve_unknown(self, unknown: str, fact: str):
        """Moves an unknown to a known fact."""
        if unknown in self.state.unknowns:
            self.state.unknowns.remove(unknown)
        self.add_fact(fact)
        
    def add_hypothesis(self, hypothesis: str):
        if hypothesis not in self.state.hypotheses:
            self.state.hypotheses.append(hypothesis)
            
    def add_evidence(self, evidence: str):
        if evidence not in self.state.evidence:
            self.state.evidence.append(evidence)
            
    def add_constraint(self, constraint: str):
        if constraint not in self.state.constraints:
            self.state.constraints.append(constraint)
            
    def set_plan(self, plan: List[str]):
        self.state.plan = plan
        
    def add_tool_result(self, tool_name: str, result: Any):
        self.state.tool_results[tool_name] = result
        
    def set_confidence(self, conf: float):
        self.state.confidence = max(0.0, min(1.0, conf))
        
    def clear(self):
        """Reset the working memory for a new task/interaction."""
        self.state = WorkingMemoryState()
