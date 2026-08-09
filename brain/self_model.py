import json

class SelfModel:
    def __init__(self):
        self.state = {
            "current_goal": None,
            "known": [],
            "unknown": [],
            "confidence_history": [],
            "recent_errors": []
        }
    
    def update_goal(self, query):
        self.state["current_goal"] = query
        
    def add_known(self, fact):
        if fact not in self.state["known"]:
            self.state["known"].append(fact)
            
    def add_unknown(self, query):
        if query not in self.state["unknown"]:
            self.state["unknown"].append(query)
            
    def record_confidence(self, confidence, correct=None):
        self.state["confidence_history"].append({
            "confidence": confidence,
            "correct": correct
        })
        
    def record_error(self, error_type):
        self.state["recent_errors"].append(error_type)
        if len(self.state["recent_errors"]) > 5:
            self.state["recent_errors"].pop(0)
            
    def display_state(self):
        print("\n--- [SELF STATE] ---")
        print(f"Goal: {self.state['current_goal']}")
        if self.state['recent_errors']:
            print(f"Recent Errors: {self.state['recent_errors'][-1]}")
        if self.state['confidence_history']:
            last_conf = self.state['confidence_history'][-1]
            print(f"Last Confidence: {last_conf['confidence']:.2f}")
        print("--------------------")
