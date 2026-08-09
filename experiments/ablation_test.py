import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.core import AIBrain

queries = [
    "What is 17 * 23?",
    "What is the capital of France?",
    "Calculate 9 * 9"
]

def run_suite(name, brain):
    print(f"\n{'='*50}\n=== {name} ===\n{'='*50}")
    for q in queries:
        ans = brain.process_query(q)
        print(f"Q: {q}\nA: {ans}")
        if brain.use_self_model:
            history = brain.self_model.state['confidence_history']
            errors = brain.self_model.state['recent_errors']
            if history:
                print(f"[State Log] Conf: {history[-1]['confidence']:.2f} | Errors Tracked: {len(errors)}")
        print("-" * 30)

run_suite("Mode A: Control (No Metacognition)", AIBrain(use_self_model=False))
run_suite("Mode B: Confidence Only (No Verifier)", AIBrain(use_self_model=True, use_confidence=True, use_verification=False))
run_suite("Mode C: Verification Only (No Confidence)", AIBrain(use_self_model=True, use_confidence=False, use_verification=True))
run_suite("Mode D: Full System (Conf + Verifier)", AIBrain(use_self_model=True, use_confidence=True, use_verification=True))
