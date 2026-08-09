import sys
import os

# Add root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.core import AIBrain

print("==================================================")
print("=== Phase 5.4: Dual-System Reasoning Test      ===")
print("==================================================\n")

brain = AIBrain(use_self_model=True, use_confidence=True, use_verification=True)

# Test 1: Simple query (Should trigger System 1)
print("\n--- Test 1: Simple Query (System 1) ---")
q1 = "Hello! Who are you?"
print(f"User: {q1}")
response1 = brain.process_query(q1)
print(f"AI: {response1.encode('ascii', 'replace').decode()}")

# Test 2: Complex Math query (Should route to System 2)
print("\n--- Test 2: Complex Math Query (System 2 Routing) ---")
q2 = "What is 15 * 12 + 8? I need you to calculate this."
print(f"User: {q2}")
response2 = brain.process_query(q2)
print(f"AI: {response2.encode('ascii', 'replace').decode()}")

# Test 3: Tricky Logic query (Should fail System 1 verification, fallback to System 2)
# We will inject a fake state into metacognition to force a failure
print("\n--- Test 3: Metacognitive Fallback (System 1 -> System 2) ---")
q3 = "What is the capital of Mars?"
print(f"User: {q3}")
response3 = brain.process_query(q3)
print(f"AI: {response3.encode('ascii', 'replace').decode()}")

print("\nSystem 2 Dual-System architecture test complete.")
