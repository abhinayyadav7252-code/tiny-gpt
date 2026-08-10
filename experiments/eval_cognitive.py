import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.core import ChaitanyaBrain
from brain.event_bus import bus, EventType
import logging
import time

def run_cognitive_benchmark():
    print("==================================================")
    print("=== PHASE 8.5: COGNITIVE REGRESSION BENCHMARK ===")
    print("==================================================\n")
    
    # Track metrics
    metrics = {
        "pipeline_successes": 0,
        "pipeline_total": 0,
        "model_accuracy": "N/A (Requires human-in-loop eval for model outputs)"
    }
    
    # 1. Initialize Brain
    print("Initializing ChaitanyaBrain...")
    brain = ChaitanyaBrain(load_model=False) # Running in Mock Mode for Architecture Test
    
    # ----------------------------------------------------
    # TEST 1: Semantic Memory Pipeline
    # ----------------------------------------------------
    print("\n--- TEST 1: Semantic Memory Pipeline ---")
    metrics["pipeline_total"] += 1
    
    # Inject a fact directly into semantic memory to bypass the verification gate
    print("Injecting fact: 'Project X uses the Blue Protocol'")
    brain.memory_manager.semantic.add_fact("Project X uses the Blue Protocol", source="test_injection")
    
    # Track event
    memory_retrieved_correctly = False
    def track_memory(event_type, payload):
        nonlocal memory_retrieved_correctly
        memories = payload.get("memories", {})
        sem_mems = memories.get("semantic", [])
        if any("Blue Protocol" in mem['content'] for mem in sem_mems):
            memory_retrieved_correctly = True

    bus.subscribe(EventType.MEMORY_RETRIEVED, track_memory)
    
    answer = brain.process("What protocol does Project X use?")
    
    if memory_retrieved_correctly:
        print("[PASS] Semantic Memory successfully retrieved and routed to Executive.")
        metrics["pipeline_successes"] += 1
    else:
        print("[FAIL] Semantic Memory failed to route.")
        
    bus.unsubscribe(EventType.MEMORY_RETRIEVED, track_memory)

    # ----------------------------------------------------
    # TEST 2: Episodic Memory Pipeline
    # ----------------------------------------------------
    print("\n--- TEST 2: Episodic Memory Pipeline ---")
    metrics["pipeline_total"] += 1
    
    brain.process("My favorite color is neon green.")
    
    episodic_retrieved = False
    def track_episodic(event_type, payload):
        nonlocal episodic_retrieved
        memories = payload.get("memories", {})
        ep_mems = memories.get("episodic", [])
        if any("neon green" in mem['content'] for mem in ep_mems):
            episodic_retrieved = True
            
    bus.subscribe(EventType.MEMORY_RETRIEVED, track_episodic)
    brain.process("What did I say my favorite color was?")
    
    if episodic_retrieved:
        print("[PASS] Episodic Memory successfully recorded and retrieved.")
        metrics["pipeline_successes"] += 1
    else:
        print("[FAIL] Episodic Memory failed to route.")
        
    bus.unsubscribe(EventType.MEMORY_RETRIEVED, track_episodic)

    # ----------------------------------------------------
    # TEST 3: Procedural Skill Pipeline (Tool Call)
    # ----------------------------------------------------
    print("\n--- TEST 3: Procedural Skill Pipeline ---")
    metrics["pipeline_total"] += 1
    
    tool_called = False
    def track_tool(event_type, payload):
        nonlocal tool_called
        if payload.get("tool_name") == "calculator":
            tool_called = True
            
    bus.subscribe(EventType.TOOL_CALLED, track_tool)
    
    # We hack the working memory hypotheses in the event handler to force the rule-based policy
    def force_math_hypothesis(event_type, payload):
        brain.executive.wm.add_hypothesis("Needs math evaluation")
        
    bus.subscribe(EventType.PLAN_CREATED, force_math_hypothesis)
    
    brain.process("What is 15 * 24?")
    
    if tool_called:
        print("[PASS] Executive Controller correctly triggered a tool call.")
        metrics["pipeline_successes"] += 1
    else:
        print("[FAIL] Tool call failed to trigger.")
        
    bus.unsubscribe(EventType.TOOL_CALLED, track_tool)
    bus.unsubscribe(EventType.PLAN_CREATED, force_math_hypothesis)

    # ----------------------------------------------------
    # TEST 4: Continuous Learning Pipeline
    # ----------------------------------------------------
    print("\n--- TEST 4: Continuous Learning Pipeline ---")
    metrics["pipeline_total"] += 1
    
    checkpoint_created = False
    def track_checkpoint(event_type, payload):
        nonlocal checkpoint_created
        if "candidate" in payload.get("checkpoint_path", ""):
            checkpoint_created = True
            
    bus.subscribe(EventType.CHECKPOINT_CREATED, track_checkpoint)
    
    # Trigger an error
    brain.report_error("Model hallucinated a fake math formula.", {"query": "calculate area of circle"})
    
    # Manually trigger consolidation (usually runs async in background)
    print("Triggering background consolidation...")
    brain.consolidator.build_replay_dataset()
    
    if checkpoint_created:
        print("[PASS] Error queued, dataset built, and Candidate Checkpoint dispatched.")
        metrics["pipeline_successes"] += 1
    else:
        print("[FAIL] Learning pipeline failed to generate candidate.")
        
    bus.unsubscribe(EventType.CHECKPOINT_CREATED, track_checkpoint)
    
    # ----------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------
    print("\n==================================================")
    print("COGNITIVE BENCHMARK RESULTS")
    print(f"Pipeline Success Rate: {metrics['pipeline_successes']} / {metrics['pipeline_total']}")
    print(f"Model Capability Accuracy: {metrics['model_accuracy']}")
    print("==================================================")
    
    brain.shutdown()

if __name__ == "__main__":
    run_cognitive_benchmark()
