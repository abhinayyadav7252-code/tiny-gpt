from brain.event_bus import bus, EventType
import logging

class RegressionGate:
    """
    Human-in-the-Loop gatekeeper.
    Prevents unverified models from deploying to production.
    Evaluates candidates across multiple dimensions (Math, Factual, Hallucination).
    """
    def __init__(self):
        self.logger = logging.getLogger("ChaitanyaEventBus")
        bus.subscribe(EventType.CHECKPOINT_CREATED, self._handle_new_candidate)
        
    def _handle_new_candidate(self, event_type, payload):
        checkpoint_path = payload.get("checkpoint_path")
        self.logger.info(f"RegressionGate: New candidate checkpoint detected at {checkpoint_path}")
        
        # 1. Run automated evaluations
        eval_results = self._run_automated_evals(checkpoint_path)
        
        # 2. Pause and wait for Human Approval
        self.logger.warning("=== HUMAN APPROVAL REQUIRED ===")
        self.logger.warning(f"Candidate: {checkpoint_path}")
        self.logger.warning(f"Eval Results: {eval_results}")
        self.logger.warning("Please review the metrics. Autonomous deployment is DISABLED.")
        
        # In a real system, this would block or put the candidate in a 'pending_review' state.
        
    def _run_automated_evals(self, checkpoint_path):
        # Mocks running eval_suite.py
        return {
            "math_accuracy": 0.95,
            "factual_accuracy": 0.88,
            "tool_use_accuracy": 0.92,
            "hallucination_rate": 0.05,
            "abstention_rate": 0.12
        }
        
    def approve_candidate(self, checkpoint_path):
        """Called by a human to approve deployment."""
        self.logger.info(f"RegressionGate: Checkpoint {checkpoint_path} APPROVED. Deploying...")
        # Move candidate to production path
        
    def reject_candidate(self, checkpoint_path, reason: str):
        """Called by a human to reject deployment."""
        self.logger.info(f"RegressionGate: Checkpoint {checkpoint_path} REJECTED. Reason: {reason}")
        bus.publish(EventType.CHECKPOINT_REJECTED, {"checkpoint_path": checkpoint_path, "reason": reason})
