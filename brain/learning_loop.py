import json
import os
from brain.event_bus import bus, EventType
import logging

class LearningLoop:
    """
    Subscribes to errors and mistakes. Does NOT train the model immediately.
    Instead, queues structured experiences for the consolidation process.
    """
    def __init__(self, queue_path="data/learning_queue.jsonl"):
        self.queue_path = queue_path
        self.logger = logging.getLogger("ChaitanyaEventBus")
        
        bus.subscribe(EventType.ERROR_DETECTED, self._handle_error)
        
    def _handle_error(self, event_type, payload):
        """
        Receives an error, formats it as a learning experience, and queues it.
        """
        error_msg = payload.get("error", "Unknown error")
        context = payload.get("context", {})
        
        # Classification (Mocked for now, later done by a small prompt)
        category = "general_error"
        if "math" in error_msg.lower(): category = "arithmetic"
        elif "tool" in error_msg.lower(): category = "tool_use"
        
        experience_record = {
            "type": "correction",
            "category": category,
            "error_msg": error_msg,
            "context_state": context,
            "status": "pending_consolidation"
        }
        
        self._enqueue(experience_record)
        self.logger.info(f"Queued learning experience: {category}")
        
    def _enqueue(self, record):
        os.makedirs(os.path.dirname(self.queue_path), exist_ok=True)
        with open(self.queue_path, "a") as f:
            f.write(json.dumps(record) + "\n")
