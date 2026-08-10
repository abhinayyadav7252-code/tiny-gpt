from enum import Enum, auto
import collections
import logging

class EventType(Enum):
    USER_INPUT = auto()
    PLAN_CREATED = auto()
    MEMORY_RETRIEVED = auto()
    TOOL_CALLED = auto()
    TOOL_RESULT = auto()
    ANSWER_GENERATED = auto()
    ANSWER_VERIFIED = auto()
    ERROR_DETECTED = auto()
    MEMORY_CREATED = auto()
    MEMORY_PROMOTED = auto()
    CHECKPOINT_CREATED = auto()
    CHECKPOINT_REJECTED = auto()

class EventBus:
    """
    Central Nervous System of Chaitanya v2.
    Decouples cognitive subsystems (Executive, Memory, Learning, Tools).
    """
    def __init__(self):
        self._subscribers = collections.defaultdict(list)
        # Configure local logger for the event bus
        self.logger = logging.getLogger("ChaitanyaEventBus")
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - [EventBus] - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
            self.logger.setLevel(logging.INFO)

    def subscribe(self, event_type: EventType, callback):
        """Register a callback for a specific event type."""
        self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: EventType, callback):
        """Remove a callback for a specific event type."""
        if callback in self._subscribers[event_type]:
            self._subscribers[event_type].remove(callback)

    def publish(self, event_type: EventType, payload: dict = None):
        """
        Publish an event to all subscribers.
        Payload should be a dictionary containing event-specific data.
        """
        if payload is None:
            payload = {}
            
        self.logger.info(f"Event Dispatched: {event_type.name} | Payload keys: {list(payload.keys())}")
        
        for callback in self._subscribers.get(event_type, []):
            try:
                callback(event_type, payload)
            except Exception as e:
                self.logger.error(f"Error in subscriber {callback.__name__} for {event_type.name}: {e}")

# Global singleton event bus instance
bus = EventBus()
