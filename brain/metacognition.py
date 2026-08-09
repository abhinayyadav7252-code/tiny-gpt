import math

class MetacognitiveEvaluator:
    def __init__(self):
        # M4: Objective Verification Dataset
        self.objective_truths = {
            "17 * 23": "391",
            "capital of France": "Paris",
            "User's name": "Abhinav",
            "learning Python": "Abhinav"
        }

    def calculate_confidence(self, mean_log_prob):
        # M3: Convert mean log probability to a normalized 0-1 confidence score
        # Geometric mean of probabilities = exp(mean_log_prob)
        return math.exp(mean_log_prob)

    def evaluate_output(self, query, generated_response, mean_log_prob):
        confidence = self.calculate_confidence(mean_log_prob)
        
        # M4: Objective Error Detection
        error_detected = False
        error_reason = None
        
        # Check against known objective truths
        for q, expected in self.objective_truths.items():
            if q.lower() in query.lower():
                if expected.lower() not in generated_response.lower():
                    error_detected = True
                    error_reason = f"Objective mismatch: Expected '{expected}'"
                    break

        # Basic format errors (e.g. broken calc blocks)
        if "[CALC:" in generated_response and "]" not in generated_response:
            error_detected = True
            error_reason = "Malformed tool invocation"
            
        return confidence, error_detected, error_reason
        
    def self_correct(self, confidence, error_detected, error_reason):
        # M5: Verification-based Self-Correction Logic
        # Instead of regenerating blindly, we issue a metacognitive fallback
        if error_detected:
            return f"[Self-Correction] I generated a response but caught an error: {error_reason}."
        if confidence < 0.15: # Threshold for our tiny model
            return "[Self-Correction] My confidence is too low. I don't have enough evidence."
        
        return None # No correction needed
