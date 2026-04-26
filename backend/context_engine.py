import json
import os
from typing import Dict, Any

# Load rules from rules.json
RULES_FILE = os.path.join(os.path.dirname(__file__), "rules.json")
try:
    with open(RULES_FILE, "r", encoding="utf-8") as f:
        SOUND_RULES = json.load(f)
except Exception as e:
    print(f"Warning: Could not load rules.json. {e}")
    SOUND_RULES = {}

# Keep track of repetitions for escalation
repetition_tracker = {}

def analyze_situation(label: str, confidence: float, time_of_day: str, location_mode: str, force_urgency: str = "") -> Dict[str, Any]:
    """
    Analyzes the sound event within its current context to determine the appropriate alert level and action.
    """
    # Default fallback
    result = {
        "description": f"Detected {label}",
        "urgency_level": "green",
        "recommended_action": "Be aware of surroundings",
        "icon": "🔊"
    }

    if label not in SOUND_RULES:
        return result

    rule_data = SOUND_RULES[label]
    icon = rule_data.get("icon", "🔊")
    levels = rule_data.get("levels", {})
    
    # 1. Base level is normal
    current_level = force_urgency if force_urgency else "normal"

    if not force_urgency:
        # 2. Track repetitions
        global repetition_tracker
        if label not in repetition_tracker:
            repetition_tracker[label] = 1
        else:
            repetition_tracker[label] += 1
            
        # Escalate based on repetitions
        reps = repetition_tracker[label]
        if reps >= 3:
            current_level = "critical"
            repetition_tracker[label] = 0 # reset after critical alert
        elif reps == 2:
            current_level = "medium"

    # Special contexts (bump up level based on environment)
    # Night time generally bumps normal to medium.
    # But if we are at the office at night, it's highly suspicious, bump to critical immediately!
    if time_of_day == "night":
        if location_mode == "office" and current_level in ["normal", "medium"]:
            current_level = "critical"
        elif current_level == "normal":
            current_level = "medium"
    
    # Final check: if force_urgency was provided, we strictly stick to it (bypass all above)
    if force_urgency:
        current_level = force_urgency

    # Get the specific details for the current level
    level_data = levels.get(current_level, levels.get("normal", {}))

    result["description"] = level_data.get("situation", f"Detected {label}")
    result["urgency_level"] = current_level
    result["recommended_action"] = level_data.get("action", "Acknowledge")
    result["icon"] = icon

    # Confidence scaling
    if confidence < 0.6:
        result["description"] += " (Low confidence)"
        if result["urgency_level"] == "critical":
            result["urgency_level"] = "medium" 
            
    return result
