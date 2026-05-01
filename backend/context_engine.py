import json
import os
import numpy as np
from typing import Dict, Any, List, Optional

# Load rules from rules.json
RULES_FILE = os.path.join(os.path.dirname(__file__), "rules.json")
CUSTOM_SOUNDS_FILE = os.path.join(os.path.dirname(__file__), "custom_embeddings.json")

try:
    with open(RULES_FILE, "r", encoding="utf-8") as f:
        SOUND_RULES = json.load(f)
except Exception as e:
    print(f"Warning: Could not load rules.json. {e}")
    SOUND_RULES = {}

# Keep track of repetitions for escalation
repetition_tracker = {}

def analyze_situation(label: str, confidence: float, history: List[str], time_of_day: str = "day", location_mode: str = "home", force_urgency: str = "") -> Dict[str, Any]:
    """
    Analyzes sound events with temporal logic and confidence filtering.
    """
    # 1. Base result
    result = {
        "label": label,
        "description": f"Detected {label.replace('_', ' ')}",
        "urgency_level": "green",
        "recommended_action": "Check surroundings",
        "icon": "🔊",
        "color_code": "#4CAF50"
    }

    # 2. Doorbell Logic (Strategic Escalation)
    if label == "doorbell":
        count_last_5 = history.count("doorbell")
        result["icon"] = "🔔"
        if count_last_5 >= 3:
            result["urgency_level"] = "red"
            result["description"] = "URGENT: Constant Doorbell!"
            result["recommended_action"] = "Immediate Action Required"
            result["color_code"] = "#F44336"
        elif count_last_5 >= 2:
            result["urgency_level"] = "yellow"
            result["description"] = "Repeated Doorbell"
            result["recommended_action"] = "Someone is waiting"
            result["color_code"] = "#FFC107"
        else:
            result["description"] = "Doorbell detected"
            result["recommended_action"] = "Check the door"

    # 3. Siren / Alarm (High Urgency)
    elif label == "siren":
        result["icon"] = "🚨"
        result["urgency_level"] = "red"
        result["description"] = "Emergency Siren Detected"
        result["recommended_action"] = "Clear path / Seek safety"
        result["color_code"] = "#F44336"

    elif label == "alarm":
        result["icon"] = "⏰"
        result["urgency_level"] = "yellow"
        result["description"] = "Alarm Triggered"
        result["recommended_action"] = "Attention required"
        result["color_code"] = "#FFC107"

    # 4. Contextual Overrides (Night mode)
    if time_of_day == "night" and result["urgency_level"] == "green":
        result["urgency_level"] = "yellow"
        result["color_code"] = "#FFC107"

    return result

# --- CUSTOM SOUND SUPPORT ---

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)

def match_custom_sound(embedding: np.ndarray, threshold: float = 0.85) -> Optional[str]:
    if not os.path.exists(CUSTOM_SOUNDS_FILE):
        return None
    try:
        with open(CUSTOM_SOUNDS_FILE, "r") as f:
            custom_db = json.load(f)
        best_match = None
        max_sim = -1
        for name, stored_emb in custom_db.items():
            sim = cosine_similarity(embedding, np.array(stored_emb))
            if sim > max_sim:
                max_sim = sim
                best_match = name
        if max_sim >= threshold:
            return best_match
    except:
        pass
    return None

def save_custom_sound(name: str, embedding: np.ndarray):
    custom_db = {}
    if os.path.exists(CUSTOM_SOUNDS_FILE):
        with open(CUSTOM_SOUNDS_FILE, "r") as f:
            custom_db = json.load(f)
    custom_db[name] = embedding.tolist()
    with open(CUSTOM_SOUNDS_FILE, "w") as f:
        json.dump(custom_db, f)
