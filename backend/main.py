from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime
import uuid
import os

from context_engine import analyze_situation
from ml_service import process_audio_or_simulate

app = FastAPI(title="SoundScene API")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DetectRequest(BaseModel):
    audio_data: str = "" 
    simulate_label: str = ""
    time_of_day: str = "day" # day, night
    location_mode: str = "home" # home, office, public
    force_urgency: str = "" # normal, medium, critical

# In-memory history storage
event_history: List[Dict[str, Any]] = []

user_profile = {"name": "", "location": "home"}

class ProfileRequest(BaseModel):
    name: str
    location: str = "home"

class CustomSoundRequest(BaseModel):
    label: str
    icon: str
    description: str
    category: str = "General"
    audio_b64: str = ""

@app.post("/profile")
async def update_profile(request: ProfileRequest):
    user_profile["name"] = request.name
    user_profile["location"] = request.location
    return {"status": "success"}

@app.get("/profile")
async def get_profile():
    return user_profile

@app.post("/custom_sound")
async def add_custom_sound(request: CustomSoundRequest):
    from context_engine import SOUND_RULES, RULES_FILE
    import json
    import os
    import base64
    
    # Save audio if present
    if request.audio_b64:
        try:
            if "," in request.audio_b64:
                header, encoded = request.audio_b64.split(",", 1)
            else:
                encoded = request.audio_b64
                
            audio_data = base64.b64decode(encoded)
            save_dir = os.path.join(os.path.dirname(__file__), "../dataset/custom")
            os.makedirs(save_dir, exist_ok=True)
            file_path = os.path.join(save_dir, f"{request.label}.webm")
            with open(file_path, "wb") as f:
                f.write(audio_data)
        except Exception as e:
            print(f"Failed to save audio file: {e}")

    SOUND_RULES[request.label] = {
        "icon": request.icon,
        "category": request.category,
        "is_custom": True,
        "levels": {
            "normal": {
                "situation": request.description,
                "action": "Check this custom event",
                "base_confidence": 0.95
            },
            "medium": {
                "situation": f"Repeated: {request.description}",
                "action": "Investigate",
                "base_confidence": 0.97
            },
            "critical": {
                "situation": f"Persistent: {request.description}",
                "action": "Immediate Action",
                "base_confidence": 0.99
            }
        }
    }
    try:
        with open(RULES_FILE, "w", encoding="utf-8") as f:
            json.dump(SOUND_RULES, f, indent=4)
    except Exception:
        pass
        
    return {"status": "success"}

@app.get("/custom_sounds")
async def get_custom_sounds():
    from context_engine import SOUND_RULES
    customs = [k for k, v in SOUND_RULES.items() if v.get("is_custom")]
    return {"sounds": customs}

@app.post("/detect")
async def detect_sound(request: DetectRequest):
    from context_engine import SOUND_RULES
    # 1. Pass through ML Service (or simulation)
    ml_result = process_audio_or_simulate(request.audio_data, request.simulate_label)
    
    if not ml_result:
        return {"status": "no_event"}

    # 2. Context Engine analysis
    situation = analyze_situation(
        label=ml_result["label"],
        confidence=ml_result["confidence"],
        time_of_day=request.time_of_day,
        location_mode=request.location_mode,
        force_urgency=request.force_urgency
    )
    
    # 3. Create event response
    is_custom = SOUND_RULES.get(ml_result["label"], {}).get("is_custom", False)
    event = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "sound": ml_result["label"],
        "confidence": round(ml_result["confidence"], 2),
        "situation": situation["description"],
        "urgency": situation["urgency_level"],
        "action": situation["recommended_action"],
        "icon": situation["icon"],
        "is_custom": is_custom
    }
    
    # 4. Save to history (keep last 5)
    event_history.insert(0, event)
    if len(event_history) > 5:
        event_history.pop()
        
    return event

@app.post("/detect_custom")
async def detect_custom(request: Dict[str, str]):
    from context_engine import SOUND_RULES
    label = request.get("label")
    if not label or label not in SOUND_RULES:
        raise HTTPException(status_code=404, detail="Custom sound not found")
    
    rule = SOUND_RULES[label]
    event = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "sound": label,
        "confidence": 0.95,
        "situation": rule["levels"]["normal"]["situation"],
        "urgency": "medium",
        "action": rule["levels"]["normal"]["action"],
        "icon": rule.get("icon", "🔊"),
        "is_custom": True
    }
    event_history.insert(0, event)
    if len(event_history) > 5:
        event_history.pop()
    return event

@app.get("/history")
async def get_history():
    return {"history": event_history}

@app.post("/reset")
async def reset_app():
    global event_history
    from context_engine import repetition_tracker
    event_history.clear()
    repetition_tracker.clear()
    return {"status": "success"}

# Serve frontend
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "../frontend")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")
