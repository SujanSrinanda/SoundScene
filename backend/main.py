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
from ml_service import process_audio

app = FastAPI(title="SoundScene API")

# Task 7: CORS Fix (Critical for Production/Ngrok)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request, call_next):
    # Task 8: Debug Logging (Backend)
    print(f"DEBUG: [API] {request.method} {request.url.path}")
    response = await call_next(request)
    print(f"DEBUG: [API] Response status: {response.status_code}")
    return response

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
    displayName: str
    location: str = "home"

class CustomSoundRequest(BaseModel):
    label: str
    icon: str
    description: str
    category: str = "General"
    audio_b64: str = ""

@app.post("/save-profile")
async def update_profile(request: ProfileRequest):
    print(f"Received profile update: displayName={request.displayName}, location={request.location}")
    try:
        if not request.displayName or not request.displayName.strip():
            return {"success": False, "message": "Display name is required"}
            
        user_profile["name"] = request.displayName.strip()
        user_profile["location"] = request.location
        
        return {
            "success": True, 
            "message": "Profile updated successfully"
        }
    except Exception as e:
        return {
            "success": False, 
            "message": f"Server error: {str(e)}"
        }

@app.get("/profile")
async def get_profile():
    return user_profile

@app.post("/custom_sound")
async def add_custom_sound(request: CustomSoundRequest):
    from context_engine import SOUND_RULES, RULES_FILE, save_custom_sound
    from ml_service import inference_system
    import json
    import os
    import base64
    
    # 1. Decode and Save Audio File
    try:
        encoded = request.audio_b64.split(",")[-1] if "," in request.audio_b64 else request.audio_b64
        audio_bytes = base64.b64decode(encoded)
        
        # 2. Extract Embedding for Fingerprinting (Task: Custom Sound Support)
        features, embedding = inference_system.extract_features(audio_bytes)
        if embedding is not None:
            save_custom_sound(request.label, embedding)
            print(f"DEBUG: [ML] Fingerprinted custom sound: {request.label}")

        # Save metadata to rules.json
        SOUND_RULES[request.label] = {
            "icon": request.icon,
            "category": request.category,
            "is_custom": True,
            "levels": {
                "normal": {"situation": request.description, "action": "Check surroundings"}
            }
        }
        with open(RULES_FILE, "w", encoding="utf-8") as f:
            json.dump(SOUND_RULES, f, indent=4)
            
        return {"success": True, "message": f"Learned sound: {request.label}"}
    except Exception as e:
        print(f"DEBUG: [API] Custom Sound Error: {e}")
        return {"success": False, "message": str(e)}

@app.get("/custom_sounds")
async def get_custom_sounds():
    from context_engine import SOUND_RULES
    customs = [k for k, v in SOUND_RULES.items() if v.get("is_custom")]
    return {"sounds": customs}
@app.post("/detect")
async def detect_sound(request: DetectRequest):
    print(f"DEBUG: [Backend] Received detect request: audio_data length={len(request.audio_data)}, time_of_day={request.time_of_day}, location_mode={request.location_mode}")
    try:
        if request.simulate_label:
            # Simulate ML output with expected keys
            ml_result = analyze_situation(request.simulate_label, 0.99, [request.simulate_label]*5, request.time_of_day, request.location_mode)
            ml_result["confidence"] = 0.99
        else:
            # Task 2 & 8: Process real audio using the global system
            print("DEBUG: [Backend] Processing audio...")
            ml_result = process_audio(request.audio_data)
            print("DEBUG: [Backend] ML result:", ml_result)

        if not ml_result:
            print("DEBUG: [Backend] No ML result, returning analyzing")
            return {"status": "analyzing"}

        # Task 5: Preserve and return valid JSON structure
        event = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            **ml_result
        }
        print("DEBUG: [Backend] Prediction output:", event)
        return event

    except Exception as e:
        print(f"DEBUG: [Backend] Request Error: {e}")
        return {"status": "error", "message": str(e)}


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
