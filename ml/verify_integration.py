import requests
import base64
import os
import time

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:8001/detect"
# Find a real audio file to test (Siren from US8K)
TEST_WAV = "UrbanSound8K/audio/fold1/7061-6-0-0.wav" # Fold 1, class 6 (siren)

def test_integration():
    print(f"🔍 Testing E2E Integration with: {TEST_WAV}")
    
    if not os.path.exists(TEST_WAV):
        print(f"❌ Test file not found: {TEST_WAV}")
        return

    # 1. Encode file to Base64
    with open(TEST_WAV, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode('utf-8')
    
    payload = {
        "audio_data": f"data:audio/wav;base64,{audio_b64}",
        "location_mode": "home"
    }

    # 2. Send request to local server
    try:
        print("📡 Sending audio to server...")
        start_time = time.time()
        response = requests.post(API_URL, json=payload)
        latency = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Server Response ({latency:.2f}ms):")
            print(result)
            
            # Verify required fields for frontend
            required_keys = ["description", "urgency_level", "recommended_action", "icon"]
            missing = [k for k in required_keys if k not in result]
            
            if not missing:
                print("\n🔥 INTEGRATION VERIFIED: System classified sound and returned valid Situation Card.")
            else:
                print(f"\n⚠️ WARNING: Response missing frontend keys: {missing}")
                
        else:
            print(f"❌ Server Error {response.status_code}: {response.text}")

    except Exception as e:
        print(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    test_integration()
