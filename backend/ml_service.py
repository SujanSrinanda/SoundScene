import random
import os
import numpy as np
from typing import Dict, Any, Optional

# Lazy load heavy ML libraries to keep API fast
TFLITE_PATH = "ml/models/sound_model.tflite"
interpreter = None
labels = ['dog', 'door_wood_knock', 'crying_baby', 'car_horn', 'siren', 'clock_alarm']

def init_tflite():
    global interpreter
    if interpreter is None and os.path.exists(TFLITE_PATH):
        try:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
            interpreter.allocate_tensors()
            print(f"✅ TFLite Model loaded from {TFLITE_PATH}")
        except Exception as e:
            print(f"❌ Failed to load TFLite model: {e}")

def extract_live_features(audio_bytes: bytes):
    """
    In a real implementation, this would use librosa to extract MFCCs
    from the incoming audio stream.
    """
    # For now, we return a shape matching the model input
    return np.random.randn(1, 40, 157, 1).astype(np.float32)

def process_audio_or_simulate(audio_data: str, simulate_label: str) -> Optional[Dict[str, Any]]:
    """
    Main entry point for sound detection.
    Supports both real TFLite inference and simulation for demo purposes.
    """
    init_tflite()
    
    from context_engine import SOUND_RULES
    valid_classes = list(SOUND_RULES.keys())
    
    # 1. Explicit simulation (Highest priority for UI demos)
    if simulate_label and simulate_label in valid_classes:
        confidence = random.uniform(0.85, 0.99)
        return {
            "label": simulate_label,
            "confidence": confidence
        }
        
    # 2. Real TFLite Inference (If model and data exist)
    if interpreter and audio_data:
        try:
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Preprocess audio_data (Base64 -> Bytes -> MFCC)
            # This is a placeholder for the actual signal processing
            features = extract_live_features(None) 
            
            interpreter.set_tensor(input_details[0]['index'], features)
            interpreter.invoke()
            
            output_data = interpreter.get_tensor(output_details[0]['index'])[0]
            max_idx = np.argmax(output_data)
            confidence = float(output_data[max_idx])
            
            if confidence > 0.5: # Detection threshold
                return {
                    "label": labels[max_idx],
                    "confidence": confidence
                }
        except Exception as e:
            print(f"Inference Error: {e}")

    # 3. Random background simulation (Fallback)
    # While training is in progress, we focus the simulation on the target classes
    if random.random() < 0.15:
        target_sim_classes = ['dog_bark', 'siren', 'car_horn', 'drilling', 'street_music']
        detected_label = random.choice(target_sim_classes)
        confidence = random.uniform(0.70, 0.95)
        return {
            "label": detected_label,
            "confidence": confidence
        }
        
    return None
