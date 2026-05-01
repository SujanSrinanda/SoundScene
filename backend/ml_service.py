import os
import base64
import io
import numpy as np
import torch
import librosa
from typing import List, Dict, Any, Optional
from collections import Counter
from context_engine import analyze_situation, match_custom_sound

# --- CONFIGURATION ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ml", "models", "robust_model.pt")
PROTOTYPES_PATH = os.path.join(os.path.dirname(__file__), "..", "ml", "models", "class_prototypes.npz")
SAMPLE_RATE = 16000
DURATION = 1.0
CONFIDENCE_THRESHOLD = 0.60
NOISE_FLOOR_WINDOW = 20
NOISE_FLOOR_MULTIPLIER = 2.5
SPIKE_MULTIPLIER = 3.0
MIN_NOISE_THRESHOLD = 0.0003
SUSPICIOUS_CONFIDENCE_THRESHOLD = 0.90
WINDOW_SIZE = 5
TEMPORAL_WINDOW = 3
MIN_STABLE_COUNT = 1
HYBRID_HIGH_THRESHOLD = 0.80
HYBRID_MEDIUM_THRESHOLD = 0.50
FUSION_WEIGHT_MODEL = 0.6
FUSION_WEIGHT_COSINE = 0.4
CLASS_THRESHOLDS = {"siren": 0.70, "dog_bark": 0.50, "doorbell": 0.65, "knock": 0.60, "alarm": 0.65}

class AudioInferenceSystem:
    def __init__(self):
        self.model = None
        self.prediction_buffer = []
        self.rms_history = []
        self.labels = ["alarm", "dog_bark", "doorbell", "knock", "siren"]
        self.load_model()
        self.load_similarity_memory()

    def load_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                self.model = torch.jit.load(MODEL_PATH)
                self.model.eval()
                print(f"DEBUG: [ML] PyTorch Model Loaded: {MODEL_PATH}")
            except Exception as e:
                print(f"DEBUG: [ML] ERROR loading model: {e}")

    def load_similarity_memory(self):
        self.class_prototypes = {}
        self.class_vectors = {}

        if not os.path.exists(PROTOTYPES_PATH):
            print(f"DEBUG: [ML] Prototype memory missing: {PROTOTYPES_PATH}")
            return

        try:
            data = np.load(PROTOTYPES_PATH, allow_pickle=True)
            class_names = data["class_names"].tolist()
            prototypes = data["class_prototypes"]
            vectors = data["class_vectors"]

            for idx, name in enumerate(class_names):
                self.class_prototypes[name] = self.normalize_vector(prototypes[idx])
                self.class_vectors[name] = np.array(vectors[idx], dtype=np.float32)

            print(f"DEBUG: [ML] Loaded similarity memory for classes: {class_names}")
        except Exception as e:
            print(f"DEBUG: [ML] Error loading similarity memory: {e}")

    def normalize_vector(self, vector):
        vec = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vec) + 1e-9
        return vec / norm

    def get_adaptive_threshold(self):
        if not self.rms_history:
            return MIN_NOISE_THRESHOLD
        noise_floor = float(np.mean(self.rms_history))
        return max(noise_floor * NOISE_FLOOR_MULTIPLIER, MIN_NOISE_THRESHOLD)

    def record_rms(self, rms: float):
        self.rms_history.append(rms)
        if len(self.rms_history) > NOISE_FLOOR_WINDOW:
            self.rms_history.pop(0)

    def build_no_sound_result(self, reason: str = "No significant sound detected") -> Dict[str, Any]:
        return {
            "sound": "no_sound",
            "label": "no_sound",
            "description": reason,
            "urgency_level": "green",
            "recommended_action": "Standby",
            "icon": "🔇",
            "confidence": 0.0
        }

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))

    def find_best_similarity(self, feature_vec: np.ndarray):
        if not self.class_prototypes:
            return None, 0.0

        best_label = None
        best_score = -1.0
        for label, prototype in self.class_prototypes.items():
            sim = self.cosine_similarity(feature_vec, prototype)
            if sim > best_score:
                best_score = sim
                best_label = label

        return best_label, best_score

    def find_max_similarity(self, feature_vec: np.ndarray):
        if not self.class_vectors:
            return None, 0.0

        best_label = None
        best_score = -1.0
        for label, vectors in self.class_vectors.items():
            if vectors.size == 0:
                continue
            scores = vectors.dot(feature_vec)
            local_best = float(np.max(scores))
            if local_best > best_score:
                best_score = local_best
                best_label = label

        return best_label, best_score

    def hybrid_decision(self, feature_vec: np.ndarray, model_outputs=None):
        cosine_label, cosine_score = self.find_max_similarity(feature_vec)
        if cosine_label is None:
            cosine_label, cosine_score = self.find_best_similarity(feature_vec)

        model_label = None
        model_conf = 0.0

        if model_outputs is not None:
            probs = torch.softmax(model_outputs, dim=1)[0]
            confidence, idx = torch.max(probs, dim=0)
            model_label = self.labels[idx.item()]
            model_conf = float(confidence.item())

        # CASE 1: strong cosine match
        if cosine_score >= HYBRID_HIGH_THRESHOLD:
            print(f"DEBUG: [ML] Fast cosine path: {cosine_label} ({cosine_score:.4f})")
            return cosine_label, max(cosine_score, 0.65)

        # CASE 2: medium similarity, combine with model
        if cosine_score >= HYBRID_MEDIUM_THRESHOLD and model_outputs is not None:
            fused_score = (FUSION_WEIGHT_MODEL * model_conf) + (FUSION_WEIGHT_COSINE * cosine_score)
            chosen_label = model_label if model_label == cosine_label else (model_label if model_conf >= cosine_score else cosine_label)
            print(f"DEBUG: [ML] Hybrid fusion: cosine={cosine_label}({cosine_score:.4f}), model={model_label}({model_conf:.4f}), fused={fused_score:.4f}")
            return chosen_label, fused_score

        # CASE 3: fallback to model only
        if model_outputs is not None:
            print(f"DEBUG: [ML] Model-only path: {model_label} ({model_conf:.4f})")
            return model_label, model_conf

        return cosine_label, cosine_score

    def smooth_prediction(self, label: str):
        self.prediction_buffer.append(label)
        if len(self.prediction_buffer) > TEMPORAL_WINDOW:
            self.prediction_buffer.pop(0)

        counts = Counter(self.prediction_buffer)
        stable_label, count = counts.most_common(1)[0]
        if len(self.prediction_buffer) == TEMPORAL_WINDOW and count < 2:
            print(f"DEBUG: [ML] Temporal validation failed; suppressing unstable label: {label}")
            return "no_sound"

        if stable_label != label and count >= 2:
            print(f"DEBUG: [ML] Temporal smoothing selected stable label: {stable_label} (count={count})")
            return stable_label
        return label

    def extract_features(self, audio_bytes):
        try:
            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE, duration=DURATION)
            rms = float(np.sqrt(np.mean(y**2)))
            print(f"DEBUG: [ML] Loaded audio: shape={y.shape}, sr={sr}, rms={rms:.6f}")
            y = librosa.util.fix_length(y, size=int(SAMPLE_RATE * DURATION))
            
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=1024, hop_length=256)
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)
            log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)
            
            features = torch.tensor(log_mel).unsqueeze(0).unsqueeze(0).float()
            feature_vec = self.normalize_vector(log_mel.flatten())
            return features, feature_vec, rms
        except Exception as e:
            print(f"DEBUG: [ML] Feature Extraction Error: {e}")
            return None, None, 0.0

    def run_inference(self, audio_b64: str) -> Optional[Dict[str, Any]]:
        import time
        start_time = time.time()
        print(f"DEBUG: [ML] Starting inference, audio_b64 length: {len(audio_b64)}")
        if not audio_b64 or not self.model:
            print("DEBUG: [ML] No audio data or model not loaded")
            return None

        try:
            # Handle Data URI prefix
            if "," in audio_b64:
                audio_b64 = audio_b64.split(",")[-1]
                
            audio_bytes = base64.b64decode(audio_b64)
            print(f"DEBUG: [ML] Decoded audio bytes: {len(audio_bytes)}")
            decode_time = time.time()
            print(f"DEBUG: [ML] Decode time: {(decode_time - start_time)*1000:.2f}ms")
            
            features, feature_vec, rms = self.extract_features(audio_bytes)
            extract_time = time.time()
            threshold = self.get_adaptive_threshold()
            print(f"DEBUG: [ML] Feature extraction time: {(extract_time - decode_time)*1000:.2f}ms")
            print(f"DEBUG: [ML] RMS filter check: {rms:.6f}, adaptive threshold: {threshold:.6f}")

            if features is None:
                print("DEBUG: [ML] Feature extraction failed")
                return None

            spike_detected = rms > threshold * SPIKE_MULTIPLIER
            if not spike_detected and rms < threshold:
                print("DEBUG: [ML] Adaptive silence detected, skipping model inference")
                self.record_rms(rms)
                return self.build_no_sound_result()

            self.record_rms(rms)
            embedding_vec = None
            with torch.no_grad():
                if hasattr(self.model, 'get_embedding'):
                    embedding_vec = self.model.get_embedding(features).cpu().numpy()[0]
                    embedding_vec = self.normalize_vector(embedding_vec)

            active_feature_vec = embedding_vec if embedding_vec is not None else feature_vec
            cosine_label, cosine_score = self.hybrid_decision(active_feature_vec, None)

            outputs = None
            if cosine_score < HYBRID_HIGH_THRESHOLD:
                with torch.no_grad():
                    outputs = self.model(features)
                    model_label, model_conf = self.hybrid_decision(active_feature_vec, outputs)
            else:
                model_label, model_conf = cosine_label, cosine_score

            final_label = model_label
            final_conf = model_conf

            if rms < (threshold * 2) and final_conf > SUSPICIOUS_CONFIDENCE_THRESHOLD:
                print(f"DEBUG: [ML] Suspicious high-confidence low-energy prediction: {final_label} ({final_conf:.4f}) at rms={rms:.6f}")
                return self.build_no_sound_result("Low energy with suspicious confidence")

            if final_conf < CONFIDENCE_THRESHOLD:
                print(f"DEBUG: [ML] Confidence below threshold: {final_conf:.4f}")
                return self.build_no_sound_result("Uncertain detection")

            final_label = self.smooth_prediction(final_label)
            if final_label == "no_sound":
                print("DEBUG: [ML] Temporal validation suppressed prediction")
                return self.build_no_sound_result()

            print(f"DEBUG: [ML] Final: {final_label} ({final_conf:.4f}) - Total time: {(time.time() - start_time)*1000:.2f}ms")

            result = analyze_situation(final_label, final_conf, self.prediction_buffer)
            if result:
                result["sound"] = final_label
                result["confidence"] = final_conf
                print(f"DEBUG: [ML] Situation analysis result: {result}")
                return result
            else:
                print("DEBUG: [ML] No situation result")

        except Exception as e:
            print(f"DEBUG: [ML] Inference Error: {e}")
            
        return None

# Global Instance
inference_system = AudioInferenceSystem()

def process_audio(audio_data: str) -> Optional[Dict[str, Any]]:
    print(f"DEBUG: [ML] Processing audio data, length: {len(audio_data)}")
    return inference_system.run_inference(audio_data)
