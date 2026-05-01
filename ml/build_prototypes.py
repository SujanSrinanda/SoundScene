import os
import sys
import numpy as np
import torch
import librosa

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ml.train_robust import load_metadata, TARGET_CLASSES

MODEL_PATH = os.path.join(ROOT_DIR, "ml", "models", "robust_model.pt")
OUTPUT_PATH = os.path.join(ROOT_DIR, "ml", "models", "class_prototypes.npz")
SAMPLE_RATE = 16000
DURATION = 1.0
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256


def normalize(vec):
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec) + 1e-9
    return vec / norm


def extract_features(path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, duration=DURATION)
    audio = librosa.util.fix_length(audio, size=int(SAMPLE_RATE * DURATION))
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)
    return torch.tensor(log_mel).unsqueeze(0).unsqueeze(0).float()


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = torch.jit.load(MODEL_PATH)
    model.eval()

    metadata = load_metadata()
    class_vectors = {label: [] for label in TARGET_CLASSES}

    print(f"Building prototype memory from {len(metadata)} samples...")
    for idx, row in metadata.iterrows():
        path = row["path"]
        label = row["label"]
        if not os.path.exists(path):
            print(f"Skipping missing file: {path}")
            continue

        features = extract_features(path)
        with torch.no_grad():
            if hasattr(model, 'get_embedding'):
                embedding = model.get_embedding(features).cpu().numpy()[0]
            else:
                raise RuntimeError("Model does not expose get_embedding")
        class_vectors[label].append(normalize(embedding))

    class_names = []
    prototype_list = []
    vectors_list = []

    for label in TARGET_CLASSES:
        samples = np.stack(class_vectors[label]) if class_vectors[label] else np.zeros((1, 128), dtype=np.float32)
        prototype = normalize(np.mean(samples, axis=0))
        class_names.append(label)
        prototype_list.append(prototype)
        vectors_list.append(samples)
        print(f"  {label}: {samples.shape[0]} vectors")

    np.savez_compressed(
        OUTPUT_PATH,
        class_names=np.array(class_names, dtype=object),
        class_prototypes=np.stack(prototype_list),
        class_vectors=np.array(vectors_list, dtype=object)
    )

    print(f"Saved class prototype memory to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
