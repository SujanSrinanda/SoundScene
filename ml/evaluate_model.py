import argparse
import os
import random
import sys
import numpy as np
import torch
import librosa
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ml.train_robust import load_metadata, TARGET_CLASSES

SAMPLE_RATE = 16000
DURATION = 2.0
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256

MODEL_PATHS = [
    "ml/models/robust_model.pt",
    "ml/models/best_model.pth"
]


def load_model(path: str):
    if path.endswith(".pt"):
        return torch.jit.load(path, map_location="cpu")

    # If a state dict exists instead, build a compatible model
    from ml.train_robust import LightweightCNN
    model = LightweightCNN(len(TARGET_CLASSES))
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def preprocess_audio(path: str):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, duration=DURATION)
    audio = librosa.util.fix_length(audio, size=int(SAMPLE_RATE * DURATION))
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)
    return torch.tensor(log_mel).unsqueeze(0).unsqueeze(0).float()


def predict(model, tensor):
    with torch.no_grad():
        output = model(tensor)
        if output.shape[-1] == 1:
            output = output.view(-1)
        probs = torch.softmax(output, dim=1) if output.ndim == 2 else torch.softmax(output.unsqueeze(0), dim=1)
        return int(torch.argmax(probs, dim=1).item())


def main(args):
    metadata = load_metadata()
    if metadata.empty:
        raise RuntimeError("No dataset metadata found. Make sure ESC-50 and UrbanSound8K are present.")

    model_path = args.model
    if not model_path:
        model_path = next((p for p in MODEL_PATHS if os.path.exists(p)), None)
    if not model_path:
        raise FileNotFoundError("No model file found. Train the model first using ml/train_robust.py.")

    print(f"Loading model from: {model_path}")
    model = load_model(model_path)
    model.eval()

    entries = metadata.to_dict(orient="records")
    if args.random_samples > 0:
        entries = random.sample(entries, min(args.random_samples, len(entries)))

    y_true = []
    y_pred = []
    mistakes = []

    for idx, row in enumerate(entries, 1):
        path = row["path"]
        label = row["label"]
        if not os.path.exists(path):
            print(f"Skipping missing file: {path}")
            continue

        tensor = preprocess_audio(path)
        pred_idx = predict(model, tensor)
        pred_label = TARGET_CLASSES[pred_idx]
        y_true.append(label)
        y_pred.append(pred_label)

        if label != pred_label:
            mistakes.append((path, label, pred_label))
        if args.verbose and idx % 10 == 0:
            print(f"Processed {idx}/{len(entries)} samples")

    if len(y_true) == 0:
        raise RuntimeError("No valid samples were processed.")

    print("\n=== Evaluation Results ===")
    print(f"Samples evaluated: {len(y_true)}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

    if args.report:
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=TARGET_CLASSES, zero_division=0))

    if args.confusion:
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred, labels=TARGET_CLASSES))

    if mistakes:
        print(f"\nMispredicted samples: {len(mistakes)}")
        for path, true_label, pred_label in mistakes[:10]:
            print(f" {path} -> true={true_label}, pred={pred_label}")
        if len(mistakes) > 10:
            print(f" ...and {len(mistakes) - 10} more")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the robust sound model against dataset samples.")
    parser.add_argument("--model", type=str, default="", help="Path to the model file (defaults to ml/models/robust_model.pt)")
    parser.add_argument("--random-samples", type=int, default=0, help="Evaluate a random subset of samples from the dataset")
    parser.add_argument("--report", action="store_true", help="Print classification report")
    parser.add_argument("--confusion", action="store_true", help="Print confusion matrix")
    parser.add_argument("--verbose", action="store_true", help="Print progress while evaluating")
    args = parser.parse_args()
    main(args)
