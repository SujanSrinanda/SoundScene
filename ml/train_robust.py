import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# --- CONFIGURATION ---
SAMPLE_RATE = 16000
DURATION = 2.0
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256
BATCH_SIZE = 32
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HARD_NEGATIVE_CLASSES = ["siren", "drilling", "alarm"]
HARD_NEGATIVE_PROB = 0.35
BALANCE_TARGET = True
DOG_BARK_AUGMENT = True
DOG_BARK_AUGMENT_PROB = 0.7
DOG_BARK_BOOST_RATIO = 1.5

ESC50_PATH = "ESC-50-master"
US8K_PATH = "UrbanSound8K"
FEATURE_CACHE = "ml/models/feature_cache.npz"

CLASS_MAP = {
    "doorbell": {"esc50": "church_bells"},
    "knock": {"esc50": "door_wood_knock"},
    "siren": {"esc50": "siren", "us8k": "siren"},
    "dog_bark": {"esc50": "dog", "us8k": "dog_bark"},
    "alarm": {"esc50": "clock_alarm"}
}

TARGET_CLASSES = sorted(list(CLASS_MAP.keys()))

def load_metadata():
    esc_df = pd.read_csv(os.path.join(ESC50_PATH, "meta/esc50.csv"))
    rows = []
    for label, sources in CLASS_MAP.items():
        if "esc50" in sources:
            subset = esc_df[esc_df["category"] == sources["esc50"]]
            for _, r in subset.iterrows():
                rows.append({"path": os.path.join(ESC50_PATH, "audio", r["filename"]), "label": label})
        if "us8k" in sources:
            us8k_df = pd.read_csv(os.path.join(US8K_PATH, "metadata/UrbanSound8K.csv"))
            subset = us8k_df[us8k_df["class"] == sources["us8k"]]
            for _, r in subset.iterrows():
                rows.append({"path": os.path.join(US8K_PATH, "audio", f"fold{r['fold']}", r['slice_file_name']), "label": label})
    return pd.DataFrame(rows)

def balance_samples(X, y):
    if not BALANCE_TARGET:
        return X, y

    counts = {label: np.sum(y == idx) for idx, label in enumerate(TARGET_CLASSES)}
    target_count = max(counts.values())
    balanced_X, balanced_y = [], []

    for idx, label in enumerate(TARGET_CLASSES):
        indices = np.where(y == idx)[0]
        if len(indices) == 0:
            continue
        sampled = np.random.choice(indices, size=target_count, replace=len(indices) < target_count)
        balanced_X.append(X[sampled])
        balanced_y.append(y[sampled])

    target_count = max(max(counts.values()), int(counts.get('dog_bark', 0) * DOG_BARK_BOOST_RATIO))
    balanced_X = np.concatenate(balanced_X, axis=0)
    balanced_y = np.concatenate(balanced_y, axis=0)

    perm = np.random.permutation(len(balanced_y))
    return balanced_X[perm], balanced_y[perm]


def add_background_noise(audio, snr_db=20):
    noise = np.random.randn(len(audio))
    audio_power = np.sum(audio ** 2)
    noise_power = np.sum(noise ** 2)
    scale = np.sqrt(audio_power / (noise_power + 1e-9)) * (10 ** (-snr_db / 20))
    return audio + noise * scale


def augment_dog_bark(audio):
    augmented = []
    if np.random.random() < DOG_BARK_AUGMENT_PROB:
        shifted = librosa.effects.pitch_shift(audio, SAMPLE_RATE, n_steps=np.random.uniform(-2.0, 2.0))
        shifted = librosa.util.fix_length(shifted, size=int(SAMPLE_RATE * DURATION))
        augmented.append(shifted)
    if np.random.random() < DOG_BARK_AUGMENT_PROB:
        rate = np.random.uniform(0.9, 1.1)
        stretched = librosa.effects.time_stretch(audio, rate)
        stretched = librosa.util.fix_length(stretched, size=int(SAMPLE_RATE * DURATION))
        augmented.append(stretched)
    if np.random.random() < DOG_BARK_AUGMENT_PROB:
        noisy = add_background_noise(audio, snr_db=np.random.uniform(15, 30))
        augmented.append(noisy)
    return augmented


def precompute_features():
    if os.path.exists(FEATURE_CACHE):
        print("Loading cached features...")
        data = np.load(FEATURE_CACHE, allow_pickle=True)
        return data['X'], data['y']

    df = load_metadata()
    X, y = [], []
    label_to_idx = {l: i for i, l in enumerate(TARGET_CLASSES)}
    
    print(f"Precomputing features for {len(df)} samples...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            audio, _ = librosa.load(row["path"], sr=SAMPLE_RATE, duration=DURATION)
            audio = librosa.util.fix_length(audio, size=int(SAMPLE_RATE * DURATION))

            def push_feature(audio_sample):
                mel = librosa.feature.melspectrogram(y=audio_sample, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
                log_mel = librosa.power_to_db(mel, ref=np.max)
                log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)
                X.append(log_mel)
                y.append(label_to_idx[row["label"]])

            push_feature(audio)
            if row["label"] == "dog_bark":
                for aug_audio in augment_dog_bark(audio):
                    push_feature(aug_audio)
        except Exception:
            continue
            
    X = np.array(X)
    y = np.array(y)
    X, y = balance_samples(X, y)
    os.makedirs(os.path.dirname(FEATURE_CACHE), exist_ok=True)
    np.savez_compressed(FEATURE_CACHE, X=X, y=y)
    return X, y

class FastDataset(Dataset):
    def __init__(self, X, y, is_training=False):
        self.X = torch.tensor(X).unsqueeze(1).float()
        self.y = torch.tensor(y).long()
        self.is_training = is_training
        self.class_indices = {
            label: np.where(y == idx)[0]
            for idx, label in enumerate(TARGET_CLASSES)
        }
        self.hard_negative_indices = np.concatenate(
            [self.class_indices.get(c, np.array([], dtype=int)) for c in HARD_NEGATIVE_CLASSES]
        ) if any(len(self.class_indices.get(c, [])) > 0 for c in HARD_NEGATIVE_CLASSES) else np.array([], dtype=int)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        label_idx = int(self.y[idx].item())
        label_name = TARGET_CLASSES[label_idx]

        if self.is_training:
            if label_name == "dog_bark" and len(self.hard_negative_indices) > 0 and np.random.random() < HARD_NEGATIVE_PROB:
                other = self.X[np.random.choice(self.hard_negative_indices)].clone()
                x = 0.85 * x + 0.15 * other

            if np.random.random() > 0.5:
                mask_size = np.random.randint(2, 12)
                start = np.random.randint(0, x.shape[1] - mask_size)
                x[:, start:start+mask_size, :] = 0
            if np.random.random() > 0.5:
                mask_size = np.random.randint(5, 25)
                start = np.random.randint(0, x.shape[2] - mask_size)
                x[:, :, start:start+mask_size] = 0

            if np.random.random() > 0.5:
                x = x + 0.02 * torch.randn_like(x)

        return x, self.y[idx]

class LightweightCNN(nn.Module):
    def __init__(self, num_classes):
        super(LightweightCNN, self).__init__()
        # Task 3: Optimized Architecture (Conv2D -> BatchNorm -> ReLU -> MaxPool)
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5), # Task 4: Dropout for overfitting
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        embedding = self.net(x)
        return self.classifier(embedding)

    @torch.jit.export
    def get_embedding(self, x):
        embedding = self.net(x)
        return embedding.view(embedding.size(0), -1)

def train(max_epochs=EPOCHS, target_accuracy=0.0, patience=7):
    X, y = precompute_features()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    train_loader = DataLoader(FastDataset(X_train, y_train, is_training=True), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(FastDataset(X_val, y_val), batch_size=BATCH_SIZE)
    
    model = LightweightCNN(len(TARGET_CLASSES)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print("Starting fast training...")
    best_acc = 0
    best_epoch = 0
    all_preds = []
    all_labels = []
    for epoch in range(max_epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            optimizer.step()
            
        model.eval()
        correct = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                outputs = model(bx)
                _, pred = torch.max(outputs, 1)
                correct += (pred == by).sum().item()
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(by.cpu().numpy())
        
        acc = correct / len(y_val)
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            best_model = model.cpu()
            torch.jit.script(best_model).save("ml/models/robust_model.pt")
            model.to(DEVICE)
        
        print(f"Epoch {epoch+1} | Val Acc: {acc:.4f} | Best Acc: {best_acc:.4f}")

        if target_accuracy and acc >= target_accuracy:
            print(f"Target accuracy of {target_accuracy:.4f} reached at epoch {epoch+1}.")
            break

        if epoch - best_epoch >= patience:
            print(f"No improvement for {patience} epochs. Early stopping.")
            break

    print(f"Training finished. Best Val Acc: {best_acc:.4f}")

    # Task 8: Error analysis using confusion matrix for target classes
    try:
        cm = confusion_matrix(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, target_names=TARGET_CLASSES, digits=4)
        print("Validation Confusion Matrix:\n", cm)
        print("Validation Classification Report:\n", report)
    except Exception as e:
        print(f"Error computing confusion analysis: {e}")

    return best_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the robust sound model until target accuracy is met.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Maximum number of training epochs")
    parser.add_argument("--target-accuracy", type=float, default=0.0, help="Stop when validation accuracy reaches this threshold")
    parser.add_argument("--max-attempts", type=int, default=1, help="Retry training up to this many independent attempts")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience in epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    best_overall = 0.0
    for attempt in range(1, args.max_attempts + 1):
        print(f"\n=== Training Attempt {attempt}/{args.max_attempts} ===")
        best_acc = train(max_epochs=args.epochs, target_accuracy=args.target_accuracy, patience=args.patience)
        best_overall = max(best_overall, best_acc)
        if args.target_accuracy and best_acc >= args.target_accuracy:
            print(f"Target accuracy achieved: {best_acc:.4f}")
            break
        if attempt < args.max_attempts:
            print(f"Target not reached: best acc {best_acc:.4f}. Retrying...\n")

    print(f"\nTraining complete. Best overall validation accuracy: {best_overall:.4f}")
