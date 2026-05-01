import os
import numpy as np
import pandas as pd
import librosa
import random
from sklearn.model_selection import train_test_split
from collections import Counter
import soundfile as sf  # For quality control

# --- CONFIGURATION ---
SAMPLE_RATE = 16000
DURATION = 1.0  # 1 second clips
N_MELS = 128
N_FFT = int(0.025 * SAMPLE_RATE)  # 25ms window
HOP_LENGTH = int(0.010 * SAMPLE_RATE)  # 10ms hop

ESC50_PATH = "ESC-50-master"
US8K_PATH = "UrbanSound8K"
FREESOUND_PATH = "Freesound"  # Optional local freesound samples

TARGET_CLASSES = ["doorbell", "knock", "siren", "dog_bark", "alarm"]
HARD_NEGATIVES = {
    "siren": ["car_horn", "engine_idling", "drilling"],
    "dog_bark": ["drilling", "jackhammer", "children_playing"],
    "alarm": ["siren", "car_horn", "engine_idling"],
    "doorbell": ["knock", "telephone", "speech"],
    "knock": ["doorbell", "telephone", "speech"]
}

CLASS_MAPPING = {
    "esc50": {
        "doorbell": ["church_bells"],
        "knock": ["door_wood_knock"],
        "siren": ["siren"],
        "dog_bark": ["dog"],
        "alarm": ["clock_alarm"]
    },
    "us8k": {
        "doorbell": [],
        "knock": ["door_wood_knock"],
        "siren": ["siren"],
        "dog_bark": ["dog_bark"],
        "alarm": []
    }
}

# --- DATA LOADING ---
def load_esc50_metadata():
    esc_df = pd.read_csv(os.path.join(ESC50_PATH, "meta/esc50.csv"))
    rows = []
    for label, esc_categories in CLASS_MAPPING["esc50"].items():
        for cat in esc_categories:
            subset = esc_df[esc_df["category"] == cat]
            for _, r in subset.iterrows():
                rows.append({
                    "path": os.path.join(ESC50_PATH, "audio", r["filename"]),
                    "label": label,
                    "source": "esc50",
                    "fold": None
                })
    return pd.DataFrame(rows)

def load_us8k_metadata():
    us8k_df = pd.read_csv(os.path.join(US8K_PATH, "metadata/UrbanSound8K.csv"))
    rows = []
    for label, us8k_classes in CLASS_MAPPING["us8k"].items():
        for cls in us8k_classes:
            subset = us8k_df[us8k_df["class"] == cls]
            for _, r in subset.iterrows():
                rows.append({
                    "path": os.path.join(US8K_PATH, "audio", f"fold{r['fold']}", r['slice_file_name']),
                    "label": label,
                    "source": "us8k",
                    "fold": r['fold']
                })
    return pd.DataFrame(rows)

def load_freesound_metadata():
    # Optional: if you have local freesound samples
    if not os.path.exists(FREESOUND_PATH):
        return pd.DataFrame()
    # Assume a metadata.csv in FREESOUND_PATH
    try:
        fs_df = pd.read_csv(os.path.join(FREESOUND_PATH, "metadata.csv"))
        rows = []
        for _, r in fs_df.iterrows():
            if r["class"] in TARGET_CLASSES:
                rows.append({
                    "path": os.path.join(FREESOUND_PATH, r["filename"]),
                    "label": r["class"],
                    "source": "freesound",
                    "fold": None
                })
        return pd.DataFrame(rows)
    except:
        return pd.DataFrame()

def load_all_metadata():
    esc_df = load_esc50_metadata()
    us8k_df = load_us8k_metadata()
    fs_df = load_freesound_metadata()
    combined = pd.concat([esc_df, us8k_df, fs_df], ignore_index=True)
    return combined

# --- QUALITY CONTROL ---
def is_audio_valid(path):
    try:
        y, sr = librosa.load(path, sr=None, duration=1.0)
        if len(y) == 0 or np.max(np.abs(y)) == 0:
            return False
        # Check for clipping (values too close to 1.0 or -1.0)
        if np.max(np.abs(y)) > 0.99:
            return False
        return True
    except:
        return False

def trim_silence(y, sr):
    # Trim leading/trailing silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    return y_trimmed

# --- STANDARDIZATION ---
def standardize_audio(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=DURATION)
    y = trim_silence(y, sr)
    y = librosa.util.fix_length(y, size=int(SAMPLE_RATE * DURATION))
    y = librosa.util.normalize(y)
    return y

# --- AUGMENTATION FUNCTIONS ---
def add_background_noise(y, noise_level=0.1):
    noise = np.random.randn(len(y)) * noise_level
    return y + noise

def time_shift(y, sr, shift_ms=200):
    shift_samples = int((shift_ms / 1000) * sr)
    return np.roll(y, shift_samples)

def volume_scaling(y, scale_range=(0.5, 1.5)):
    scale = np.random.uniform(*scale_range)
    return y * scale

def mixup(y1, y2, alpha=0.2):
    return alpha * y1 + (1 - alpha) * y2

def time_stretch(y, rate_range=(0.9, 1.1)):
    rate = np.random.uniform(*rate_range)
    return librosa.effects.time_stretch(y, rate)

def pitch_shift(y, sr, semitones_range=(-2, 2)):
    semitones = np.random.uniform(*semitones_range)
    return librosa.effects.pitch_shift(y, sr, n_steps=semitones)

def apply_augmentation(y, sr):
    # Apply random augmentations
    augmentations = [
        lambda y: add_background_noise(y),
        lambda y: time_shift(y, sr),
        lambda y: volume_scaling(y),
        lambda y: time_stretch(y),
        lambda y: pitch_shift(y, sr)
    ]
    random.shuffle(augmentations)
    for aug in augmentations[:2]:  # Apply 2 random augmentations
        y = aug(y)
    return y

# --- FEATURE EXTRACTION ---
def extract_log_mel(y, sr):
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)
    return log_mel

# --- DATASET BUILDER ---
class BalancedAudioDataset:
    def __init__(self, metadata_df, target_samples_per_class=1000, augment=True):
        self.metadata = metadata_df.copy()
        self.target_samples = target_samples_per_class
        self.augment = augment
        self.balance_classes()

    def balance_classes(self):
        counts = Counter(self.metadata["label"])
        print("Original class counts:", dict(counts))

        balanced_rows = []
        for label in TARGET_CLASSES:
            label_data = self.metadata[self.metadata["label"] == label]
            current_count = len(label_data)
            if current_count == 0:
                print(f"Warning: No samples for {label}")
                continue

            if current_count < self.target_samples:
                # Oversample with augmentation
                multiplier = int(np.ceil(self.target_samples / current_count))
                augmented = []
                for _ in range(multiplier):
                    sample = label_data.sample(n=min(current_count, self.target_samples - len(augmented)), replace=True)
                    augmented.extend(sample.to_dict('records'))
                    if len(augmented) >= self.target_samples:
                        break
                balanced_rows.extend(augmented[:self.target_samples])
            else:
                # Downsample
                balanced_rows.extend(label_data.sample(n=self.target_samples, replace=False).to_dict('records'))

        self.metadata = pd.DataFrame(balanced_rows)
        print("Balanced class counts:", dict(Counter(self.metadata["label"])))

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        path = row["path"]
        label = row["label"]

        if not is_audio_valid(path):
            # Return a zero feature if invalid
            feature = np.zeros((N_MELS, int(DURATION * SAMPLE_RATE // HOP_LENGTH) + 1))
            return feature, TARGET_CLASSES.index(label)

        y = standardize_audio(path)
        if self.augment:
            y = apply_augmentation(y, SAMPLE_RATE)

        feature = extract_log_mel(y, SAMPLE_RATE)
        return feature, TARGET_CLASSES.index(label)

# --- DATA SPLIT ---
def split_dataset(metadata_df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Group by fold/source to avoid leakage
    groups = metadata_df.groupby(["source", "fold"]).groups
    group_keys = list(groups.keys())

    train_keys, temp_keys = train_test_split(group_keys, train_size=train_ratio, random_state=42)
    val_keys, test_keys = train_test_split(temp_keys, train_size=val_ratio / (val_ratio + test_ratio), random_state=42)

    train_df = metadata_df.loc[metadata_df.apply(lambda r: (r["source"], r["fold"]) in train_keys, axis=1)]
    val_df = metadata_df.loc[metadata_df.apply(lambda r: (r["source"], r["fold"]) in val_keys, axis=1)]
    test_df = metadata_df.loc[metadata_df.apply(lambda r: (r["source"], r["fold"]) in test_keys, axis=1)]

    return train_df, val_df, test_df

# --- MAIN PREPARATION ---
def prepare_dataset():
    print("Loading metadata...")
    metadata = load_all_metadata()
    print(f"Total samples: {len(metadata)}")

    # Filter valid audio
    valid_metadata = metadata[metadata["path"].apply(is_audio_valid)]
    print(f"Valid samples: {len(valid_metadata)}")

    # Add hard negatives (map some classes to confuse the model)
    for target, negatives in HARD_NEGATIVES.items():
        for neg in negatives:
            neg_samples = valid_metadata[valid_metadata["label"] == neg]
            if len(neg_samples) > 0:
                # Label them as the target to force separation
                neg_samples = neg_samples.copy()
                neg_samples["label"] = target
                valid_metadata = pd.concat([valid_metadata, neg_samples])

    train_df, val_df, test_df = split_dataset(valid_metadata)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Create balanced datasets
    train_dataset = BalancedAudioDataset(train_df, augment=True)
    val_dataset = BalancedAudioDataset(val_df, augment=False)
    test_dataset = BalancedAudioDataset(test_df, augment=False)

    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    train_ds, val_ds, test_ds = prepare_dataset()
    print("Dataset preparation complete.")
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")
    print(f"Test samples: {len(test_ds)}")