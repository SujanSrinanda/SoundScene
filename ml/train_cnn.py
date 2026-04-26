import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Configuration
DATA_BASE = "UrbanSound8K/audio/"
META_PATH = "UrbanSound8K/metadata/UrbanSound8K.csv"
SAMPLE_RATE = 16000
DURATION = 4
N_MFCC = 40
N_FFT = 1024
HOP_LENGTH = 512

# Target Classes
TARGET_CLASSES = ['dog_bark', 'siren', 'car_horn', 'drilling', 'street_music']

# --- DATASET & PREPROCESSING ---

def apply_augmentation(audio):
    # 1. Add background noise
    if np.random.random() > 0.5:
        noise = np.random.randn(len(audio))
        audio = audio + 0.005 * noise * np.random.uniform()
    # 2. Time shifting
    if np.random.random() > 0.5:
        shift = np.random.randint(SAMPLE_RATE * 0.2)
        audio = np.roll(audio, shift)
    # 3. Volume scaling
    audio = audio * np.random.uniform(0.8, 1.2)
    return audio

class UrbanSoundDataset(Dataset):
    def __init__(self, df, base_path, augment=False):
        self.df = df
        self.df = df.reset_index(drop=True)
        self.base_path = base_path
        self.augment = augment
        self.le = LabelEncoder()
        self.df['label_encoded'] = self.le.fit_transform(self.df['class'])
        print("Extracting MFCC + Delta + Delta-Delta features...", flush=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx % 100 == 0:
            print(f"Progress: {idx}/{len(self)} samples processed...", flush=True)

        row = self.df.iloc[idx]
        file_path = os.path.join(self.base_path, f"fold{row['fold']}", row['slice_file_name'])
        
        # Preprocessing
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION, mono=True)
        if self.augment:
            audio = apply_augmentation(audio)
            
        target_len = SAMPLE_RATE * DURATION
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]

        # Features: MFCC + Delta + Delta-Delta
        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        features = np.concatenate([mfcc, delta, delta2], axis=0) # (120, frames)
        
        return torch.tensor(features).unsqueeze(0).float(), torch.tensor(row['label_encoded']).long()

# --- MODEL ARCHITECTURE ---

class LightweightCNN(nn.Module):
    def __init__(self, num_classes):
        super(LightweightCNN, self).__init__()
        # Conv Block 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Conv Block 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # GAP + Classifier
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# --- TRAINING LOOP ---

def train():
    print("Initializing UrbanSound8K Robust Pipeline (PyTorch Edition)...")
    
    # Load and Filter Metadata
    df = pd.read_csv(META_PATH)
    df = df[df['class'].isin(TARGET_CLASSES)]
    
    # Stratified Split
    train_df, rem_df = train_test_split(df, train_size=0.7, stratify=df['class'], random_state=42)
    val_df, test_df = train_test_split(rem_df, test_size=0.5, stratify=rem_df['class'], random_state=42)
    
    train_loader = DataLoader(UrbanSoundDataset(train_df, DATA_BASE, augment=True), batch_size=32, shuffle=True)
    val_loader = DataLoader(UrbanSoundDataset(val_df, DATA_BASE), batch_size=32)
    test_loader = DataLoader(UrbanSoundDataset(test_df, DATA_BASE), batch_size=32)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightweightCNN(len(TARGET_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    patience = 7
    trigger_times = 0
    
    print("\nStarting training...")
    for epoch in range(40):
        model.train()
        train_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / len(val_df)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger_times = 0
            torch.save(model.state_dict(), 'ml/models/best_model.pth')
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping!")
                break

    print("\nTraining complete. Model saved to ml/models/best_model.pth")
    print("Note: In this environment (Python 3.14), PyTorch is used for maximum stability.")

if __name__ == "__main__":
    os.makedirs("ml/models", exist_ok=True)
    train()
