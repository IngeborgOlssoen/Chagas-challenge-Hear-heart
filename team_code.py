#!/usr/bin/env python

import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import shutil
from glob import glob

from helper_code import *
from base_model import AudioClassifier, CRNN, ResNet

# --- Constants ---
FIXED_LENGTH = 3000  # pad or crop ECG signals to this length
DATA_DIR = "/content/drive/MyDrive/chagas_datasets"

# --- Custom Dataset Class ---
class ECGFromCSV(Dataset):
    def __init__(self, folder_path, augment=False):
        self.base_dir = folder_path
        self.augment = augment
        self.records = [f.replace(".hea", "") for f in os.listdir(folder_path) if f.endswith(".hea")]

    def augment_signal(self, signal):
        noise = np.random.normal(0, 0.01, size=signal.shape)
        signal += noise
        signal *= np.random.uniform(0.9, 1.1)
        signal = np.roll(signal, shift=np.random.randint(-100, 100), axis=1)
        return signal

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record_id = self.records[idx]
        record_path = os.path.join(self.base_dir, record_id)

        signal, _ = load_signals(record_path)
        signal = signal.T[:12, :]

        if signal.shape[1] < FIXED_LENGTH:
            signal = np.pad(signal, ((0, 0), (0, FIXED_LENGTH - signal.shape[1])), mode='constant')
        else:
            signal = signal[:, :FIXED_LENGTH]

        label = get_label(load_header(record_path + ".hea"))

        if self.augment and label == 0:
            signal = self.augment_signal(signal)

        signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(float(label), dtype=torch.float32)
        return signal, label

# --- Feature Extraction ---
def extract_features(record):
    header = load_header(record)
    age = np.array([get_age(header)])

    sex = get_sex(header)
    sex_one_hot = np.zeros(3, dtype=bool)
    if sex.lower().startswith('f'):
        sex_one_hot[0] = 1
    elif sex.lower().startswith('m'):
        sex_one_hot[1] = 1
    else:
        sex_one_hot[2] = 1

    signal, fields = load_signals(record)
    channels = fields['sig_name']
    reference_channels = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    signal = reorder_signal(signal, channels, reference_channels)

    signal_mean = np.nanmean(np.where(np.isfinite(signal), signal, np.nan), axis=0)
    signal_std = np.nanstd(np.where(np.isfinite(signal), signal, 0), axis=0)

    return age, sex_one_hot, signal_mean, signal_std

# --- Training Function ---
def train_model(data_folder, model_folder, verbose):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Using device: {device}")

    dataset = ECGFromCSV(data_folder, augment=True)
    val_size = int(0.15 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

    models = {
        "audio_classifier": AudioClassifier().to(device),
        "crnn": CRNN().to(device),
        "resnet": ResNet().to(device),
    }

    patience, min_delta = 4, 0.001
    for model_name, model in models.items():
        model_path = os.path.join(model_folder, f"{model_name}_model.pt")
        if os.path.exists(model_path):
            print(f"⏭️ Skipping {model_name}, already trained.")
            continue

        print(f"Training {model_name}...")

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        best_loss = float('inf')
        best_model = None
        epochs_without_improvement = 0

        train_losses = []
        val_losses = []

        for epoch in range(100):
            model.train()
            total_loss = 0

            for idx, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                record_id = dataset.records[train_set.indices[idx]]
                record = os.path.join(dataset.base_dir, record_id)

                age, sex_1hot, mean, std = extract_features(record)
                age_tensor = torch.tensor([[float(age)]], dtype=torch.float32).to(device)
                sex_tensor = torch.from_numpy(sex_1hot.astype(np.float32)).unsqueeze(0).to(device)
                mean_tensor = torch.from_numpy(mean.astype(np.float32)).unsqueeze(0).to(device)
                std_tensor = torch.from_numpy(std.astype(np.float32)).unsqueeze(0).to(device)
                x1 = torch.cat([age_tensor, sex_tensor, mean_tensor, std_tensor], dim=1)

                out = model(x, x1) if model_name == 'resnet' else model(x)
                loss = criterion(out.view(-1), y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)

            # --- Validation ---
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for idx, (x, y) in enumerate(val_loader):
                    x, y = x.to(device), y.to(device)
                    record_id = dataset.records[val_set.indices[idx]]
                    record = os.path.join(dataset.base_dir, record_id)
                    age, sex_1hot, mean, std = extract_features(record)
                    age_tensor = torch.tensor([[float(age)]], dtype=torch.float32).to(device)
                    sex_tensor = torch.from_numpy(sex_1hot.astype(np.float32)).unsqueeze(0).to(device)
                    mean_tensor = torch.from_numpy(mean.astype(np.float32)).unsqueeze(0).to(device)
                    std_tensor = torch.from_numpy(std.astype(np.float32)).unsqueeze(0).to(device)
                    x1 = torch.cat([age_tensor, sex_tensor, mean_tensor, std_tensor], dim=1)
                    out = model(x, x1) if model_name == 'resnet' else model(x)
                    val_loss += criterion(out.view(-1), y).item()

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            if verbose:
                print(f"{model_name} - Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            if best_loss - avg_val_loss > min_delta:
                best_loss = avg_val_loss
                best_model = model.state_dict()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"⏹️ Early stopping at epoch {epoch+1}")
                    break

        model.load_state_dict(best_model)
        os.makedirs(model_folder, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(model_folder, f"{model_name}_model.pt"))

        pd.DataFrame({"train_loss": train_losses, "val_loss": val_losses}).to_csv(
            os.path.join(model_folder, f"{model_name}_loss_curve.csv"), index=False
        )

        print(f"✅ Saved {model_name}.")

    print("🎉 Training completed for all models!")


def run_model(record, models, verbose):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    signal, _ = load_signals(record)
    signal = signal.T[:12, :]

    if signal.shape[1] < FIXED_LENGTH:
        signal = np.pad(signal, ((0, 0), (0, FIXED_LENGTH - signal.shape[1])), mode='constant')
    else:
        signal = signal[:, :FIXED_LENGTH]

    signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    age, sex_1hot, mean, std = extract_features(record)
    age_tensor = torch.tensor([[float(age)]], dtype=torch.float32).to(device)
    sex_tensor = torch.from_numpy(sex_1hot.astype(np.float32)).unsqueeze(0).to(device)
    mean_tensor = torch.from_numpy(mean.astype(np.float32)).unsqueeze(0).to(device)
    std_tensor = torch.from_numpy(std.astype(np.float32)).unsqueeze(0).to(device)
    x1 = torch.cat([age_tensor, sex_tensor, mean_tensor, std_tensor], dim=1)

    weights = {"audio_classifier": 0.45, "crnn": 0.10, "resnet": 0.45}
    weighted_probs = []
    model_predictions = {}

    with torch.no_grad():
        for name, model in models.items():
            output = model(signal, x1) if name == 'resnet' else model(signal)
            prob = torch.sigmoid(output).item()
            pred = int(prob >= 0.5)
            model_predictions[name] = (pred, prob)
            weighted_probs.append(prob * weights[name])

    raw_avg_prob = sum(weighted_probs) / sum(weights.values())
    avg_prob = temperature_scale(raw_avg_prob, T=3.0)
    final_pred = int(avg_prob >= 0.65)

    if verbose:
        print("🔍 Model predictions:")
        for name, (pred, prob) in model_predictions.items():
            print(f"{name:>15}: pred={pred}, prob={prob:.4f}, weight={weights[name]:.2f}")
        print(f"\n🧩 Final avg prob: {avg_prob:.4f} → Prediction: {final_pred}")

    # --- Save prediction output in required format ---
    # Save output in required format
    output_dir = "holdout_outputs"
    os.makedirs(output_dir, exist_ok=True)
    record_id = os.path.basename(record).replace(".hea", "")
    file_path = os.path.join(output_dir, f"{record_id}.txt")
    with open(file_path, "w") as f:
        f.write(f"# Chagas label: {final_pred}\n")
        f.write(f"# Chagas probability: {avg_prob:.4f}\n")
    
    return final_pred, avg_prob


def temperature_scale(prob, T=2.0):
    logit = np.log(prob / (1 - prob))
    scaled_logit = logit / T
    return 1 / (1 + np.exp(-scaled_logit))



# --- Model Loading ---
def load_model(model_folder, verbose):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = {
        "audio_classifier": AudioClassifier().to(device),
        "crnn": CRNN().to(device),
        "resnet": ResNet().to(device),
    }
    for name in models:
        path = os.path.join(model_folder, f"{name}_model.pt")
        models[name].load_state_dict(torch.load(path, map_location=device))
        models[name].eval()
    return models

def save_model(model_folder, model):
    joblib.dump({'model': model}, os.path.join(model_folder, 'model.pt'), protocol=0)
