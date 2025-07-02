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
DATA_DIR = "dataset"

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

# --- Temperature Scaling ---
def temperature_scale(prob, T=2.0):
    logit = np.log(prob / (1 - prob))
    scaled_logit = logit / T
    return 1 / (1 + np.exp(-scaled_logit))

# --- Run Model ---
def run_model(record, models, verbose):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    signal, _ = load_signals(record)
    signal = signal.T[:12, :]

    if signal.shape[1] < FIXED_LENGTH:
        signal = np.pad(signal, ((0, 0), (0, FIXED_LENGTH - signal.shape[1])), mode='constant')
    else:
        signal = signal[:, :FIXED_LENGTH]

    if isinstance(models, dict):
        # Ensemble mode
        signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        age, sex_1hot, mean, std = extract_features(record)
        x1 = torch.cat([
            torch.tensor([[float(age)]], dtype=torch.float32).to(device),
            torch.from_numpy(sex_1hot.astype(np.float32)).unsqueeze(0).to(device),
            torch.from_numpy(mean.astype(np.float32)).unsqueeze(0).to(device),
            torch.from_numpy(std.astype(np.float32)).unsqueeze(0).to(device)
        ], dim=1)

        weights = {"audio_classifier": 0.45, "crnn": 0.10, "resnet": 0.45}
        weighted_probs = []
        model_predictions = {}

        with torch.no_grad():
            for name, model in models.items():
                output = model(signal_tensor, x1) if name == 'resnet' else model(signal_tensor)
                prob = torch.sigmoid(output).item()
                pred = int(prob >= 0.5)
                model_predictions[name] = (pred, prob)
                weighted_probs.append(prob * weights[name])

        raw_avg_prob = sum(weighted_probs) / sum(weights.values())
        avg_prob = temperature_scale(raw_avg_prob, T=3.0)
        final_pred = int(avg_prob >= 0.65)

        if verbose:
            for name, (pred, prob) in model_predictions.items():
                print(f"{name:>15}: pred={pred}, prob={prob:.4f}, weight={weights[name]:.2f}")
            print(f"\nFinal avg prob: {avg_prob:.4f} → Prediction: {final_pred}")

    else:
        # Single model mode
        signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)  # [1, 12, 3000]
        signal_tensor = signal_tensor.transpose(1, 2).to(device)  # [1, 3000, 12]
        with torch.no_grad():
            prob = models(signal_tensor).item()
        avg_prob = temperature_scale(prob, T=3.0)
        final_pred = int(avg_prob >= 0.65)

        if verbose:
            print(f"final_model.pt: prob={avg_prob:.4f} → Prediction: {final_pred}")

    output_dir = "holdout_outputs"
    os.makedirs(output_dir, exist_ok=True)
    record_id = os.path.basename(record).replace(".hea", "")
    with open(os.path.join(output_dir, f"{record_id}.txt"), "w") as f:
        f.write(f"# Chagas label: {final_pred}\n")
        f.write(f"# Chagas probability: {avg_prob:.4f}\n")

    return final_pred, avg_prob

# --- Load Model ---
def load_model(model_folder, verbose):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    final_model_path = os.path.join(model_folder, "final_model.pt")
    if os.path.exists(final_model_path):
        from s4d import S4D
        class S4Model(nn.Module):
            def __init__(self, d_input=12, d_output=1, d_model=128, n_layers=4, dropout=0.1, prenorm=False):
                super().__init__()
                self.encoder = nn.Linear(d_input, d_model)
                self.s4_layers = nn.ModuleList([
                    S4D(d_model, dropout=dropout, transposed=True) for _ in range(n_layers)
                ])
                self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
                self.dropouts = nn.ModuleList([nn.Dropout1d(dropout) for _ in range(n_layers)])
                self.decoder = nn.Linear(d_model, d_output)

            def forward(self, x):
                x = self.encoder(x).transpose(-1, -2)
                for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
                    z = layer(x)[0]
                    z = dropout(z)
                    x = norm((z + x).transpose(-1, -2)).transpose(-1, -2)
                x = x.transpose(-1, -2).mean(dim=1)
                return torch.sigmoid(self.decoder(x))

        model = S4Model().to(device)
        model.load_state_dict(torch.load(final_model_path, map_location=device))
        model.eval()
        return model
    else:
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
