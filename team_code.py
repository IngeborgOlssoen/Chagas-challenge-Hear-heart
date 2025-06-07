#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys

from helper_code import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd


from base_model import AudioClassifier


################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.


FIXED_LENGTH = 3000  # pad or crop ECG signals to this length

class ECGFromCSV(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        signal_path = row['signal_path']
        label = row['label']

        signal, _ = load_signals(signal_path)
        signal = signal.T[:12, :]  # [12, L]

        L = signal.shape[1]
        if L < FIXED_LENGTH:
            signal = np.pad(signal, ((0, 0), (0, FIXED_LENGTH - L)), mode='constant')
        else:
            signal = signal[:, :FIXED_LENGTH]

        signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)  # [1, 12, L]
        label = torch.tensor(label, dtype=torch.float32)
        return signal, label

    

# Train your model.
def train_model(data_folder, model_folder, verbose):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = ECGFromCSV(data_folder)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = AudioClassifier().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x).squeeze(1)
            loss = criterion(torch.sigmoid(out), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if verbose:
            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    os.makedirs(model_folder, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_folder, 'model.pt'))


# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudioClassifier().to(device)  # or whatever model class you're using
    model.load_state_dict(torch.load(os.path.join(model_folder, 'model.pt'), map_location=device, weights_only=False))
    model.eval()
    return model



# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    signal, _ = load_signals(record)
    signal = signal.T[:12, :]
    L = signal.shape[1]
    if L < FIXED_LENGTH:
        pad_width = FIXED_LENGTH - L
        signal = np.pad(signal, ((0, 0), (0, pad_width)), mode='constant')
    else:
        signal = signal[:, :FIXED_LENGTH]
    signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 12, L]

    with torch.no_grad():
        output = model(signal)
        prob = torch.sigmoid(output).item()
        pred = int(prob >= 0.5)

    return pred, prob

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Extract your features.
def extract_features(record):
    header = load_header(record)

    # Extract the age from the record.
    age = get_age(header)
    age = np.array([age])

    # Extract the sex from the record and represent it as a one-hot encoded vector.
    sex = get_sex(header)
    sex_one_hot_encoding = np.zeros(3, dtype=bool)
    if sex.casefold().startswith('f'):
        sex_one_hot_encoding[0] = 1
    elif sex.casefold().startswith('m'):
        sex_one_hot_encoding[1] = 1
    else:
        sex_one_hot_encoding[2] = 1

    # Extract the source from the record (but do not use it as a feature).
    source = get_source(header)

    # Load the signal data and fields. Try fields.keys() to see the fields, e.g., fields['fs'] is the sampling frequency.
    signal, fields = load_signals(record)
    channels = fields['sig_name']

    # Reorder the channels in case they are in a different order in the signal data.
    reference_channels = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    num_channels = len(reference_channels)
    signal = reorder_signal(signal, channels, reference_channels)

    # Compute two per-channel features as examples.
    signal_mean = np.zeros(num_channels)
    signal_std = np.zeros(num_channels)

    for i in range(num_channels):
        num_finite_samples = np.sum(np.isfinite(signal[:, i]))
        if num_finite_samples > 0:
            signal_mean[i] = np.nanmean(signal)
        else:
            signal_mean = 0.0
        if num_finite_samples > 1:
            signal_std[i] = np.nanstd(signal)
        else:
            signal_std = 0.0

    # Return the features.

    return age, sex_one_hot_encoding, source, signal_mean, signal_std

# Save your trained model.
def save_model(model_folder, model):
    d = {'model': model}
    filename = os.path.join(model_folder, 'model.pt')
    joblib.dump(d, filename, protocol=0)