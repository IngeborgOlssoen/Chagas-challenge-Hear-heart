#!/usr/bin/env python

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




from base_model import AudioClassifier, CRNN, ResNet


FIXED_LENGTH = 3000  # pad or crop ECG signals to this length


class ECGFromCSV(Dataset):
    def __init__(self, csv_path, augment=False):
        self.df = pd.read_csv(csv_path)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        signal_path = row['signal_path']
        label = row['label']

        signal, _ = load_signals(signal_path)
        signal = signal.T[:12, :]  # [12, L]

        # --- Pad/crop to fixed length ---
        L = signal.shape[1]
        if L < FIXED_LENGTH:
            signal = np.pad(signal, ((0, 0), (0, FIXED_LENGTH - L)), mode='constant')
        else:
            signal = signal[:, :FIXED_LENGTH]

        # --- AUGMENTATION ---
        if self.augment and label == 0:
            signal = self.augment_signal(signal)

        signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)  # [1, 12, L]
        label = torch.tensor(float(label), dtype=torch.float32)  # shape: []
        return signal, label

    def augment_signal(self, signal):
        # Add small Gaussian noise
        noise = np.random.normal(0, 0.01, size=signal.shape)
        signal += noise

        # Optional: amplitude scaling (random between 0.9x and 1.1x)
        factor = np.random.uniform(0.9, 1.1)
        signal *= factor

        # Optional: random circular shift (time-wise)
        shift = np.random.randint(-100, 100)
        signal = np.roll(signal, shift=shift, axis=1)

        return signal

    def get_signal_path(self, idx):
        return self.df.iloc[idx]['signal_path']


# Modify train_model to use the feature extraction function
from copy import deepcopy

def train_model(data_folder, model_folder, verbose):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Using device: {device}")

    patience = 7
    min_delta = 0.001


    dataset = ECGFromCSV(data_folder, augment=True)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    models = {
        "audio_classifier": AudioClassifier().to(device),
        "crnn": CRNN().to(device),
        "resnet": ResNet().to(device),
    }

    for model_name, model in models.items():
        print(f"Training {model_name} model...")
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()

        best_loss = float('inf')
        best_model = None
        epochs_without_improvement = 0

        for epoch in range(100):
            model.train()
            total_loss = 0

            for idx, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                record_path = dataset.df.iloc[idx]['signal_path']
                age, sex_one_hot_encoding, signal_mean, signal_std = extract_features(record_path)

                age_tensor = torch.tensor(np.array(age), dtype=torch.float32).reshape(1, -1).to(device)
                sex_tensor = torch.tensor(np.array(sex_one_hot_encoding), dtype=torch.float32).reshape(1, -1).to(device)
                signal_mean_tensor = torch.tensor(np.array(signal_mean), dtype=torch.float32).reshape(1, -1).to(device)
                signal_std_tensor = torch.tensor(np.array(signal_std), dtype=torch.float32).reshape(1, -1).to(device)

                x1 = torch.cat([age_tensor, sex_tensor, signal_mean_tensor, signal_std_tensor], dim=1)

                out = model(x, x1) if model_name == 'resnet' else model(x)
                out = out.view(-1)
                loss = criterion(out, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            if verbose:
                print(f"{model_name} - Epoch {epoch+1}, Loss: {avg_loss:.4f}")

            # Early stopping
            if best_loss - avg_loss > min_delta:
              best_loss = avg_loss
              best_model = deepcopy(model.state_dict())
              epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"⏹️ Early stopping at epoch {epoch+1}")
                    break

        model.load_state_dict(best_model)
        os.makedirs(model_folder, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(model_folder, f'{model_name}_model.pt'))
        print(f"✅ Saved {model_name} after training.")

    print("🎉 Training completed for all models!")



# Modify train_model to use the feature extraction function
#def train_model(data_folder, model_folder, verbose):
#    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#    
#    print(f"💻 Using device: {device}")
#
#    from copy import deepcopy
#
#    patience = 3
#    best_loss = float('inf')
#    epochs_without_improvement = 0
#
#    dataset = ECGFromCSV(data_folder, augment=True)
#    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
#
#
#    # List of models to train independently
#    models = {
#
#        
#        "audio_classifier": AudioClassifier().to(device),
#        "crnn": CRNN().to(device),
#        "resnet": ResNet().to(device),
#        
#        
#    }
#
#    # Loss and optimizer
#    criterion = nn.BCEWithLogitsLoss()
#    optimizer = optim.Adam(models['audio_classifier'].parameters(), lr=1e-3)  # Initial optimizer setup
#
#    for model_name, model in models.items():
#        print(f"Training {model_name} model...")
#        optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Recreate optimizer for each model
#        criterion = nn.BCEWithLogitsLoss()
#
#        # Training loop
#        for epoch in range(10):  # Number of epochs
#            model.train()
#            total_loss = 0
#
#            for idx, (x, y) in enumerate(train_loader):
#                x, y = x.to(device), y.to(device)  # Move data to the device (GPU/CPU)
#                
#                # Get the path of the current record (this is from the dataset)
#                record_path = dataset.df.iloc[idx]['signal_path']  # Accessing the path correctly from the DataFrame
#                
#                # Extract wide features using the pre-made function
#                # Modify this line in train_model
#                age, sex_one_hot_encoding, signal_mean, signal_std = extract_features(record_path)
#
#                age = float(age)  # <-- ensure it's not a NumPy array
#
#
#                # Convert each feature to a tensor and ensure they have the correct shapes
#                age_tensor = torch.tensor([[age]], dtype=torch.float32).to(device)  # Shape: [1, 1]
#                #print(age_tensor.shape)
#                sex_tensor = torch.tensor(np.array(sex_one_hot_encoding), dtype=torch.float32).unsqueeze(0).to(device)
#                signal_mean_tensor = torch.tensor([signal_mean], dtype=torch.float32).to(device)   # Shape: [1, 12]
#                #print(signal_mean_tensor.shape)
#                signal_std_tensor = torch.tensor([signal_std], dtype=torch.float32).to(device)     # Shape: [1, 12]
#                #print(signal_std_tensor.shape)
#
#                # Now concatenate them along dim=1 (features axis)
#                x1 = torch.cat([age_tensor, sex_tensor, signal_mean_tensor, signal_std_tensor], dim=1)  # Concatenate along features axis
#                
#                # Print the shape of the concatenated result
#                #print(f"Shape of concatenated x1: {x1.shape}")
#
#                # For the ResNet model, pass both x and x1
#                
#                out = model(x, x1) if model_name == 'resnet' else model(x)
#                out = out.view(-1)
#
#
#                loss = criterion(out, y)  # logits go directly into BCEWithLogitsLoss  # Calculate loss
#                optimizer.zero_grad()  # Zero the gradients before backprop
#                loss.backward()  # Backpropagate the loss
#                optimizer.step()  # Update model parameters
#
#                total_loss += loss.item()  # Accumulate loss
#
#            if verbose:
#                print(f"{model_name} - Epoch {epoch+1}, Loss: {total_loss:.4f}")
#
#        # Save the model after training
#        os.makedirs(model_folder, exist_ok=True)
#        torch.save(model.state_dict(), os.path.join(model_folder, f'{model_name}_model.pt'))
#
#    print("Training completed for all models!")



# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, models, verbose):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    signal, _ = load_signals(record)
    signal = signal.T[:12, :]  # Only use the first 12 channels

    L = signal.shape[1]
    if L < FIXED_LENGTH:
        signal = np.pad(signal, ((0, 0), (0, FIXED_LENGTH - L)), mode='constant')
    else:
        signal = signal[:, :FIXED_LENGTH]

    signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 12, L]

    # Extract wide features
    age, sex_one_hot_encoding, signal_mean, signal_std = extract_features(record)

    age = float(age)
    age_tensor = torch.tensor([[age]], dtype=torch.float32).to(device)
    sex_tensor = torch.tensor([sex_one_hot_encoding], dtype=torch.float32).to(device)
    signal_mean_tensor = torch.tensor([signal_mean], dtype=torch.float32).to(device)
    signal_std_tensor = torch.tensor([signal_std], dtype=torch.float32).to(device)

    x1 = torch.cat([age_tensor, sex_tensor, signal_mean_tensor, signal_std_tensor], dim=1)

    # 🧠 Define model weights based on training loss
    weights = {
        "audio_classifier": 0.45,
        "crnn": 0.10,         # Lower weight due to higher loss
        "resnet": 0.45
    }

    model_predictions = {}
    weighted_probs = []
    total_weight = 0.0
    
    with torch.no_grad():
        for model_name, model in models.items():
            if model_name == 'resnet':
                output = model(signal, x1)
            else:
                output = model(signal)
    
            prob = torch.sigmoid(output).item()
            pred = int(prob >= 0.5)
    
            model_predictions[model_name] = (pred, prob)
    
            weighted_probs.append(prob * weights[model_name])
            total_weight += weights[model_name]
    
    # Compute final prediction *only after all are done*
    avg_prob = sum(weighted_probs) / total_weight
    final_pred = int(avg_prob >= 0.5)
    




    if verbose:
        print("🔍 Model predictions (weighted):")
        for name, (p, pr) in model_predictions.items():
            print(f"{name:>15}: pred={p}, prob={pr:.4f}, weight={weights[name]:.2f}")
        print(f"\n🧩 Weighted avg prob: {avg_prob:.4f} → Final Pred: {final_pred}")

    return final_pred, avg_prob






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
            signal_mean[i] = np.nanmean(signal[:, i])  # ✅ mean of channel i

        else:
            signal_mean[i] = 0.0
        if num_finite_samples > 1:
            signal_std[i] = np.nanstd(signal[:, i])  # ✅ std of channel i

        else:
            signal_std[i] = 0.0

    # Return only the necessary features: age, sex_one_hot_encoding, signal_mean, and signal_std
    return age, sex_one_hot_encoding, signal_mean, signal_std



# Load your trained models
def load_model(model_folder, verbose):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models = {
        "audio_classifier": AudioClassifier().to(device),
        "crnn": CRNN().to(device),
        "resnet": ResNet().to(device),
    }

    loaded_models = {}
    for model_name, model in models.items():
        model.load_state_dict(torch.load(os.path.join(model_folder, f'{model_name}_model.pt'), map_location=device))
        model.eval()  # Set the model to evaluation mode
        loaded_models[model_name] = model

    return loaded_models



# Save your trained model.
def save_model(model_folder, model):
    d = {'model': model}
    filename = os.path.join(model_folder, 'model.pt')
    joblib.dump(d, filename, protocol=0)
