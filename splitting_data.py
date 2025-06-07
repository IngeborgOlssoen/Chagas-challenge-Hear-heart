import os
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from helper_code import load_text, load_label
from imblearn.under_sampling import RandomUnderSampler

# === Configuration ===
dataset_dirs = {
    "samitrop": "samitrop_output",
    "ptbxl": "ptbxl_output",
}

output_base = Path("dataset")
train_dir = output_base / "train"
val_dir = output_base / "val"
holdout_dir = Path("holdout_data")

# Create output directories
for d in [train_dir, val_dir, holdout_dir]:
    os.makedirs(d, exist_ok=True)

# === Symlink one record's files ===
def symlink_single_record(record, dest_dir, remove_label=False):
    for ext in ['.hea', '.dat', '.txt']:
        src_file = record.with_suffix(ext)
        if src_file.exists():
            dst_file = dest_dir / src_file.name
            try:
                if not dst_file.exists():
                    os.symlink(src_file.resolve(), dst_file)
            except FileExistsError:
                pass

            if remove_label and ext == '.txt':
                text = load_text(src_file)
                text['label'] = None
                with open(dst_file, 'w') as f:
                    f.write(str(text))

# === Parallel symlinking with progress ===
def symlink_records_parallel(records, dest_dir, remove_label=False, label=""):
    os.makedirs(dest_dir, exist_ok=True)
    total = len(records)
    print(f"[{label}] Starting symlink of {total} records to {dest_dir}...")

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {
            executor.submit(symlink_single_record, record, dest_dir, remove_label): i
            for i, record in enumerate(records)
        }

        for i, future in enumerate(as_completed(futures)):
            if (i + 1) % 100 == 0 or (i + 1) == total:
                print(f"[{label}] Processed {i + 1}/{total} records")

    print(f"[{label}] Done.\n")

# === Undersample combined train set ===
def undersample_records(records):
    X, y = [], []

    for record in records:
        try:
            label = load_label(str(record))  # Convert Path to string
            print(f"{record.name} → {label}")
            if label is not None:
                X.append(record)
                y.append(label)
        except Exception as e:
            print(f"Skipping {record.name}: {e}")

    if not X:
        print("No labels found for undersampling.")
        return records

    print("Original class distribution:", Counter(y))

    if len(set(y)) < 2:
        print("Only one class found — skipping undersampling.")
        return X

    X_indices = list(range(len(X)))
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample([[i] for i in X_indices], y)

    selected_indices = [i[0] for i in X_resampled]
    undersampled_records = [X[i] for i in selected_indices]

    print("Resampled class distribution:", Counter(y_resampled))
    print(f"Undersampled: from {len(records)} to {len(undersampled_records)} records.")
    return undersampled_records

# === Main dataset split and processing ===
all_train_records = []
all_val_records = []
all_test_records = []

for dataset_name, dataset_path in dataset_dirs.items():
    dataset_path = Path(dataset_path)
    record_list = sorted([f.with_suffix('') for f in dataset_path.glob("*.hea")])
    random.shuffle(record_list)

    n = len(record_list)
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)

    train_records = record_list[:n_train]
    val_records = record_list[n_train:n_train + n_val]
    test_records = record_list[n_train + n_val:]

    print(f"\nProcessing dataset: {dataset_name}")
    all_train_records += train_records
    all_val_records += val_records
    all_test_records += test_records

# 🔁 Perform undersampling *after* merging both datasets
train_records_sampled = undersample_records(all_train_records)
symlink_records_parallel(train_records_sampled, train_dir, label="combined - train (undersampled)")
symlink_records_parallel(all_val_records, val_dir, label="combined - val")
symlink_records_parallel(all_test_records, holdout_dir, remove_label=True, label="combined - holdout")
