import os
import requests

BASE_URL = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-1.0.3"
SAVE_DIR = "ptbxl_input"
os.makedirs(SAVE_DIR, exist_ok=True)

# 2. Download selected record files from records500/
record_ids = [f"records500/{i:05d}_ld.dat" for i in range(50)]  # Adjust range as needed

total = len(record_ids)
for idx, record in enumerate(record_ids, start=1):
    print(f"Downloading {idx}/{total}: {record}")
    url = f"{BASE_URL}/{record}"
    local_path = os.path.join(SAVE_DIR, os.path.basename(record))
    r = requests.get(url)
    if r.status_code == 200:
        with open(local_path, 'wb') as f:
            f.write(r.content)
    else:
        print(f"Failed to download {record} (status code {r.status_code})")
