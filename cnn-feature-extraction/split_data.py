from pathlib import Path
import numpy as np
import sys 
from concurrent.futures import ProcessPoolExecutor
sys.path.append("..")
from util import load_split_ids

def process_patient(patient):
    if patient.is_dir() and patient.name in include_patients:
        patient_id = patient.name

        for hour_file in patient.iterdir():
            hour_id = hour_file.stem.split("_")[2]
            hour_data = np.load(hour_file)

            # Splitting the data into 5-minute segments and discarding the last segment if it's shorter than 5 minutes
            segments = [hour_data[i:i+data_per_seg] for i in range(0, len(hour_data), data_per_seg) if len(hour_data[i:i+data_per_seg]) == data_per_seg]

            save_folder = save_path / patient_id
            save_folder.mkdir(parents=True, exist_ok=True)

            for idx, segment in enumerate(segments):
                filename = f"{patient_id}_{hour_id}_{idx:03}_EEG.npy"
                filepath = save_folder / filename
                np.save(filepath, segment)

path_to_data = Path("/media/hdd1/i-care/processed")
save_path = Path("/media/hdd1/i-care/five-minutes")
fs = 100
seg_len = 300 #seconds
data_per_seg = fs * seg_len

# Get the list of patients to include in this program
split_path = Path("/home/bc299/icare/artifacts")
train_ids, val_ids, test_ids = load_split_ids(split_path)
include_patients = train_ids + val_ids + test_ids

# Using ProcessPoolExecutor to parallelize the processing
with ProcessPoolExecutor(max_workers=24) as executor:
    list(executor.map(process_patient, path_to_data.iterdir()))
