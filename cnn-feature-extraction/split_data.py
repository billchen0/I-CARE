from pathlib import Path
import numpy as np


path_to_data = Path("/media/hdd1/i-care/processed")
save_path = Path("/media/hdd1/i-care/10s")
fs = 100
seg_len = 10 #seconds
data_per_seg = fs * seg_len

# Loop through the participants
for patient in path_to_data.iterdir():
    patient_id = patient.name

    for hour_file in patient.iterdir():
        hour_id = hour_file.stem.split("_")[2]
        hour_data = np.load(hour_file)

        num_segments = len(hour_data) // data_per_seg
        segments = np.array_split(hour_data, num_segments)

        save_folder = save_path / patient_id
        save_folder.mkdir(parents=True, exist_ok=True)

        for idx, segment in enumerate(segments):
            filename = f"{patient_id}_{hour_id}_{idx:03}_EEG.npy"
            filepath = save_folder / filename
            np.save(filepath, segment)