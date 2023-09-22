from pathlib import Path
import ast
import numpy as np


path_to_data = Path("/media/hdd1/i-care/processed")
save_path = Path("/media/hdd1/i-care/10s")
fs = 100
seg_len = 10 #seconds
data_per_seg = fs * seg_len

# Get the list of patients to include in this program
include_patients = ['0858', '0726', '0955', '0760', '0866', '0526', '0483', '0741', '0395', '0975', '0375', '0894', '0612', '0591', '0332', '0573', '0433', '0873', '0442', '0377', '0796', '0981', '0937', '0565', '0452', '0971', '0794', '0847', '0813', '0766', '0571', '0672', '0907', '0772', '0892', '0518', '0781', '0951', '0614', '0637', '0490', '0960', '0411', '0469', '0353', '0750', '0428']

# Loop through the participants
for patient in path_to_data.iterdir():
    if patient.is_dir() and patient.name in include_patients:
        patient_id = patient.name

        for hour_file in patient.iterdir():
            hour_id = hour_file.stem.split("_")[2]
            hour_data = np.load(hour_file)

            num_segments = len(hour_data) // data_per_seg
            try:
                segments = np.array_split(hour_data, num_segments)
            except ValueError:
                print(f"Encountered number section=0 error, patient id: {patient_id}")
                continue

            save_folder = save_path / patient_id
            save_folder.mkdir(parents=True, exist_ok=True)

            for idx, segment in enumerate(segments):
                filename = f"{patient_id}_{hour_id}_{idx:03}_EEG.npy"
                filepath = save_folder / filename
                np.save(filepath, segment)