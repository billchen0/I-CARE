import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
import sys
sys.path.append("..")
from util import load_split_ids

def check_eeg_length(eeg_file):
    eeg = np.load(eeg_file)
    if eeg.shape[0] > 1000:
        eeg = eeg[:1000]  # Truncate segments larger than 1000
        np.save(eeg_file, eeg)
    elif 800 <= eeg.shape[0] < 1000:
        # Impute missing values with previous values
        diff = 1000 - eeg.shape[0]
        eeg = np.concatenate((eeg, eeg[-diff:]))
        np.save(eeg_file, eeg)
    elif eeg.shape[0] < 800:
        eeg_file.unlink()  # Remove the file
        return eeg_file.parent.name  # Return patient name for discarded segments
    return None

if __name__ == "__main__":
    print(f"Number of CPUs: {cpu_count()}")
    data_path = Path("/media/hdd1/i-care/ten-seconds")

    for patient in data_path.iterdir():
        patient_path = data_path / patient
        patient_files = list(patient_path.iterdir())
        with Pool(cpu_count()) as p:
            discarded_results = p.map(check_eeg_length, patient_files)
        
        discarded_count = len([res for res in discarded_results if res is not None])
        print(f"Processing completed for patient {patient_path.name}. {discarded_count} segments discarded.")
