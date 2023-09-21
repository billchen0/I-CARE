import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count


def check_eeg_length(eeg_file):
    eeg = np.load(eeg_file)
    if eeg.shape[0] < 1000:
        return eeg_file.name
    return None


if __name__ == "__main__":
    print(f"Number of CPUs: {cpu_count()}")
    path_to_data = Path("/media/hdd1/i-care/10s")
    all_eeg_files = [eeg_file for patient in path_to_data.iterdir() for eeg_file in patient.iterdir()]

    with Pool(cpu_count()) as p:
        results = p.map(check_eeg_length, all_eeg_files)
    
    short_seq = [res for res in results if res is not None]

    print(f"Number of sequences that are short: {len(short_seq)}")
