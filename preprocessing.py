from pathlib import Path
from scipy.signal import butter, sosfiltfilt, resample
import numpy as np
import wfdb
from util import list_patient_id


def main(path_to_data: Path):
    for p in ["0858", "0726"]:
    #for p in list_patient_id(path_to_data):
        eeg_path = path_to_data / p
        for eeg_segment in sorted(set([seg.parent / seg.stem for seg in eeg_path.glob("*EEG.mat")])):
            # Read in EEG segment, separating the signal and the channels
            eeg_record = wfdb.io.rdrecord(eeg_segment)
            eeg_signal = eeg_record.p_signal
            eeg_channels = eeg_record.sig_name
            fs = eeg_record.fs
            # Remove excess channels and reorder the channels to a standardized order
            reordered_signal = match_channels(eeg_signal, eeg_channels)
            # Filter signal with 6th order butterworth bandpass filter
            filtered_signal = butter_bandpass_filter(reordered_signal, fs=fs)
            # Downsample the data to 100 Hz and perform z-score normalization
            fs_ds = 100
            processed_signal = downsample_normalize(filtered_signal, fs, fs_ds)
            # Save the segment
            save_path = Path(str(eeg_segment).replace("training", "processed"))
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path, processed_signal)
            print(f"{str(save_path)} saved.")


def match_channels(signal, channels):
    # Define the standard order of the 19-channel configuration
    standard_channels = ["Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "Cz", "C3", "C4",
                         "T3", "T4", "T5", "T6", "Pz", "P3", "P4", "O1", "O2"]
    # Create dictionary to map channel names to their indices
    order_dict = {ch: idx for idx, ch in enumerate(channels)}
    # Reorder the channels in signal based on standard order and select standard channels
    reordered_signal = signal[:, [order_dict[ch] for ch in standard_channels]]

    return reordered_signal


def butter_bandpass_filter(signal, fs, lowcut=0.5, highcut=30, order=6):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype="band", output="sos")
    filtered_signal = sosfiltfilt(sos, signal, axis=0)

    return filtered_signal


def downsample_normalize(signal, fs_original, fs_downsample):
    num_samples_downsampled = int(signal.shape[0] * fs_downsample / fs_original)
    downsampled_signal = resample(signal, num_samples_downsampled)
    
    # Z-score normalization
    mean = np.mean(signal, axis=0)
    std = np.std(signal, axis=0)
    norm_signal = (downsampled_signal - mean) / std

    return norm_signal


if __name__ == "__main__":
    root_path = Path("/media/hdd1/i-care/training")
    main(root_path)