from scipy.signal import butter, sosfiltfilt, resample
import numpy as np

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

