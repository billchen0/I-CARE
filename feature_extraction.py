from pathlib import Path
import neurokit2 as nk
import numpy as np


def main(root_path):
    # Loop through all the patients
    load_path = root_path / "processed"
    for patient in load_path.iterdir():
        # Create a list to hold all extracted features for this patient
        all_features_list = []
        # Loop through the hourly segments for the current patient
        for hr_seg in sorted(patient.iterdir()):
            # Read in the hourly segment
            eeg_data = np.load(hr_seg)
            total_samp = eeg_data.shape[0]
            window_samp = 5*60*100
            # Loop through each segment with with a 5-min window and extract features
            for start in range(0, total_samp, window_samp):
                end = start + window_samp
                # Ignore segments that don't have the full length
                if end <= total_samp:
                    segment = eeg_data[start:end, :]
                # Extract features from the segment
                features = extract_features(segment)
                all_features_list.append(features)
        # Convert the list fo features to a Numpy array
        all_features = np.array(all_features_list).T
        # Save the features
        save_path = root_path / "features" / f"{patient.name}" 
        save_path.mkdir(parents=True, exist_ok=True)
        np.save(save_path / "manual-features.npy", all_features)



def extract_features(eeg_seg: np.ndarray):
    # Change the dimensions to (channel, timestamps)
    eeg_seg = eeg_seg.T
    # Extract power band density
    band_powers = nk.eeg_power(eeg_seg, 
                               100, 
                               frequency_band=["Delta", "Theta", "Alpha", "Beta"])
    # Average power band across channels
    avg_band_power = band_powers.mean(numeric_only=True).values
    # Add in alpha-delta ratio as a feature
    ab_ratio =  avg_band_power[2] / avg_band_power[0]
    # Compute the Shannon entropy
    avg_entropy = np.mean(shannon_entropy(eeg_seg))
    # Compute regularity
    avg_regularity = np.mean(compute_reg_multichannel(eeg_seg))
    # Compute burst supression ratio
    avg_bsr = np.mean(compute_bsr_multichannel(eeg_seg))
    # Compute epileptiform discharge frequency
    # Combine into a feature vector for the segment
    features = np.concatenate([avg_band_power, [ab_ratio, avg_entropy, avg_bsr, avg_regularity]])

    return features


def shannon_entropy(eeg_seg: np.ndarray, num_bins: int=100):
    entropies = []
    for channel in eeg_seg:
        # Split values into bin
        try:
            counts, bin_edges = np.histogram(channel, bins=num_bins)
        except ValueError:
            continue
        # Calculate the probabilities
        probs = counts / float(len(channel))
        # Filter out zero probabilites to avoid log(0)
        probs = probs[probs > 0]
        # Compute entropy for this channel
        entropy = -np.sum(probs * np.log2(probs))
        entropies.append(entropy)

    return entropies


def compute_bsr_multichannel(eeg_seg, alpha=0.01, threshold_factor=0.5):
    bsr_values = []
    for eeg_channel in eeg_seg:
        bsr_values.append(compute_bsr(eeg_channel, alpha, threshold_factor))

    return bsr_values
    

def compute_bsr(eeg_channel, alpha=0.01, threshold_factor=0.5):
    # Initialize values
    mean = eeg_channel[0]
    variance = 0
    suppressed_samples = 0

    for x_t in eeg_channel:
        # Recrusive mean estimation
        mean = (1-alpha) * mean + alpha * x_t
        # Recursive variance estimation
        variance = (1-alpha) * variance + alpha * (x_t-mean)**2
        # Threshold to classify burst vs. suppression
        if variance < threshold_factor:
            suppressed_samples += 1
    
    # Compute burst suppression ratio
    bsr = suppressed_samples / len(eeg_channel)

    return bsr


def compute_reg_multichannel(eeg_seg, sampling_rate=100):
    reg_values = []
    for eeg_channel in eeg_seg:
        reg_values.append(compute_regularity(eeg_channel, sampling_rate))

    return reg_values


def compute_regularity(eeg_channel, sampling_rate=100):
    # Square the EEG signal
    squared_eeg = np.square(eeg_channel)
    # Apply a moving-average filter with a window of 0.5s
    window_length = int(0.5 * sampling_rate)
    smoothed_eeg = np.convolve(squared_eeg, np.ones(window_length) / window_length, mode="valid")
    # Sort the smoothed signal in descending order
    sorted_eeg = np.sort(smoothed_eeg)[::-1]
    # Calculate Regularity
    N = len(sorted_eeg)
    i = np.arange(1, N+1)
    numerator = np.sum(np.square(i) * sorted_eeg)
    denominator = (1/3) * N**2 * np.sum(sorted_eeg)
    regularity = np.sqrt(numerator / denominator)

    return regularity


if __name__ == "__main__":
    root_path = Path("/media/hdd1/i-care")
    main(root_path)