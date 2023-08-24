from pathlib import Path
import neurokit2 as nk
import numpy as np

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
    # Compute burst supression ratio
    avg_bsr = np.mean(compute_bsr_multichannel(eeg_seg))
    # Compute epileptiform discharge frequency
    # Combine into a feature vector for the segment
    features = np.concatenate([avg_band_power, [ab_ratio, avg_entropy, avg_bsr]])

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

