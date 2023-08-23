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
    # Compute epileptiform discharge frequency
    # Combine into a feature vector for the segment
    features = np.concatenate([avg_band_power, [ab_ratio, avg_entropy]])

    return features


def shannon_entropy(eeg_seg: np.ndarray, num_bins: int=100):
    entropies = []
    for i, channel in enumerate(eeg_seg):
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