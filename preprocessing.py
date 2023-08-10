def match_channels(signal, channels):
    # Define the standard order of the 19-channel configuration
    standard_channels = ["Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "Cz", "C3", "C4",
                         "T3", "T4", "T5", "T6", "Pz", "P3", "P4", "O1", "O2"]
    # Create dictionary to map channel names to their indices
    order_dict = {ch: idx for idx, ch in enumerate(channels)}
    # Reorder the channels in signal based on standard order
    reordered_signal = signal[:, [order_dict[ch] for ch in standard_channels]]
    
    return reordered_signal