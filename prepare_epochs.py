from pathlib import Path
import numpy as np


def get_unique_hour_files(path):
    all_files = sorted(list(path.iterdir()))
    seen_hours = set()
    unique_hour_files = []
    for file in all_files:
        hour = file.stem.split("_")[2]
        if hour not in seen_hours:
            unique_hour_files.append(file)
            seen_hours.add(hour)
        
    return unique_hour_files


# Group the hours list with selected 6 hour epochs
def segment_hours_into_epochs(hour_list):
    # Initialize a list of 12 epochs with None
    epochs = [None] * 12
    
    start_idx = None
    current_epoch = None

    for idx, hour in enumerate(hour_list):
        # Determine which epoch the hour belongs to
        epoch_idx = hour // 6

        # If we're starting a new epoch or at the beginning of the list
        if current_epoch is None or current_epoch != epoch_idx:
            # If we have a starting index, this means we just finished an epoch
            if start_idx is not None:
                epochs[current_epoch] = (start_idx, idx - 1)
            start_idx = idx
            current_epoch = epoch_idx

    # Handle the case for the last hour in the list
    if start_idx is not None:
        epochs[current_epoch] = (start_idx, len(hour_list) - 1)
    
    return epochs


def combine_arrays(arr_list):
    # Initialize the result array with nan values
    combined_arr = np.full((8, 72), np.nan)
    # Calculate the position to start inserting arrays
    start_col = 72 - (len(arr_list) * 12)
    
    for arr in arr_list:
        if arr.shape[1] < 12:
            padding_cols = 12 - arr.shape[1]
            arr = np.hstack([arr, np.full((8, padding_cols), np.nan)])
        
        # Insert into the combined array
        combined_arr[:, start_col:start_col+12] = arr
        start_col += 12
    
    return combined_arr


def find_nearrest_valid(array_list, index, position):
    # Search backwards
    for i in range(index-1, -1, -1):
        if not np.isnan(array_list[i][position]):
            return array_list[i][position]
    
    # Search forwards
    for i in range(index+1, len(array_list)):
        if not np.isnan(array_list[i][position]):
            return array_list[i][position]
    
    # If no valid value is found, return 0
    return 0


def main():
    path_to_data = Path("/media/nvme1/icare-data/features")
    for patient in path_to_data.iterdir():
        files = get_unique_hour_files(patient)
        hours = [int(f.stem.split("_")[2]) for f in files]
        # Remove features above 72 hours to index error.
        hours = [h for h in hours if h < 72]
        epoch_idx = segment_hours_into_epochs(hours)

        all_epoch_features = []
        for epoch in epoch_idx:
            if epoch is not None:
                feature_files = files[epoch[0]:epoch[1]+1]
                features = []
                for feat_file in feature_files:
                    feat = np.load(feat_file)
                    features.append(feat)
                combined_features = combine_arrays(features)
                all_epoch_features.append(combined_features)
            # If the current epoch doesn't have any data, append an empty array
            else:
                all_epoch_features.append(np.full((8, 72), np.nan))
        
        for i, epoch in enumerate(all_epoch_features):
            nan_positions = np.argwhere(np.isnan(epoch))
            
            for position in nan_positions:
                valid_value = find_nearrest_valid(all_epoch_features, i, tuple(position))
                epoch[tuple(position)] = valid_value

            # If it's not the first array, create the concatenated array
            if i >= 1:
                # Find the average of all previous arrays
                avg_previous_epochs = np.mean(np.stack(all_epoch_features[:i]), axis=0)
                # Concatenate the average array with the current array
                concat_epoch = np.hstack((avg_previous_epochs, epoch))
                # Save the combined epochs to designated path
                save_path = Path(f"/media/nvme1/icare-data/6h-features/{patient.name}")
                save_path.mkdir(parents=True, exist_ok=True)
                if i == 1:
                    filename = f"0{i*6}_{i*6+6}_features.npy"
                else:
                    filename = f"{i*6}_{i*6+6}_features.npy"
                np.save(save_path / filename, concat_epoch)


if __name__ == "__main__":
    main()