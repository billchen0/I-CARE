import re
import bisect
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import wfdb


### I/O Helper Functions ###

# Function to load patient clinical data
def load_clinical_data(root_path: Path, patient_id: str):
    load_path = root_path / patient_id
    data_file = list(load_path.glob("*.txt"))[0]
    with open(data_file, "r") as f:
        data = f.read().split("\n")[:-1]

    return data


def get_segments_by_hour(load_path: Path, start_hour: int, end_hour: int):
    # Obtain all hourly segements for the participant
    all_files = sorted(list(load_path.iterdir()))
    # Get the hours for each element in the list
    all_hours = [int(f.name.split("_")[2]) for f in all_files]
    # Find the start and end indices
    start_index = bisect.bisect_left(all_hours, start_hour)
    end_index = bisect.bisect_right(all_hours, end_hour)
    # Subset the files to find segments within determined index
    selected = all_files[start_index:end_index]
    
    return selected
    

### Compile Data Helper Functions ###

def compile_clinical_data(root_path: Path):
    patients = list_patient_id(root_path)
    patient_data = []
    for patient in patients:
        # Add patient metadata as a list to the data list
        patient_data.append(load_clinical_data(root_path, patient))

    # Extract key and values from each metadata list
    formatted_patient_data = []
    for p_data in patient_data:
        formatted_p_data = {item.split(":")[0].strip(): item.split(":")[1].strip() for item in p_data}
        formatted_patient_data.append(formatted_p_data)
    
    # Create dataframe from formatted data and convert numeric columns to appropriate type
    df = pd.DataFrame(formatted_patient_data)
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["TTM"] = pd.to_numeric(df["TTM"], errors="coerce")
    df["ROSC"] = pd.to_numeric(df["ROSC"], errors="coerce")

    return df


def compile_signal_metadata(root_path: Path):
    patients = list_patient_id(root_path)
    # Store metadata dictionaries
    all_eeg_metadata = []
    for patient in patients:
        p_path = root_path / patient
        metadata_dict = {}
        # Store duration data for each segment
        durations = []
        for segment in sorted(set([p.stem for p in p_path.glob("*EEG.mat")])):
            seg_path = p_path / segment
            eeg_metadata = wfdb.io.rdheader(seg_path)
            # Calculate duration of segment using signal length and fs
            duration = (eeg_metadata.sig_len / eeg_metadata.fs) / 60 #[minutes]
            durations.append(duration)
        
        # Add metadata that is constant for different segments
        metadata_dict["Patient"] = eeg_metadata.record_name[:4]
        metadata_dict["Channels"] = eeg_metadata.sig_name
        metadata_dict["Channel Count"] = len(eeg_metadata.sig_name)
        metadata_dict["Utility Frequency"] = int(eeg_metadata.comments[0].split(" ")[2])
        metadata_dict["Fs"] = eeg_metadata.fs
        metadata_dict["EEG Duration"] = int(eeg_metadata.record_name[5:8])
        metadata_dict["Avg Duration"] = sum(durations) / len(durations)
        metadata_dict["Max Duration"] = max(durations)
        metadata_dict["Min Duration"] = min(durations)
        all_eeg_metadata.append(metadata_dict)
    
    # Convert list of dictionary to dataframe
    metadata_df = pd.DataFrame(all_eeg_metadata)

    return metadata_df


### Plotting Helper Functions ###

def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        # Adjust the col_width to make the first column wider
        size = (np.array(data.shape[::-1]) + np.array([2, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    mpl_table = ax.table(cellText=data.reset_index().values, bbox=bbox, colLabels=[''] + list(data.columns), **kwargs)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
    return ax.get_figure(), ax

### Others ###

# Function to find all patient id in training set
def list_patient_id(root_path: Path):
    patient_list = []
    patient_dirs = [p for p in root_path.iterdir() if re.fullmatch(r"\d{4}", p.name)]
    for patient in patient_dirs:
        patient_list.append(patient.name)

    return list(patient_list)


# Function to pad sequence to largest sequence length
def pad_sequence(arr_list: List[np.array], pad_value=0):
    # Calculate max sequence length
    max_len = max(arr.shape[1] for arr in arr_list)
    
    padded_arr_list = []
    for arr in arr_list:
        # Calculate the number of positions to pad
        padding_length = max_len - arr.shape[1]
        # Create new array filled with NaN
        padding_arr = np.full((arr.shape[0], padding_length), np.nan)
        padded_array = np.concatenate([arr, padding_arr], axis=1)
        padded_arr_list.append(padded_array)

    return padded_arr_list