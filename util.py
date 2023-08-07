import re
from pathlib import Path
import pandas as pd
from scipy.io import loadmat


### I/O Helper Functions ###

# Function to load patient clinical data
def load_clinical_data(root_path: Path, patient_id: str):
    load_path = root_path / patient_id
    data_file = list(load_path.glob("*.txt"))[0]
    with open(data_file, "r") as f:
        data = f.read().split("\n")[:-1]

    return data


# Function to randomly load single patient EEG data
def load_single_patient_eeg(root_path: Path, patient_id: str):
    load_path = root_path / patient_id
    # Find all EEG .mat files in patient directory
    eeg_segment = list(load_path.glob("*EEG.mat"))[0]
    eeg_signal = loadmat(eeg_segment)["val"]

    return eeg_signal


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


### Plotting Helper Functions ###


### Others ###

# Function to find all patient id in training set
def list_patient_id(root_path: Path):
    patient_list = []
    patient_dirs = [p for p in root_path.iterdir() if re.fullmatch(r"\d{4}", p.name)]
    for patient in patient_dirs:
        patient_list.append(patient.name)

    return patient_list


# Functions to randomly sample a patient id
def sample_patient(count):
    pass