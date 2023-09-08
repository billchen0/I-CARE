import torch 
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from util import transform_to_binary_labels


class ManualFeatureDataset(Dataset):
    def __init__(self, root_dir, labels_csv, patient_ids=None):
        self.root_dir = root_dir
        self.labels_df = pd.read_csv(labels_csv, dtype={"Patient": str}, index_col=0)["CPC"]
        self.labels_df = self.labels_df[self.labels_df.index.isin(patient_ids)]
        self.labels_df = self.labels_df.apply(lambda x: transform_to_binary_labels(x))
        self.patient_ids = self.labels_df.index.tolist() if patient_ids is None else patient_ids

    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        patient_dir = self.root_dir / patient_id
        patient_files = sorted([f for f in patient_dir.iterdir()])

        data = []
        for file in patient_files:
            features = np.load(file)
            data.append(features)
        data = torch.tensor(data).float()

        label = torch.tensor(self.labels_df.loc[patient_id])

        return data, label