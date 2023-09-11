import torch 
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
from sklearn.model_selection import train_test_split
from util import transform_to_binary_labels


class ManualFeatureDataset(Dataset):
    def __init__(self, root_dir, labels_csv, patient_ids=None):
        self.root_dir = root_dir
        self.labels_df = pd.read_csv(labels_csv, dtype={"Patient": str}, index_col=0)["CPC"]
        self.labels_df = self.labels_df[self.labels_df.index.isin(patient_ids)]
        # Preapre the patient_ids list
        self.patient_ids = self.labels_df.index.tolist()
        self.data = []
        self.labels = []

        for patient_id in self.patient_ids:
            patient_dir = self.root_dir / patient_id
            patient_files = sorted([f for f in patient_dir.iterdir()])
            for file in patient_files:
                features = np.load(file)
                self.data.append(features)
                self.labels.append(transform_to_binary_labels(self.labels_df.loc[patient_id]))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx]).float()
        label = torch.tensor(self.labels[idx])
        return data, label
    

class ManualFeatureDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, labels_csv, batch_size=32):
        super().__init__()
        self.root_dir = root_dir
        self.labels_csv = labels_csv
        self.batch_size = batch_size

    def setup(self, stage: str=None):
        # List all patient ids
        all_patient_ids = [dir_.name for dir_ in self.root_dir.iterdir()]
        train_ids, temp_ids = train_test_split(all_patient_ids, test_size=0.3, random_state=42)
        val_ids, test_ids = train_test_split(temp_ids, test_size=2/3, random_state=42)

        # Create datasets for each split based on patient ids
        self.train_dataset = ManualFeatureDataset(self.root_dir, 
                                                  self.labels_csv, 
                                                  patient_ids=train_ids)
        self.val_dataset = ManualFeatureDataset(self.root_dir, 
                                                self.labels_csv, 
                                                patient_ids=val_ids)
        self.test_dataset = ManualFeatureDataset(self.root_dir, 
                                                 self.labels_csv, 
                                                 patient_ids=test_ids)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size)
