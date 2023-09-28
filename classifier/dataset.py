import torch 
import pandas as pd
from pathlib import Path
import multiprocessing
import numpy as np
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
import sys 
sys.path.append("/home/bc299/icare")
from util import transform_to_binary_labels, load_split_ids


class ManualFeatureDataset(Dataset):
    def __init__(self, root_dir, labels_csv, patient_ids=None, per_epoch=False):
        self.root_dir = root_dir
        self.labels_df = pd.read_csv(labels_csv, dtype={"Patient": str}, index_col=0)["CPC"]
        self.labels_df = self.labels_df[self.labels_df.index.isin(patient_ids)]
        # Preapre the patient_ids list
        self.patient_ids = self.labels_df.index.tolist()
        self.per_epoch = per_epoch

        # For testing sets, organize the batches for each epoch
        if per_epoch:
            num_epochs = 11
            self.data = [[] for _ in range(num_epochs)]
            self.labels = [[] for _ in range(num_epochs)]
            # Loop through each patient
            for patient_id in self.patient_ids:
                patient_dir = self.root_dir / patient_id
                patient_files = sorted([f for f in patient_dir.iterdir()])
                # Loop through each file and organize them by epoch
                for i, file in enumerate(patient_files):
                    features = np.load(file)
                    self.data[i].append(features)
                    self.labels[i].append(transform_to_binary_labels(self.labels_df.loc[patient_id]))
        else:
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
        if self.per_epoch:
            return len(self.data[0]) * len(self.data)
        else:
            return len(self.data)
    
    def __getitem__(self, idx):
        if self.per_epoch:
            epoch_idx = idx // len(self.patient_ids)
            patient_idx = idx % len(self.patient_ids)
            data = torch.tensor(self.data[epoch_idx][patient_idx]).float()
            label = torch.tensor(self.labels[epoch_idx][patient_idx])
        else:
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
        # Get patient ids for train validation and test
        split_path = Path("/home/bc299/icare/artifacts")
        train_ids, val_ids, test_ids = load_split_ids(split_path)

        # Create datasets for each split based on patient ids
        self.train_dataset = ManualFeatureDataset(self.root_dir,
                                                  self.labels_csv,
                                                  patient_ids=train_ids)
        self.val_dataset = ManualFeatureDataset(self.root_dir, 
                                                self.labels_csv, 
                                                patient_ids=val_ids)
        self.test_dataset = ManualFeatureDataset(self.root_dir, 
                                                 self.labels_csv, 
                                                 patient_ids=test_ids,
                                                 per_epoch=True)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size,
                          num_workers=multiprocessing.cpu_count(),
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          num_workers=multiprocessing.cpu_count(),
                          batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          num_workers=multiprocessing.cpu_count(),
                          batch_size=len(self.test_dataset.patient_ids))
