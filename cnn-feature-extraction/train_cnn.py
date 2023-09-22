import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import numpy as np

import sys
sys.path.append("..")
from util import load_split_ids


class EEGSegmentDataset(Dataset):
    def __init__(self, data_dir, patient_ids):
        self.data_files = []
        for patient_id in patient_ids:
            patient_folder = data_dir / patient_id
            self.data_files.extend(sorted(patient_folder.iterdir()))
        
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        segment_file = self.data_files[idx]
        segment = np.load(segment_file)
        return torch.tensor(segment, dtype=torch.float32)
    

class EEGDataModule(pl.LightningDataModule):
    def __init__(self, train_ids, val_ids, data_dir, batch_size=32):
        super().__init__()
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = EEGSegmentDataset(self.data_dir, self.train_ids)
        self.val_dataset = EEGSegmentDataset(self.data_dir, self.val_ids)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    

class EEGNetAutoencoder(nn.Module):
    def __init__(self, receptive_field=64, filter_size=8, dropout=0.5, D=2):
        super().__init__()

        # Encoder
        self.temporal = nn.Sequential(
            nn.Conv2d(1, filter_size, kernel_size=[1, receptive_field], stride=1, bias=False, padding="same"),
            nn.BatchNorm2d(filter_size)
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(filter_size, filter_size * D, kernel_size=[8, 1], bias=False, groups=filter_size),
            nn.BatchNorm2d(filter_size * D),
            nn.ELU(True)
        )
        self.separable = nn.Sequential(
            nn.Conv2d(filter_size * D, filter_size * D, kernel_size=[1, 16], padding="same", groups=filter_size * D, bias=False),
            nn.Conv2d(filter_size * D, filter_size * D, kernel_size=[1, 1], padding="same", groups=1, bias=False),
            nn.BatchNorm2d(filter_size * D),
            nn.ELU(True)
        )
        self.avgpool1 = nn.AvgPool2d([1, 5], stride=[1, 5], padding=0)
        self.avgpool2 = nn.AvgPool2d([1, 5], stride=[1, 5], padding=0)
        self.dropout = nn.Dropout(dropout)

        # Decoder
        self.uppool1 = nn.Upsample(scale_factor=(1, 5), mode="nearest")
        self.uppool2 = nn.Upsample(scale_factor=(1, 5), mode="nearest")
        self.de_separable = nn.Sequential(
            nn.ConvTranspose2d(filter_size * D, filter_size * D, kernel_size=[1, 16], padding='same', groups=filter_size * D, bias=False),
            nn.ConvTranspose2d(filter_size * D, filter_size * D, kernel_size=[1, 1], padding='same', groups=1, bias=False),
            nn.BatchNorm2d(filter_size * D),
            nn.ELU(True),
        )
        self.de_spatial = nn.Sequential(
            nn.ConvTranspose2d(filter_size * D, filter_size, kernel_size=[8, 1], bias=False, groups=filter_size),
            nn.BatchNorm2d(filter_size),
            nn.ELU(True),
        )
        self.de_temporal = nn.Sequential(
            nn.ConvTranspose2d(filter_size, 1, kernel_size=[1, receptive_field], stride=1, bias=False, padding='same'),
            nn.BatchNorm2d(1),
        )
    
    def forward(self, x):
        # Encoder
        encoded = self.temporal(x)
        encoded = self.spatial(encoded)
        encoded = self.avgpool1(encoded)
        encoded = self.dropout(encoded)
        encoded = self.separable(encoded)
        encoded = self.avgpool2(encoded)
        encoded = self.dropout(encoded)
        
        # Decoder
        decoded = self.uppool1(encoded)
        decoded = self.de_separable(decoded)
        decoded = self.dropout(decoded)
        decoded = self.uppool2(decoded)
        decoded = self.de_spatial(decoded)
        decoded = self.de_temporal(decoded)

        return decoded


class EEGNetAutoencoderModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-5):
        super().__init__()
        
        self.model = EEGNetAutoencoder()
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x = batch
        reconstructed = self(x)
        loss = F.mse_loss(reconstructed, x)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        reconstructed = self(x)
        loss = F.mse_loss(reconstructed, x)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    

path_to_split = Path("/home/bc299/icare/artifacts")
train_ids, val_ids, test_ids = load_split_ids(path_to_split)

data_module = EEGDataModule(train_ids, val_ids, path_to_split)
model = EEGNetAutoencoderModel()

trainer = pl.Trainer(max_epochs=2)
trainer.fit(model, data_module)

# Save model


