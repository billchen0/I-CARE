import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class EEGDataset(Dataset):
    def __init__(self, labels_csv, data_dir, patient_ids):
        self.labels_df = pd.read_csv(labels_csv, dtype={"Patient": str}, index_col=0)["CPC"]
        self.labels_df = self.labels_df[self.labels_df.index.isin(patient_ids)]
        self.data_dir = data_dir

    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        ...

