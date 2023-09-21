import lightning.pytorch as pl
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import sys
sys.path.append("..")
from util import load_split_ids

path_to_split = Path("/home/bc299/icare/artifacts")
train_ids, val_ids, test_ids = load_split_ids(path_to_split)

print(len(train_ids))
print(len(val_ids))
print(len(test_ids))