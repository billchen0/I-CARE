# Training Hyperparameters
BATCH_SIZE = 128
INPUT_SIZE = 14
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100

# Logger
PROJECT_NAME = "icare"
MODEL_NAME = "transformer"
FEATURES = "eeg+ecg"
BEST_MODEL_NAME = "best-transformer-eeg+ecg"

# Dataset
DATA_DIR = "/media/nvme1/icare-data/6h-combined"
LABEL_DIR = "/home/bc299/icare/artifacts/patient_data.csv"