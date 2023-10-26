# Training Hyperparameters
BATCH_SIZE = 128
# ! change
INPUT_SIZE = 296
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100

# Logger
# ! change
PROJECT_NAME = "icare-final"
# ! change
MODEL_NAME = "bilstm"
# ! change
FEATURES = "combined"
BEST_MODEL_NAME = f"best-{MODEL_NAME}-{FEATURES}"

# Dataset
# ! change
DATA_DIR = "/media/hdd1/i-care/6h-combined"
LABEL_DIR = "/home/bc299/icare/artifacts/patient_data.csv"