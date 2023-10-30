# Training Hyperparameters
BATCH_SIZE = 128
# ! change
INPUT_SIZE = 144
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100

# Logger
# ! change
PROJECT_NAME = "icare-final"
# ! change
MODEL_NAME = "transformer"
# ! change
FEATURES = "ae"
BEST_MODEL_NAME = f"best-{MODEL_NAME}-{FEATURES}"

# Dataset
# ! change
DATA_DIR = "/media/hdd1/i-care/6h-ae"
LABEL_DIR = "/home/bc299/icare/artifacts/patient_data.csv"