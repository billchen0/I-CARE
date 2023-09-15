from pathlib import Path
from sklearn.model_selection import train_test_split

import config
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from dataset import ManualFeatureDataModule
from model import BiLSTMClassifierModule

def main():
    root_dir = Path(config.DATA_DIR)
    labels_csv = Path(config.LABEL_DIR)
    # Setup data module
    dm = ManualFeatureDataModule(root_dir, labels_csv, batch_size=config.BATCH_SIZE)
    dm.setup()
    # Setup model module and training procedure
    model = BiLSTMClassifierModule(input_size=config.INPUT_SIZE,
                                   hidden_size=256,
                                   num_layers=10,
                                   dropout=0.3,
                                   learning_rate=config.LEARNING_RATE
                                   )
    logger = WandbLogger(project=config.PROJECT_NAME)
    trainer = Trainer(max_epochs=config.NUM_EPOCHS, logger=logger)
    # Train and test model
    trainer.fit(model, dm)
    trainer.test(model, dm)


if __name__ == "__main__":
    main()