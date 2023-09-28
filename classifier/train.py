from pathlib import Path
import config as config
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
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
                                   hidden_size=128,
                                   num_layers=10,
                                   dropout=0.5,
                                   learning_rate=config.LEARNING_RATE
                                   )
    logger = WandbLogger(project=config.PROJECT_NAME, name=config.RUN_NAME)
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=10, verbose=True, mode="min")
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", 
                                          dirpath="checkpoints", 
                                          filename=config.BEST_MODEL_NAME, 
                                          save_top_k=1, 
                                          mode="min")
    trainer = Trainer(max_epochs=config.NUM_EPOCHS, 
                      logger=logger,
                      callbacks=[early_stop_callback, checkpoint_callback])
    # Train and test model
    trainer.fit(model, dm)
    trainer.test(model, dm)


if __name__ == "__main__":
    main()