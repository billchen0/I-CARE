from pathlib import Path
import config as config
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from dataset import ManualFeatureDataModule
from model import BiLSTMClassifierModule
from transformer import TransformerClassifierModule
import wandb


def main():
    # Initialize wandb run
    run = wandb.init(project=config.PROJECT_NAME)
    hyperparameters = f"{config.MODEL_NAME}_{config.FEATURES}"
    run.name = hyperparameters
    run.save()

    root_dir = Path(config.DATA_DIR)
    labels_csv = Path(config.LABEL_DIR)
    # Setup data module
    dm = ManualFeatureDataModule(root_dir, labels_csv, batch_size=config.BATCH_SIZE)
    dm.setup()
    ### BiLSTM Model
    # Setup model with hyperparameter from WandB config
    # hidden_size = wandb.config.hidden_size
    # num_layers = wandb.config.num_layers
    # dropout = wandb.config.dropout
    # model = BiLSTMClassifierModule(input_size=config.INPUT_SIZE,
    #                                hidden_size=hidden_size,
    #                                num_layers=num_layers,
    #                                dropout=dropout,
    #                                learning_rate=config.LEARNING_RATE
    #                                )
    ### Transformer Model
    d_model = wandb.config.d_model
    num_layers = wandb.config.num_layers
    nhead = wandb.config.nhead
    dropout=wandb.config.dropout
    model = TransformerClassifierModule(input_size=config.INPUT_SIZE,
                                        d_model=d_model,
                                        nhead=nhead,
                                        num_layers=num_layers,
                                        dropout=dropout,
                                        learning_rate=config.LEARNING_RATE)
    ### TCN Model
    # Setup logger and callbacks
    logger = WandbLogger(project=config.PROJECT_NAME)
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=20, verbose=True, mode="min")
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

    # Close the WandB run
    # wandb.finish()


if __name__ == "__main__":
    main()