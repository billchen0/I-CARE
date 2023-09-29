import torch
import wandb
import lightning.pytorch as pl
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import (BinaryAUROC,
    BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score)
from sklearn.metrics import roc_curve

class TransformerClassifier(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, dropout, num_classes=2):
        super().__init__()

        self.embedding_linear = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.ReLU(),
        )
        self.embedding_conv = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        self.pe = nn.Parameter(torch.randn(1, d_model))
        self.transformer = nn.Transformer(d_model, nhead, num_layers, dropout=dropout)
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        x = self.embedding_linear(x)
        x = self.embedding_conv(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1) + self.pe
        x = x.permute(1, 0, 2)
        out = self.transformer.encoder(x)
        out = out[-1, :, :]
        out = self.fc(out)
        return F.log_softmax(out, dim=1)
    

class TransformerClassifierModule(pl.LightningModule):
    def __init__(self, input_size, d_model, nhead, num_layers, dropout, learning_rate):
        super().__init__()

        self.model = TransformerClassifier(input_size, d_model, nhead, num_layers, dropout)
        self.learning_rate = learning_rate
        self.test_step_outputs = []
        self.save_hyperparameters()
        # Initialize performance metrics
        metrics = MetricCollection([BinaryAccuracy(),
                                    BinaryAUROC(),
                                    BinaryRecall(),
                                    BinaryPrecision(),
                                    BinaryF1Score()])
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")

        # Setup per 6h evaluation on testing set
        self.epoch_names = [str(x) for x in range(12, 72+1, 6)]
        self.test_epoch_auc = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        output = self.train_metrics(preds, y)
        self.log_dict(output)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)

        output = self.val_metrics(preds, y)
        self.log_dict(output)
        self.log("val_loss", loss)
        return loss

    # The test step will evaluate the performance for each 6h epoch
    # assuming that the 6h epochs are in order and the batch size is 1.
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_step_outputs.append({
            "predictions": preds,
            "logits": logits,
            "labels": y,
            "batch_idx": batch_idx
        })
        return loss
    
    def on_test_epoch_end(self):
        # Store aggregated labels and predictions
        aggregated_labels = {name: [] for name in self.epoch_names}
        aggregated_preds = {name: [] for name in self.epoch_names}
        aggregated_logits = {name: [] for name in self.epoch_names}
        # Aggregate labels and predictions based on batch_idx (6h epochs)
        for output in self.test_step_outputs:
            batch_idx = output["batch_idx"]
            epoch_name = self.epoch_names[batch_idx]
            aggregated_labels[epoch_name].extend(output["labels"].cpu().numpy())
            aggregated_preds[epoch_name].extend(output["predictions"].cpu().numpy())
            aggregated_logits[epoch_name].extend(output["logits"][:, 1].cpu().numpy())
        # Compute and log metrics for each 6h epoch
        for epoch_name in self.epoch_names:
            y = torch.tensor(aggregated_labels[epoch_name])
            preds = torch.tensor(aggregated_preds[epoch_name])
            logits = torch.tensor(aggregated_logits[epoch_name])
            # Compute and save AUROC
            auroc = BinaryAUROC()
            test_auroc = auroc(preds, y)
            self.test_epoch_auc.append(test_auroc)
            # Compute and save ROC curves
            fpr, tpr, thresholds = roc_curve(y, logits, pos_label=1)
            table_data = list(zip(fpr, tpr, thresholds))
            table = wandb.Table(data=table_data, columns=["TPR", "FPR", "Thresholds"])
            wandb.log({f"ROC_{epoch_name}": table})
        # Free memory
        self.test_step_outputs.clear()
    
    def on_test_end(self):
        data = [[epoch, value] for (epoch, value) in zip(self.epoch_names, self.test_epoch_auc)]
        table = wandb.Table(data=data, columns=["epoch", "test_auc"])
        wandb.log({
            "test_auc_per_epoch": wandb.plot.line(table, "epoch", "test_auc", title="AUROC at 6h Epochs")
        })

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                    mode="min", 
                                                                    factor=0.1, 
                                                                    patience=5, 
                                                                    verbose=True),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1
        }
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler}