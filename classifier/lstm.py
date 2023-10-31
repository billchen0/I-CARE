import torch
import wandb
import lightning.pytorch as pl
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import (BinaryAUROC,
    BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score)
from sklearn.metrics import roc_curve

class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_classes=2):
        super().__init__()

        # BiLISTM Layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        # Dropout layer
        self.dropout = nn.Dropout()

        # Fully connected layer
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        # Only take the last output for classification
        out = out[:, -1, :]
        out = self.fc(out)

        return F.log_softmax(out, dim=1)
    

class BiLSTMClassifierModule(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, dropout, learning_rate):
        super().__init__()

        self.model = BiLSTMClassifier(input_size, hidden_size, num_layers, dropout)
        self.learning_rate = learning_rate
        self.test_step_outputs = []
        self.save_hyperparameters()
        # Initialize performance metrics
        metrics = MetricCollection([BinaryAccuracy(),
                                    BinaryAUROC(),
                                    BinaryRecall(),
                                    BinaryPrecision(),
                                    BinaryF1Score()])
        self.train_metrics = metrics.clone(prefix="Training ")
        self.val_metrics = metrics.clone(prefix="Validation ")

        # Setup per 6h evaluation on testing set
        self.epoch_names = [str(x) for x in range(12, 72+1, 6)]
        self.test_epoch_auc = []
        self.test_epoch_acc = []
        self.test_epoch_f1 = []

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
            accuracy = BinaryAccuracy()
            f1score = BinaryF1Score()
            auroc = BinaryAUROC()
            test_auroc = auroc(preds, y)
            test_accuracy = accuracy(preds, y)
            test_f1score = f1score(preds, y)
            self.test_epoch_auc.append(test_auroc)
            self.test_epoch_acc.append(test_accuracy)
            self.test_epoch_f1.append(test_f1score)
            # Compute and save ROC curves
            fpr, tpr, thresholds = roc_curve(y, logits, pos_label=1)
            table_data = list(zip(fpr, tpr, thresholds))
            table = wandb.Table(data=table_data, columns=["TPR", "FPR", "Thresholds"])
            wandb.log({f"ROC_{epoch_name}": table})
        # Free memory
        self.test_step_outputs.clear()
    
    def on_test_end(self):
        # Plot the evaluation metrics for each epoch
        auroc = [[epoch, value] for (epoch, value) in zip(self.epoch_names, self.test_epoch_auc)]
        auroc_table = wandb.Table(data=auroc, columns=["epoch", "test_auc"])
        acc = [[epoch, value] for (epoch, value) in zip(self.epoch_names, self.test_epoch_acc)]
        acc_table = wandb.Table(data=acc, columns=["epoch", "test_acc"])
        wandb.log({"test_acc_table": acc_table})
        f1 = [[epoch, value] for (epoch, value) in zip(self.epoch_names, self.test_epoch_f1)]
        f1_table = wandb.Table(data=f1, columns=["epoch", "test_f1"])
        wandb.log({"test_f1_table": f1_table})
        wandb.log({
            "test_auc_per_epoch": wandb.plot.line(auroc_table, "epoch", "test_auc", title="AUROC at 6h Epochs")
        })
        # Plot the predicted probability for each epoch
        pred_probs = [[epoch, prob] for epoch, prob in zip(self.epoch_names, self.test_pred_probs)]
        table_probs = wandb.Table(data=pred_probs, columns=["epoch", "predicted_probability"])
        wandb.log({
            "predicted_prob_per_epoch": wandb.plot.line(table_probs, 
                                                        "epoch", 
                                                        "predicted_probability", 
                                                        title="Predicted Probability at 6h Epochs")
        })

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr=self.learning_rate, 
                                     weight_decay=1e-5)
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