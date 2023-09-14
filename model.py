import torch
import torchmetrics
import lightning.pytorch as pl
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch.nn.functional as F

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
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.test_acc = torchmetrics.Accuracy(task="binary")

        self.train_auc = torchmetrics.AUROC(task="binary")
        self.val_auc = torchmetrics.AUROC(task="binary")
        self.test_auc = torchmetrics.AUROC(task="binary")

        self.train_f1 = torchmetrics.F1Score(task="binary")
        self.val_f1 = torchmetrics.F1Score(task="binary")
        self.test_f1 = torchmetrics.F1Score(task="binary")
        
        self.test_roc = torchmetrics.ROC(task="binary")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        self.train_acc(preds, y)
        self.train_auc(preds, y)
        self.train_f1(preds, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_acc", self.train_acc, prog_bar=True, on_epoch=True)
        self.log("train_auc", self.train_auc, prog_bar=True)
        self.log("train_f1", self.train_f1, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.val_acc(preds, y)
        self.val_auc(preds, y)
        self.val_f1(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        self.log("val_auc", self.val_auc, prog_bar=True)
        self.log("val_f1", self.val_f1, prog_bar=True)
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
            "labels": y,
            "batch_idx": batch_idx
        })
        return loss
    
    def on_test_epoch_end(self):
        # Create a list of epoch names based on data
        epoch_names = [str(x) for x in range(12, 72+1, 6)]
        # Store aggregated labels and predictions
        aggregated_labels = {name: [] for name in epoch_names}
        aggregated_preds = {name: [] for name in epoch_names}
        # Aggregate labels and predictions based on batch_idx (6h epochs)
        for output in self.test_step_outputs:
            batch_idx = output["batch_idx"]
            epoch_name = epoch_names[batch_idx]
            aggregated_labels[epoch_name].extend(output["labels"].cpu().numpy())
            aggregated_preds[epoch_name].extend(output["predictions"].cpu().numpy())
        # Compute and log metrics for each 6h epoch
        for epoch_name in epoch_names:
            y = torch.tensor(aggregated_labels[epoch_name])
            preds = torch.tensor(aggregated_preds[epoch_name])
            acc = accuracy_score(y, preds)
            self.test_auc(preds, y)
            self.test_f1(preds, y)
            
            self.log(f"test_acc_{epoch_name}", acc)
            self.log(f"test_auc_{epoch_name}", self.test_auc)
            self.log(f"test_f1_{epoch_name}", self.test_f1)
        # Free memory
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer