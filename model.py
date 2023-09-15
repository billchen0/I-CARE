import torch
import lightning.pytorch as pl
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import (BinaryAUROC,
    BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score)

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
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")

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
            print(f"labels: {y}")
            print(f"predictions: {preds}")
        # Free memory
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer