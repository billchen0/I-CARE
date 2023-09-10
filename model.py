import torch.nn as nn


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

        self.log_softmax = nn.LogSoftmax()

    def forward(self, x):
        out, _ = self.lstm(x)
        # Only take the last output for classification
        out = out[:, -1, :]
        out = self.fc(out)

        return self.log_softmax(out, dim=1)