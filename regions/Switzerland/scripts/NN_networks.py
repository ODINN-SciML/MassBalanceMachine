import torch.nn as nn
import torch


class BiggerNetwork(nn.Module):
    def __init__(self, input_dim, layer0, layer1, layer2, layer3, dropout):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(input_dim, layer0), nn.ReLU(),
                                   nn.Dropout(dropout),
                                   
                                   nn.Linear(layer0, layer1), nn.ReLU(),
                                   nn.Dropout(dropout),
                                   
                                   nn.Linear(layer1, layer2), nn.ReLU(),
                                   nn.Dropout(dropout),
                                   
                                   nn.Linear(layer2, layer3), nn.ReLU(),
                                   nn.Dropout(dropout), nn.Linear(layer3, 1))

    def forward(self, x):
        return self.model(x)


class FlexibleNetwork(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout=0.2, use_batchnorm=False):
        """
        input_dim: int, number of input features
        hidden_layers: list of int, sizes of each hidden layer
        dropout: float, dropout probability for all layers (or a list for per-layer dropout)
        use_batchnorm: bool, whether to use batch normalization after each layer
        """
        super().__init__()

        layers = []
        current_dim = input_dim

        if not isinstance(dropout, list):
            dropout = [dropout] * len(hidden_layers)

        for i, (hidden_dim, drop_rate) in enumerate(zip(hidden_layers, dropout)):
            layers.append(nn.Linear(current_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(drop_rate))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, 1))  # Output layer

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class SimpleModel(nn.Module):

    def __init__(self, input_dim, layer0=16, layer1=8):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, layer0),
            nn.ReLU(),
            nn.Linear(layer0, layer1),
            nn.ReLU(),
            nn.Linear(layer1, 1),
        )

    def forward(self, x):
        return self.model(x)
    