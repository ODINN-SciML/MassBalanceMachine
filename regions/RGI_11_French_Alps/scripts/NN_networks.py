import torch.nn as nn
import torch


class BiggerNetwork(nn.Module):

    def __init__(self, input_dim, layer0, layer1, layer2, layer3, dropout):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, layer0),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer0, layer1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer1, layer2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer2, layer3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer3, 1),
        )

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


class ClimateTopoNet(nn.Module):
    def __init__(
        self,
        n_topo: int,
        n_climate: int,
        hidden_dim_0: int = 64,
        hidden_dim_1: int = 32,
        final_dim: int = 16,
    ):
        super().__init__()
        self.n_topo = n_topo
        self.n_climate = n_climate

        # Topographical subnetwork
        self.topo_net = nn.Sequential(
            nn.Linear(n_topo, hidden_dim_0),
            nn.ReLU(),
            nn.Linear(hidden_dim_0, hidden_dim_1),
            nn.ReLU(),
        )

        # Climate subnetwork
        self.climate_net = nn.Sequential(
            nn.Linear(n_climate, hidden_dim_0),
            nn.ReLU(),
            nn.Linear(hidden_dim_0, hidden_dim_1),
            nn.ReLU(),
        )

        # Final network after merging
        self.final_net = nn.Sequential(
            nn.Linear(2 * hidden_dim_1, final_dim),
            nn.ReLU(),
            nn.Linear(final_dim, 1),
        )

    def forward(self, x):
        # x: (batch_size, n_topo + n_climate)
        topo = x[:, : self.n_topo]
        climate = x[:, self.n_topo : self.n_topo + self.n_climate]

        topo_out = self.topo_net(topo)
        climate_out = self.climate_net(climate)

        combined = torch.cat([topo_out, climate_out], dim=1)
        return self.final_net(combined)


class PeriodSpecificNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim_0: int = 64,
        hidden_dim_1: int = 32,
        hidden_dim_2: int = 32,
    ):
        super().__init__()
        self.input_dim = input_dim

        self.shared_net = nn.Sequential(
            nn.Linear(
                input_dim - 1, hidden_dim_0
            ),  # -1 remove PERIOD_INDICATOR, its just used as a switch
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim_0, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Annual-specific branch
        self.annual_net = nn.Sequential(
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim_2, 1),
        )

        # Winter-specific branch
        self.winter_net = nn.Sequential(
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim_2, 1),
        )

    def forward(self, x):
        period_indicator = x[:, -1]  # 0=annual, 1=winter
        features = x[:, :-1]  # All features except period indicator

        # Shared feature extraction
        shared_features = self.shared_net(features)

        # Get predictions from both branches
        annual_pred = self.annual_net(shared_features)
        winter_pred = self.winter_net(shared_features)

        period_indicator = period_indicator.unsqueeze(1)  # Shape: (batch_size, 1)

        # For annual (0): use annual_pred, for winter (1): use winter_pred
        output = (1 - period_indicator) * annual_pred + period_indicator * winter_pred

        return output


class PeriodSpecificNetBigger(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim_0: int = 128,
        hidden_dim_1: int = 96,
        hidden_dim_2: int = 64,
        hidden_dim_3: int = 32,
    ):
        super().__init__()
        self.input_dim = input_dim

        self.shared_net = nn.Sequential(
            nn.Linear(
                input_dim - 1, hidden_dim_0
            ),  # -1 remove PERIOD_INDICATOR, its just used as a switch
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim_0, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Annual-specific branch
        self.annual_net = nn.Sequential(
            nn.Linear(hidden_dim_2, hidden_dim_3),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim_3, 1),
        )

        # Winter-specific branch
        self.winter_net = nn.Sequential(
            nn.Linear(hidden_dim_2, hidden_dim_3),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim_3, 1),
        )

    def forward(self, x):
        period_indicator = x[:, -1]  # 0=annual, 1=winter
        features = x[:, :-1]  # All features except period indicator

        # Shared feature extraction
        shared_features = self.shared_net(features)

        # Get predictions from both branches
        annual_pred = self.annual_net(shared_features)
        winter_pred = self.winter_net(shared_features)

        period_indicator = period_indicator.unsqueeze(1)  # Shape: (batch_size, 1)

        # For annual (0): use annual_pred, for winter (1): use winter_pred
        output = (1 - period_indicator) * annual_pred + period_indicator * winter_pred

        return output


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
