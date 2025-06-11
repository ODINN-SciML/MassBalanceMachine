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
    def __init__(self,
                 n_topo: int,
                 n_climate: int,
                 hidden_dim_0: int = 64,
                 hidden_dim_1: int = 32,
                 final_dim: int = 16):
        super().__init__()
        self.n_topo = n_topo
        self.n_climate = n_climate

        # Topographical subnetwork
        self.topo_net = nn.Sequential(nn.Linear(n_topo, hidden_dim_0),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim_0, hidden_dim_1),
                                      nn.ReLU())

        # Climate subnetwork
        self.climate_net = nn.Sequential(nn.Linear(n_climate, hidden_dim_0),
                                         nn.ReLU(),
                                         nn.Linear(hidden_dim_0, hidden_dim_1),
                                         nn.ReLU())

        # Final network after merging
        self.final_net = nn.Sequential(
            nn.Linear(2 * hidden_dim_1, final_dim),
            nn.ReLU(),
            nn.Linear(final_dim, 1),
        )

    def forward(self, x):
        # x: (batch_size, n_topo + n_climate)
        topo = x[:, :self.n_topo]
        climate = x[:, self.n_topo:self.n_topo + self.n_climate]

        topo_out = self.topo_net(topo)
        climate_out = self.climate_net(climate)

        combined = torch.cat([topo_out, climate_out], dim=1)
        return self.final_net(combined)