import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class SimpleTemporalGNN(nn.Module):

    def __init__(
        self,
        num_features,
        hidden_dim,
        window_size,
    ):
        super().__init__()

        self.window_size = window_size

        self.gcn1 = GCNConv(
            num_features,
            hidden_dim
        )

        self.gcn2 = GCNConv(
            hidden_dim,
            hidden_dim
        )

        self.temporal = nn.Sequential(

            nn.Linear(
                hidden_dim * window_size,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                1
            )
        )

    def forward(
        self,
        x,
        edge_index,
    ):

        B, T, N, F_dim = x.shape

        x_flattened = x.view(B * T * N, F_dim)

        h = self.gcn1(x_flattened, edge_index)
        h = F.relu(h)

        h = self.gcn2(h, edge_index)
        h = F.relu(h)

        h = h.view(B, T, N, -1)

        h = h.permute(0, 2, 1, 3)

        h = h.reshape(B, N, T * h.shape[-1])

        out = self.temporal(h)
        out = out.squeeze(-1)

        return out