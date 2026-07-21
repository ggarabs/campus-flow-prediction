import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import TGCN

class TemporalGCN(nn.Module):
    def __init__(self, num_features, hidden_dim, window_size, forecast_horizon):
        super(TemporalGCN, self).__init__()
        
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        
        self.tgcn = TGCN(in_channels=num_features, out_channels=hidden_dim)
        
        self.linear = nn.Linear(hidden_dim, forecast_horizon)

    def forward(self, x, edge_index):
        B, T, N, F_in = x.shape

        edge_index_batched = edge_index.repeat(1, B)
        shift = torch.arange(B, device=x.device).repeat_interleave(edge_index.shape[1]) * N
        edge_index_batched = edge_index_batched + shift.unsqueeze(0)
        
        h = None
        for t in range(T):
                xt = x[:, t, :, :].reshape(B*N, F_in)
                h = self.tgcn(X=xt, edge_index=edge_index_batched, H=h)
                h = F.relu(h)
            
        out = self.linear(h)
        out = out.reshape(B, N, self.forecast_horizon)
                            
        return F.relu(out)