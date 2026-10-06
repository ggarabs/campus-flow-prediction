import torch
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import TGCN


class TemporalGCN(nn.Module):
    def __init__(self, num_features, hidden_dim, window_size, forecast_horizon, dropout):
        super().__init__()

        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon

        self.tgcn = TGCN(in_channels=num_features, out_channels=hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, forecast_horizon)

    def forward(self, x, edge_index):
        B, T, N, F_in = x.shape

        edge_index_batched = edge_index.repeat(1, B)
        shift = torch.arange(B, device=x.device).repeat_interleave(edge_index.shape[1]) * N
        edge_index_batched = edge_index_batched + shift.unsqueeze(0)

        h = None
        for t in range(T):
            xt = x[:, t, :, :].reshape(B * N, F_in)
            h = self.tgcn(X=xt, edge_index=edge_index_batched, H=h)

        h = self.dropout(h)
        out = self.linear(h)
        out = out.reshape(B, N, self.forecast_horizon)

        return out


class Seq2SeqTGCN(nn.Module):
    def __init__(self, num_features, hidden_dim, forecast_horizon, dropout=0.2):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.hidden_dim = hidden_dim

        self.encoder = TGCN(in_channels=num_features, out_channels=hidden_dim)

        self.decoder = TGCN(in_channels=1, out_channels=hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.proj_out = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        B, T_in, N, F_in = x.shape

        edge_index_batched = edge_index.repeat(1, B)
        shift = torch.arange(B, device=x.device).repeat_interleave(edge_index.shape[1]) * N
        edge_index_batched = edge_index_batched + shift.unsqueeze(0)

        h = None
        for t in range(T_in):
            xt = x[:, t, :, :].reshape(B * N, F_in)
            h = self.encoder(X=xt, edge_index=edge_index_batched, H=h)

        current_y = x[:, -1, :, 0:1].reshape(B * N, 1)

        outputs = []
        for t in range(self.forecast_horizon):
            h = self.decoder(X=current_y, edge_index=edge_index_batched, H=h)
            h_drop = self.dropout(h)

            pred_step = self.proj_out(h_drop)
            outputs.append(pred_step)

            current_y = pred_step

        out = torch.stack(outputs, dim=1)
        out = out.reshape(B, N, self.forecast_horizon)

        return out
