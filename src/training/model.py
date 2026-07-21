import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import TGCN

class TemporalGCN(nn.Module):
    def __init__(self, num_features, hidden_dim, window_size):
        super(TemporalGCN, self).__init__()
        
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        
        self.tgcn = TGCN(in_channels=num_features, out_channels=hidden_dim)
        
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        B, T, N, F_in = x.shape
        
        batch_predictions = []
        
        for b in range(B):
            h = None
            
            for t in range(T):
                xt = x[b, t]
                h = self.tgcn(X=xt, edge_index=edge_index, H=h)
                h = F.relu(h)
            
            out = self.linear(h)
            batch_predictions.append(out)
            
        final_out = torch.stack(batch_predictions, dim=0)
                
        return F.relu(final_out)