from torch.utils.data import Dataset

class TemporalGraphDataset(Dataset):
    def __init__(
        self,
        X,
        window_size=12,
        target_feature_idx=0,
    ):
        self.X = X
        self.window_size = window_size
        self.target_feature_idx = target_feature_idx

    def __len__(self):
        return len(self.X) - self.window_size

    def __getitem__(self, idx):
        x = self.X[
            idx : idx + self.window_size
        ]

        y = self.X[
            idx + self.window_size,
            :,
            self.target_feature_idx,
        ]

        return x, y