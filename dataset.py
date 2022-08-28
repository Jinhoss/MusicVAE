from torch.utils.data import Dataset
import torch

class GrooveDataset(Dataset):
    def __init__(self, x):
        super().__init__()
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.Tensor(self.x[idx].todense())