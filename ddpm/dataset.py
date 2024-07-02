import torch
from torch.utils.data import Dataset


class DdpmDataset(Dataset):
    def __init__(self, X, y, device):
        super().__init__()
        self.X = X
        self.y = y
        self.device = device
        self.pad = torch.nn.ConstantPad2d((2, 2, 2, 2), 0)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        sample = self.X[item]
        sample = self.pad(sample)
        sample = sample.unsqueeze(0).to(torch.float32).to(self.device)
        return sample
