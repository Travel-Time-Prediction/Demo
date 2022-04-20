import numpy as np
from torch.utils.data import Dataset

class TravelTimeDataset(Dataset):
    def __init__(self, x, y):
        super(TravelTimeDataset, self).__init__()
        self.x = x.reshape((x.shape[0], x.shape[1], 1))
        # self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return (self.x[index], self.y[index])