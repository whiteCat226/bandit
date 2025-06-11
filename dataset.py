## dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

class PathMNISTDataset(Dataset):
    def __init__(self, npz_file, split='train'):
        data = np.load(npz_file)
        self.images = data[f'{split}_images']
        self.labels = data[f'{split}_labels'].flatten()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32).permute(2, 0, 1) / 255.0
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

def split_dataset(dataset, num_clients=3):
    total = len(dataset)
    client_size = total // num_clients
    subsets = []
    for i in range(num_clients):
        start = i * client_size
        end = (i + 1) * client_size if i < num_clients - 1 else total
        subsets.append(Subset(dataset, list(range(start, end))))
    return subsets
