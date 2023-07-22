import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class NasaDataset(Dataset):

    def __init__(self, dataset_path: str, transform = None):
        self.dataset = pd.read_csv(dataset_path)
        self.ruls = self.dataset.pop('RUL').values
        self.machine_ids = self.dataset.pop('unit_number')
        self.dataset = self.dataset.values
        self.transfrom = transform
    
    def get_input_shape(self):
        return self.dataset.shape[1]

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            return idx.tolist()
        
        sample = {'sensors': self.dataset[idx], 'rul': self.ruls[idx], 'machine_id': self.machine_ids[idx]}

        if self.transfrom:
            sample = self.transfrom(sample)

        return sample
    


class SimpleAE(nn.Module):

    def __init__(self, input_shape, layers_sizes: tuple):
        super(SimpleAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features=layers_sizes[0]),
            nn.SiLU(),
            nn.Linear(in_features=layers_sizes[0], out_features=layers_sizes[1]),
            nn.SiLU(),
            nn.Linear(in_features=layers_sizes[1], out_features=layers_sizes[2])
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=layers_sizes[2], out_features=layers_sizes[1]),
            nn.SiLU(),
            nn.Linear(in_features=layers_sizes[1], out_features=layers_sizes[2]),
            nn.ReLU(),
            nn.Linear(in_features=layers_sizes[2], out_features=input_shape)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded