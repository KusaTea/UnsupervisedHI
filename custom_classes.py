import numpy as np
import pandas as pd
import random

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


def seed_everything(seed, verbose: bool = False):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)
    if verbose:
        print('Seeds are fixed')


class NasaDataset(Dataset):

    def __init__(self, dataset_path: str = None, dataset_dict: dict = None,
                 transform = None, device: torch.device = None):
        if dataset_path:
            self.dataset = pd.read_csv(dataset_path)
            self.ruls = torch.FloatTensor(self.dataset.pop('RUL').values)
            self.machine_ids = torch.FloatTensor(self.dataset.pop('unit_number').values)
            self.dataset = torch.FloatTensor(self.dataset.values)
        
        elif dataset_dict:
            self.dataset = dataset_dict['sensors']
            self.ruls = dataset_dict['rul']
            self.machine_ids = dataset_dict['machine_id']

        self.transfrom = transform

        if device:
            self.to(device)
    
    def to(self, device):
        self.dataset = self.dataset.to(device)
        self.ruls = self.ruls.to(device)
        self.machine_ids = self.machine_ids.to(device)

    def get_input_shape(self) -> int:
        return self.dataset.shape[1]
    
    def get_whole_dataset(self) -> tuple:
        return self.dataset, self.machine_ids, self.ruls

    def __len__(self) -> int:
        return self.dataset.shape[0]

    def __getitem__(self, idx: int) -> dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = {'sensors': self.dataset[idx], 'rul': self.ruls[idx], 'machine_id': self.machine_ids[idx]}

        if self.transfrom:
            sample = self.transfrom(sample)

        return sample


class AnomalyDataset(Dataset):

    def __init__(self, dataset_dict: dict = None,
                 transform = None, device: torch.device = None):
        
        self.dataset = dataset_dict['sensors']
        self.ruls = dataset_dict['rul']
        self.machine_ids = dataset_dict['machine_id']

        self.transfrom = transform

        if device:
            self.to(device)
    
    def to(self, device):
        self.dataset = self.dataset.to(device)
        self.ruls = self.ruls.to(device)
        self.machine_ids = self.machine_ids.to(device)

    def get_input_shape(self) -> int:
        return self.dataset.shape[1]
    
    def get_whole_dataset(self) -> tuple:
        return self.dataset, self.machine_ids, self.ruls

    def __len__(self) -> int:
        return self.dataset.shape[0]

    def __getitem__(self, indeces: int or list or torch.Tensor) -> dict:
        if torch.is_tensor(indeces):
            indeces = indeces.tolist()
        
        sample = {'sensors': self.dataset[indeces], 'rul': self.ruls[indeces], 'machine_id': self.machine_ids[indeces]}

        if self.transfrom:
            sample = self.transfrom(sample)

        return sample
    
    def get_indeces(self, machine_id: int):
        return torch.where(self.machine_ids == machine_id)[0]
    

class AnomalyLoader:
    def __init__(self, dataset: AnomalyDataset, batch_size: int = None):
        self.dataset = dataset
        self.batch_size = batch_size if batch_size else len(dataset)

        self.idx = 0
        self.machine_ids = torch.unique(dataset.machine_ids)
        self.current_machine_id = self.machine_ids[self.idx]
        self.current_start = -batch_size
        self.current_indeces = dataset.get_indeces(self.current_machine_id)
    
    def __iter__(self):
        return self

    def __next__(self):
        self.current_start += self.batch_size
        if self.current_start >= len(self.current_indeces):
            self.idx += 1
            if self.idx < len(self.machine_ids):
                self.current_machine_id = self.machine_ids[self.idx]
                self.current_indeces = self.dataset.get_indeces(self.current_machine_id)
                self.current_start = 0
            else:
                raise StopIteration
        return self.dataset[self.current_indeces[self.current_start:self.current_start + self.batch_size]]


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
            nn.SiLU(),
            nn.Linear(in_features=layers_sizes[2], out_features=input_shape)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def split_dataset(dataset: NasaDataset, test_size: float = None, train_size: float = None):

    def __splitter(sensors, machine_ids, ruls, mask: np.array) -> NasaDataset:
        dataset_dict = {
            'sensors': sensors[mask],
            'machine_id': machine_ids[mask],
            'rul': ruls[mask]
        }

        return NasaDataset(dataset_dict=dataset_dict)


    if train_size:
        assert train_size < 1 and train_size > 0, "\'train_size\' has to be in interval (0.0, 1.0)"
        test_size = 1 - train_size
    
    assert test_size < 1 and test_size > 0, "\'test_size\' has to be in interval (0.0, 1.0)"

    sensors, machine_ids, ruls = dataset.get_whole_dataset()
    uniq_ids = torch.unique(machine_ids)
    test_machines = torch.tensor(np.random.choice(uniq_ids, int(test_size*len(uniq_ids))))
    test_mask = torch.isin(machine_ids, test_machines)
    train_dataset = __splitter(sensors, machine_ids, ruls, torch.logical_not(test_mask))
    test_dataset = __splitter(sensors, machine_ids, ruls, test_mask)
    return train_dataset, test_dataset


def split_anomaly_normal(dataset: NasaDataset):
    def __splitter(sensors, machine_ids, ruls, ids: torch.Tensor) -> NasaDataset:
        dataset_dict = {
            'sensors': sensors[ids],
            'machine_id': machine_ids[ids],
            'rul': ruls[ids]
        }

        return dataset_dict
    

    sensors, machine_ids, ruls = dataset.get_whole_dataset()
    normal_ids = torch.where(ruls == 125)[0]
    anomaly_ids = torch.where(ruls < 125)[0]
    normal_data = NasaDataset(dataset_dict=__splitter(sensors, machine_ids, ruls, normal_ids))
    anomaly_data = AnomalyDataset(dataset_dict=__splitter(sensors, machine_ids, ruls, anomaly_ids))
    return normal_data, anomaly_data