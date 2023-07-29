import numpy as np
import pandas as pd
import random

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple


class NasaDataset(Dataset):

    '''Custom pytorch Dataset class. Works with NASA-CMAPSS dataset.
File must have .csv format and contain "unit_number", "RUL" and sensors columns.
It returns dictionary {"sensors": pytorch.Tensor, "rul": pytorch.Tensor, "machine_id": pytorch.Tensor}.'''

    def __init__(self, dataset_path: str = None, dataset_dict: dict = None,
                 transform = None, targets: np.array = None, device: torch.device = None):
        if dataset_path:
            self.dataset = pd.read_csv(dataset_path)
            self.ruls = torch.FloatTensor(self.dataset.pop('RUL').values)
            self.machine_ids = torch.FloatTensor(self.dataset.pop('unit_number').values)
            self.dataset = torch.FloatTensor(self.dataset.values)
        
        elif dataset_dict:
            self.dataset = dataset_dict['sensors']
            self.ruls = dataset_dict['rul']
            self.machine_ids = dataset_dict['machine_id']

        self.targets = targets if isinstance(targets, type(None)) else torch.FloatTensor(targets)

        self.transfrom = transform

        if device:
            self.device = device
            self.to(self.device)
        else:
            self.device = self.ruls.device.type
    
    def to(self, device):
        '''Transfers all tensors to "device".'''
        self.dataset = self.dataset.to(device)
        self.ruls = self.ruls.to(device)
        self.machine_ids = self.machine_ids.to(device)

        if not isinstance(self.targets, type(None)):
            self.targets = self.targets.to(device)

    def get_input_shape(self) -> int:
        '''Returns input shape for neural network.'''
        return self.dataset.shape[1]
    
    def get_whole_dataset(self) \
        -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor] or \
            Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        if not isinstance(self.targets, type(None)):
            return self.dataset, self.machine_ids, self.ruls, self.targets
        else:
            return self.dataset, self.machine_ids, self.ruls

    def __len__(self) -> int:
        return self.dataset.shape[0]

    def __getitem__(self, idx: int) -> dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = {'sensors': self.dataset[idx], 'rul': self.ruls[idx], 'machine_id': self.machine_ids[idx]}

        if not isinstance(self.targets, type(None)):
            sample.update({'targets': self.targets[idx]})

        if self.transfrom:
            sample = self.transfrom(sample)

        return sample

    def get_indeces(self, machine_id: int) -> torch.Tensor:
        '''Returns indeces of data points of a certain machine.'''
        return torch.where(self.machine_ids == machine_id)[0]
    

class AnomalyLoader:

    '''Loads data by batch for each machine.'''

    def __init__(self, dataset: NasaDataset, batch_size: int = None):
        self.dataset = dataset
        self.batch_size = batch_size if batch_size else len(dataset)

    def reset_everything(self):
        self.idx = 0
        self.machine_ids = torch.unique(self.dataset.machine_ids)
        self.current_machine_id = self.machine_ids[self.idx]
        self.current_start = -self.batch_size
        self.current_indeces = self.dataset.get_indeces(self.current_machine_id)
    
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


class WindowedLoader:

    '''Loads data by batch for each machine.'''

    def __init__(self, dataset: NasaDataset, batch_size: int = None, window_size: int = 20, for_conv: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size if batch_size else len(dataset)
        self.window_size = window_size
        self.device = self.dataset.device
        self.for_conv = for_conv

        self.reset_everything()
    
    def reset_everything(self):
        self.idx = 0
        self.machine_ids = torch.unique(self.dataset.machine_ids)
        self.current_machine_id = self.machine_ids[self.idx]
        self.current_start = -1
        self.current_indeces = self.dataset.get_indeces(self.current_machine_id)
        self.flag = False
    
    def __iter__(self):
        return self

    def __next__(self):
        if not self.flag:
            rul_batch, sensors_batch, machine_id_batch, indeces_batch = list(), list(), list(), list()
            # Get batch
            for _ in range(self.batch_size):
                # Do step
                self.current_start += 1
                
                # Go to the next machine if got all data of the current machine
                if self.current_start >= len(self.current_indeces) - self.window_size + 1:
                    self.idx += 1
                    if self.idx < len(self.machine_ids):
                        self.current_machine_id = self.machine_ids[self.idx]
                        self.current_indeces = self.dataset.get_indeces(self.current_machine_id)
                        self.current_start = 0
                    # If all machines were observed stop iteration
                    else:
                        self.flag = True
                        break
                
                # Get windowed data
                inds = self.current_indeces[self.current_start: self.current_start + self.window_size]
                dta = self.dataset[inds]
                rul_batch.append(dta['rul'])
                sensors_batch.append(dta['sensors'])
                machine_id_batch.append(dta['machine_id'])
                indeces_batch.append(inds)

            try:
                sample = {
                    'sensors': torch.stack(sensors_batch),
                    'rul': torch.stack(rul_batch),
                    'machine_id': torch.stack(machine_id_batch),
                    'indeces': torch.stack(indeces_batch)
                }

                if self.for_conv:
                    sample['sensors'] = torch.transpose(sample['sensors'], 1, 2)
                
                return sample

            except RuntimeError:
                return self.__next__()
        
        else:
            self.reset_everything()
            raise StopIteration


class SimpleAE(nn.Module):

    '''Custom pytorch Autoencoder class.'''

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
    
    def forward(self, x) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class AEConstructor(nn.Module):

    '''Creates autoencoder by obtained characteristics.'''

    def __init__(self, input_shape: int, window_size: int, layers_sizes: tuple):
        super(AEConstructor, self).__init__()
        self.encoder = nn.Sequential(nn.Flatten(),
                                     nn.Linear(in_features=input_shape*window_size, out_features=layers_sizes[0]))
        for in_features, out_features in zip(layers_sizes[:-1], layers_sizes[1:]):
            linear, activation = self.ae_block(in_features, out_features)
            self.encoder.append(activation)
            self.encoder.append(linear)

        self.decoder = nn.Sequential()
        for in_features, out_features in zip(layers_sizes[:0:-1], layers_sizes[-2::-1]):
            linear, activation = self.ae_block(in_features, out_features)
            self.decoder.append(linear)
            self.decoder.append(activation)
        self.decoder.append(nn.Linear(in_features=layers_sizes[0], out_features=input_shape*window_size))
        self.decoder.append(nn.Unflatten(1, (window_size, input_shape)))
    
    @staticmethod
    def ae_block(in_features: int, out_features: int) -> Tuple[nn.Linear, nn.SiLU]:
        linear = nn.Linear(in_features=in_features, out_features=out_features)
        activation = nn.SiLU()
        return linear, activation
    
    def forward(self, x) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class CAE(nn.Module):
    def __init__(self, input_channels: int, layers: tuple, conv_kernel: int = 5, conv_stride: int  = 1,
                 pool_kernel: int = 3, pool_stride: int = 2, unconv_kernels: list = (5, 5), unconv_stride: int = 1):
        super(CAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=layers[0],
                      kernel_size=conv_kernel, stride=conv_stride),
            nn.SiLU(),
            nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_stride),
            nn.Conv1d(in_channels=layers[0], out_channels=layers[1],
                      kernel_size=conv_kernel, stride=conv_stride),
            nn.SiLU(),
            nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_stride),
            nn.Flatten(),
            nn.Linear(in_features=layers[2], out_features=layers[3]),
            nn.BatchNorm1d(layers[3]),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=layers[3], out_features=layers[4])
            )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=layers[4], out_features=layers[3]),
            nn.BatchNorm1d(layers[3]),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=layers[3], out_features=layers[2]),
            nn.SiLU(),
            nn.Unflatten(dim=1, unflattened_size=torch.Size((layers[1], -1))),
            nn.ConvTranspose1d(in_channels=layers[1], out_channels=layers[0],
                               kernel_size=unconv_kernels[0], stride=unconv_stride),
            nn.SiLU(),
            nn.ConvTranspose1d(in_channels=layers[0], out_channels=input_channels,
                               kernel_size=unconv_kernels[1], stride=unconv_stride)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded




class LossAndMetric(nn.Module):

    '''Model evaluating block.'''

    def __init__(self, loss_func: object, metric_func: object,
                 scaler: object = None, transform = None, transform_dct: dict = {}):
        super(LossAndMetric, self).__init__()
        self.loss_func = loss_func
        self.metric_func = metric_func
        self.scaler = scaler
        self.transform = transform
        self.transform_dct = transform_dct
    
    
    def forward(self, predicted_values: torch.Tensor, true_values: torch.Tensor) -> tuple:
        if self.transform:
            predicted_values = self.transform(predicted_values, **self.transform_dct)
            true_values = self.transform(true_values, **self.transform_dct)
        
        loss = self.loss_func(predicted_values, true_values)

        if self.scaler:
            metric = self.metric_func(self.scaler.inverse_transform(predicted_values),
                                 self.scaler.inverse_transform(true_values))
        else:
            metric = self.metric_func(predicted_values, true_values)
        
        return loss, metric