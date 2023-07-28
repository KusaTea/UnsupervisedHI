import pickle
import os

import numpy as np
import random

import torch

from custom_classes import NasaDataset
from typing import Tuple


def save_object(obj: object, path: str):
    '''Saves any objets to .pkl format file.'''
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_object(path: str) -> object:
    '''Loads objets from .pkl format file.'''
    with open(path, 'rb') as f:
        return pickle.load(f)
    

def check_path(pth):
    '''Check if the path exists, otherwise creates the path.'''
    if not os.path.exists(pth):
        os.makedirs(pth)


def seed_everything(seed, verbose: bool = False):

    '''This function fixes all seeds to make experiment repeatability.'''

    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)
    if verbose:
        print('Seeds are fixed')


def split_dataset(dataset: NasaDataset, test_size: float = None, train_size: float = None) -> Tuple[NasaDataset, NasaDataset]:

    '''Splits one NasaDatset to two NasaDatasets in test_size:train_size proportion.'''

    def __splitter(sensors: torch.FloatTensor, machine_ids: torch.FloatTensor,
                   ruls: torch.FloatTensor, mask: np.array, targets: torch.FloatTensor = None) -> NasaDataset:
        dataset_dict = {
            'sensors': sensors[mask],
            'machine_id': machine_ids[mask],
            'rul': ruls[mask]
        }
        
        return NasaDataset(dataset_dict=dataset_dict, targets=targets)


    if train_size:
        assert train_size < 1 and train_size > 0, "\'train_size\' has to be in interval (0.0, 1.0)"
        test_size = 1 - train_size
    
    assert test_size < 1 and test_size > 0, "\'test_size\' has to be in interval (0.0, 1.0)"

    if isinstance(dataset.targets, type(None)):
        sensors, machine_ids, ruls = dataset.get_whole_dataset()
        targets = None
    else:
        sensors, machine_ids, ruls, targets = dataset.get_whole_dataset()
    uniq_ids = torch.unique(machine_ids)
    test_machines = torch.tensor(np.random.choice(uniq_ids, int(test_size*len(uniq_ids))))
    test_mask = torch.isin(machine_ids, test_machines)
    train_dataset = __splitter(sensors, machine_ids, ruls, torch.logical_not(test_mask), targets=targets)
    test_dataset = __splitter(sensors, machine_ids, ruls, test_mask, targets=targets)
    return train_dataset, test_dataset


def split_anomaly_normal(dataset: NasaDataset) -> Tuple[NasaDataset, NasaDataset]:

    '''Separates normal and anomaly data. Point is considered as anomaly if it's RUL less than 125.'''

    def __splitter(sensors, machine_ids, ruls, ids: torch.Tensor) -> NasaDataset:
        dataset_dict = {
            'sensors': sensors[ids],
            'machine_id': machine_ids[ids],
            'rul': ruls[ids]
        }

        return dataset_dict
    
    if isinstance(dataset.targets, type(None)):
        sensors, machine_ids, ruls = dataset.get_whole_dataset()
    else:
        sensors, machine_ids, ruls, targets = dataset.get_whole_dataset()
    normal_ids = torch.where(ruls == 125)[0]
    anomaly_ids = torch.where(ruls < 125)[0]
    normal_data = NasaDataset(dataset_dict=__splitter(sensors, machine_ids, ruls, normal_ids),
                              targets=None if isinstance(dataset.targets, type(None)) else targets[normal_ids])
    anomaly_data = NasaDataset(dataset_dict=__splitter(sensors, machine_ids, ruls, anomaly_ids),
                               targets=None if isinstance(dataset.targets, type(None)) else targets[anomaly_ids])
    return normal_data, anomaly_data


def split_anomaly_normal23(dataset: NasaDataset) -> Tuple[NasaDataset, NasaDataset]:

    '''Separates normal and anomaly data. Point is marked anomalous if its index is equal to 2/3 of the data length with RUL 125.'''

    def __splitter(sensors, machine_ids, ruls, ids: torch.Tensor) -> NasaDataset:
        dataset_dict = {
            'sensors': sensors[ids],
            'machine_id': machine_ids[ids],
            'rul': ruls[ids]
        }

        return dataset_dict
    

    if isinstance(dataset.targets, type(None)):
        sensors, machine_ids, ruls = dataset.get_whole_dataset()
        targets = None
    else:
        sensors, machine_ids, ruls, targets = dataset.get_whole_dataset()
    normal_ids, anomaly_ids = list(), list()
    for id in machine_ids.unique():
        r_indeces = torch.where(machine_ids == id)[0]
        r = ruls[machine_ids == id]
        mixed = r_indeces[torch.where(r == 125)[0]]
        normal = mixed[:2*len(mixed)//3]
        anomaly = r_indeces[torch.where(r < 125)[0]]
        anomaly = torch.hstack((mixed[2*len(mixed)//3:], anomaly))
        normal_ids.append(normal)
        anomaly_ids.append(anomaly)

    normal_ids = torch.hstack(normal_ids)
    anomaly_ids = torch.hstack(anomaly_ids)

    normal_data = NasaDataset(dataset_dict=__splitter(sensors, machine_ids, ruls, normal_ids),
                              targets=None if isinstance(dataset.targets, type(None)) else targets[normal_ids])
    anomaly_data = NasaDataset(dataset_dict=__splitter(sensors, machine_ids, ruls, anomaly_ids),
                               targets=None if isinstance(dataset.targets, type(None)) else targets[anomaly_ids])
    return normal_data, anomaly_data