from custom_classes import NasaDataset
import torch


def standard_scaling(dataset: NasaDataset):
    dta = dataset.dataset
    m = dta.mean(axis=0, keepdim=True)
    std = dta.std(axis=0, keepdim=True)
    dataset.dataset = (dta - m) / std
    assert not torch.isin(torch.inf, dataset.dataset)
    assert not torch.isin(torch.nan, dataset.dataset)

def min_max_scaling(dataset: NasaDataset):
    dta = dataset.dataset
    min = dta.min(axis=0, keepdim=True)
    max = dta.max(axis=0, keepdim=True)
    std = (dta - min) / (max - min)
    dataset.dataset = std * (max - min) + max
    assert not torch.isin(torch.inf, dataset.dataset)
    assert not torch.isin(torch.nan, dataset.dataset)