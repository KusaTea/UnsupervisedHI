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
    mn = dta.min(axis=0, keepdim=True).values
    mx = dta.max(axis=0, keepdim=True).values
    dataset.dataset = (dta - mn) / (mx - mn)
    assert not torch.isin(torch.inf, dataset.dataset)
    assert not torch.isin(torch.nan, dataset.dataset)