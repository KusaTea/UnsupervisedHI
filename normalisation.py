from custom_classes import NasaDataset
import copy
import torch


class StandardScaler:
    def fit(self, dataset: NasaDataset):
        self.m = dataset.dataset.mean(axis=0, keepdim=True)
        self.std = dataset.dataset.std(axis=0, keepdim=True)
        assert not torch.any(self.std == 0), "There are std=0, check dataset"

    def transform(self, dataset: NasaDataset) -> NasaDataset:
        dataset = copy.deepcopy(dataset)
        dataset.dataset = (dataset.dataset - self.m) / self.std
        return dataset


    def fit_transform(self, dataset: NasaDataset) -> NasaDataset:
        self.fit(dataset)
        return self.transform(dataset)

    def inverse_transform(self, dataset: NasaDataset):
        dataset = copy.deepcopy(dataset)
        dataset.dataset = (dataset.dataset * self.std) + self.m
        return dataset


class MinMaxScaler:
    def fit(self, dataset: NasaDataset):
        self.mn = dataset.dataset.min(axis=0, keepdim=True).values
        self.mx = dataset.dataset.max(axis=0, keepdim=True).values


    def transform(self, dataset: NasaDataset) -> NasaDataset:
        dataset = copy.deepcopy(dataset)
        dataset.dataset = (dataset.dataset - self.mn) / (self.mx - self.mn)
        return dataset


    def fit_transform(self, dataset: NasaDataset) -> NasaDataset:
        self.fit(dataset)
        return self.transform(dataset)

    def inverse_transform(self, dataset: NasaDataset):
        dataset = copy.deepcopy(dataset)
        dataset.dataset = dataset.dataset * (self.mx - self.mn) + self.mn
        return dataset