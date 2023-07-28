import torch


class StandardScaler:
    def fit(self, dataset: torch.Tensor):
        self.m = dataset.mean(axis=0, keepdim=True)
        self.std = dataset.std(axis=0, keepdim=True)
        assert not torch.any(self.std == 0), "There are std=0, check dataset"

    def transform(self, dataset: torch.Tensor) -> torch.Tensor:
        dataset = dataset[:,:]
        dataset = (dataset - self.m) / self.std
        return dataset


    def fit_transform(self, dataset: torch.Tensor) -> torch.Tensor:
        self.fit(dataset)
        return self.transform(dataset)

    def inverse_transform(self, dataset: torch.Tensor) -> torch.Tensor:
        dataset = dataset[:,:]
        dataset = (dataset * self.std) + self.m
        return dataset


class MinMaxScaler:
    def fit(self, dataset: torch.Tensor):
        self.mn = dataset.min(axis=0, keepdim=True).values
        self.mx = dataset.max(axis=0, keepdim=True).values


    def transform(self, dataset: torch.Tensor) -> torch.Tensor:
        dataset = dataset[:,:]
        dataset = (dataset - self.mn) / (self.mx - self.mn)
        return dataset


    def fit_transform(self, dataset: torch.Tensor) -> torch.Tensor:
        self.fit(dataset)
        return self.transform(dataset)

    def inverse_transform(self, dataset: torch.Tensor) -> torch.Tensor:
        dataset = dataset[:,:]
        dataset = dataset * (self.mx - self.mn) + self.mn
        return dataset
    

class ErrorScaler:

    '''Scaler to get Health Index Curves from Mean Squared Errors.'''

    def fit(self, dataset: torch.Tensor):
        self.mn = dataset.min(axis=0, keepdim=True).values
        self.mx = dataset.max(axis=0, keepdim=True).values

    def transform(self, dataset: torch.Tensor) -> torch.Tensor:
        dataset = torch.clone(dataset)
        dataset = (-dataset + self.mx) / (self.mx - self.mn)
        return dataset

    def fit_transform(self, dataset: torch.Tensor) -> torch.Tensor:
        self.fit(dataset)
        return self.transform(dataset)

    def inverse_transform(self, dataset: torch.Tensor) -> torch.Tensor:
        dataset = torch.clone(dataset)
        dataset = -(dataset * (self.mx - self.mn) - self.mx)
        return dataset