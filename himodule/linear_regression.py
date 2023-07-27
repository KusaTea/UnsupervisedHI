import torch
from torch import nn


class LinearRegression(nn.Module):

    def __init__(self, input_shape: int):
        super(LinearRegression, self).__init__()
        model = nn.Sequential(nn.Linear(in_features=input_shape, out_features=1))

    def forward(self, x):
        return self.model(x)
