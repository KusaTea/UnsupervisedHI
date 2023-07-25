import torch
from torch import nn


class MAPE(nn.Module):

    def __init__(self):
        super(MAPE, self).__init__()
    
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not isinstance(target, torch.Tensor):
            target = torch.FloatTensor(target)
        err = (target - output).abs() / target * 100
        return err.mean()