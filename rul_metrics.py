import numpy as np
import torch
from torch import nn

####################
# JUST TO BE
# The simplest way to estimate is mean distance between points
# The less the better

# pytorch.nn.L1loss
def mean_absolute_distance(test_rul: np.array, target_rul: np.array) -> float:
    return (test_rul - target_rul).mean()

# To avoid situation of bad results but 0 distance, because of equal negative and positive values
# pytorch.nn.MSELoss
def mean_squared_distance(test_rul: np.array, target_rul: np.array) -> float:
    return ((test_rul - target_rul)**2).mean()
####################

####################
# The more confident approach - calculate square between two curves
# WORKS ONLY WITH LATENT WITH SIZE (1,)


class SBCLoss(nn.Module):

    def __init__(self):
        super(SBCLoss, self).__init__()
    
    def forward(self, output: torch.Tensor, target: torch.Tensor):
        if not isinstance(target, torch.Tensor):
            target = torch.FloatTensor(target)
        loss_func = self.sbc_multispace
        loss = loss_func(output, target)
        return loss

    @classmethod
    def square_between_curves(cls, test_rul: torch.Tensor, target_rul: torch.Tensor) -> float:

        """Calculates square between two curves with equal x coordinates of points.
        If there warning of zero or incorrect division is appeared,
        don't worry, everything is ok - these situations are provided."""

        difference = target_rul - test_rul
        intersections = torch.nan_to_num(difference[:-1] / difference[1:], 0)
        intersections[intersections == torch.inf] = 0
        intersections = torch.sign(intersections) + 1
        difference = torch.abs(difference)

        trapeziums = torch.where(intersections > 0)[0]
        squares_trapeziums = 0.5*(difference[trapeziums] + difference[trapeziums+1])

        triangles = torch.where(intersections == 0)[0]
        x_of_intersections = 1 / (difference[triangles] / difference[triangles+1] + 1)
        x_of_intersections = torch.vstack((x_of_intersections, 1 - x_of_intersections))
        squares_triangles = 0.5*(x_of_intersections[0]*difference[triangles]) + \
            0.5*(x_of_intersections[1]*difference[triangles+1])
        return squares_trapeziums.sum() + squares_triangles.sum()

    # TODO: add weighted averaging somehow

    @classmethod
    def sbc_multispace(cls, prediction: torch.Tensor, target: torch.Tensor):
        squares = list()
        # TODO: check feature dimension and change code: cycle has to walk along features dimension, not points
        for pred, trg in zip(prediction, target):
            squares.append(cls.square_between_curves(pred, trg))
        return np.average(squares)
####################


####################
class RMSELoss(nn.Module):
    
    def __init__(self):
        super().__init__(RMSELoss, self)
    
    def forward(self, output: torch.Tensor, target: torch.Tensor):
        if not isinstance(target, torch.Tensor):
            target = torch.FloatTensor(target)
        loss_func = torch.nn.MSELoss()
        loss = torch.sqrt(loss_func(output, target))
        return loss
####################


####################
class RULScore(nn.Module):

    def __init__(self):
        super(RULScore, self).__init__()
    
    def forward(self, output: torch.Tensor, target: torch.Tensor):
        if not isinstance(target, torch.Tensor):
            target = torch.FloatTensor(target)
        err = target - output

        less = torch.where(err < 0)[0]
        greater = torch.where(err >= 0)[0]

        loss = (torch.exp(-(err[less] / 10)) - 1).sum() + (torch.exp(err[greater] / 13) - 1).sum()
        return loss
####################