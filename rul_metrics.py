import numpy as np

####################
# JUST TO BE
# The simplest way to estimate is mean distance between points
# The less the better

def mean_absolute_distance(test_rul: np.array, target_rul: np.array) -> float:
    return (test_rul - target_rul).mean()

# To avoid situation of bad results but 0 distance, because of equal negative and positive values
def mean_squared_distance(test_rul: np.array, target_rul: np.array) -> float:
    return ((test_rul - target_rul)**2).mean()
####################

####################
# The more confident approach - calculate square between two curves
# WORKS ONLY WITH LATENT WITH SIZE (1,)

def square_between_curves(test_rul: np.array, target_rul: np.array) -> float:

    """Calculates square between two curves with equal x coordinates of points.
    If there warning of zero or incorrect division is appeared,
    don't worry, everything is ok - these situations are provided."""

    difference = target_rul - test_rul
    intersections = np.nan_to_num(difference[:-1] / difference[1:], 0)
    intersections[intersections == np.inf] = 0
    intersections = np.sign(intersections) + 1
    print(intersections)
    difference = np.abs(difference)

    trapeziums = np.where(intersections > 0)[0]
    squares_trapeziums = 0.5*(difference[trapeziums] + difference[trapeziums+1])

    triangles = np.where(intersections == 0)[0]
    x_of_intersections = 1 / (difference[triangles] / difference[triangles+1] + 1)
    x_of_intersections = np.vstack((x_of_intersections, 1 - x_of_intersections))
    squares_triangles = 0.5*(x_of_intersections[0]*difference[triangles]) + \
        0.5*(x_of_intersections[1]*difference[triangles+1])
    return squares_trapeziums.sum() + squares_triangles.sum()
####################

####################
# TODO: add weighted averaging somehow

def sbc_multispace(prediction, target):
    squares = list()
    # TODO: check feature dimension and change code: cycle has to walk along features dimension, not points
    for pred, trg in zip(prediction, target):
        squares.append(square_between_curves)
    return np.mean(squares)
####################