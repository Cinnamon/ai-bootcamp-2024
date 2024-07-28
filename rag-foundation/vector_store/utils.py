import numpy as np


def get_rank(arr: np.ndarray):
    orders = arr.argsort()[::-1]
    ranks = orders.argsort() + 1

    return ranks
