# Created on 11.11.2022
# author: Hasal
# without using numpy for inverse, multiplication and norm calculations
# only works for a x_dim = 3, f_dim=3 , 3x3 jacobian matrix
# numpy is used to avoid OverflowError: math range error - of python's default,
# exceeding of maximum limit to store data

import math
import numpy as np
from typing import List
from base_problem import BaseProblem


# testing - using the given function, for example,
class ExampleProblem(BaseProblem):
    """ an example problem """

    def __init__(self) -> None:
        super().__init__()

    def fitness_func(self, x: List) -> List:
        """The function """

        x = x.flatten()

        f1 = x[0]/x[1] + x[2]/x[0]
        f2 = 0.5*(x[1]**3) - 250*x[1]*x[2] - 75000*(x[2]**2)
        f3 = math.exp(-x[2]) + x[2]*math.exp(1)
        return np.array([[f1], [f2], [f3]])
