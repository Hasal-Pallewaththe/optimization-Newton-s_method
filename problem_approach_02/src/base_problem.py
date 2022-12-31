# Created on 11.11.2022
# author: Hasal
# fitness_func f and x are considered as column vectors

import numpy as np
from typing import List


class BaseProblem:
    """ a generalized problem with all necessary tools
    for roots calculation using Newton's method """

    def __init__(
                self,
                x_dim: int = 3,
                f_dim: int = 3) -> None:

        self.x_dim = x_dim
        self.f_dim = f_dim

    # setter method
    def set_x_dim(self, x_dim: int) -> None:
        self.x_dim = x_dim

    # setter method
    def set_f_dim(self, f_dim: int) -> None:
        self.f_dim = f_dim

    def fitness_func(self, x: List) -> List:
        """an arbitrary function is implemented this needs to be modified"""

        x = x.flatten()

        f1 = x[0] + x[1]
        f2 = x[1] + x[2]
        f3 = x[2] + x[0]
        return np.array([[f1], [f2], [f3]])

    def evolve(
            self,
            x0: List,
            max_itter: int = 30,
            x_tol: float = 1E-6,
            step: float = 1E-6) -> List:
        """ calculates the solution for x (roots),  where f(x)=0
            using the newtons methods, iteratively
        """

        x = np.array(x0)
        x = x[:, np.newaxis]

        if x.size != self.x_dim:
            raise ValueError("please check the x-dimention and initial guess")
        elif self.f_dim != self.x_dim:
            raise ValueError("both f-dimention and x-dimention should be equal")
        else:
            try:
                self.fitness_func(x)
            except ValueError:
                print("please check the fitness func")
        if np.size(self.fitness_func(x)) != self.f_dim:
            raise ValueError("please check the f-dimention")

        for it in range(max_itter):

            # jacobian
            jac = []
            # step size
            h = step
            for i in range(self.f_dim):
                jac_row = []
                for j in range(self.x_dim):

                    # standard unit vector
                    ej = np.zeros(self.x_dim)
                    ej[j] = 1
                    ej = ej[:, np.newaxis]

                    # if condition to simplyfy the jacobian matrix
                    if j < i:
                        df_dx = 0
                    else:
                        df_dx = (self.fitness_func(x + h*ej)[i, 0] - self.fitness_func(x)[i, 0])/h

                    jac_row.append(df_dx)

                jac.append(jac_row)

            jac = np.array(jac)

            delta_x = np.matmul(np.linalg.inv(jac), self.fitness_func(x))
            delta_x_2norm = np.linalg.norm(delta_x, ord=2)

            # check condition ||delta_x|| 2 norm
            if delta_x_2norm < x_tol:
                break

            x = x - delta_x

        return list(x.flatten())
