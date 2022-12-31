# Created on 11.11.2022
# author: Hasal
# without using numpy for inverse, multiplication and norm calculations
# only works for a x_dim = 3, f_dim=3 , 3x3 jacobian matrix
# numpy is used to avoid OverflowError: math range error - of python's default,
# exceeding of maximum limit to store data

import math
import numpy as np
from typing import List


class BaseProblem:
    """ a generalized problem with all necessary tools
    for roots calculation using Newton's method """

    def __init__(self) -> None:

        self.x_dim = 3
        self.f_dim = 3

    @staticmethod
    def inverse_mat(a):
        """Inversion  of a 3x3 upper triangular matrix """
        # adjoint of the matrix is the transpose of cofactor matrix
        adj = [
                [a[1][1]*a[2][2], -a[0][1]*a[2][2], a[0][1]*a[1][2]-a[1][1]*a[0][2]],
                [0, a[0][0]*a[2][2], -a[0][0]*a[1][2]],
                [0, 0, a[0][0]*a[1][1]]
            ]
        # determinant of a upper triangualer 3x3 matrix
        det = (a[0][0]*a[1][1]*a[2][2])

        inv = [
            [adj[0][0]/det, adj[0][1]/det, adj[0][2]/det],
            [0, adj[1][1]/det, adj[1][2]/det],
            [0, 0, adj[2][2]/det]
        ]
        return inv

    @staticmethod
    def matmul(a, b):
        """multiplication of a 3x3 matrix with a 3x1 matrix"""
        mult = [
            [a[0][0]*b[0][0] + a[0][1]*b[1][0] + a[0][2]*b[2][0]],
            [a[1][0]*b[0][0] + a[1][1]*b[1][0] + a[1][2]*b[2][0]],
            [a[2][0]*b[0][0] + a[2][1]*b[1][0] + a[2][2]*b[2][0]],
        ]
        return mult

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
            raise ValueError("please check the initial guess")
        else:
            try:
                self.fitness_func(x)
            except ValueError:
                print("please check the fitness func")

        for it in range(max_itter):

            # jacobian
            jac = []
            # step size
            h = step
            for i in range(self.f_dim):
                jac_row = []
                for j in range(self.x_dim):

                    # standard unit vector
                    ej = [0, 0, 0]
                    ej[j] = 1
                    ej = [[ej[0]], [ej[1]], [ej[2]]]

                    # if condition to simplyfy the jacobian matrix
                    if j < i:
                        df_dx = 0
                    else:
                        fx_hej = np.array([[x[0][0]+ej[0][0]*h],
                                           [x[1][0]+ej[1][0]*h],
                                           [x[2][0]+ej[2][0]*h]])
                        df_dx = (self.fitness_func(fx_hej)[i, 0] - self.fitness_func(x)[i, 0])/h

                    jac_row.append(df_dx)

                jac.append(jac_row)

            jac = np.array(jac)

            delta_x = BaseProblem.matmul(BaseProblem.inverse_mat(jac), self.fitness_func(x))
            delta_x_2norm = math.sqrt(delta_x[0][0]**2 + delta_x[1][0]**2 + delta_x[2][0]**2)

            # check condition ||delta_x|| 2 norm
            if delta_x_2norm < x_tol:
                break

            x = x - delta_x

        return list(x.flatten())
