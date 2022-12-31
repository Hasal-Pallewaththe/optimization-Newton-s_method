# Created on 11.11.2022
# author: Hasal

from example_problem import ExampleProblem

f = ExampleProblem()
f.set_f_dim(3)
f.set_x_dim(3)
x_solution = f.evolve(x0=[10, 100, 0], max_itter=50)
print("solution: ", x_solution)
