#ifndef BASEPROBLEM_H
#define BASEPROBLEM_H

#include <eigen3/Eigen/Core>
#include <iostream>


class BaseProblem {

    // a generic class with necessary tools for both problem definition and calculation
    public:

        Eigen:: MatrixXd fitnessFunc(Eigen:: MatrixXd x);

        Eigen:: MatrixXd evolve(Eigen:: MatrixXd x0, int max_itter, double x_tol, double step);

};

#endif  // BASEPROBLEM_H
