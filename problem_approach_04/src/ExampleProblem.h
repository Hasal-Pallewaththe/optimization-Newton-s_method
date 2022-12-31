#ifndef EXAMPLEPROBLEM_H
#define EXAMPLEPROBLEM_H

#include <eigen3/Eigen/Core>
#include <math.h>
#include "BaseProblem.h"


class ExampleProblem : public BaseProblem {
    // an example problem - for testing
    // the given function is implemented here
    public:
        const static int x_dim = 3;
        const static int f_dim = 3;

        Eigen:: MatrixXd fitnessFunc(Eigen:: MatrixXd x) {

            double f1 = x(0)/x(1) + x(2)/x(0);
            double f2 = 0.5*x(1)*x(1)*x(1)  - 250*x(1)*x(2) - 75000*x(2)*x(2);
            double f3 = exp(-x(2)) + x(2)*exp(1);

            Eigen:: Matrix <double , 3, 1 > f ;
            f <<    f1,
                    f2,
                    f3;

            return f;
        }

};

#endif  //EXAMPLEPROBLEM_H