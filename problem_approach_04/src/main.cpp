/*
Created on 11.11.2022
@author: Hasal
*/

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include "ExampleProblem.h"

using namespace std;
using namespace Eigen;


class Problem: public ExampleProblem {

    public:

        MatrixXd evolve(MatrixXd x0, int max_itter = 50, double x_tol = 1E-6, double step = 1E-6) {
            /* calculates the solution for x (roots),  where f(x)=0
             using the newtons methods, iteratively */

            Matrix <double , x_dim, 1 > x ;
            x = x0;

            for(int it = 0; it < max_itter; it++) {
            cout << "iteration: " << it <<  endl;

            // jacobian
            Matrix <double , f_dim, x_dim> jac ;
            // step size
            float h = step;

                for(int i = 0; i < f_dim; i++) {


                    for(int j = 0; j < x_dim; j++) {

                        // unit vector
                        MatrixXd ej = MatrixXd::Zero(x_dim, 1);
                        ej(j) = 1;

                        // if condition to simplyfy the jacobian matrix
                        if (j < i) {
                            double df_dx = 0;
                            jac(i, j) = df_dx;
                        } else {
                            double df_dx = (this->fitnessFunc(x + h*ej)(i, 0) - this->fitnessFunc(x)(i, 0))/h;
                            jac(i, j) = df_dx;
                        }

                    }
                }

                Matrix <double , x_dim, 1 > delta_x ;
                delta_x = jac.inverse()*(this->fitnessFunc(x));
                double delta_x_2norm = delta_x.norm();

                // check condition ||delta_x|| 2 norm
                if (delta_x_2norm < x_tol){
                    break;
                }

                x = x - delta_x;

            }


            return x;
        }
};


int main() {

    Problem prob1;
    Matrix <double , 3, 1 > x0 ;
    x0 << 10,
          100,
          0;

    Matrix <double , 3, 1 > solution ;

    solution = prob1.evolve(x0, 100);
    cout << "The solution is:\n" << solution << endl;

    return 0;
}
