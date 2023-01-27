//========================================================================================
// Radiation FEM_N code for Athena
// Copyright (C) 2023 Maitraya Bhattacharyya <mbb6217@psu.edu> and David Radice <dur566@psu.edu>
// AthenaXX copyright(C) James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_linalg.cpp
//  \brief implementation of the matrix inverse routines

#include <cmath>
#include <stdexcept>
#include "radiation_femn/radiation_femn.hpp"

namespace radiationfemn {

    template<size_t N>
    void RadiationFEMN::CGSolve(double (&A)[N][N], double (&b)[N], double (&xinit)[N], double (&x)[N], double tolerance) {
        /*!
         * \brief Solve a linear system of equations using the Conjugate gradient method.
         *
         * Solves the system:
         *              A x + b = 0
         * where A is a positive definite N x N symmetric matrix and x, b are N x 1 column vectors.
         *
         * Inputs:
         *      A: An N x N symmetric positive definite matrix of type double
         *      b: An N x 1 comlumn vector of type double.
         *      xinit: The initial guess of the solution, N x 1 column vector of doubles.
         *      x: The solution of the linear system of equations.
         *      tolerance: The accuracy upto which the solution is calculated.
         */

        double residual[N] = {0.};
        double residual_norm(0.);
        double p[N] = {0.};

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                residual[i] += A[i][j] * xinit[j];
            }
            residual[i] += - b[i];

            p[i] = - residual[i];
            residual_norm += residual[i] * residual[i];
            x[i] = xinit[i];
        }

        residual_norm = sqrt(residual_norm);
        int niters(0);

        while (residual_norm >= tolerance) {

            double A_dot_p[N] = {0.};
            double res_dot_res(0.);
            double p_dot_Ap(0.);

            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    A_dot_p[i] += A[j][i] * p[j];
                }
                res_dot_res += residual[i] * residual[i];
                p_dot_Ap += p[i] * A_dot_p[i];
            }

            double alpha = res_dot_res / p_dot_Ap;
            double res_dot_res_new(0.);

            for (int i = 0; i < N; i++) {
                x[i] = x[i] + alpha * p[i];
                residual[i] = residual[i] + alpha * A_dot_p[i];
                res_dot_res_new += residual[i] * residual[i];
            }

            double beta = res_dot_res_new / res_dot_res;
            residual_norm = 0.;

            for (int i = 0; i < N; i++) {
                p[i] = - residual[i] + beta * p[i];
                residual_norm += residual[i] * residual[i];
            }

            residual_norm = sqrt(residual_norm);

            niters++;

            if(niters > 1e3) {
                throw std::runtime_error("The number of iterations for the Conjugate Gradient method exceeds the maximum allowed number!");
            }
        }

    }

    template<size_t N>
    void RadiationFEMN::CGMatrixInverse(double (&mat)[N][N], double (&guess)[N][N], double (&matinv)[N][N]) {

        for (int i = 0; i < N; i++) {

            double b[N] = {0.};
            double xinit[N] = {0.};
            double x[N] = {0.};

            for (int j = 0; j < N; j++) {
                b[j] = (i == j);
                xinit[j] = guess[j][i];
            }

            CGSolve(mat, b, xinit, x);

            for (int j = 0; j < N; j++) {
                matinv[j][i] = x[j];
            }
        }
    }
} // namespace radiationfemn