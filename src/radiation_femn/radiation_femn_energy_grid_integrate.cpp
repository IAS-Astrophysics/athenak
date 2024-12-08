//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_energy_grid_integrate.cpp
//  \brief implements functions to integrate on the energy grid

#include <iostream>
#include <gsl/gsl_integration.h>

#include "athena.hpp"
#include "radiation_femn.hpp"
#include "radiation_femn/radiation_femn_energy_grid.hpp"

namespace raditionfemn
{
    inline Real energy_basis_fd(int m, Real x)
    {
        return 1. / (Kokkos::exp(x) + 1.);
    }

    inline Real energy_basis_fd_exp(int m, Real x)
    {
        return 1. / (Kokkos::exp(-x) + 1.);
    }

    inline Real der_energy_basis_fd(int m, Real x)
    {
        return 1. / (-2. - 2. * Kokkos::cosh(x));
    }

    inline Real IntegrateMatrixEnergyGaussLaguerre(int m, int n,
                                                   const HostArray1D<Real>&
                                                   scheme_weights,
                                                   const HostArray2D<Real>& scheme_points,
                                                   int matrixchoice)
    {
        Real result = 0.;

        switch (matrixchoice)
        {
        case 0:
            for (size_t i = 0; i < scheme_weights.size(); i++)
            {
                result += scheme_weights(i)
                    * scheme_points(i, 0) * scheme_points(i, 0)
                    * energy_basis_fd_exp(m, scheme_points(i, 0))
                    * energy_basis_fd(n, scheme_points(i, 0));
            }
            break;
        case 1:
            for (size_t i = 0; i < scheme_weights.size(); i++)
            {
                result += scheme_weights(i)
                    * scheme_points(i, 0) * scheme_points(i, 0) * scheme_points(i, 0)
                    * energy_basis_fd_exp(m, scheme_points(i, 0))
                    * der_energy_basis_fd(n, scheme_points(i, 0));
            }
            break;
        default: result = -42.;
        }
        return result;
    }

    void GenerateQuadratureGaussLaguerre(int degree, int alpha,
                                         const HostArray1D<Real>& scheme_weights,
                                         const HostArray2D<Real>& scheme_points) {

    }
} // namespace radiationfemn
