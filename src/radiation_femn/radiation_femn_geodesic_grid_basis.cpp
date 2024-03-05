//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_basis.cpp
//  \brief implements radiation FEM/FPN basis functions and helpers functions

#include <iostream>
#include <gsl/gsl_sf_legendre.h>
#include <gsl/gsl_sf.h>

#include "athena.hpp"
#include "radiation_femn/radiation_femn_geodesic_grid.hpp"

namespace radiationfemn
{
    /* Convert Barycentric coordinates to Cartesian coordinates given vertices of triangle
     *
     * Inputs:
     * (x1,y1,z1), (x2,y2,z2), (x3,y3,z3):  the three triangle vertices in cartesian coordinates
     * (xi1, xi2, xi3):the barycentric coordinates of a point inside the triangle
     *
     * Output:
     * (xval, yval, zval): the cartesian coordinates of the point
     */
    inline void BarycentricToCartesian(Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3,
                                       Real xi1, Real xi2, Real xi3, Real& xval, Real& yval, Real& zval)
    {
        xval = xi1 * x1 + xi2 * x2 + xi3 * x3;
        yval = xi1 * y1 + xi2 * y2 + xi3 * y3;
        zval = xi1 * z1 + xi2 * z2 + xi3 * z3;
    }

    /* Given index numbers of two vertices, finds if they share an edge and if so, return triangle info
     *
     * If a = b, this return all triangles that share the vertex
     *
     * Inputs:
     * a, b: index number of vertices
     * triangles: triangle information of the geodesic grid
     *
     * Outputs:
     * edge_triangles: the vertex information of shared edge(s)
     * is_edge: bool for if the vertices share an edge or not
     */
    void FindTriangles(int a, int b, const HostArray2D<int>& triangles, HostArray2D<int>& edge_triangles, bool& is_edge)
    {
        is_edge = false;
        Kokkos::realloc(edge_triangles, 6, 3);
        Kokkos::deep_copy(edge_triangles, -42.);

        if (a == b)
        {
            size_t index{0};
            for (size_t i = 0; i < triangles.size() / 3; i++)
            {
                {
                    if (triangles(i, 0) == a || triangles(i, 1) == a || triangles(i, 2) == a)
                    {
                        is_edge = true;
                        edge_triangles(index, 0) = triangles(i, 0);
                        edge_triangles(index, 1) = triangles(i, 1);
                        edge_triangles(index, 2) = triangles(i, 2);
                        index++;
                    }
                }
            }
        }
        else if (a != b)
        {
            size_t index{0};
            for (size_t i = 0; i < triangles.size() / 3; i++)
            {
                if ((triangles(i, 0) == a && triangles(i, 1) == b) || (triangles(i, 0) == a && triangles(i, 2) == b) || (triangles(i, 0) == b && triangles(i, 1) == a)
                    || (triangles(i, 0) == b && triangles(i, 2) == a) || (triangles(i, 1) == a && triangles(i, 2) == b) || (triangles(i, 1) == b && triangles(i, 2) == a))
                {
                    is_edge = true;
                    edge_triangles(index, 0) = triangles(i, 0);
                    edge_triangles(index, 1) = triangles(i, 1);
                    edge_triangles(index, 2) = triangles(i, 2);
                    index++;
                }
            }
        }
    }


    /* FEM basis functions (in barycentric coordinates)
     *
     * basis_choice: 1 => overlapping tent, 2 => small tent, 3 => overlapping honeycomb, 4 => small honeycomb
     * basis_index: 1 => basis peaked at xi1 = 1, 2 => basis peaked at xi2 = 1, 3 => basis peaked at xi3 = 1
     */
    using fem_basis_type = Real (*)(Real, Real, Real);
    const fem_basis_type fem_basis_fn[3][4] = {
        {fem_overtent_index1, fem_smalltent_index1, fem_overhoney_index1, fem_smallhoney_index1},
        {fem_overtent_index2, fem_smalltent_index2, fem_overhoney_index2, fem_smallhoney_index2},
        {fem_overtent_index3, fem_smalltent_index3, fem_overhoney_index3, fem_smallhoney_index3}
    };

    inline Real fem_basis(Real xi1, Real xi2, Real xi3, int basis_index, int basis_choice)
    {
        return fem_basis_fn[basis_index - 1][basis_choice - 1](xi1, xi2, xi3);
    }

    // Overlapping tent basis
    inline Real fem_overtent_index1(Real xi1, Real xi2, Real xi3) { return 2. * xi1 + xi2 + xi3 - 1.; }
    inline Real fem_overtent_index2(Real xi1, Real xi2, Real xi3) { return xi1 + 2. * xi2 + xi3 - 1.; }
    inline Real fem_overtent_index3(Real xi1, Real xi2, Real xi3) { return xi1 + xi2 + 2. * xi3 - 1.; }

    // Small tent basis
    inline Real fem_smalltent_index1(Real xi1, Real xi2, Real xi3) { return (xi1 >= 0.5) * (xi1 - xi2 - xi3); }
    inline Real fem_smalltent_index2(Real xi1, Real xi2, Real xi3) { return (xi2 >= 0.5) * (xi2 - xi3 - xi1); }
    inline Real fem_smalltent_index3(Real xi1, Real xi2, Real xi3) { return (xi3 >= 0.5) * (xi3 - xi1 - xi2); }

    // Overlapping honeycomb
    inline Real fem_overhoney_index1(Real xi1, Real xi2, Real xi3) { return 1.; }
    inline Real fem_overhoney_index2(Real xi1, Real xi2, Real xi3) { return 1.; }
    inline Real fem_overhoney_index3(Real xi1, Real xi2, Real xi3) { return 1.; }

    // Small honeycomb
    inline Real fem_smallhoney_index1(Real xi1, Real xi2, Real xi3) { return (xi1 >= xi2) * (xi1 > xi3) * 1.; }
    inline Real fem_smallhoney_index2(Real xi1, Real xi2, Real xi3) { return (xi2 >= xi3) * (xi2 > xi1) * 1.; }
    inline Real fem_smallhoney_index3(Real xi1, Real xi2, Real xi3) { return (xi3 >= xi1) * (xi3 > xi2) * 1.; }

    /* Derivative of FEM basis wrt barycentric coordinates: dpsi/dxi
     *
     * basis_index: 1 => basis peaked at xi1 = 1, 2 => basis peaked at xi2 = 1, 3 => basis peaked at xi3 = 1
     * xi_index: 1 => xi1, 2 => xi2, 3 => xi3
     */
    const Real fem_basis_derivative[3][3] = {
        {2., 1., 1.},
        {1., 2., 1.},
        {1., 1., 2.}
    };

    Real dfem_dxi(Real xi1, Real xi2, Real xi3, int basis_index, int xi_index)
    {
        return fem_basis_derivative[basis_index - 1][xi_index - 1];
    }

    /* Product of two FEM basis functions evaluated in a triangular element: psi_a psi_b
     *
     * a, b: vertex index over which basis is peaked
     * t1, t2, t3: vertex indices of the triangular element being considered
     * basis_choice: type of basis function (1 => overlapping tent, 2 => small tent, 3 => overlapping honeycomb, 4 => small honeycomb)
     */
    Real fem_basis_ab(int a, int b, int t1, int t2, int t3, Real xi1, Real xi2, Real xi3, int basis_choice)
    {
        int basis_index_a = (a == t1) * 1 + (a == t2) * 2 + (a == t3) * 3;
        int basis_index_b = (b == t1) * 1 + (b == t2) * 2 + (b == t3) * 3;
        return fem_basis(xi1, xi2, xi3, basis_index_a, basis_choice) * fem_basis(xi1, xi2, xi3, basis_index_b, basis_choice);
    }

    /* A FEM basis function evaluated in a triangular element: psi_a
     *
     * a: vertex index over which basis is peaked
     * t1, t2, t3: vertex indices of the triangular element being considered
     * basis_choice: type of basis function (1 => overlapping tent, 2 => small tent, 3 => overlapping honeycomb, 4 => small honeycomb)
     */
    Real fem_basis_a(int a, int t1, int t2, int t3, Real xi1, Real xi2, Real xi3, int basis_choice)
    {
        int basis_index_a = (a == t1) * 1 + (a == t2) * 2 + (a == t3) * 3;
        return fem_basis(xi1, xi2, xi3, basis_index_a, basis_choice);
    }

    /* FPN basis: real spherical harmonics
     *
     * Note: gsl_sf_legendre_sphPlm computes sqrt((2l+1)/(4 pi)) sqrt((l-m)!/(l+m)!) P_m^l(x)
     *
     * Computes:
     *
     * Ylm = sqrt((2l+1)/(2 pi)) sqrt((l-m)!/(l+m)!) cos (m phi) P_m^l(cos theta), m > 0
     *     = sqrt((2l+1)/(4 pi)) P_m^l(cos theta), m = 0
     *     = sqrt((2l+1)/(2 pi)) sqrt((l-|m|)!/(l+|m|)!) sin (|m| phi) P_|m|^l(cos theta), m < 0
     */
    Real fpn_basis_lm(int l, int m, Real phi, Real theta)
    {
        Real result = 0.;
        if (m > 0)
        {
            result = sqrt(2.) * cos(m * phi) * gsl_sf_legendre_sphPlm(l, m, cos(theta));
        }
        else if (m == 0)
        {
            result = gsl_sf_legendre_sphPlm(l, 0, cos(theta));
        }
        else
        {
            result = sqrt(2.) * sin(abs(m) * phi) * gsl_sf_legendre_sphPlm(l, abs(m), cos(theta));
        }
        return result;
    }

    /* Derivative of FPN basis wrt angle
     *
     * var_index: 1 => phi, 2 => theta)
     */
    using dfpn_dOmega_fn = Real (*)(int, int, Real, Real);
    const dfpn_dOmega_fn dfpn_domega_fns[2] = {dfpn_dphi, dfpn_dtheta};

    Real dfpn_dOmega(int l, int m, Real phi, Real theta, int var_index)
    {
        return dfpn_domega_fns[var_index - 1](l, m, phi, theta);
    }

    // dYlm/dphi = - m Yl-m
    inline Real dfpn_dphi(int l, int m, Real phi, Real theta)
    {
        return -m * fpn_basis_lm(l, -m, phi, theta);
    }

    /* dYlm/dtheta
     *
     * Note: Uses (1-x^2) dP^m_l/dx = (m-l-1) P^m_l+1 + (l+1)xP^m_l
     *       When x = -1, 1, dP^m_l/dx = 0
     *       When m = -k < 0, dP^-k_l/dx = (-1)^k (l-k)!/(l+k)! dP^k_l/dx
     */
    inline Real dfpn_dtheta(int l, int m, Real phi, Real theta)
    {
        Real result = 0.;
        const Real x = cos(theta);

        if (abs(x - 1.) < 1e-14)
        {
            return 0.;
        }

        Real der_legendre = -sin(theta) * ((m - l - 1.) * gsl_sf_legendre_sphPlm(l + 1, abs(m), x) + (l + 1) * x * gsl_sf_legendre_sphPlm(l, abs(m), x)) / (1. - x * x);

        if (m > 0)
        {
            result = sqrt(2.) * cos(m * phi) * der_legendre;
        }
        else if (m == 0)
        {
            result = der_legendre;
        }
        else
        {
            result = pow(-1, -m) * (gsl_sf_fact(l + m) / gsl_sf_fact(l - m)) * sqrt(2.) * sin(abs(m) * phi) * der_legendre;
        }
        return result;
    }

    /* Inverse Jacobian P^itilde_ihat [e = 1, itilde = (phi, theta), ihat = (x,y,z)]
     *
     * P^itilde_ihat = [-cosec(theta)*sin(phi)  cosec(theta)*cos(phi)   0]
     *                 [cos(theta)*cos(phi)     cos(theta)*sin(phi)     -sin(theta)]
     */
    using inv_jac_fn = Real (*)(Real, Real);
    const inv_jac_fn inv_jac_fns[2][3] = {
        {
            [](Real phi, Real theta) { return -sin(phi) / sin(theta); },
            [](Real phi, Real theta) { return cos(phi) / sin(theta); },
            [](Real phi, Real theta) { return 0.0; }
        },
        {
            [](Real phi, Real theta) { return cos(phi) * cos(theta); },
            [](Real phi, Real theta) { return sin(phi) * cos(theta); },
            [](Real phi, Real theta) { return -sin(theta); }
        }
    };

    Real inv_jac_itilde_ihat(Real phi, Real theta, int tilde_index, int hat_index)
    {
        return inv_jac_fns[tilde_index - 1][hat_index - 1](phi, theta);
    }

    // x-component of momentum (e = 1): cos(phi) sin(theta)
    Real cos_phi_sin_theta(Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3, Real xi1, Real xi2, Real xi3)
    {
        Real xval, yval, zval;
        BarycentricToCartesian(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3, xval, yval, zval);

        Real rval = sqrt(xval * xval + yval * yval + zval * zval);
        Real thetaval = acos(zval / rval);
        Real phival = atan2(yval, xval);

        return cos(phival) * sin(thetaval);
    }

    // y-component of momentum (e = 1): sin(phi) sin(theta)
    Real sin_phi_sin_theta(Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3, Real xi1, Real xi2, Real xi3)
    {
        Real xval, yval, zval;
        BarycentricToCartesian(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3, xval, yval, zval);

        Real rval = sqrt(xval * xval + yval * yval + zval * zval);
        Real thetaval = acos(zval / rval);
        Real phival = atan2(yval, xval);

        return sin(phival) * sin(thetaval);
    }

    // z-component of momentum (e = 1): cos(theta)
    Real cos_theta(Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3, Real xi1, Real xi2, Real xi3)
    {
        Real xval, yval, zval;
        BarycentricToCartesian(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3, xval, yval, zval);

        Real rval = sqrt(xval * xval + yval * yval + zval * zval);
        Real thetaval = acos(zval / rval);

        return cos(thetaval);
    }

    // Momentum p^mu (e = 1) for FEM
    using mom_fem_fn = Real (*)(Real, Real, Real, Real, Real, Real, Real, Real, Real, Real, Real, Real);
    const mom_fem_fn mom_fem_fns[4] = {
        [](Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3, Real xi1, Real xi2, Real xi3) { return 1.; },
        cos_phi_sin_theta,
        sin_phi_sin_theta,
        cos_theta
    };

    Real mom(int mu, Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3, Real xi1, Real xi2, Real xi3)
    {
        return mom_fem_fns[mu](x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3);
    }

    // Momentum p^mu (e = 1) for FPN
    using fpn_phitheta_fn = Real (*)(Real, Real);
    const fpn_phitheta_fn mom_fpn_fns[4] = {
        [](Real phi, Real theta) { return 1.; },
        [](Real phi, Real theta) { return cos(phi) * sin(theta); },
        [](Real phi, Real theta) { return sin(phi) * sin(theta); },
        [](Real phi, Real theta) { return cos(theta); }
    };

    Real mom(int mu, Real phi, Real theta)
    {
        return mom_fpn_fns[mu](phi, theta);
    }


    /* Derivative of FEM basis with respect to cartesian coordinates on the unit sphere
     *
     * dpsi/dx^ihat = dpsi/dxi^i dxi^i/dx^ihat
     *
     * where dxi^i/dx^ihat is the inverse of the matrix:
     *
     * dx^ihat/dxi^i = [x1 x2 x3]
     *                 [y1 y2 y3]
     *                 [z1 z2 z3]
     */
    using dfem_dxihat_fn = Real (*)(Real, Real, Real, Real, Real, Real, Real, Real, Real, Real, Real, Real, int);
    const dfem_dxihat_fn dfem_dxihat_fns[3] = {
        [](Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3, Real xi1, Real xi2, Real xi3, int basis_index_a)
        {
            return dfem_dxi(xi1, xi2, xi3, basis_index_a, 1) * (y3 * z2 - y2 * z3)
                / (x3 * y2 * z1 - x2 * y3 * z1 - x3 * y1 * z2 + x1 * y3 * z2 + x2 * y1 * z3 - x1 * y2 * z3)
                + dfem_dxi(xi1, xi2, xi3, basis_index_a, 2) * (y3 * z1 - y1 * z3)
                / (-(x3 * y2 * z1) + x2 * y3 * z1 + x3 * y1 * z2 - x1 * y3 * z2 - x2 * y1 * z3 + x1 * y2 * z3)
                + dfem_dxi(xi1, xi2, xi3, basis_index_a, 3) * (y2 * z1 - y1 * z2)
                / (x3 * y2 * z1 - x2 * y3 * z1 - x3 * y1 * z2 + x1 * y3 * z2 + x2 * y1 * z3 - x1 * y2 * z3);
        },
        [](Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3, Real xi1, Real xi2, Real xi3, int basis_index_a)
        {
            return dfem_dxi(xi1, xi2, xi3, basis_index_a, 1) * (x3 * z2 - x2 * z3)
                / (-(x3 * y2 * z1) + x2 * y3 * z1 + x3 * y1 * z2 - x1 * y3 * z2 - x2 * y1 * z3 + x1 * y2 * z3)
                + dfem_dxi(xi1, xi2, xi3, basis_index_a, 2) * (x3 * z1 - x1 * z3)
                / (x3 * y2 * z1 - x2 * y3 * z1 - x3 * y1 * z2 + x1 * y3 * z2 + x2 * y1 * z3 - x1 * y2 * z3)
                + dfem_dxi(xi1, xi2, xi3, basis_index_a, 3) * (x2 * z1 - x1 * z2)
                / (-(x3 * y2 * z1) + x2 * y3 * z1 + x3 * y1 * z2 - x1 * y3 * z2 - x2 * y1 * z3 + x1 * y2 * z3);
        },
        [](Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3, Real xi1, Real xi2, Real xi3, int basis_index_a)
        {
            return dfem_dxi(xi1, xi2, xi3, basis_index_a, 1) * (x3 * y2 - x2 * y3)
                / (x3 * y2 * z1 - x2 * y3 * z1 - x3 * y1 * z2 + x1 * y3 * z2 + x2 * y1 * z3 - x1 * y2 * z3)
                + dfem_dxi(xi1, xi2, xi3, basis_index_a, 2) * (x3 * y1 - x1 * y3)
                / (-(x3 * y2 * z1) + x2 * y3 * z1 + x3 * y1 * z2 - x1 * y3 * z2 - x2 * y1 * z3 + x1 * y2 * z3)
                + dfem_dxi(xi1, xi2, xi3, basis_index_a, 3) * (x2 * y1 - x1 * y2)
                / (x3 * y2 * z1 - x2 * y3 * z1 - x3 * y1 * z2 + x1 * y3 * z2 + x2 * y1 * z3 - x1 * y2 * z3);
        }
    };

    inline Real dfem_dxihat(Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3, Real xi1, Real xi2, Real xi3, int basis_index_a, int idx)
    {
        return dfem_dxihat_fns[idx - 1](x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3, basis_index_a);
    }

    /* Computes R^jhat_ihat = delta^jhat_ihat - n^jhat n_ihat where n^ihat is the normal to the unit sphere
     *
     * R = [1-cos^2(phi)*sin^2(theta)           -cos(phi)*sin^2(theta)*sin(phi) -cos(theta)*cos(phi)*sin(theta)]
     *     [-cos(phi)*sin^2(theta)*sin(phi)     1-sin^2(theta)*sin^2(phi)       -cos(theta)*sin(theta)*sin(phi)]
     *     [-cos(theta)*cos(phi)*sin(theta)     -cos(theta)*sin(theta)*sin(phi)  sin^2(theta)]
     */
    const fpn_phitheta_fn matrix_jac_r[3][3] = {
        {
            [](Real phival, Real thetaval) { return 1. - sin(thetaval) * sin(thetaval) * cos(phival) * cos(phival); },
            [](Real phival, Real thetaval) { return -sin(thetaval) * sin(thetaval) * sin(phival) * cos(phival); },
            [](Real phival, Real thetaval) { return -sin(thetaval) * cos(thetaval) * cos(phival); }
        },
        {
            [](Real phival, Real thetaval) { return -sin(thetaval) * sin(thetaval) * sin(phival) * cos(phival); },
            [](Real phival, Real thetaval) { return 1. - sin(thetaval) * sin(thetaval) * sin(phival) * sin(phival); },
            [](Real phival, Real thetaval) { return -sin(thetaval) * cos(thetaval) * sin(phival); }
        },
        {
            [](Real phival, Real thetaval) { return -sin(thetaval) * cos(thetaval) * cos(phival); },
            [](Real phival, Real thetaval) { return -sin(thetaval) * cos(thetaval) * sin(phival); },
            [](Real phival, Real thetaval) { return sin(thetaval) * sin(thetaval); }
        }

    };

    inline Real matrix_r(Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3, Real xi1, Real xi2, Real xi3, int i, int ihat)
    {
        Real xval, yval, zval;
        BarycentricToCartesian(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3, xval, yval, zval);

        Real rval = sqrt(xval * xval + yval * yval + zval * zval);
        Real thetaval = acos(zval / rval);
        Real phival = atan2(yval, xval);

        return matrix_jac_r[i - 1][ihat - 1](phival, thetaval);
    }

    /* Derivative of the FEM basis functions with respect to cartesian momentum in co-moving frame
     *
     * dpsi/dp^ihat = R^jhat_ihat dpsi/dx^jhat
     *
     * where R^jhat_ihat = delta^jhat_ihat - n^jhat n_ihat (n^ihat is the normal to the unit sphere)
     *
     * Note: basis_choice is redundant for now, only works for overlapping tent!
     */
    Real dfem_dpihat(int ihat, int a, int t1, int t2, int t3, Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3,
                     Real xi1, Real xi2, Real xi3, int basis_choice)
    {
        int basis_index_a = (a == t1) * 1 + (a == t2) * 2 + (a == t3) * 3;

        return dfem_dxihat(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3, basis_index_a, 1) * matrix_r(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3, 1, ihat)
            + dfem_dxihat(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3, basis_index_a, 2) * matrix_r(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3, 2, ihat)
            + dfem_dxihat(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3, basis_index_a, 3) * matrix_r(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3, 3, ihat);
    }
} // namespace radiationfemn
