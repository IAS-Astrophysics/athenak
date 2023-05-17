#ifndef ATHENA_RADIATION_FEMN_GEODESIC_GRID_MATRICES_HPP
#define ATHENA_RADIATION_FEMN_GEODESIC_GRID_MATRICES_HPP

//========================================================================================
// GR radiation code for AthenaK with FEM_N & FP_N
// Copyright (C) 2023 Maitraya Bhattacharyya <mbb6217@psu.edu> and David Radice <dur566@psu.edu>
// AthenaXX copyright(C) James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn.hpp
//  \brief Matrices for the FEM_N and FP_N schemes for GR Boltzmann transport

#include "athena.hpp"

namespace radiationfemn {

// ---------------------------------------------------------------------
// Geodesic grid generation functions (radiation_femn_geodesic_grid.cpp)
void
GeodesicGridBaseGenerate(int &geogrid_level,
                         int &geogrid_num_points,
                         int &geogrid_num_edges,
                         int &geogrid_num_triangles,
                         HostArray1D<Real> &x,
                         HostArray1D<Real> &y,
                         HostArray1D<Real> &z,
                         HostArray1D<Real> &r,
                         HostArray1D<Real> &theta,
                         HostArray1D<Real> &phi,
                         HostArray2D<int> &edges,
                         HostArray2D<int> &triangles);    // Generate base geodesic grid
void GeodesicGridRefine(int &geogrid_level,
                        int &geogrid_num_points,
                        int &geogrid_num_edges,
                        int &geogrid_num_triangles,
                        HostArray1D<Real> &x,
                        HostArray1D<Real> &y,
                        HostArray1D<Real> &z,
                        HostArray1D<Real> &r,
                        HostArray1D<Real> &theta,
                        HostArray1D<Real> &phi,
                        HostArray2D<int> &edges,
                        HostArray2D<int> &triangles);          // Refine geodesic grid by one level
void CartesianToSpherical(double xvar,
                          double yvar,
                          double zvar,
                          double &rvar,
                          double &thetavar,
                          double &phivar);   // Convert from Cartesian to spherical coordinates
int FindEdgesIndex(int e1, int e2, HostArray2D<int> &edges);  // Given two edge indices, find

// ---------------------------------------------------------------------------
// basis functions & helper functions (radiation_femn_geodesic_grid_basis.cpp)
void BarycentricToCartesian(double x1,
                            double y1,
                            double z1,
                            double x2,
                            double y2,
                            double z2,
                            double x3,
                            double y3,
                            double z3,
                            double xi1,
                            double xi2,
                            double xi3,
                            double &xval,
                            double &yval,
                            double &zval);

// Type 1: 'Overlapping tent' (Default FEM_N choice)
double FEMBasis1Type1(double xi1, double xi2, double xi3);
double FEMBasis2Type1(double xi1, double xi2, double xi3);
double FEMBasis3Type1(double xi1, double xi2, double xi3);

// Type 2: 'Non-overlapping tent'
double FEMBasis1Type2(double xi1, double xi2, double xi3);
double FEMBasis2Type2(double xi1, double xi2, double xi3);
double FEMBasis3Type2(double xi1, double xi2, double xi3);

// Type 3: 'Overlapping honeycomb'
double FEMBasis1Type3(double xi1, double xi2, double xi3);
double FEMBasis2Type3(double xi1, double xi2, double xi3);
double FEMBasis3Type3(double xi1, double xi2, double xi3);

// Type 4: 'Non-overlapping honeycomb' (S_N choice)
double FEMBasis1Type4(double xi1, double xi2, double xi3);
double FEMBasis2Type4(double xi1, double xi2, double xi3);
double FEMBasis3Type4(double xi1, double xi2, double xi3);

// FEM basis, pick from type
double FEMBasis(double xi1, double xi2, double xi3, int basis_index, int basis_choice);

// some other useful functions
double FEMBasisABasisB(int a, int b, int t1, int t2, int t3, double xi1, double xi2, double xi3, int basis_choice);
double FEMBasisA(int a, int t1, int t2, int t3, double xi1, double xi2, double xi3, int basis_choice);
double dFEMBasisdxi(double xi1, double xi2, double xi3, int basis_index, int basis_choice, int xi_index);

double PartialFEMBasisBwithoute(int ihat,
                                int a,
                                int t1,
                                int t2,
                                int t3,
                                double x1,
                                double y1,
                                double z1,
                                double x2,
                                double y2,
                                double z2,
                                double x3,
                                double y3,
                                double z3,
                                double xi1,
                                double xi2,
                                double xi3,
                                int basis_choice);

double CosPhiSinTheta(double x1,
                      double y1,
                      double z1,
                      double x2,
                      double y2,
                      double z2,
                      double x3,
                      double y3,
                      double z3,
                      double xi1,
                      double xi2,
                      double xi3);
double SinPhiSinTheta(double x1,
                      double y1,
                      double z1,
                      double x2,
                      double y2,
                      double z2,
                      double x3,
                      double y3,
                      double z3,
                      double xi1,
                      double xi2,
                      double xi3);
double CosTheta(double x1,
                double y1,
                double z1,
                double x2,
                double y2,
                double z2,
                double x3,
                double y3,
                double z3,
                double xi1,
                double xi2,
                double xi3);
double SinPhiCosecTheta(double x1,
                        double y1,
                        double z1,
                        double x2,
                        double y2,
                        double z2,
                        double x3,
                        double y3,
                        double z3,
                        double xi1,
                        double xi2,
                        double xi3);
double CosPhiCosTheta(double x1,
                      double y1,
                      double z1,
                      double x2,
                      double y2,
                      double z2,
                      double x3,
                      double y3,
                      double z3,
                      double xi1,
                      double xi2,
                      double xi3);
double CosPhiCosecTheta(double x1,
                        double y1,
                        double z1,
                        double x2,
                        double y2,
                        double z2,
                        double x3,
                        double y3,
                        double z3,
                        double xi1,
                        double xi2,
                        double xi3);
double SinPhiCosTheta(double x1,
                      double y1,
                      double z1,
                      double x2,
                      double y2,
                      double z2,
                      double x3,
                      double y3,
                      double z3,
                      double xi1,
                      double xi2,
                      double xi3);
double SinTheta(double x1,
                double y1,
                double z1,
                double x2,
                double y2,
                double z2,
                double x3,
                double y3,
                double z3,
                double xi1,
                double xi2,
                double xi3);
double mom_by_energy(int mu,
                     double x1,
                     double y1,
                     double z1,
                     double x2,
                     double y2,
                     double z2,
                     double x3,
                     double y3,
                     double z3,
                     double xi1,
                     double xi2,
                     double xi3);

// Find triangles which share an edge
void FindTriangles(int a, int b, const HostArray2D<int> &triangles, HostArray2D<int> edge_triangles, bool &is_edge);

// -------------------------------------------------------------------------------------------------------------------------------------------
// Integration routines over geodesic grid (radiation_femn_geodesic_grid_matrix_integrate.cpp and radiation_femn_geodesic_grid_quadrature.cpp)
void LoadQuadrature(int &scheme_num_points, HostArray1D<Real> scheme_weights, HostArray2D<Real> scheme_points);
double CalculateDeterminantJacobian(double x1,
                                    double y1,
                                    double z1,
                                    double x2,
                                    double y2,
                                    double z2,
                                    double x3,
                                    double y3,
                                    double z3,
                                    double xi1,
                                    double xi2,
                                    double xi3);
double IntegrateMatrixSphericalTriangle(int a,
                                        int b,
                                        int basis,
                                        int t1,
                                        int t2,
                                        int t3,
                                        const HostArray1D<Real> &x,
                                        const HostArray1D<Real> &y,
                                        const HostArray1D<Real> &z,
                                        const HostArray1D<Real> &scheme_weights,
                                        const HostArray2D<Real> &scheme_points,
                                        int matrixnumber,
                                        int nu = -42,
                                        int mu = -42,
                                        int ihat = -42);
double IntegrateMatrix(int a,
                       int b,
                       int basis,
                       const HostArray1D<Real> &x,
                       const HostArray1D<Real> &y,
                       const HostArray1D<Real> &z,
                       const HostArray1D<Real> &scheme_weights,
                       const HostArray2D<Real> &scheme_points,
                       const HostArray2D<int> &triangles,
                       int matrixchoice);
double RealSphericalHarmonic(int l, int m, double phi, double theta);
double IntegrateMatrixFPN(int la,
                          int ma,
                          int lb,
                          int mb,
                          const HostArray1D<Real> &scheme_weights,
                          const HostArray2D<Real> &scheme_points,
                          int matrixchoice);
} // namespace radiationfemn

#endif //ATHENA_RADIATION_FEMN_GEODESIC_GRID_MATRICES_HPP