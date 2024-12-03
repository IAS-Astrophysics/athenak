//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file bns_nurates_calc_opacities.cpp
//! \brief problem generator for calculating opacities with nurates

// C headers

// C++ headers
#include <cmath>      // sqrt()
#include <cstdio>     // fopen(), fprintf(), freopen()
#include <cstring>    // strcmp()
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <string>     // c_str()
#include <random>
#include <chrono>

// Athena++ headers
#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "bns_nurates/opacities.hpp"
#include "bns_nurates/m1_opacities.hpp"
#include "bns_nurates/integration.hpp"
#include "bns_nurates/distribution.hpp"
#include "bns_nurates/constants.hpp"
#include "bns_nurates/functions.hpp"

void TestM1OpacitiesBenchmarks(int nx, int mb_nx, const char* filepath, bool print_out) {

  printf("Data_directory: %s\n", filepath);

  FILE *fptr;
  fptr = fopen(filepath, "r");
  if (fptr == NULL) {
    printf("%s: The file %s does not exist!\n", __FILE__, filepath);
    exit(1);
  }

  // store columns here
  int num_data = 102;
  double e_nu;
  Kokkos::View<int*, LayoutWrapper, HostMemSpace> h_zone("zone", num_data);
  Kokkos::View<double*, LayoutWrapper, HostMemSpace> h_r("r", num_data);
  Kokkos::View<double*, LayoutWrapper, HostMemSpace> h_rho("rho", num_data);
  Kokkos::View<double*, LayoutWrapper, HostMemSpace> h_T("T", num_data);
  Kokkos::View<double*, LayoutWrapper, HostMemSpace> h_Ye("Ye", num_data);
  Kokkos::View<double*, LayoutWrapper, HostMemSpace> h_mu_e("mu_e", num_data);
  Kokkos::View<double*, LayoutWrapper, HostMemSpace> h_mu_hat("mu_hat", num_data);
  Kokkos::View<double*, LayoutWrapper, HostMemSpace> h_Yh("Yh", num_data);
  Kokkos::View<double*, LayoutWrapper, HostMemSpace> h_Ya("Ya", num_data);
  Kokkos::View<double*, LayoutWrapper, HostMemSpace> h_Yp("Yp", num_data);
  Kokkos::View<double*, LayoutWrapper, HostMemSpace> h_Yn("Yn", num_data);
  Kokkos::View<double*, LayoutWrapper, HostMemSpace> h_em_nue("em_nue", num_data);
  Kokkos::View<double*, LayoutWrapper, HostMemSpace> h_l_nue_inv("l_nue_inv", num_data);
  Kokkos::View<double*, LayoutWrapper, HostMemSpace> h_em_anue("em_anue", num_data);
  Kokkos::View<double*, LayoutWrapper, HostMemSpace> h_l_anue_inv("l_anue_inv", num_data);

  // read in the data file
  int i = 0;
  char line[1000];
  while (fgets(line, sizeof(line), fptr) != NULL) {
    if (line[1] == '#' && i == 0) {
      sscanf(line + 14, "%lf\n", &e_nu);
      continue;
    } else if (line[1] == '#' && i != 0) {
      continue;
    }

    sscanf(line, "%d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
           &h_zone(i), &h_r(i), &h_rho(i), &h_T(i), &h_Ye(i), &h_mu_e(i), &h_mu_hat(i), &h_Yh[i], &h_Ya[i], &h_Yp(i), &h_Yn(i), &h_em_nue(i), &h_l_nue_inv(i), &h_em_anue(i), &h_l_anue_inv(i));
    i++;
  }

  fclose(fptr);

  // construct arrays based on meshblock dimensions
  int npts = mb_nx * mb_nx * mb_nx;
  Kokkos::View<double*, LayoutWrapper, HostMemSpace> mb_h_rho("mb_h_rho", npts);
  Kokkos::View<double*, LayoutWrapper, HostMemSpace> mb_h_T("mb_h_T", npts);
  Kokkos::View<double*, LayoutWrapper, HostMemSpace> mb_h_Ye("mb_h_Ye", npts);
  Kokkos::View<double*, LayoutWrapper, HostMemSpace> mb_h_mu_e("mb_h_mu_e", npts);
  Kokkos::View<double*, LayoutWrapper, HostMemSpace> mb_h_mu_hat("mb_h_mu_hat", npts);
  Kokkos::View<double*, LayoutWrapper, HostMemSpace> mb_h_Yp("mb_h_Yp", npts);
  Kokkos::View<double*, LayoutWrapper, HostMemSpace> mb_h_Yn("mb_h_Yn", npts);

  DvceArray1D<Real> mb_d_rho("mb_d_rho", npts);
  DvceArray1D<Real> mb_d_T("mb_d_T", npts);
  DvceArray1D<Real> mb_d_Ye("mb_d_Ye", npts);
  DvceArray1D<Real> mb_d_mu_e("mb_d_mu_e", npts);
  DvceArray1D<Real> mb_d_mu_hat("mb_d_mu_hat", npts);
  DvceArray1D<Real> mb_d_Yp("mb_d_Yp", npts);
  DvceArray1D<Real> mb_d_Yn("mb_d_Yn", npts);

  printf("Building data arrays %d points ...\n", npts);

  std::random_device random_device;
  std::mt19937 generate(random_device());
  std::uniform_int_distribution<> uniform_int_distribution(0, num_data - 1);
  for (int i = 0; i < npts; i++) {
    int index = uniform_int_distribution(generate);
    //printf("i: %d, random index: %d\n", i, index);

    mb_h_rho(i) = h_rho(index);
    mb_h_T(i) = h_T(index);
    mb_h_Ye(i) = h_Ye(index);
    mb_h_mu_e(i) = h_mu_e(index);
    mb_h_mu_hat(i) = h_mu_hat(index);
    mb_h_Yp(i) = h_Yp(index);
    mb_h_Yn(i) = h_Yn(index);
  }

  Kokkos::deep_copy(mb_d_rho, mb_h_rho);
  Kokkos::deep_copy(mb_d_T, mb_h_T);
  Kokkos::deep_copy(mb_d_Ye, mb_h_Ye);
  Kokkos::deep_copy(mb_d_mu_e, mb_h_mu_e);
  Kokkos::deep_copy(mb_d_mu_hat, mb_h_mu_hat);
  Kokkos::deep_copy(mb_d_Yp, mb_h_Yp);
  Kokkos::deep_copy(mb_d_Yn, mb_h_Yn);

  printf("Data arrays generated.\n");

  DvceArray1D<Real> d_diff_distribution("diff_distribution", num_data);
  DvceArray2D<Real> d_coeffs_eta_0("coeffs_eta_0", num_data, 4);
  DvceArray2D<Real> d_coeffs_eta("coeffs_eta", num_data, 4);
  DvceArray2D<Real> d_coeffs_kappa_0_a("coeffs_kappa_0_a", num_data, 4);
  DvceArray2D<Real> d_coeffs_kappa_a("coeffs_kappa_a", num_data, 4);
  DvceArray2D<Real> d_coeffs_kappa_s("coeffs_kappa_s", num_data, 4);

  DvceArray1D<Real> h_diff_distribution("h_diff_distribution", num_data);
  DvceArray2D<Real> h_coeffs_eta_0("h_coeffs_eta_0", num_data, 4);
  DvceArray2D<Real> h_coeffs_eta("h_coeffs_eta", num_data, 4);
  DvceArray2D<Real> h_coeffs_kappa_0_a("h_coeffs_kappa_0_a", num_data, 4);
  DvceArray2D<Real> h_coeffs_kappa_a("h_coeffs_kappa_a", num_data, 4);
  DvceArray2D<Real> h_coeffs_kappa_s("h_coeffs_kappa_s", num_data, 4);

  printf("Generating quadratures with %d points ...\n", nx);
  MyQuadrature my_quad = {.type = kGauleg,
      .alpha = -42.,
      .dim = 1,
      .nx = nx,
      .ny = 1,
      .nz = 1,
      .x1 = 0.,
      .x2 = 1.,
      .y1 = -42.,
      .y2 = -42.,
      .z1 = -42.,
      .z2 = -42.,
      .points = {0},
      .w = {0}};
  GaussLegendre(&my_quad);
  printf("Quadratures generated.\n");

  auto start = std::chrono::high_resolution_clock::now();
  par_for("test_opacities_benchmarks_compute_opacities_loop", DevExeSpace(), 0, npts -1,
    KOKKOS_LAMBDA(const int &i) {
    //printf("Entered in Kokkos parallel_for loop\n");

    GreyOpacityParams my_grey_opacity_params;
    my_grey_opacity_params.opacity_flags = {.use_abs_em          = 1,
        .use_pair            = 1,
        .use_brem            = 1,
        .use_inelastic_scatt = 1,
        .use_iso             = 1};

    // populate EOS parameters from table
    my_grey_opacity_params.eos_pars.mu_e = mb_d_mu_e(i);
    my_grey_opacity_params.eos_pars.mu_p = 0.;
    my_grey_opacity_params.eos_pars.mu_n = mb_d_mu_hat(i) + kQ;
    my_grey_opacity_params.eos_pars.temp = mb_d_T(i);
    my_grey_opacity_params.eos_pars.yp = mb_d_Yp(i);
    my_grey_opacity_params.eos_pars.yn = mb_d_Yn(i);
    my_grey_opacity_params.eos_pars.nb = mb_d_rho(i) / kMu;


    // Opacity parameters (corrections all switched off)
    my_grey_opacity_params.opacity_pars = {.use_dU             = false,
        .use_dm_eff         = false,
        .use_WM_ab          = false,
        .use_WM_sc          = false,
        .use_decay          = false,
        .use_BRT_brem       = false,
        .use_NN_medium_corr = false,
        .neglect_blocking   = false};


    // Distribution parameters
    my_grey_opacity_params.distr_pars =
        NuEquilibriumParams(&my_grey_opacity_params.eos_pars); // consider neutrino distribution function at equilibrium

    // M1 parameters
    ComputeM1DensitiesEq(&my_grey_opacity_params.eos_pars,
                         &my_grey_opacity_params.distr_pars,
                         &my_grey_opacity_params.m1_pars);

    for (int idx = 0; idx < total_num_species; idx++) {
      my_grey_opacity_params.m1_pars.chi[idx] = 1. / 3.;
      my_grey_opacity_params.m1_pars.J[idx] =
          my_grey_opacity_params.m1_pars.J[idx] * kMeV;
    }

    //printf("Computing M1 coefficients\n");

    M1Opacities coeffs = ComputeM1Opacities(&my_quad, &my_quad, &my_grey_opacity_params);
    auto testval = coeffs.eta[id_nue];
    if(print_out)
    {
      printf("%d %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e\n",
              i,
              coeffs.eta_0[id_nue], coeffs.eta_0[id_anue], coeffs.eta_0[id_nux], coeffs.eta_0[id_anux],
              coeffs.eta[id_nue], coeffs.eta[id_anue], coeffs.eta[id_nux], coeffs.eta[id_anux],
              coeffs.kappa_0_a[id_nue], coeffs.kappa_0_a[id_anue], coeffs.kappa_0_a[id_nux], coeffs.kappa_0_a[id_anux],
              coeffs.kappa_a[id_nue], coeffs.kappa_a[id_anue], coeffs.kappa_a[id_nux], coeffs.kappa_a[id_anux],
              coeffs.kappa_s[id_nue], coeffs.kappa_s[id_anue], coeffs.kappa_s[id_nux], coeffs.kappa_s[id_anux]);
    }
  });

  Kokkos::fence();

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  double time_taken_seconds = duration.count();

  printf("Opacities computed.\n");
  printf("Number of quadrature points = %d\n", nx);
  printf("Meshblock Nx = %d\n", mb_nx);
  printf("[Nx x Nx x Nx] = %d\n", npts);
  printf("Total time taken = %lf\n", time_taken_seconds);
  printf("zone-cycles/second = %e\n", double(npts)/time_taken_seconds);

}

void ProblemGenerator::CalcM1Opacities(ParameterInput *pin, const bool restart)
{

  printf("=================================================== \n");
  printf("Benchmark tests for bns_nurates ... \n");
  printf("=================================================== \n");

  int nx, mb_nx;

  auto filepath = pin->GetString("problem", "input_data_path");
  auto print_out = pin->GetOrAddBoolean("problem", "print_data", false);
  std::cout << "Enter the number of quadrature points: ";
  std::cin >> nx;

  if (std::cin.fail()) {
    std::cerr << "Invalid input!" << std::endl;
  }

  std::cout << "Enter Nx for meshblock with [Nx x Nx x Nx] points: ";
  std::cin >> mb_nx;

  if (std::cin.fail()) {
    std::cerr << "Invalid input!" << std::endl;
  }

  TestM1OpacitiesBenchmarks(nx, mb_nx, filepath.c_str(), print_out);

}