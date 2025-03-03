//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rad_m1_hybridsj.cpp
//  \brief 1D beam for grey M1

// C++ headers

// Athena++ headers
#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "radiation_m1/radiation_m1_params.hpp"
#include "radiation_m1/radiation_m1_roots_hybridsj.hpp"

class HybridsjFunc {
public:
  KOKKOS_INLINE_FUNCTION
  void operator()(const Real (&x)[M1_MULTIROOTS_DIM],
                  Real (&f)[M1_MULTIROOTS_DIM],
                  Real (&J)[M1_MULTIROOTS_DIM][M1_MULTIROOTS_DIM],
                  radiationm1::SrcParams &pars) {
    Real A = Kokkos::pow(10, 4);

    for (int i = 0; i < M1_MULTIROOTS_DIM; i++) {
      f[i] = 0;
      for (int j = 0; j < M1_MULTIROOTS_DIM; j++) {
        J[i][j] = 0;
      }
    }

    f[0] = A * x[0] * x[1] - 1;
    f[1] = Kokkos::exp(-x[0]) + Kokkos::exp(-x[1]) - (1. + 1. / A);

    J[0][0] = A * x[1];
    J[0][1] = A * x[0];
    J[1][0] = -Kokkos::exp(-x[0]);
    J[1][1] = -Kokkos::exp(-x[1]);
  }
};

KOKKOS_INLINE_FUNCTION
void print_fdf(Real (&x)[M1_MULTIROOTS_DIM], Real (&f)[M1_MULTIROOTS_DIM],
               Real (&J)[M1_MULTIROOTS_DIM][M1_MULTIROOTS_DIM]) {
  /*
  std::cout << "x: " << std::endl;
  for (int i = 0; i < M1_MULTIROOTS_DIM; ++i) {
    std::cout << x[i] << " ";
  }
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << "f: " << std::endl;
  for (int i = 0; i < M1_MULTIROOTS_DIM; ++i) {
    std::cout << f[i] << " ";
  }
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << "J: " << std::endl;
  for (int i = 0; i < M1_MULTIROOTS_DIM; ++i) {
    for (int j = 0; j < M1_MULTIROOTS_DIM; ++j) {
      std::cout << J[i][j] << " ";
    }
    std::cout << std::endl;
  }
  */
}

KOKKOS_INLINE_FUNCTION
void print_f(Real (&x)[M1_MULTIROOTS_DIM], Real (&f)[M1_MULTIROOTS_DIM],
             Real (&J)[M1_MULTIROOTS_DIM][M1_MULTIROOTS_DIM]) {
  /*
  for (int i = 0; i < M1_MULTIROOTS_DIM; ++i) {
    std::cout << x[i] << " ";
  }
  for (int i = 0; i < M1_MULTIROOTS_DIM; ++i) {
    std::cout << f[i] << " ";
  }
  std::cout << std::endl;
  */
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::UserProblem(ParameterInput *pin)
//  \brief Sets initial conditions for radiation M1 beams test
void ProblemGenerator::RadiationM1HybridsjTest(ParameterInput *pin,
                                               const bool restart) {
  if (restart)
    return;
  /*
  Real A[M1_MULTIROOTS_DIM][M1_MULTIROOTS_DIM] = {
      {0.42274137, 0.22774634, 0.51149515, 0.83359161},
      {0.31137829, 0.93582124, 0.54958705, 0.64094763},
      {0.30909202, 0.61688019, 0.80969002, 0.07436079},
      {0.0283099, 0.76613643, 0.18623546, 0.33547237}};

  Real Q[M1_MULTIROOTS_DIM][M1_MULTIROOTS_DIM]{};
  Real R[M1_MULTIROOTS_DIM][M1_MULTIROOTS_DIM]{};

  radiationm1::qr_factorize(A, Q, R);

  std::cout << "QR decomposition A = QR:" << std::endl;
  std::cout << std::endl;
  std::cout << "A:" << std::endl;
  std::cout << std::endl;
  for (int i = 0; i < M1_MULTIROOTS_DIM; ++i) {
    for (int j = 0; j < M1_MULTIROOTS_DIM; ++j) {
      std::cout << A[i][j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  std::cout << "Q:" << std::endl;
  std::cout << std::endl;
  for (int i = 0; i < M1_MULTIROOTS_DIM; ++i) {
    for (int j = 0; j < M1_MULTIROOTS_DIM; ++j) {
      std::cout << Q[i][j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  std::cout << "R:" << std::endl;
  std::cout << std::endl;
  for (int i = 0; i < M1_MULTIROOTS_DIM; ++i) {
    for (int j = 0; j < M1_MULTIROOTS_DIM; ++j) {
      std::cout << R[i][j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl; */
  /*
  std::cout << "Testing Powell's Hybrid method:" << std::endl;
  HybridsjFunc func;
  radiationm1::HybridsjState state{};
  radiationm1::HybridsjParams pars{};
  radiationm1::SrcParams src_params;
  pars.x[0] = 1;
  pars.x[1] = 0;
  radiationm1::HybridsjSignal ierr =
      radiationm1::HybridsjInitialize(func, state, src_params);
  print_f(pars.x, pars.f, pars.J);

  int iter = 0;
  int maxiter = 200;
  Real epsabs = 1e-15;
  Real epsrel = 1e-5;
  do {
    ierr = radiationm1::HybridsjIterate(func, state, src_params);
    print_f(pars.x, pars.f, pars.J);
    iter++;

    ierr = radiationm1::HybridsjTestDelta(pars.dx, pars.x, epsabs, epsrel);
  } while (ierr == radiationm1::HYBRIDSJ_CONTINUE && iter < maxiter);

  printf("Iters: %d\n", iter); */

  return;
}
