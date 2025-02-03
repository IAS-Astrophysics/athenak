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
#include "radiation_m1/radiation_m1_roots_hybridsj.hpp"

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::UserProblem(ParameterInput *pin)
//  \brief Sets initial conditions for radiation M1 beams test
void ProblemGenerator::RadiationM1HybridsjTest(ParameterInput *pin,
                                               const bool restart) {
  if (restart)
    return;

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

  return;
}
