//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rad_m1_beams.cpp
//  \brief 1D beam for grey M1

// C++ headers

// Athena++ headers
#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "radiation_m1/radiation_m1.hpp"
#include "radiation_m1/radiation_m1_roots.hpp"

class BrentFunc {
public:
  KOKKOS_INLINE_FUNCTION
  Real operator()(const Real x, const Real a) { return x * x + 3. * x + 2; }
};

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::UserProblem(ParameterInput *pin)
//  \brief Sets initial conditions for radiation M1 beams test
void ProblemGenerator::RadiationM1BrentTest(ParameterInput *pin,
                                            const bool restart) {
  if (restart)
    return;

  BrentFunc f;
  Real x_lo = -1.9;
  Real x_md = -3.;
  Real x_hi = 0.1;
  Real root{};
  radiationm1::BrentState state{};

  // Initialize rootfinder
  int closure_maxiter = 64;
  Real closure_epsilon = 1e-15;
  radiationm1::BrentSignal ierr =
      BrentInitialize(f, x_lo, x_hi, root, state, 1);

  // Rootfinding
  int iter = 0;
  do {
    ++iter;
    ierr = BrentIterate(f, x_lo, x_hi, root, state, 1);

    // Some nans in the evaluation. This should not happen.
    if (ierr != radiationm1::BRENT_SUCCESS) {
      printf("Unexpected error in BrentIterate.\n");
      exit(EXIT_FAILURE);
    }
    x_md = root;
    ierr = radiationm1::BrentTestInterval(x_lo, x_hi, closure_epsilon, 0);
  } while (ierr == radiationm1::BRENT_CONTINUE && iter < closure_maxiter);

  printf("root = %.14e\n", x_md);

  if (ierr != radiationm1::BRENT_SUCCESS) {
    printf("Maximum number of iterations exceeded when computing the M1 "
           "closure\n");
  }
  return;
}
