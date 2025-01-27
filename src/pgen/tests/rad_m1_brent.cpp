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
  Real operator()(const Real x) {
    return 3. * x * x - 7. * x - 5;
  }
};

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::UserProblem(ParameterInput *pin)
//  \brief Sets initial conditions for radiation M1 beams test
void ProblemGenerator::RadiationM1BrentTest(ParameterInput *pin,
                                            const bool restart) {
  if (restart)
    return;

  BrentFunc f;
  const Real x_lo_arr[2] = {-20, 1.1};
  const Real x_hi_arr[2] = {-0.1, 6.3};

  for (int i = 0; i < 2; ++i) {
        Real x_lo = x_lo_arr[i];
        Real x_md = 0.1;
        Real x_hi = x_hi_arr[i];

        Real root{};
        radiationm1::BrentState state{};

        // Initialize rootfinder
        int closure_maxiter = 64;
        Real closure_epsilon = 1e-15;
        radiationm1::BrentSignal ierr =
            BrentInitialize(f, x_lo, x_hi, root, state);

        // Rootfinding
        int iter = 0;
        do {
          ++iter;
          ierr = BrentIterate(f, x_lo, x_hi, root, state);

          // Some nans in the evaluation. This should not happen.
          if (ierr != radiationm1::BRENT_SUCCESS) {
            printf("Unexpected error in BrentIterate.\n");
            exit(EXIT_FAILURE);
          }
          x_md = root;
          ierr = radiationm1::BrentTestInterval(x_lo, x_hi, closure_epsilon, 0);
        } while (ierr == radiationm1::BRENT_CONTINUE && iter < closure_maxiter);

        printf("[%d] root = %.14e\n", i, x_md);

        if (ierr != radiationm1::BRENT_SUCCESS) {
          printf("Maximum number of iterations exceeded\n");
        }
      }

      printf("Answer: -0.573384418151758, 2.90671775148509\n");
  return;
}
