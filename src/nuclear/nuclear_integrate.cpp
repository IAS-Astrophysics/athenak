#include <algorithm>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "athena_tensor.hpp"
#include "driver/driver.hpp"
#include "mhd/mhd.hpp"
#include "hydro/hydro.hpp"
#include "pgen/pgen.hpp"
#include "nuclear/nuclear.hpp"

namespace nuclear {
  TaskStatus Nuclear::Integrate(Driver *pdrive, int stage) {
    // This correspond to the caller that performs the integration of the network.
    // The network is integrated in consecutive substeps of size dt_nuc (set by the
    // pgen) until the whole hydro step pmesh->dt is covered.
    if (pmy_pack->pmesh->pgen->user_nuc_srcs) {
      const Real dt = pmy_pack->pmesh->dt;
      // Substep size, frozen for this hydro step (the pgen updates dt_nuc during the burn)
      const Real dt_sub = (dt_nuc > 0.0) ? dt_nuc : dt;
      // DEBUG: print the substeps on the same cycles as the driver's "elapsed=" line
      // (every ndiag cycles), followed by a blank line
      const bool print_diag = (global_variable::my_rank == 0) &&
                              (pmy_pack->pmesh->ncycle % pdrive->ndiag == 0);
      // Cap the number of substeps: never use substeps smaller than dt/max_substeps
      const Real h_min = dt/static_cast<Real>(max_substeps);
      const Real h_sub = std::max(dt_sub, h_min);
      if (dt_sub < h_min && global_variable::my_rank == 0) {
        std::cout << "### WARNING in Nuclear::Integrate: cycle=" << pmy_pack->pmesh->ncycle
                  << " dt_nuc=" << dt_sub << " would need " << dt/dt_sub
                  << " substeps; capped at max_substeps=" << max_substeps
                  << " (h=" << h_min << ")" << std::endl;
      }
      Real t = 0.0;
      int nsub = 0;  // DEBUG
      while (t < dt) {
        Real h = std::min(h_sub, dt - t);
        // absorb a roundoff-sized remainder into this substep
        if (dt - (t + h) < 1.0e-10*dt) h = dt - t;
        (pmy_pack->pmesh->pgen->user_srcs_nuc_func)(pmy_pack->pmesh, h);
        t += h;
        nsub++;  // DEBUG
        if (print_diag) {
          std::cout << "  Nuclear::Integrate substep=" << nsub << " h=" << h
                    << " sum(h)=" << t << " sum(h)-dt=" << t - dt << std::endl;
        }
      }
      if (print_diag) std::cout << std::endl;
    }

    return TaskStatus::complete;
  }
}
