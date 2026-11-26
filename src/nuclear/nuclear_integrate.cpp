#include <algorithm>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "nuclear/nuclear.hpp"

namespace nuclear {
  TaskStatus Nuclear::Integrate(Driver *pdrive, int stage) {
    // This correspond to the caller that performs the integration of the network.
    // The network is integrated in consecutive substeps of size dt_nuc (set by
    // BurnStep) until the whole hydro step pmesh->dt is covered. The burn is on
    // whenever the <nuclear> block exists (it creates this module).
    const Real dt = pmy_pack->pmesh->dt;
    // Substep size, frozen for this hydro step (BurnStep updates dt_nuc during the burn).
    // dt_nuc is 0 until the first burn sets it (also after a restart), so that hydro
    // step is covered by a single substep.
    const Real dt_sub = (dt_nuc > 0.0) ? dt_nuc : dt;
    // DEBUG: print the substeps on the same cycles as the driver's "elapsed=" line
    // (every ndiag cycles), followed by a blank line
    const bool print_diag = (global_variable::my_rank == 0) &&
                            (pmy_pack->pmesh->ncycle % pdrive->ndiag == 0);
    // Cap the number of substeps: never use substeps smaller than dt/max_substeps
    const int max_substeps = params.max_substeps;
    const Real h_min = dt/static_cast<Real>(max_substeps);
    const Real h_sub = std::max(dt_sub, h_min);
    if (dt_sub < h_min && global_variable::my_rank == 0) {
      std::cout << " Nuclear::Integrate: cycle=" << pmy_pack->pmesh->ncycle
                << " would need " << dt/dt_sub
                << " substeps; capped at max_substeps=" << max_substeps
                << " dt_sub=" << h_min << std::endl;
    }
    Real t = 0.0;
    int nsub = 0;  // DEBUG
    // Newton solver failures of this rank, summed over the substeps: a cell that fails in
    // k substeps counts k times (cell-substep failures, not distinct cells). Also the
    // first substep where any failed (elapsed time at its start, size). Failed cells keep
    // burning until the end of the hydro step; the run is stopped on all ranks after the
    // loop.
    int nfail = 0;
    Real t_fail = 0.0, h_fail = 0.0;
    while (t < dt) {
      Real h = std::min(h_sub, dt - t);
      // absorb a roundoff-sized remainder into this substep
      if (dt - (t + h) < 1.0e-10*dt) h = dt - t;
      const int nf = BurnStep(h);
      if (nf > 0 && nfail == 0) {
        t_fail = t;
        h_fail = h;
      }
      nfail += nf;
      t += h;
      nsub++;  // DEBUG
      if (print_diag) {
        std::cout << "  Nuclear::Integrate substep=" << nsub << " dt_sub=" << h
                  << " elapsed=" << t << std::endl;
      }
    }
    if (print_diag) std::cout << std::endl;

    // Cell-substep failures on this rank, kept for the per-rank report below
    const int nfail_rank = nfail;
#if MPI_PARALLEL_ENABLED
    // Same substep size on every rank: BurnStep set dt_nuc from the min(cv*T/|edot|) of
    // this rank's MeshBlocks only. Reduced once per hydro step, since only the value
    // set by the last substep is used (at the start of the next hydro step). This makes
    // the result independent of the domain decomposition.
    MPI_Allreduce(MPI_IN_PLACE, &dt_nuc, 1, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
    // Total cell-substep failures over all ranks, so every rank stops together
    MPI_Allreduce(MPI_IN_PLACE, &nfail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif

    // Stop the run if the Newton solver failed in any cell of any rank. Each rank with
    // failures reports its own count and first failing substep; rank 0 the total.
    if (nfail > 0) {
      if (nfail_rank > 0) {
        std::cout << "Nuclear::Integrate: rank " << global_variable::my_rank << ": "
                  << nfail_rank << " cell-substep failures, first in the substep "
                  << "starting at elapsed=" << t_fail << " with h=" << h_fail
                  << std::endl;
      }
      if (global_variable::my_rank == 0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Nuclear Newton solver: " << nfail
                  << " cell-substep failures over all ranks (a cell failing in k "
                  << "substeps counts k times; no convergence in nuclear/max_iter="
                  << params.max_iter << " iterations, singular matrix, or non-finite "
                  << "result)" << std::endl << "cycle=" << pmy_pack->pmesh->ncycle
                  << " time=" << pmy_pack->pmesh->time << " dt=" << dt << std::endl;
      }
      exit(EXIT_FAILURE);
    }

    return TaskStatus::complete;
  }
}
