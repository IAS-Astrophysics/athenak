//#include <iostream>
#include <algorithm>
#include <cinttypes>
#include <iostream>
#include <limits>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "adm/adm.hpp"
#include "z4c/z4c.hpp"
#include "coordinates/cell_locations.hpp"
#include "utils/lagrange_interpolator.hpp"

namespace z4c {
//----------------------------------------------------------------------------------------
//! \fn TaskStatus Z4c::PunctureTracker
//! \brief Finds the apparent horizon
TaskStatus Z4c::PunctureTracker(Driver *pdriver, int stage)
{
  if (stage == 1) {
    // load in the shift vector
    auto &z4c = pmy_pack->pz4c->z4c;
    auto &beta_u = z4c.beta_u;
    auto &ppos = pmy_pack->pz4c->ppos;
    Real betap[3] = {0., 0., 0.,};
    // Interpolate shift to puncture pos
    LagrangeInterpolator *S = nullptr;
    S = new LagrangeInterpolator(pmy_pack,ppos);

    for (int a = 0; a < 3; ++a) {
      betap[a] = S->Interpolate(beta_u,a);
    }

    // Update puncture location
    for (int a = 0; a < 3; ++a) {
      ppos[a] -= pmy_pack->pmesh->dt * betap[a];
    }

    std::cout << "Puncture Location:" << "\t" << ppos[0] <<
              "\t" << ppos[1] << "\t" << ppos[2] << std::endl;
  }
  return TaskStatus::complete;
}
}