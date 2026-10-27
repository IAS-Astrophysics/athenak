#include <algorithm>
#include <cmath>

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "mhd/mhd.hpp"
#include "hydro/hydro.hpp"
#include "pgen/pgen.hpp"
#include "nuclear/nuclear.hpp"

namespace nuclear {

TaskStatus Nuclear::NewTimeStep(Driver* pdrive, int stage) {

  Real dt_hydro{};
  Real cfl_no = pmy_pack->pmesh->cfl_no;
  if (ismhd) {
    dt_hydro = cfl_no * pmy_pack->pmhd->dtnew;
  } else if (ishydro) {
    dt_hydro = cfl_no * pmy_pack->phydro->dtnew;
  }

  dtnew = dt_hydro;
  if(!dt_nuc){
    dtnew = dt_hydro;
  } else {
    dtnew = params.edot_limiter*std::min(dt_hydro, dt_nuc);
  }
  return TaskStatus::complete;
}

}
