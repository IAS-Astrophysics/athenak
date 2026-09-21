//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file newdt.cpp
//! \brief Radiation-relaxation timestep for operator-split Q_rad coupling.
//! Linearized cooling rate of a cell with opacity per unit mass sigma_a / rho:
//!   nu_rad = 4*(gamma-1)*T^3 * (sigma_a/rho) * crat * prat / (1 + 3*(dx*chi/pi)^2)
//! The diffusion correction in the denominator (chi = total extinction) transitions
//! smoothly from the optically thin rate to the diffusion timescale at high tau.

#include <algorithm>
#include <cmath>
#include <limits>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "eos/eos.hpp"
#include "nr_radiation/nr_radiation.hpp"

namespace nr_radiation {

//----------------------------------------------------------------------------------------
//! \fn TaskStatus SC::NewTimeStep
//! \brief Compute radiation-relaxation timestep and store in dtnew.
//! Registered in stagen; Mesh::NewTimeStep() picks up dtnew in its global minimum.

TaskStatus SC::NewTimeStep(Driver *pdrive, int stage) {
  if (stage != pdrive->nexp_stages) return TaskStatus::complete;
  // the rate below reads sigma_a: refresh it from the fluid state at the end of this
  // step, not the one the solve saw at its start (a restart recomputes it from this too)
  UpdateOpacity();
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nx1 = ie - is + 1, nx2 = je - js + 1, nx3 = ke - ks + 1;
  int nmkji = (nmb1 + 1) * nx3 * nx2 * nx1;

  auto &mbsize = pmy_pack->pmb->mb_size;
  Real dxmin = std::numeric_limits<Real>::max();
  for (int m = 0; m <= nmb1; ++m) {
    dxmin = std::min(dxmin, mbsize.h_view(m).dx1);
    if (pmy_pack->pmesh->multi_d) {
      dxmin = std::min(dxmin, mbsize.h_view(m).dx2);
    }
    if (pmy_pack->pmesh->three_d) {
      dxmin = std::min(dxmin, mbsize.h_view(m).dx3);
    }
  }
  const Real diff_coeff = 3.0 / (M_PI * M_PI);

  DvceArray5D<Real> w0;
  Real gamma = 0.0;
  if (pmy_pack->phydro != nullptr) {
    w0 = pmy_pack->phydro->w0;
    gamma = pmy_pack->phydro->peos->eos_data.gamma;
  } else if (pmy_pack->pmhd != nullptr) {
    w0 = pmy_pack->pmhd->w0;
    gamma = pmy_pack->pmhd->peos->eos_data.gamma;
  } else {
    dtnew = std::numeric_limits<Real>::max();   // no fluid: no relaxation limit
    return TaskStatus::complete;
  }
  const Real gm1 = gamma - 1.0;
  const Real rate_base = 4.0 * gm1 * crat * prat;   // times sigma_a/rho and T^3 per cell

  auto chi_ = chi;
  auto sigma_a_ = sigma_a;
  Real dxmin2 = dxmin * dxmin;
  Real diff_c = diff_coeff;
  Real rb = rate_base;
  Real dt_min = std::numeric_limits<Real>::max();
  Kokkos::parallel_reduce("sc_rad_dt",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int idx, Real &ldt) {
      int m = idx / (nx3 * nx2 * nx1);
      int kji = idx - m * (nx3 * nx2 * nx1);
      int k = kji / (nx2 * nx1);
      int ji = kji - k * (nx2 * nx1);
      int j = ji / nx1;
      int i = ji - j * nx1 + is;
      j += js;
      k += ks;
      Real dens = w0(m, IDN, k, j, i);
      if (dens <= 0.0) return;
      Real temp = gm1 * w0(m, IEN, k, j, i) / dens;
      if (temp <= 0.0) return;
      Real T3 = temp * temp * temp;
      Real chiv = chi_(m, k, j, i);
      Real nu_rad = rb * (sigma_a_(m, 0, k, j, i) / dens) * T3;
      Real denom = 1.0 + diff_c * dxmin2 * chiv * chiv;
      nu_rad /= denom;
      if (nu_rad > 0.0) {
        ldt = fmin(ldt, 1.0 / nu_rad);
      }
    }, Kokkos::Min<Real>(dt_min));

  dtnew = dt_min;
  return TaskStatus::complete;
}

}  // namespace nr_radiation
