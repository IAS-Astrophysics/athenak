//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_gauge.cpp
//! \brief implementation of gauge for the Z4c class

// C++ standard headers
#include <cmath> // pow
#include <iostream>
#include <fstream>

// Athena++ headers
#include "parameter_input.hpp"
#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "mesh/mesh.hpp"
#include "z4c/z4c.hpp"
#include "coordinates/cell_locations.hpp"


namespace z4c {
//----------------------------------------------------------------------------------------
//! \fn void Z4c::GaugePreCollapsedLapse(MeshBlockPack *pmbp, ParameterInput *pin)
//! \brief set lapse from conformal factor in initial data

void Z4c::GaugePreCollapsedLapse(MeshBlockPack *pmbp, ParameterInput *pin) {
  // capture variables for the kernel
  auto &indcs = pmbp->pmesh->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int isg = is-indcs.ng; int ieg = ie+indcs.ng;
  int jsg = js-indcs.ng; int jeg = je+indcs.ng;
  int ksg = ks-indcs.ng; int keg = ke+indcs.ng;
  int nmb = pmbp->nmb_thispack;
  auto &z4c = pmbp->pz4c->z4c;
  auto &adm = pmbp->padm->adm;

  par_for("GaugePreCollapsedLapse",
  DevExeSpace(),0,nmb-1,ksg,keg,jsg,jeg,isg,ieg,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    z4c.alpha(m,k,j,i) = std::pow(adm.psi4(m,k,j,i),-0.5); // setting z4c.alpha,
                                                           // which is 0th component
                                                           // of z4c
  });
}

} // end namespace z4c
