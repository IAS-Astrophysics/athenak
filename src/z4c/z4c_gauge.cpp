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
#include "adm/adm.hpp"
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
  //For GLOOPS
  int isg = is-indcs.ng; int ieg = ie+indcs.ng;
  int jsg = js-indcs.ng; int jeg = je+indcs.ng;
  int ksg = ks-indcs.ng; int keg = ke+indcs.ng;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int nmb = pmbp->nmb_thispack;
  auto &z4c = pmbp->pz4c->z4c;
  auto &adm = pmbp->padm->adm;

  int scr_level = 0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1);
  par_for_outer("pgen one puncture",
  DevExeSpace(),scr_size,scr_level,0,nmb-1,ksg,keg,jsg,jeg,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {
    par_for_inner(member, isg, ieg, [&](const int i) {
      z4c.alpha(m,k,j,i) = std::pow(adm.psi4(m,k,j,i),-0.5); // setting z4c.alpha,
                                                             // which is 0th component
                                                             // of z4c
    });
  });
}


} // end namespace z4c
