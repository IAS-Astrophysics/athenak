//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_gauge_wave.cpp
//! \brief z4c gauge wave test


// C/C++ headers
#include <algorithm> // min, max
#include <cmath>     // sqrt()
#include <cstdio>    // fopen(), fprintf(), freopen()
#include <iostream>  // endl
#include <sstream>   // stringstream
#include <string>    // c_str()
#include <limits>

// Athena++ headers
#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/adm.hpp"
#include "mesh/mesh.hpp"
#include "z4c/z4c.hpp"
#include "driver/driver.hpp"
#include "pgen/pgen.hpp"

// function to compute errors in solution at end of run
void Z4cGaugeWaveErrors(ParameterInput *pin, Mesh *pm);

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief Sets initial conditions for gw linear wave tests

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart)
    return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pz4c == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Z4c Wave test can only be run in Z4c, but no <z4c> block "
              << "in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Prepare Initial Data

  // capture variables for the kernel
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is;
  int &ie = indcs.ie;
  int &js = indcs.js;
  int &je = indcs.je;
  int &ks = indcs.ks;
  int &ke = indcs.ke;

  int isg = is-indcs.ng; int ieg = ie+indcs.ng;
  int jsg = js-indcs.ng; int jeg = je+indcs.ng;
  int ksg = ks-indcs.ng; int keg = ke+indcs.ng;

  auto &pz4c = pmbp->pz4c;
  auto &adm = pmbp->padm->adm;
  Real x1size = pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min;
  std::cout << x1size << std::endl;

  // Wave amplitude
  Real amp = pin->GetOrAddReal("problem", "amp", 0.001);

  par_for("pgen_gauge_wave", DevExeSpace(), 0, (pmbp->nmb_thispack - 1),
      ksg, keg, jsg, jeg, isg, ieg, KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;

    int nx1 = indcs.nx1;

    Real x1v = CellCenterX(i - is, nx1, x1min, x1max);
    Real H = amp * sin(2 * M_PI * x1v / x1size);
    Real dH_dt = - amp * 2 * M_PI / x1size *
                    cos(2 * M_PI * x1v / x1size);

    adm.g_dd(m,0,0,k,j,i) = 1 - H;
    adm.g_dd(m,0,1,k,j,i) = 0;
    adm.g_dd(m,0,2,k,j,i) = 0;
    adm.g_dd(m,1,1,k,j,i) = 1;
    adm.g_dd(m,1,2,k,j,i) = 0;
    adm.g_dd(m,2,2,k,j,i) = 1;

    adm.vK_dd(m,0,0,k,j,i) = 0.5 * dH_dt / sqrt(adm.g_dd(m,0,0,k,j,i));
    adm.vK_dd(m,0,1,k,j,i) = 0;
    adm.vK_dd(m,0,2,k,j,i) = 0;
    adm.vK_dd(m,1,1,k,j,i) = 0;
    adm.vK_dd(m,1,2,k,j,i) = 0;
    adm.vK_dd(m,2,2,k,j,i) = 0;

    adm.alpha(m,k,j,i) = sqrt(1 - H);
  });
  switch (indcs.ng) {
    case 2: pmbp->pz4c->ADMToZ4c<2>(pmbp, pin);
            break;
    case 3: pmbp->pz4c->ADMToZ4c<3>(pmbp, pin);
            break;
    case 4: pmbp->pz4c->ADMToZ4c<4>(pmbp, pin);
            break;
  }
  pmbp->pz4c->Z4cToADM(pmbp);
  switch (indcs.ng) {
    case 2: pmbp->pz4c->ADMConstraints<2>(pmbp);
            break;
    case 3: pmbp->pz4c->ADMConstraints<3>(pmbp);
            break;
    case 4: pmbp->pz4c->ADMConstraints<4>(pmbp);
            break;
  }
  return;
}
