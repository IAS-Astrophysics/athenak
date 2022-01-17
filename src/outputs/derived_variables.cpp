//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file derived_variables.cpp
//! \brief Calculates various derived variables for outputs, storing them into the 
//! "derived_vars" device array located in BaseTypeOutput class.  Variables are only
//! calculated over active zones (ghost zones excluded).

#include <cstdio>
#include <cstdlib>
#include <cstring>    // strcmp
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>   // std::string, to_string()

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "outputs.hpp"

//----------------------------------------------------------------------------------------
// BaseTypeOutput::ComputeDerivedVariable()

void BaseTypeOutput::ComputeDerivedVariable(std::string name, Mesh *pm) {
  int nmb = pm->pmb_pack->nmb_thispack;
  auto &indcs = pm->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;

  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;
  auto &size = pm->pmb_pack->pmb->mb_size;
  auto &two_d = pm->two_d;

  // z-component of current density.
  // Note calculated from cell-centered fields.  This makes for a large stencil, but
  // better represents volume-averaged value within cell.
  if (name.compare("mhd_jz") == 0) {
    Kokkos::realloc(derived_var, nmb, 1, n3, n2, n1);

    auto bcc = pm->pmb_pack->pmhd->bcc0;
    par_for("fixed_x1", DevExeSpace(), 0, (nmb-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      derived_var(m,0,k,j,i) = (bcc(m,1,k,j,i+1) - bcc(m,1,k,j,i-1))/size.d_view(m).dx1;
      if (two_d) {
        derived_var(m,0,k,j,i) -=(bcc(m,0,k,j+1,i) - bcc(m,0,k,j-1,i))/size.d_view(m).dx2;
      }
    });
  }
}
