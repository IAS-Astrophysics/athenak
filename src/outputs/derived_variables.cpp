//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file derived_variables.cpp
//! \brief Calculates various derived variables for outputs, storing them into the
//! "derived_vars" device array located in BaseTypeOutput class.  Variables are only
//! calculated over active zones (ghost zones excluded). Currently implemented are:
//!   - z-component of vorticity Curl(v)_z  [non-relativistic]
//!   - magnitude of vorticity Curl(v)^2  [non-relativistic]
//!   - z-component of current density Jz  [non-relativistic]
//!   - magnitude of current density J^2  [non-relativistic]

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
  auto &multi_d = pm->multi_d;
  auto &three_d = pm->three_d;

  // z-component of vorticity.
  // Not computed in ghost zones since requires derivative
  if (name.compare("hydro_wz") == 0 ||
      name.compare("mhd_wz") == 0) {
    Kokkos::realloc(derived_var, nmb, 1, n3, n2, n1);
    auto dv = derived_var;
    auto w0_ = (name.compare("hydro_wz") == 0)?
      pm->pmb_pack->phydro->w0 : pm->pmb_pack->pmhd->w0;
    par_for("jz", DevExeSpace(), 0, (nmb-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      dv(m,0,k,j,i) = (w0_(m,IVY,k,j,i+1) - w0_(m,IVY,k,j,i-1))/size.d_view(m).dx1;
      if (multi_d) {
        dv(m,0,k,j,i) -=(w0_(m,IVX,k,j+1,i) - w0_(m,IVX,k,j-1,i))/size.d_view(m).dx2;
      }
    });
  }

  // magnitude of vorticity.
  // Not computed in ghost zones since requires derivative
  if (name.compare("hydro_w2") == 0 ||
      name.compare("mhd_w2") == 0) {
    Kokkos::realloc(derived_var, nmb, 1, n3, n2, n1);
    auto dv = derived_var;
    auto w0_ = (name.compare("hydro_w2") == 0)?
      pm->pmb_pack->phydro->w0 : pm->pmb_pack->pmhd->w0;
    par_for("jz", DevExeSpace(), 0, (nmb-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real w1 = 0.0;
      Real w2 = -(w0_(m,IVZ,k,j,i+1) - w0_(m,IVZ,k,j,i-1))/size.d_view(m).dx1;
      Real w3 =  (w0_(m,IVY,k,j,i+1) - w0_(m,IVY,k,j,i-1))/size.d_view(m).dx1;
      if (multi_d) {
        w1 += (w0_(m,IVZ,k,j+1,i) - w0_(m,IVZ,k,j-1,i))/size.d_view(m).dx2;
        w3 -= (w0_(m,IVX,k,j+1,i) - w0_(m,IVX,k,j-1,i))/size.d_view(m).dx2;
      }
      if (three_d) {
        w1 -= (w0_(m,IVY,k+1,j,i) - w0_(m,IVY,k-1,j,i))/size.d_view(m).dx3;
        w2 += (w0_(m,IVX,k+1,j,i) - w0_(m,IVX,k-1,j,i))/size.d_view(m).dx3;
      }
      dv(m,0,k,j,i) = w1*w1 + w2*w2 + w3*w3;
    });
  }

  // z-component of current density.  Calculated from cell-centered fields.
  // This makes for a large stencil, but approximates volume-averaged value within cell.
  // Not computed in ghost zones since requires derivative
  if (name.compare("mhd_jz") == 0) {
    Kokkos::realloc(derived_var, nmb, 1, n3, n2, n1);
    auto dv = derived_var;
    auto bcc = pm->pmb_pack->pmhd->bcc0;
    par_for("jz", DevExeSpace(), 0, (nmb-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      dv(m,0,k,j,i) = (bcc(m,IBY,k,j,i+1) - bcc(m,IBY,k,j,i-1))/size.d_view(m).dx1;
      if (multi_d) {
        dv(m,0,k,j,i) -=(bcc(m,IBX,k,j+1,i) - bcc(m,IBX,k,j-1,i))/size.d_view(m).dx2;
      }
    });
  }

  // magnitude of current density.  Calculated from cell-centered fields.
  // Not computed in ghost zones since requires derivative
  if (name.compare("mhd_j2") == 0) {
    Kokkos::realloc(derived_var, nmb, 1, n3, n2, n1);
    auto dv = derived_var;
    auto bcc = pm->pmb_pack->pmhd->bcc0;
    par_for("jz", DevExeSpace(), 0, (nmb-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real j1 = 0.0;
      Real j2 = -(bcc(m,IBZ,k,j,i+1) - bcc(m,IBZ,k,j,i-1))/size.d_view(m).dx1;
      Real j3 =  (bcc(m,IBY,k,j,i+1) - bcc(m,IBY,k,j,i-1))/size.d_view(m).dx1;
      if (multi_d) {
        j1 += (bcc(m,IBZ,k,j+1,i) - bcc(m,IBZ,k,j-1,i))/size.d_view(m).dx2;
        j3 -= (bcc(m,IBX,k,j+1,i) - bcc(m,IBX,k,j-1,i))/size.d_view(m).dx2;
      }
      if (three_d) {
        j1 -= (bcc(m,IBY,k+1,j,i) - bcc(m,IBY,k-1,j,i))/size.d_view(m).dx3;
        j2 += (bcc(m,IBX,k+1,j,i) - bcc(m,IBX,k-1,j,i))/size.d_view(m).dx3;
      }
      dv(m,0,k,j,i) = j1*j1 + j2*j2 + j3*j3;
    });
  }

  // divergence of B, including ghost zones
  if (name.compare("mhd_divb") == 0) {
    Kokkos::realloc(derived_var, nmb, 1, n3, n2, n1);

    // set the loop limits for 1D/2D/3D problems
    int jl = js, ju = je, kl = ks, ku = ke;
    if (multi_d) {
      jl = js-ng, ju = je+ng;
    } else if (three_d) {
      jl = js-ng, ju = je+ng, kl = ks-ng, ku = ke+ng;
    }

    auto dv = derived_var;
    auto b0 = pm->pmb_pack->pmhd->b0;
    par_for("divb", DevExeSpace(), 0, (nmb-1), kl, ku, jl, ju, (is-ng), (ie+ng),
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real divb = (b0.x1f(m,k,j,i+1) - b0.x1f(m,k,j,i))/size.d_view(m).dx1;
      if (multi_d) {
        divb += (b0.x2f(m,k,j+1,i) - b0.x2f(m,k,j,i))/size.d_view(m).dx2;
      }
      if (three_d) {
        divb += (b0.x3f(m,k+1,j,i) - b0.x3f(m,k,j,i))/size.d_view(m).dx3;
      }
      dv(m,0,k,j,i) = divb;
    });
  }
}
