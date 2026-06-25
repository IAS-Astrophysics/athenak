//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file H2_uniform.cpp
//! \brief Problem generator for chemistry problem with a uniform state using
//! the H2 network that has an analytical solution

#include <iostream>
#include <sstream>
#include <string>

#include "athena.hpp"
#include "chemistry/chemistry.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "units/units.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::H2_uniform()
//! \brief Problem Generator for the H2 test problem with a uniform state

void ProblemGenerator::H2Uniform(ParameterInput* pin, const bool restart) {
  if (restart) return;

  // capture variables for the kernel
  auto& indcs = pmy_mesh_->mb_indcs;
  int& is = indcs.is;
  int& ie = indcs.ie;
  int& js = indcs.js;
  int& je = indcs.je;
  int& ks = indcs.ks;
  int& ke = indcs.ke;
  MeshBlockPack* pmbp = pmy_mesh_->pmb_pack;
  auto& w0 = pmbp->phydro->w0;
  auto& u0 = pmbp->phydro->u0;

  // Get the input parameters from the input file
  const Real n_H = pin->GetReal("problem", "n_H");
  const Real iso_cs = pin->GetReal("hydro", "iso_sound_speed");
  HydPrim1D hydro;
  hydro.d = n_H * pmbp->punit->hydrogen_mass_cgs * pmbp->pchemistry->mu_H /
            pmbp->punit->density_cgs();
  hydro.vx = pin->GetReal("problem", "vx_kms");
  hydro.vy = 0.0;
  hydro.vz = 0.0;
  hydro.e = hydro.d * SQR(iso_cs) / (pmbp->phydro->peos->eos_data.gamma - 1.0);
  const Real init_H = pin->GetReal("problem", "init_H");
  const Real init_H2 = pin->GetReal("problem", "init_H2");

  // Assign values
  const int chem_start = pmbp->pchemistry->get_chemistry_scalars_first_idx();
  par_for(
      "pgen_H2_hydro", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), ks, ke, js,
      je, is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
        // Assign hydro values to this cell
        w0(m, IDN, k, j, i) = hydro.d;
        w0(m, IVX, k, j, i) = hydro.vx;
        w0(m, IVY, k, j, i) = hydro.vy;
        w0(m, IVZ, k, j, i) = hydro.vz;
        w0(m, IEN, k, j, i) = hydro.e;

        // Assign chemistry values to this cell
        w0(m, chem_start + chemistry::H2Network::IH2, k, j, i) = init_H2;
        w0(m, chem_start + chemistry::H2Network::IH, k, j, i) = init_H;
      });

  // Convert primitives to conserved
  pmbp->phydro->peos->PrimToCons(w0, u0, is, ie, js, je, ks, ke);

  return;
}
