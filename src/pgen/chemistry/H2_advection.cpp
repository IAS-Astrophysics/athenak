//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file H2_advection.cpp
//! \brief Problem generator for chemistry problem that advects a gaussian state
//! using the H2 network that has an analytical solution

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
//! \fn ProblemGenerator::H2_advection()
//! \brief Problem Generator for the H2 test problem that advects a gaussian
//! state

void ProblemGenerator::H2Advection(ParameterInput* pin, const bool restart) {
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
  auto& size = pmbp->pmb->mb_size;
  auto& w0 = pmbp->phydro->w0;
  auto& u0 = pmbp->phydro->u0;

  // Get the input parameters from the input file
  const Real n_H = pin->GetReal("problem", "n_H");
  const Real iso_cs = pin->GetReal("hydro", "iso_sound_speed");
  HydPrim1D hydro;
  hydro.d = n_H * pmbp->punit->hydrogen_mass_cgs * pmbp->punit->mu() /
            pmbp->punit->density_cgs();
  hydro.vx = pin->GetReal("problem", "vx_kms");
  hydro.vy = 0.0;
  hydro.vz = 0.0;
  hydro.e = hydro.d * SQR(iso_cs) / (pmbp->phydro->peos->eos_data.gamma - 1.0);
  const Real init_H = pin->GetOrAddReal("problem", "init_H", 0.0);

  // mean and std of the initial gaussian profile
  const Real gaussian_mean = pin->GetOrAddReal("problem", "gaussian_mean", 0.5);
  const Real gaussian_std = pin->GetOrAddReal("problem", "gaussian_std", 0.1);

  // Assign values
  const int chem_start =
      pmbp->pchemistry->get_chemistry_scalars_first_idx() - 1;
  par_for(
      "pgen_H2_hydro", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), ks, ke, js,
      je, is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
        // Assign hydro values to this cell
        w0(m, IDN, k, j, i) = hydro.d;
        w0(m, IVX, k, j, i) = hydro.vx;
        w0(m, IVY, k, j, i) = hydro.vy;
        w0(m, IVZ, k, j, i) = hydro.vz;
        w0(m, IEN, k, j, i) = hydro.e;

        // Compute the location
        Real& x1min = size.d_view(m).x1min;
        Real& x1max = size.d_view(m).x1max;
        int nx1 = indcs.nx1;
        const Real x = CellCenterX(i - is, nx1, x1min, x1max);

        // Compute the H and H2 abundances
        Real H2_abundance, H_abundance;
        if (x <= 1.0) {
          H_abundance =
              Kokkos::exp(-SQR(x - gaussian_mean) / (2. * SQR(gaussian_std)));
          H2_abundance = 0.5 * (1.0 - H_abundance);
        } else {
          H_abundance = init_H;
          H2_abundance = (1. - init_H) * 0.5;
        }

        // Assign chemistry values to this cell
        w0(m, chem_start + chemistry::H2Network::IH2, k, j, i) = H2_abundance;
        w0(m, chem_start + chemistry::H2Network::IH, k, j, i) = H_abundance;
      });

  // Convert primitives to conserved
  pmbp->phydro->peos->PrimToCons(w0, u0, is, ie, js, je, ks, ke);

  return;
}
