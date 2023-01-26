//========================================================================================
// Radiation FEM_N code for Athena
// Copyright (C) 2023 Maitraya Bhattacharyya <mbb6217@psu.edu> and David Radice <dur566@psu.edu>
// AthenaXX copyright(C) James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn.hpp
//  \brief implementation of the radiation FEM_N class constructor and other functions

#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "srcterms/srcterms.hpp"
#include "bvals/bvals.hpp"
#include "coordinates/coordinates.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "units/units.hpp"
#include "radiation_femn/radiation_femn.hpp"

namespace radiationfemn {
//----------------------------------------------------------------------------------------------
// class constructor, initialize parameters and data structures

    RadiationFEMN::RadiationFEMN(MeshBlockPack *ppack, ParameterInput *pin) :
            pmy_pack(ppack),
            i0("i0", 1, 1, 1, 1, 1),
            coarse_i0("ci0", 1, 1, 1, 1, 1),
            i1("i1", 1, 1, 1, 1, 1),
            iflx("iflx", 1, 1, 1, 1, 1),
            itemp("itemp", 1, 1, 1, 1, 1),
            etemp0("etemp0", 1, 1, 1, 1, 1),
            etemp1("etemp1", 1, 1, 1, 1, 1),
            mass_matrix("mm", 1, 1),
            stiffness_matrix_x("sx", 1, 1),
            stiffness_matrix_y("sy", 1, 1),
            stiffness_matrix_z("sz", 1, 1),
            stilde_matrix_x("stlidex", 1, 1),
            stilde_matrix_y("stlidey", 1, 1),
            stilde_matrix_z("stlidez", 1, 1),
            stildemod_matrix_x("stildemodx", 1, 1),
            stildemod_matrix_y("stildemody", 1, 1),
            stildemod_matrix_z("stildemodz", 1, 1),
            int_psi("int_psi", 1),
            e_source("e_source", 1),
            S_source("S_source", 1, 1),
            W_matrix("W_matrix", 1, 1),
            eta("eta", 1, 1, 1, 1),
            kappa_a("kappa_a", 1, 1, 1, 1),
            kappa_s("kappa_s", 1, 1, 1, 1),
            beam_mask("beam_mask", 1, 1, 1, 1, 1) {

        // set parfile parameters
        limiter_dg = pin->GetOrAddString("radiation_femn", "limiter_dg", "minmod2");
        fpn = pin->GetOrAddInteger("radiation_femn", "fpn", 0) == 1;

        if (!fpn) {
            matrix_path = pin->GetString("radiation_femn", "femn_matrix_location");
            lmax = -42;
            nangles = pin->GetInteger("radiation_femn", "num_angles");
            basis = pin->GetInteger("radiation_femn", "basis");
            filter_sigma_eff = -42;
            limiter_fem = pin->GetOrAddString("radiation_femn", "limiter_fem", "clp");
        } else {
            matrix_path = "-42";
            lmax = pin->GetInteger("radiation_femn", "lmax");
            nangles = (lmax + 1) * (lmax + 1);
            basis = -42;
            filter_sigma_eff = pin->GetOrAddInteger("radiation_femn", "filter_opacity", 0);
            limiter_fem = "-42";
        }

        rad_source = pin->GetOrAddInteger("radiation_femn", "sources", 0) == 1;
        beam_source = pin->GetOrAddInteger("radiation_femn", "beam_sources", 0) == 1;

        // allocate memory for matrices
        Kokkos::realloc(mass_matrix, nangles, nangles);
        Kokkos::realloc(stiffness_matrix_x, nangles, nangles);
        Kokkos::realloc(stiffness_matrix_y, nangles, nangles);
        Kokkos::realloc(stiffness_matrix_z, nangles, nangles);
        Kokkos::realloc(stilde_matrix_x, nangles, nangles);
        Kokkos::realloc(stilde_matrix_y, nangles, nangles);
        Kokkos::realloc(stilde_matrix_z, nangles, nangles);
        Kokkos::realloc(stildemod_matrix_x, nangles, nangles);
        Kokkos::realloc(stildemod_matrix_y, nangles, nangles);
        Kokkos::realloc(stildemod_matrix_z, nangles, nangles);

        // populate matrix values from file
        this->LoadMatrix(nangles, basis, "mass_matrix", mass_matrix, matrix_path);
        this->LoadMatrix(nangles, basis, "stiffness_x", stiffness_matrix_x, matrix_path);
        this->LoadMatrix(nangles, basis, "stiffness_y", stiffness_matrix_y, matrix_path);
        this->LoadMatrix(nangles, basis, "stiffness_z", stiffness_matrix_z, matrix_path);
        this->LoadMatrix(nangles, basis, "stilde_x", stilde_matrix_x, matrix_path);
        this->LoadMatrix(nangles, basis, "stilde_y", stilde_matrix_y, matrix_path);
        this->LoadMatrix(nangles, basis, "stilde_z", stilde_matrix_z, matrix_path);
        this->LoadMatrix(nangles, basis, "stildemod_x", stildemod_matrix_x, matrix_path);
        this->LoadMatrix(nangles, basis, "stildemod_y", stildemod_matrix_y, matrix_path);
        this->LoadMatrix(nangles, basis, "stildemod_z", stildemod_matrix_z, matrix_path);

        // allocate memory for evolved variables
        int nmb = ppack->nmb_thispack;
        auto &indcs = pmy_pack->pmesh->mb_indcs;
        int ncells1 = indcs.nx1 + 2 * (indcs.ng);
        int ncells2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2 * (indcs.ng)) : 1;
        int ncells3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2 * (indcs.ng)) : 1;

        Kokkos::realloc(i0, nmb, nangles, ncells3, ncells2, ncells1);
        Kokkos::realloc(i1, nmb, nangles, ncells3, ncells2, ncells1);
        Kokkos::realloc(iflx.x1f, nmb, nangles, ncells3, ncells2, ncells1);
        Kokkos::realloc(iflx.x2f, nmb, nangles, ncells3, ncells2, ncells1);
        Kokkos::realloc(iflx.x3f, nmb, nangles, ncells3, ncells2, ncells1);
        Kokkos::realloc(itemp, nmb, nangles, ncells3, ncells2, ncells1);

        // reallocate memory for the temporary intensity matrices if the clipping limiter is on
        if (limiter_fem == "clp") {
            Kokkos::realloc(etemp0, nmb, nangles, ncells3, ncells2, ncells1);
            Kokkos::realloc(etemp1, nmb, nangles, ncells3, ncells2, ncells1);
        }

        // reallocate allocate memory for evolved variables on coarse mesh
        if (ppack->pmesh->multilevel) {
            auto &indcs = pmy_pack->pmesh->mb_indcs;
            int nccells1 = indcs.cnx1 + 2 * (indcs.ng);
            int nccells2 = (indcs.cnx2 > 1) ? (indcs.cnx2 + 2 * (indcs.ng)) : 1;
            int nccells3 = (indcs.cnx3 > 1) ? (indcs.cnx3 + 2 * (indcs.ng)) : 1;
            Kokkos::realloc(coarse_i0, nmb, nangles, nccells3, nccells2, nccells1);
        }

        // only do if sources are present
        if (rad_source) {
            Kokkos::realloc(int_psi, nangles);
            Kokkos::realloc(e_source, nangles);
            Kokkos::realloc(S_source, nangles, nangles);
            Kokkos::realloc(W_matrix, nangles, nangles);
            this->CalcIntPsi();

            Kokkos::realloc(eta, nmb, ncells3, ncells2, ncells1);
            Kokkos::realloc(kappa_a, nmb, ncells3, ncells2, ncells1);
            Kokkos::realloc(kappa_s, nmb, ncells3, ncells2, ncells1);
        }

        if (beam_source) {
            Kokkos::realloc(beam_mask, nmb, nangles, ncells3, ncells2, ncells1);
        }

        // allocate boundary buffers for cell-centered variables
        pbval_i = new BoundaryValuesCC(ppack, pin, false);
        pbval_i->InitializeBuffers(nangles);
    }

//----------------------------------------------------------------------------------------------
// class constructor, initialize parameters and data structures

    RadiationFEMN::~RadiationFEMN() {
        delete pbval_i;
    }

} // namespace radiationfemn
