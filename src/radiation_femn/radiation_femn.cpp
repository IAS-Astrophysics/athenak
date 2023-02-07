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
            beam_mask("beam_mask", 1, 1, 1, 1, 1),
            num_ref(0),
            num_points(12),
            num_edges(30),
            num_triangles(20),
            x("x", 12),
            y("y", 12),
            z("z", 12),
            r("r", 12),
            theta("theta", 12),
            phi("phi", 12),
            edges("edges", 30, 2),
            triangles("triangles", 20, 3),
            scheme_weights("scheme_weights", 171),
            scheme_points("scheme_points", 171, 3) {

        // set parfile parameters
        limiter_dg = pin->GetOrAddString("radiation_femn", "limiter_dg", "minmod2");
        fpn = pin->GetOrAddInteger("radiation_femn", "fpn", 0) == 1;

        if (!fpn) {
            lmax = -42;
            nangles = pin->GetInteger("radiation_femn", "num_angles");
            basis = pin->GetInteger("radiation_femn", "basis");
            filter_sigma_eff = -42;
            limiter_fem = pin->GetOrAddString("radiation_femn", "limiter_fem", "clp");
        } else {
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

        // initialize the base grid
        double golden_ratio = (1.0 + sqrt(5.0)) / 2.0;
        double normalization_factor = sqrt(1. + golden_ratio * golden_ratio);

        x(0) = normalization_factor * 0.;
        x(1) = normalization_factor * 0.;
        x(2) = normalization_factor * 0.;
        x(3) = normalization_factor * 0.;
        x(4) = normalization_factor * 1.;
        x(5) = normalization_factor * 1.;
        x(6) = normalization_factor * -1.;
        x(7) = normalization_factor * -1.;
        x(8) = normalization_factor * golden_ratio;
        x(9) = normalization_factor * golden_ratio;
        x(10) = normalization_factor * -golden_ratio;
        x(11) = normalization_factor * -golden_ratio;

        y(0) = normalization_factor * 1.;
        y(1) = normalization_factor * 1.;
        y(2) = normalization_factor * -1.;
        y(3) = normalization_factor * -1.;
        y(4) = normalization_factor * golden_ratio;
        y(5) = normalization_factor * -golden_ratio;
        y(6) = normalization_factor * golden_ratio;
        y(7) = normalization_factor * -golden_ratio;
        y(8) = normalization_factor * 0.;
        y(9) = normalization_factor * 0.;
        y(10) = normalization_factor * 0.;
        y(11) = normalization_factor * 0.;

        z(0) = normalization_factor * golden_ratio;
        z(1) = normalization_factor * -golden_ratio;
        z(2) = normalization_factor * golden_ratio;
        z(3) = normalization_factor * -golden_ratio;
        z(4) = normalization_factor * 0.;
        z(5) = normalization_factor * 0.;
        z(6) = normalization_factor * 0.;
        z(7) = normalization_factor * 0.;
        z(8) = normalization_factor * 1.;
        z(9) = normalization_factor * -1.;
        z(10) = normalization_factor * 1.;
        z(11) = normalization_factor * -1.;

        edges(0, 0) = 2;
        edges(0, 1) = 8;
        edges(1, 0) = 1;
        edges(1, 1) = 4;
        edges(2, 0) = 4;
        edges(2, 1) = 6;
        edges(3, 0) = 3;
        edges(3, 1) = 9;
        edges(4, 0) = 4;
        edges(4, 1) = 9;
        edges(5, 0) = 10;
        edges(5, 1) = 11;
        edges(6, 0) = 5;
        edges(6, 1) = 9;
        edges(7, 0) = 6;
        edges(7, 1) = 11;
        edges(8, 0) = 0;
        edges(8, 1) = 6;
        edges(9, 0) = 7;
        edges(9, 1) = 10;
        edges(10, 0) = 0;
        edges(10, 1) = 2;
        edges(11, 0) = 0;
        edges(11, 1) = 4;
        edges(12, 0) = 3;
        edges(12, 1) = 5;
        edges(13, 0) = 1;
        edges(13, 1) = 6;
        edges(14, 0) = 5;
        edges(14, 1) = 8;
        edges(15, 0) = 3;
        edges(15, 1) = 11;
        edges(16, 0) = 1;
        edges(16, 1) = 3;
        edges(17, 0) = 3;
        edges(17, 1) = 7;
        edges(18, 0) = 0;
        edges(18, 1) = 10;
        edges(19, 0) = 7;
        edges(19, 1) = 11;
        edges(20, 0) = 2;
        edges(20, 1) = 7;
        edges(21, 0) = 0;
        edges(21, 1) = 8;
        edges(22, 0) = 5;
        edges(22, 1) = 7;
        edges(23, 0) = 1;
        edges(23, 1) = 9;
        edges(24, 0) = 2;
        edges(24, 1) = 10;
        edges(25, 0) = 1;
        edges(25, 1) = 11;
        edges(26, 0) = 8;
        edges(26, 1) = 9;
        edges(27, 0) = 6;
        edges(27, 1) = 10;
        edges(28, 0) = 2;
        edges(28, 1) = 5;
        edges(29, 0) = 4;
        edges(29, 1) = 8;


        triangles(0, 0) = 0;
        triangles(0, 1) = 6;
        triangles(0, 2) = 10;
        triangles(1, 0) = 7;
        triangles(1, 1) = 10;
        triangles(1, 2) = 11;
        triangles(2, 0) = 3;
        triangles(2, 1) = 5;
        triangles(2, 2) = 9;
        triangles(3, 0) = 1;
        triangles(3, 1) = 4;
        triangles(3, 2) = 6;
        triangles(4, 0) = 0;
        triangles(4, 1) = 4;
        triangles(4, 2) = 6;
        triangles(5, 0) = 2;
        triangles(5, 1) = 7;
        triangles(5, 2) = 10;
        triangles(6, 0) = 3;
        triangles(6, 1) = 7;
        triangles(6, 2) = 11;
        triangles(7, 0) = 6;
        triangles(7, 1) = 10;
        triangles(7, 2) = 11;
        triangles(8, 0) = 1;
        triangles(8, 1) = 6;
        triangles(8, 2) = 11;
        triangles(9, 0) = 2;
        triangles(9, 1) = 5;
        triangles(9, 2) = 8;
        triangles(10, 0) = 0;
        triangles(10, 1) = 2;
        triangles(10, 2) = 10;
        triangles(11, 0) = 4;
        triangles(11, 1) = 8;
        triangles(11, 2) = 9;
        triangles(12, 0) = 1;
        triangles(12, 1) = 3;
        triangles(12, 2) = 9;
        triangles(13, 0) = 1;
        triangles(13, 1) = 3;
        triangles(13, 2) = 11;
        triangles(14, 0) = 0;
        triangles(14, 1) = 4;
        triangles(14, 2) = 8;
        triangles(15, 0) = 0;
        triangles(15, 1) = 2;
        triangles(15, 2) = 8;
        triangles(16, 0) = 2;
        triangles(16, 1) = 5;
        triangles(16, 2) = 7;
        triangles(17, 0) = 1;
        triangles(17, 1) = 4;
        triangles(17, 2) = 9;
        triangles(18, 0) = 5;
        triangles(18, 1) = 8;
        triangles(18, 2) = 9;
        triangles(19, 0) = 3;
        triangles(19, 1) = 5;
        triangles(19, 2) = 7;

        // populate matrix values from file
        /*this->LoadMatrix(nangles, basis, "mass_matrix", mass_matrix, matrix_path);
        this->LoadMatrix(nangles, basis, "stiffness_x", stiffness_matrix_x, matrix_path);
        this->LoadMatrix(nangles, basis, "stiffness_y", stiffness_matrix_y, matrix_path);
        this->LoadMatrix(nangles, basis, "stiffness_z", stiffness_matrix_z, matrix_path);
        this->LoadMatrix(nangles, basis, "stilde_x", stilde_matrix_x, matrix_path);
        this->LoadMatrix(nangles, basis, "stilde_y", stilde_matrix_y, matrix_path);
        this->LoadMatrix(nangles, basis, "stilde_z", stilde_matrix_z, matrix_path);
        this->LoadMatrix(nangles, basis, "stildemod_x", stildemod_matrix_x, matrix_path);
        this->LoadMatrix(nangles, basis, "stildemod_y", stildemod_matrix_y, matrix_path);
        this->LoadMatrix(nangles, basis, "stildemod_z", stildemod_matrix_z, matrix_path);
        */

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
        pbval_i = new BoundaryValuesCC(ppack, pin);
        pbval_i->InitializeBuffers(nangles);
    }

//----------------------------------------------------------------------------------------------
// class constructor, initialize parameters and data structures

    RadiationFEMN::~RadiationFEMN() {
        delete pbval_i;
    }

} // namespace radiationfemn
