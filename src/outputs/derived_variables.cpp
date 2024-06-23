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
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "radiation/radiation.hpp"
#include "radiation_femn/radiation_femn.hpp"
#include "radiation/radiation_tetrad.hpp"
#include "outputs.hpp"

//----------------------------------------------------------------------------------------
// BaseTypeOutput::ComputeDerivedVariable()

void BaseTypeOutput::ComputeDerivedVariable(std::string name, Mesh *pm) {
  int nmb = pm->pmb_pack->nmb_thispack;
  auto &indcs = pm->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2 * ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2 * ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2 * ng) : 1;

  int &is = indcs.is;
  int &ie = indcs.ie;
  int &js = indcs.js;
  int &je = indcs.je;
  int &ks = indcs.ks;
  int &ke = indcs.ke;
  auto &size = pm->pmb_pack->pmb->mb_size;
  auto &multi_d = pm->multi_d;
  auto &three_d = pm->three_d;

  // z-component of vorticity.
  // Not computed in ghost zones since requires derivative
  if (name.compare("hydro_wz") == 0 ||
      name.compare("mhd_wz") == 0) {
    Kokkos::realloc(derived_var, nmb, 1, n3, n2, n1);
    auto dv = derived_var;
    auto w0_ = (name.compare("hydro_wz") == 0) ?
               pm->pmb_pack->phydro->w0 : pm->pmb_pack->pmhd->w0;
    par_for("jz", DevExeSpace(), 0, (nmb - 1), ks, ke, js, je, is, ie,
            KOKKOS_LAMBDA(int m, int k, int j, int i) {
              dv(m, 0, k, j, i) = (w0_(m, IVY, k, j, i + 1) - w0_(m, IVY, k, j, i - 1)) / size.d_view(m).dx1;
              if (multi_d) {
                dv(m, 0, k, j, i) -= (w0_(m, IVX, k, j + 1, i) - w0_(m, IVX, k, j - 1, i)) / size.d_view(m).dx2;
              }
            });
  }

  // magnitude of vorticity.
  // Not computed in ghost zones since requires derivative
  if (name.compare("hydro_w2") == 0 ||
      name.compare("mhd_w2") == 0) {
    Kokkos::realloc(derived_var, nmb, 1, n3, n2, n1);
    auto dv = derived_var;
    auto w0_ = (name.compare("hydro_w2") == 0) ?
               pm->pmb_pack->phydro->w0 : pm->pmb_pack->pmhd->w0;
    par_for("jz", DevExeSpace(), 0, (nmb - 1), ks, ke, js, je, is, ie,
            KOKKOS_LAMBDA(int m, int k, int j, int i) {
              Real w1 = 0.0;
              Real w2 = -(w0_(m, IVZ, k, j, i + 1) - w0_(m, IVZ, k, j, i - 1)) / size.d_view(m).dx1;
              Real w3 = (w0_(m, IVY, k, j, i + 1) - w0_(m, IVY, k, j, i - 1)) / size.d_view(m).dx1;
              if (multi_d) {
                w1 += (w0_(m, IVZ, k, j + 1, i) - w0_(m, IVZ, k, j - 1, i)) / size.d_view(m).dx2;
                w3 -= (w0_(m, IVX, k, j + 1, i) - w0_(m, IVX, k, j - 1, i)) / size.d_view(m).dx2;
              }
              if (three_d) {
                w1 -= (w0_(m, IVY, k + 1, j, i) - w0_(m, IVY, k - 1, j, i)) / size.d_view(m).dx3;
                w2 += (w0_(m, IVX, k + 1, j, i) - w0_(m, IVX, k - 1, j, i)) / size.d_view(m).dx3;
              }
              dv(m, 0, k, j, i) = w1 * w1 + w2 * w2 + w3 * w3;
            });
  }

  // z-component of current density.  Calculated from cell-centered fields.
  // This makes for a large stencil, but approximates volume-averaged value within cell.
  // Not computed in ghost zones since requires derivative
  if (name.compare("mhd_jz") == 0) {
    Kokkos::realloc(derived_var, nmb, 1, n3, n2, n1);
    auto dv = derived_var;
    auto bcc = pm->pmb_pack->pmhd->bcc0;
    par_for("jz", DevExeSpace(), 0, (nmb - 1), ks, ke, js, je, is, ie,
            KOKKOS_LAMBDA(int m, int k, int j, int i) {
              dv(m, 0, k, j, i) = (bcc(m, IBY, k, j, i + 1) - bcc(m, IBY, k, j, i - 1)) / size.d_view(m).dx1;
              if (multi_d) {
                dv(m, 0, k, j, i) -= (bcc(m, IBX, k, j + 1, i) - bcc(m, IBX, k, j - 1, i)) / size.d_view(m).dx2;
              }
            });
  }

  // magnitude of current density.  Calculated from cell-centered fields.
  // Not computed in ghost zones since requires derivative
  if (name.compare("mhd_j2") == 0) {
    Kokkos::realloc(derived_var, nmb, 1, n3, n2, n1);
    auto dv = derived_var;
    auto bcc = pm->pmb_pack->pmhd->bcc0;
    par_for("jz", DevExeSpace(), 0, (nmb - 1), ks, ke, js, je, is, ie,
            KOKKOS_LAMBDA(int m, int k, int j, int i) {
              Real j1 = 0.0;
              Real j2 = -(bcc(m, IBZ, k, j, i + 1) - bcc(m, IBZ, k, j, i - 1)) / size.d_view(m).dx1;
              Real j3 = (bcc(m, IBY, k, j, i + 1) - bcc(m, IBY, k, j, i - 1)) / size.d_view(m).dx1;
              if (multi_d) {
                j1 += (bcc(m, IBZ, k, j + 1, i) - bcc(m, IBZ, k, j - 1, i)) / size.d_view(m).dx2;
                j3 -= (bcc(m, IBX, k, j + 1, i) - bcc(m, IBX, k, j - 1, i)) / size.d_view(m).dx2;
              }
              if (three_d) {
                j1 -= (bcc(m, IBY, k + 1, j, i) - bcc(m, IBY, k - 1, j, i)) / size.d_view(m).dx3;
                j2 += (bcc(m, IBX, k + 1, j, i) - bcc(m, IBX, k - 1, j, i)) / size.d_view(m).dx3;
              }
              dv(m, 0, k, j, i) = j1 * j1 + j2 * j2 + j3 * j3;
            });
  }

  // divergence of B, including ghost zones
  if (name.compare("mhd_divb") == 0) {
    Kokkos::realloc(derived_var, nmb, 1, n3, n2, n1);

    // set the loop limits for 1D/2D/3D problems
    int jl = js, ju = je, kl = ks, ku = ke;
    if (multi_d) {
      jl = js - ng, ju = je + ng;
    } else if (three_d) {
      jl = js - ng, ju = je + ng, kl = ks - ng, ku = ke + ng;
    }

    auto dv = derived_var;
    auto b0 = pm->pmb_pack->pmhd->b0;
    par_for("divb", DevExeSpace(), 0, (nmb - 1), kl, ku, jl, ju, (is - ng), (ie + ng),
            KOKKOS_LAMBDA(int m, int k, int j, int i) {
              Real divb = (b0.x1f(m, k, j, i + 1) - b0.x1f(m, k, j, i)) / size.d_view(m).dx1;
              if (multi_d) {
                divb += (b0.x2f(m, k, j + 1, i) - b0.x2f(m, k, j, i)) / size.d_view(m).dx2;
              }
              if (three_d) {
                divb += (b0.x3f(m, k + 1, j, i) - b0.x3f(m, k, j, i)) / size.d_view(m).dx3;
              }
              dv(m, 0, k, j, i) = divb;
            });
  }

  // radiation moments
  if (name.compare(0, 3, "rad") == 0 && pm->pmb_pack->pradfemn == NULL) {
    // Determine if coordinate and/or fluid frame moments required
    bool needs_coord_only = (name.compare("rad_coord") == 0);
    bool needs_fluid_only = (name.compare("rad_fluid") == 0);
    bool needs_both = !(needs_coord_only || needs_fluid_only);
    int mom_var_size = (needs_both) ? 20 : 10;
    int moments_offset = (needs_both) ? 10 : 0;
    Kokkos::realloc(derived_var, nmb, mom_var_size, n3, n2, n1);
    auto dv = derived_var;

    // Coordinates
    auto &coord = pm->pmb_pack->pcoord->coord_data;
    bool &flat = coord.is_minkowski;
    Real &spin = coord.bh_spin;

    // Radiation
    int nang1 = pm->pmb_pack->prad->prgeo->nangles - 1;
    auto nh_c_ = pm->pmb_pack->prad->nh_c;
    auto tet_c_ = pm->pmb_pack->prad->tet_c;
    auto tetcov_c_ = pm->pmb_pack->prad->tetcov_c;
    auto solid_angles_ = pm->pmb_pack->prad->prgeo->solid_angles;
    auto i0_ = pm->pmb_pack->prad->i0;
    auto norm_to_tet_ = pm->pmb_pack->prad->norm_to_tet;

    // Select either Hydro or MHD (if fluid enabled)
    DvceArray5D<Real> w0_;
    if (pm->pmb_pack->phydro != nullptr) {
      w0_ = pm->pmb_pack->phydro->w0;
    } else if (pm->pmb_pack->pmhd != nullptr) {
      w0_ = pm->pmb_pack->pmhd->w0;
    }

    par_for("moments", DevExeSpace(), 0, (nmb - 1), 0, (n3 - 1), 0, (n2 - 1), 0, (n1 - 1),
            KOKKOS_LAMBDA(int m, int k, int j, int i) {
              Real &x1min = size.d_view(m).x1min;
              Real &x1max = size.d_view(m).x1max;
              Real x1v = CellCenterX(i - is, indcs.nx1, x1min, x1max);

              Real &x2min = size.d_view(m).x2min;
              Real &x2max = size.d_view(m).x2max;
              Real x2v = CellCenterX(j - js, indcs.nx2, x2min, x2max);

              Real &x3min = size.d_view(m).x3min;
              Real &x3max = size.d_view(m).x3max;
              Real x3v = CellCenterX(k - ks, indcs.nx3, x3min, x3max);

              // Extract components of metric
              Real glower[4][4], gupper[4][4];
              ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);

              // coordinate component n^0
              Real n0 = tet_c_(m, 0, 0, k, j, i);

              // set coordinate frame components
              for (int n1 = 0, n12 = 0; n1 < 4; ++n1) {
                for (int n2 = n1; n2 < 4; ++n2, ++n12) {
                  dv(m, n12, k, j, i) = 0.0;
                  for (int n = 0; n <= nang1; ++n) {
                    Real nmun1 = 0.0;
                    Real nmun2 = 0.0;
                    Real n_0 = 0.0;
                    for (int d = 0; d < 4; ++d) {
                      nmun1 += tet_c_(m, d, n1, k, j, i) * nh_c_.d_view(n, d);
                      nmun2 += tet_c_(m, d, n2, k, j, i) * nh_c_.d_view(n, d);
                      n_0 += tetcov_c_(m, d, 0, k, j, i) * nh_c_.d_view(n, d);
                    }
                    dv(m, n12, k, j, i) += (nmun1 * nmun2 * (i0_(m, n, k, j, i) / (n0 * n_0)) *
                                            solid_angles_.d_view(n));
                  }
                }
              }

              if (needs_fluid_only || needs_both) {
                // stash coordinate frame moments
                Real moments_coord[4][4];
                moments_coord[0][0] = dv(m, 0, k, j, i);
                moments_coord[0][1] = dv(m, 1, k, j, i);
                moments_coord[0][2] = dv(m, 2, k, j, i);
                moments_coord[0][3] = dv(m, 3, k, j, i);
                moments_coord[1][1] = dv(m, 4, k, j, i);
                moments_coord[1][2] = dv(m, 5, k, j, i);
                moments_coord[1][3] = dv(m, 6, k, j, i);
                moments_coord[2][2] = dv(m, 7, k, j, i);
                moments_coord[2][3] = dv(m, 8, k, j, i);
                moments_coord[3][3] = dv(m, 9, k, j, i);
                moments_coord[1][0] = moments_coord[0][1];
                moments_coord[2][0] = moments_coord[0][2];
                moments_coord[3][0] = moments_coord[0][3];
                moments_coord[2][1] = moments_coord[1][2];
                moments_coord[3][1] = moments_coord[1][3];
                moments_coord[3][2] = moments_coord[2][3];

                // fluid velocity in tetrad frame
                Real uu1 = w0_(m, IVX, k, j, i);
                Real uu2 = w0_(m, IVY, k, j, i);
                Real uu3 = w0_(m, IVZ, k, j, i);
                Real q = glower[1][1] * uu1 * uu1 + 2.0 * glower[1][2] * uu1 * uu2 + 2.0 * glower[1][3] * uu1 * uu3
                         + glower[2][2] * uu2 * uu2 + 2.0 * glower[2][3] * uu2 * uu3
                         + glower[3][3] * uu3 * uu3;
                Real uu0 = sqrt(1.0 + q);
                Real u_tet_[4];
                u_tet_[0] = (norm_to_tet_(m, 0, 0, k, j, i) * uu0 + norm_to_tet_(m, 0, 1, k, j, i) * uu1 +
                             norm_to_tet_(m, 0, 2, k, j, i) * uu2 + norm_to_tet_(m, 0, 3, k, j, i) * uu3);
                u_tet_[1] = (norm_to_tet_(m, 1, 0, k, j, i) * uu0 + norm_to_tet_(m, 1, 1, k, j, i) * uu1 +
                             norm_to_tet_(m, 1, 2, k, j, i) * uu2 + norm_to_tet_(m, 1, 3, k, j, i) * uu3);
                u_tet_[2] = (norm_to_tet_(m, 2, 0, k, j, i) * uu0 + norm_to_tet_(m, 2, 1, k, j, i) * uu1 +
                             norm_to_tet_(m, 2, 2, k, j, i) * uu2 + norm_to_tet_(m, 2, 3, k, j, i) * uu3);
                u_tet_[3] = (norm_to_tet_(m, 3, 0, k, j, i) * uu0 + norm_to_tet_(m, 3, 1, k, j, i) * uu1 +
                             norm_to_tet_(m, 3, 2, k, j, i) * uu2 + norm_to_tet_(m, 3, 3, k, j, i) * uu3);

                // Construct Lorentz boost from tetrad frame to orthonormal fluid frame
                Real tet_to_fluid[4][4];
                tet_to_fluid[0][0] = u_tet_[0];
                tet_to_fluid[0][1] = -u_tet_[1];
                tet_to_fluid[0][2] = -u_tet_[2];
                tet_to_fluid[0][3] = -u_tet_[3];
                tet_to_fluid[1][1] = u_tet_[1] * u_tet_[1] / (1.0 + u_tet_[0]) + 1.0;
                tet_to_fluid[1][2] = u_tet_[1] * u_tet_[2] / (1.0 + u_tet_[0]);
                tet_to_fluid[1][3] = u_tet_[1] * u_tet_[3] / (1.0 + u_tet_[0]);
                tet_to_fluid[2][2] = u_tet_[2] * u_tet_[2] / (1.0 + u_tet_[0]) + 1.0;
                tet_to_fluid[2][3] = u_tet_[2] * u_tet_[3] / (1.0 + u_tet_[0]);
                tet_to_fluid[3][3] = u_tet_[3] * u_tet_[3] / (1.0 + u_tet_[0]) + 1.0;
                tet_to_fluid[1][0] = tet_to_fluid[0][1];
                tet_to_fluid[2][0] = tet_to_fluid[0][2];
                tet_to_fluid[3][0] = tet_to_fluid[0][3];
                tet_to_fluid[2][1] = tet_to_fluid[1][2];
                tet_to_fluid[3][1] = tet_to_fluid[1][3];
                tet_to_fluid[3][2] = tet_to_fluid[2][3];

                // set tetrad frame moments
                for (int n1 = 0, n12 = 0; n1 < 4; ++n1) {
                  for (int n2 = n1; n2 < 4; ++n2, ++n12) {
                    dv(m, moments_offset + n12, k, j, i) = 0.0;
                    for (int m1 = 0; m1 < 4; ++m1) {
                      for (int m2 = 0; m2 < 4; ++m2) {
                        dv(m, moments_offset + n12, k, j, i) += (tetcov_c_(m, n1, m1, k, j, i) *
                                                                 tetcov_c_(m, n2, m2, k, j, i) *
                                                                 moments_coord[m1][m2]);
                      }
                    }
                  }
                }
                dv(m, moments_offset + 1, k, j, i) *= -1.0;
                dv(m, moments_offset + 2, k, j, i) *= -1.0;
                dv(m, moments_offset + 3, k, j, i) *= -1.0;

                // stash tetrad frame moments
                Real moments_tetrad[4][4];
                moments_tetrad[0][0] = dv(m, moments_offset + 0, k, j, i);
                moments_tetrad[0][1] = dv(m, moments_offset + 1, k, j, i);
                moments_tetrad[0][2] = dv(m, moments_offset + 2, k, j, i);
                moments_tetrad[0][3] = dv(m, moments_offset + 3, k, j, i);
                moments_tetrad[1][1] = dv(m, moments_offset + 4, k, j, i);
                moments_tetrad[1][2] = dv(m, moments_offset + 5, k, j, i);
                moments_tetrad[1][3] = dv(m, moments_offset + 6, k, j, i);
                moments_tetrad[2][2] = dv(m, moments_offset + 7, k, j, i);
                moments_tetrad[2][3] = dv(m, moments_offset + 8, k, j, i);
                moments_tetrad[3][3] = dv(m, moments_offset + 9, k, j, i);
                moments_tetrad[1][0] = moments_tetrad[0][1];
                moments_tetrad[2][0] = moments_tetrad[0][2];
                moments_tetrad[3][0] = moments_tetrad[0][3];
                moments_tetrad[2][1] = moments_tetrad[1][2];
                moments_tetrad[3][1] = moments_tetrad[1][3];
                moments_tetrad[3][2] = moments_tetrad[2][3];

                // set R^{\mu \nu} (fluid frame)
                for (int n1 = 0, n12 = 0; n1 < 4; ++n1) {
                  for (int n2 = n1; n2 < 4; ++n2, ++n12) {
                    dv(m, moments_offset + n12, k, j, i) = 0.0;
                    for (int m1 = 0; m1 < 4; ++m1) {
                      for (int m2 = 0; m2 < 4; ++m2) {
                        dv(m, moments_offset + n12, k, j, i) += (tet_to_fluid[n1][m1] *
                                                                 tet_to_fluid[n2][m2] *
                                                                 moments_tetrad[m1][m2]);
                      }
                    }
                  }
                }
              }
            });
  }

  // radiation number density
  if (name.compare("rad_femn_N") == 0) {
    Kokkos::realloc(derived_var, nmb, 1, n3, n2, n1);
    auto dv = derived_var;
    auto &f0_ = pm->pmb_pack->pradfemn->f0;
    auto &L_mu_muhat_ = pm->pmb_pack->pradfemn->L_mu_muhat0;
    auto &Q_matrix_ = pm->pmb_pack->pradfemn->Q_matrix;
    auto &e_source_nominv_ = pm->pmb_pack->pradfemn->e_source_nominv;
    auto &energy_grid_ = pm->pmb_pack->pradfemn->energy_grid;
    auto &num_points_ = pm->pmb_pack->pradfemn->num_points;
    auto &num_energy_bins_ = pm->pmb_pack->pradfemn->num_energy_bins;
    auto &num_species_ = pm->pmb_pack->pradfemn->num_species;

    // Compute sum_{B} S_n F^nA M_AB where S_n = (e_{n+1}^4 - e_{n}^4)/4
    // @TODO: currently only works for one species of neutrino -- generalize!
    int scr_level = 0;
    int scr_size = 1;
    int species = 0;
    par_for_outer("rad_femn_N_compute", DevExeSpace(), scr_size, scr_level, 0, nmb - 1, ks, ke, js, je, is, ie,
                  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j, const int i) {
                    Real temp_sum = 0.;
                    Kokkos::parallel_reduce(Kokkos::TeamVectorRange(member, 0, num_energy_bins_ * num_points_), [=](const int enA, Real &partial_sum) {
                      int en = int(enA / num_points_);
                      int A = enA - en * num_points_;

                      int nuenang = radiationfemn::IndicesUnited(species, en, A, num_species_, num_energy_bins_, num_points_);
                      Real Ven = (num_energy_bins_ > 1) ? (pow(energy_grid_(en + 1), 3) - pow(energy_grid_(en), 3)) / 3.0 : 1.;

                      Real ql_term = 0;
                      for (int idx = 0; idx < 4; idx++){
                        ql_term += Q_matrix_(idx, A) * L_mu_muhat_(m, 0, idx, k, j, i);
                      }
                      partial_sum += Ven * f0_(m, nuenang, k, j, i) * ql_term;
                    }, temp_sum);
                    dv(m, 0, k, j, i) = temp_sum;
                  });
  }

  // radiation energy density for FEM_N
  if (name.compare("rad_femn_E") == 0 && pm->pmb_pack->pradfemn->fpn == 0) {
    Kokkos::realloc(derived_var, nmb, 1, n3, n2, n1);
    auto dv = derived_var;
    auto &f0_ = pm->pmb_pack->pradfemn->f0;
    auto &e_source_nominv_ = pm->pmb_pack->pradfemn->e_source_nominv;
    auto &energy_grid_ = pm->pmb_pack->pradfemn->energy_grid;
    auto &num_points_ = pm->pmb_pack->pradfemn->num_points;
    auto &num_energy_bins_ = pm->pmb_pack->pradfemn->num_energy_bins;
    auto &num_species_ = pm->pmb_pack->pradfemn->num_species;

    // Compute sum_{B} S_n F^nA M_AB where S_n = (e_{n+1}^4 - e_{n}^4)/4
    // @TODO: currently only works for one species of neutrino -- generalize!
    int scr_level = 0;
    int scr_size = 1;
    int species = 0;
    par_for_outer("rad_femn_E_compute_femn", DevExeSpace(), scr_size, scr_level, 0, nmb - 1, ks, ke, js, je, is, ie,
                  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j, const int i) {

                    Real temp_sum = 0.;
                    Kokkos::parallel_reduce(Kokkos::TeamVectorRange(member, 0, num_energy_bins_ * num_points_), [=](const int enA, Real &partial_sum) {

                      int en = int(enA / num_points_);
                      int A = enA - en * num_points_;

                      int nuenang = radiationfemn::IndicesUnited(species, en, A, num_species_, num_energy_bins_, num_points_);
                      Real Sen = (num_energy_bins_ > 1) ? (pow(energy_grid_(en + 1), 4) - pow(energy_grid_(en), 4)) / 4.0 : 1.;
                      partial_sum += Sen * e_source_nominv_(A) * f0_(m, nuenang, k, j, i);

                    }, temp_sum);

                    dv(m, 0, k, j, i) = temp_sum;
                  });
  }

  if (name.compare("rad_femn_E") == 0 && pm->pmb_pack->pradfemn->fpn != 0) {
    Kokkos::realloc(derived_var, nmb, 1, n3, n2, n1);
    auto dv = derived_var;
    auto &f0_ = pm->pmb_pack->pradfemn->f0;
    auto neng = pm->pmb_pack->pradfemn->num_energy_bins;
    auto nmodes = pm->pmb_pack->pradfemn->num_points;
    auto &energy_grid_ = pm->pmb_pack->pradfemn->energy_grid;
    auto &num_energy_bins_ = pm->pmb_pack->pradfemn->num_energy_bins;

    // Compute sum_{B} sqrt(4 pi) S_n F^n0 where S_n = (e_{n+1}^4 - e_{n}^4)/4
    int scr_level = 0;
    int scr_size = 1;
    par_for_outer("rad_femn_E_compute_fpn", DevExeSpace(), scr_size, scr_level, 0, nmb - 1, ks, ke, js, je, is, ie,
                  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j, const int i) {

                    Real Eval = 0.;
                    Kokkos::parallel_reduce(Kokkos::TeamVectorRange(member, 0, neng), [&](const int en, Real &partial_sum) {
                      Real Sen = (num_energy_bins_ > 1) ? (pow(energy_grid_(en + 1), 4) - pow(energy_grid_(en), 4)) / 4.0 : 1.;
                      partial_sum += Sen * 2. * sqrt(M_PI) * f0_(m, en * nmodes, k, j, i);
                    }, Eval);
                    member.team_barrier();

                    dv(m, 0, k, j, i) = Eval;
                  });
  }

  if (name.compare("rad_femn_flux") == 0 && pm->pmb_pack->pradfemn->fpn != 0 && pm->pmb_pack->pradfemn->m1_flag != false) {
    Kokkos::realloc(derived_var, nmb, 3, n3, n2, n1);
    auto dv = derived_var;
    auto &f0_ = pm->pmb_pack->pradfemn->f0;
    auto neng = pm->pmb_pack->pradfemn->num_energy_bins;
    auto nmodes = pm->pmb_pack->pradfemn->num_points;
    auto &energy_grid_ = pm->pmb_pack->pradfemn->energy_grid;
    auto &num_energy_bins_ = pm->pmb_pack->pradfemn->num_energy_bins;

    int scr_level = 0;
    int scr_size = 1;
    // only works for a single energy bin and single species
    par_for_outer("rad_femn_E_compute_flux_m1", DevExeSpace(), scr_size, scr_level, 0, nmb - 1, ks, ke, js, je, is, ie,
                  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j, const int i) {
                    dv(m, 0, k, j, i) = -Kokkos::sqrt(4. * M_PI / 3.0) * f0_(m, 3, k, j, i);
                    dv(m, 1, k, j, i) = -Kokkos::sqrt(4. * M_PI / 3.0) * f0_(m, 1, k, j, i);
                    dv(m, 2, k, j, i) = Kokkos::sqrt(4. * M_PI / 3.0) * f0_(m, 2, k, j, i);
                  });
  }
}
