#include <iostream>
#include <cmath>
#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "pgen.hpp"
#include "units/units.hpp"
#include "utils/sn_scheduler.hpp"
#include "particles/particles.hpp"

//===========================================================================//
//                               Globals                                     //
//===========================================================================//

void SNTestSource(Mesh* pm, const Real bdt);
void SNTestHistory(HistoryData *pdata, Mesh *pm);

namespace {
  Real r_inj;
  Real e_sn;
  Real m_ej;
  DvceArray2D<Real> sn_centers_buffer;

  int last_sn_detect_cycle = -1;
  int num_sn_this_cycle;
}

//===========================================================================//
//                               Initialize                                  //
//===========================================================================//

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;

  user_srcs_func = SNTestSource;
  user_hist_func = SNTestHistory;

  // Read SN parameters
  r_inj = pin->GetReal("SN", "r_inj");
  const Real sphere_vol = (4.0/3.0)*M_PI*std::pow(r_inj, 3);
  const Real E_def = 1e51;
  const Real M_def = 8.4;
  e_sn = pin->GetOrAddReal("SN", "E_sn", E_def)*pmbp->punit->erg()/sphere_vol;
  m_ej = pin->GetOrAddReal("SN", "M_ej", M_def)*pmbp->punit->msun()/sphere_vol;

  // Initialize particle tracking
  pmy_mesh_->CountParticles();
  sn_centers_buffer = DvceArray2D<Real>("sn_centers", 3, pmy_mesh_->nprtcl_total);

  if (global_variable::my_rank == 0) {
    std::cout << std::endl;
    std::cout << "=== SN Test ===" << std::endl;
    std::cout << "r_inj  = " << r_inj << std::endl;
    std::cout << "e_sn   = " << e_sn << " (code density)" << std::endl;
    std::cout << "m_ej   = " << m_ej << " (code density)" << std::endl;
    std::cout << "E_tot  = " << e_sn*sphere_vol << " (code energy)" << std::endl;
    std::cout << "M_tot  = " << m_ej*sphere_vol << " (code mass)" << std::endl;
    std::cout << std::endl;
  }

  if (restart) return;

  // Uniform box initial conditions
  Real rho0 = pin->GetReal("problem", "rho0");
  Real pres0 = pin->GetReal("problem", "pres0");
  auto &eos = pmbp->phydro->peos->eos_data;
  Real gm1 = eos.gamma - 1.0;

  auto &u0 = pmbp->phydro->u0;
  int nmb = pmbp->nmb_thispack;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;

  par_for("sn_test_init", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    u0(m, IDN, k, j, i) = rho0;
    u0(m, IM1, k, j, i) = 0.0;
    u0(m, IM2, k, j, i) = 0.0;
    u0(m, IM3, k, j, i) = 0.0;
    u0(m, IEN, k, j, i) = pres0/gm1;
  });
}

//===========================================================================//
//                              Source Term                                   //
//===========================================================================//

void SNTestSource(Mesh* pm, const Real bdt) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int nmb1 = pmbp->nmb_thispack - 1;
  auto &size = pmbp->pmb->mb_size;

  auto &u0 = pmbp->phydro->u0;
  Real beta = bdt / pm->dt;
  Real dr = r_inj;

  // Detect SNe once per cycle
  if (pm->ncycle != last_sn_detect_cycle) {
    last_sn_detect_cycle = pm->ncycle;

    auto &pr = pmbp->ppart->prtcl_rdata;
    auto &pi = pmbp->ppart->prtcl_idata;
    int npart = pmbp->ppart->nprtcl_thispack;
    int nrdata = pmbp->ppart->nrdata;
    auto gids = pmbp->gids;
    Real time = pm->time;
    Real unit_time = pmbp->punit->time_cgs();

    auto &sn_centers = sn_centers_buffer;
    Kokkos::View<int> d_counter("sn_counter");

    par_for("sn_check", DevExeSpace(), 0, npart-1, KOKKOS_LAMBDA(const int p) {
      Real next_sn_time = pr(IPT_NEXT_SN, p);

      if (time > next_sn_time) {
        pi(NSN, p) += 1;
        int sn_idx = pi(NSN, p);
        Real par_t_create = pr(IPT_CREATE, p);
        Real cluster_mass = pr(IPMASS, p);
        pr(IPT_NEXT_SN, p) = GetNthSNTime(cluster_mass, par_t_create, unit_time, sn_idx);

        int idx = Kokkos::atomic_fetch_add(&d_counter(), 1);
        int m = pi(PGID, p) - gids;

        Real x1min = size.d_view(m).x1min;
        Real x1max = size.d_view(m).x1max;
        Real x2min = size.d_view(m).x2min;
        Real x2max = size.d_view(m).x2max;
        Real x3min = size.d_view(m).x3min;
        Real x3max = size.d_view(m).x3max;

        sn_centers(0, idx) = fmin(fmax(pr(IPX,p), x1min+dr), x1max-dr);
        sn_centers(1, idx) = fmin(fmax(pr(IPY,p), x2min+dr), x2max-dr);
        sn_centers(2, idx) = fmin(fmax(pr(IPZ,p), x3min+dr), x3max-dr);
      }
    });

    DevExeSpace().fence();
    Kokkos::deep_copy(num_sn_this_cycle, d_counter);
    std::cout << "  [SN] " << num_sn_this_cycle << " SN at time=" << time << std::endl;
  }

  if (num_sn_this_cycle > 0) {
    int num_sn = num_sn_this_cycle;
    Real e_sn_ = e_sn * beta;
    Real m_ej_ = m_ej * beta;
    auto &sn_centers = sn_centers_buffer;

    par_for("sn_inject", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real x1min = size.d_view(m).x1min;
      Real x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

      Real x2min = size.d_view(m).x2min;
      Real x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

      Real x3min = size.d_view(m).x3min;
      Real x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

      for (int sn = 0; sn < num_sn; ++sn) {
        Real dx = x1v - sn_centers(0, sn);
        Real dy = x2v - sn_centers(1, sn);
        Real dz = x3v - sn_centers(2, sn);
        Real r = sqrt(dx*dx + dy*dy + dz*dz);

        if (r <= dr) {
          u0(m, IDN, k, j, i) += m_ej_;
          u0(m, IEN, k, j, i) += e_sn_;
        }
      }
    });
  }
}

//===========================================================================//
//                            History Output                                  //
//===========================================================================//

void SNTestHistory(HistoryData *pdata, Mesh *pm) {
  pdata->nhist = 2;
  pdata->label[0] = "tot-mass";
  pdata->label[1] = "tot-E";

  auto &u0_ = pm->pmb_pack->phydro->u0;
  auto &size = pm->pmb_pack->pmb->mb_size;

  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int is = indcs.is; int nx1 = indcs.nx1;
  int js = indcs.js; int nx2 = indcs.nx2;
  int ks = indcs.ks; int nx3 = indcs.nx3;
  int nmb1 = pm->pmb_pack->nmb_thispack - 1;

  array_sum::GlobalSum sum_this_mb;
  Kokkos::parallel_reduce("sn_hist",
    Kokkos::RangePolicy<>(DevExeSpace(), 0,
      (nmb1+1)*nx3*nx2*nx1),
    KOKKOS_LAMBDA(const int idx, array_sum::GlobalSum &mb_sum) {
      int m = idx / (nx3*nx2*nx1);
      int rem = idx % (nx3*nx2*nx1);
      int k = ks + rem / (nx2*nx1);
      rem = rem % (nx2*nx1);
      int j = js + rem / nx1;
      int i = is + rem % nx1;

      Real x1min = size.d_view(m).x1min;
      Real x1max = size.d_view(m).x1max;
      Real dx1 = (x1max - x1min)/static_cast<Real>(nx1);
      Real x2min = size.d_view(m).x2min;
      Real x2max = size.d_view(m).x2max;
      Real dx2 = (x2max - x2min)/static_cast<Real>(nx2);
      Real x3min = size.d_view(m).x3min;
      Real x3max = size.d_view(m).x3max;
      Real dx3 = (x3max - x3min)/static_cast<Real>(nx3);
      Real vol = dx1*dx2*dx3;

      array_sum::GlobalSum hvars;
      hvars.the_array[0] = vol*u0_(m, IDN, k, j, i);
      hvars.the_array[1] = vol*u0_(m, IEN, k, j, i);

      for (int n = 2; n < NREDUCTION_VARIABLES; ++n) {
        hvars.the_array[n] = 0.0;
      }

      mb_sum += hvars;
    }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb));
  Kokkos::fence();

  for (int n = 0; n < pdata->nhist; ++n) {
    pdata->hdata[n] = sum_this_mb.the_array[n];
  }
}
