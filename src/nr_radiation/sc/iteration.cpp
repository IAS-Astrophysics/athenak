//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file iteration.cpp
//! \brief Task assembly, the per-solve opacity/source update, boundary-exchange tasks,
//! and SolveTransfer: the sweep-then-exchange loop iterated until J stops changing.
//! sc_bvals exchanges ir and srad (source S).

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "eos/eos.hpp"
#include "driver/driver.hpp"
#include "nr_radiation/nr_radiation.hpp"

namespace nr_radiation {

//----------------------------------------------------------------------------------------
//! \fn void SC::AssembleSCTasks
//! \brief Register the solve, coupling and timestep tasks and the three exchange lists.

void SC::AssembleSCTasks(std::map<std::string, std::shared_ptr<TaskList>> tl) {
  TaskID none(0);
  hydro::Hydro *phyd = pmy_pack->phydro;
  mhd::MHD *pmhd = pmy_pack->pmhd;

  id.sc_solve = tl["before_timeintegrator"]->AddTask(&SC::SolveTransfer,
                                                      this, none);

  if (affect_fluid) {
    if (phyd != nullptr) {
      id.sc_qrad = tl["stagen"]->InsertTask(&SC::AddQrad, this,
                                             phyd->id.rkupdt, phyd->id.srctrms);
      id.sc_newdt = tl["stagen"]->AddTask(&SC::NewTimeStep, this, phyd->id.newdt);
    } else if (pmhd != nullptr) {
      id.sc_qrad = tl["stagen"]->InsertTask(&SC::AddQrad, this,
                                             pmhd->id.rkupdt, pmhd->id.srctrms);
      id.sc_newdt = tl["stagen"]->AddTask(&SC::NewTimeStep, this, pmhd->id.newdt);
    }
  }

  // sc_bvals: dual CC exchange for ir (nvar=nang_tot) and srad (nvar=1)
  // InitRecv both from none -> Restrict -> Send -> Recv -> PhysBCs (after both Recvs)
  // -> Prolong -> Clear*
  auto &vtl = tl["sc_bvals"];
  id.srad_irecv = vtl->AddTask(&SC::InitRecvSrad, this, none);
  id.ir_irecv = vtl->AddTask(&SC::InitRecvIr, this, none);

  id.srad_rest = vtl->AddTask(&SC::RestrictSrad, this, id.srad_irecv);
  id.ir_rest = vtl->AddTask(&SC::RestrictIr, this, id.ir_irecv);

  id.srad_send = vtl->AddTask(&SC::SendSrad, this, id.srad_rest);
  id.ir_send = vtl->AddTask(&SC::SendIr, this, id.ir_rest);

  id.srad_recv = vtl->AddTask(&SC::RecvSrad, this, id.srad_send);
  id.ir_recv = vtl->AddTask(&SC::RecvIr, this, id.ir_send);

  TaskID both_recv = id.srad_recv | id.ir_recv;
  id.ir_bcs = vtl->AddTask(&SC::ApplyPhysicalBCsIr, this, both_recv);
  id.srad_bcs = vtl->AddTask(&SC::ApplyPhysicalBCsSrad, this, id.ir_bcs);

  id.srad_prol = vtl->AddTask(&SC::ProlongateSrad, this, id.srad_bcs);
  id.ir_prol = vtl->AddTask(&SC::ProlongateIr, this, id.srad_bcs);

  TaskID both_prol = id.srad_prol | id.ir_prol;
  id.srad_csend = vtl->AddTask(&SC::ClearSendSrad, this, both_prol);
  id.ir_csend = vtl->AddTask(&SC::ClearSendIr, this, id.srad_csend);
  id.srad_crecv = vtl->AddTask(&SC::ClearRecvSrad, this, id.ir_csend);
  id.ir_crecv = vtl->AddTask(&SC::ClearRecvIr, this, id.srad_crecv);

  // sc_bvals_srad: the srad half alone, for the start of a solve. B changes between
  // solves (the fluid moved), ir does not, so the source needs refreshing while ir's
  // ghosts are still the ones the previous solve's post-sweep exchange left. Reuses the
  // same wrappers; only the chain is new.
  auto &stl = tl["sc_bvals_srad"];
  id.sronly_irecv = stl->AddTask(&SC::InitRecvSrad, this, none);
  id.sronly_rest = stl->AddTask(&SC::RestrictSrad, this, id.sronly_irecv);
  id.sronly_send = stl->AddTask(&SC::SendSrad, this, id.sronly_rest);
  id.sronly_recv = stl->AddTask(&SC::RecvSrad, this, id.sronly_send);
  id.sronly_bcs = stl->AddTask(&SC::ApplyPhysicalBCsSrad, this, id.sronly_recv);
  id.sronly_prol = stl->AddTask(&SC::ProlongateSrad, this, id.sronly_bcs);
  id.sronly_csend = stl->AddTask(&SC::ClearSendSrad, this, id.sronly_prol);
  id.sronly_crecv = stl->AddTask(&SC::ClearRecvSrad, this, id.sronly_csend);

  // sc_flxcor: coarse-fine correction of hflx. Run once per solve, after the face flux
  // is built, on a multilevel mesh only.
  auto &ftl = tl["sc_flxcor"];
  // InitFluxRecv is NOT a task here: SolveTransfer posts it before the iteration loop,
  // so every rank has posted before any rank sends and ClearFluxSend always clears.
  id.hflx_send = ftl->AddTask(&SC::SendHflx, this, none);
  id.hflx_recv = ftl->AddTask(&SC::RecvHflx, this, id.hflx_send);
  id.hflx_csend = ftl->AddTask(&SC::ClearSendHflx, this, id.hflx_recv);
  id.hflx_crecv = ftl->AddTask(&SC::ClearRecvHflx, this, id.hflx_csend);
}

//----------------------------------------------------------------------------------------
//! \fn void SC::UpdateOpacity / SC::UpdateEmission
//! \brief Per-cell opacity and LTE emission of the current fluid state over ALL cells
//! including ghosts (the sweep reads both at ghost footpoints): the enrolled hook, else
//! sigma_a = kappa_a * rho and brad = T^4. Both run at the start of every solve; the
//! opacity also wherever the relaxation timestep is evaluated. Neither is state: nothing
//! is packed by AMR or checkpointed.

void SC::UpdateOpacity() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto chi_ = chi;
  auto sigma_a_ = sigma_a;
  Real kappa_a_ = kappa_a;

  if (user_opacity_func != nullptr) {
    user_opacity_func(pmy_pack);
  } else {
    DvceArray5D<Real> w0;
    if (pmy_pack->phydro != nullptr) {
      w0 = pmy_pack->phydro->w0;
    } else {
      w0 = pmy_pack->pmhd->w0;
    }
    par_for("sc_sigma_a", DevExeSpace(), 0, nmb1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real dens = fmax(w0(m,IDN,k,j,i), 0.0);
      sigma_a_(m,0,k,j,i) = kappa_a_ * dens;
    });
  }
  par_for("sc_chi", DevExeSpace(), 0, nmb1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    chi_(m,k,j,i) = sigma_a_(m,0,k,j,i);
  });
}

void SC::UpdateEmission() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto brad_ = brad;

  if (user_emission_func != nullptr) {
    user_emission_func(pmy_pack);
  } else {
    // ideal-gas temperature T = (gamma-1) e / rho
    DvceArray5D<Real> w0;
    Real gm1 = 0.0;
    if (pmy_pack->phydro != nullptr) {
      w0 = pmy_pack->phydro->w0;
      gm1 = pmy_pack->phydro->peos->eos_data.gamma - 1.0;
    } else {
      w0 = pmy_pack->pmhd->w0;
      gm1 = pmy_pack->pmhd->peos->eos_data.gamma - 1.0;
    }
    par_for("sc_emission", DevExeSpace(), 0, nmb1, 0,(n3-1), 0,(n2-1), 0,(n1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real dens = fmax(w0(m,IDN,k,j,i), 0.0);
      Real temp = (dens > 0.0) ? (gm1 * w0(m,IEN,k,j,i) / dens) : 0.0;
      brad_(m,k,j,i) = temp * temp * temp * temp;
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn void SC::UpdateSource
//! \brief The source the sweep reads: S = brad without scattering. Kept as its own array
//! because it is exchanged and packed by AMR as a CC field.

void SC::UpdateSource() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto srad_ = srad;
  auto brad_ = brad;
  par_for("sc_srad", DevExeSpace(), 0, nmb1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    srad_(m,0,k,j,i) = brad_(m,k,j,i);
  });
}

//----------------------------------------------------------------------------------------
//! \fn void SC::ApplyPhysicalBCs
//! \brief Intensity physical BCs only (RadiationBCs + optional user_bcs).

void SC::ApplyPhysicalBCs() {
  if (!(pmy_pack->pmesh->strictly_periodic)) {
    pbval_ir->RadiationBCs(pmy_pack, pbval_ir->i_in, ir);
    ApplyReflectBCs();   // reflect is not handled by RadiationBCs; see below
  }
  if (pmy_pack->pmesh->pgen != nullptr && pmy_pack->pmesh->pgen->user_bcs) {
    (pmy_pack->pmesh->pgen->user_bcs_func)(pmy_pack->pmesh);
  }
}

//----------------------------------------------------------------------------------------
//! \fn int MirrorRay
//! \brief The ray with direction cosine `axis` negated: the same |mu| triple, in the
//! octant with that sign bit flipped, same in-octant index. axis_bit = 1<<axis; x is
//! bit 0 in the 1D, 2D and 3D octant encodings alike (see SCAngularGrid::BuildCarlsonND).

KOKKOS_INLINE_FUNCTION
int MirrorRay(int angg, int nang, int axis_bit) {
  const int oct = angg / nang;
  return ((oct ^ axis_bit) * nang) + (angg - oct * nang);
}

//----------------------------------------------------------------------------------------
//! \fn void SC::ApplyReflectBCs
//! \brief Mirror the intensity into the ghost zones of reflecting physical boundaries:
//! ghost is-i-1 takes ray angg from active cell is+i of the ray with the normal direction
//! cosine negated (MirrorRay). Exact for the Carlson set, which is closed under these
//! reflections. Only ghosts are written and only interior cells read. Lives here rather
//! than in bvals/RadiationBCs because the ray mirror is a property of this angular grid.

void SC::ApplyReflectBCs() {
  auto &pm = pmy_pack->pmesh;
  // only faces below 2*ndim are filled by Mesh; the rest are indeterminate
  const int ndim = pang->ndim;
  bool any = false;
  for (int f = 0; f < 2*ndim; ++f) {
    if (pm->mesh_bcs[f] == BoundaryFlag::reflect) { any = true; }
  }
  if (!any) return;

  auto &indcs = pm->mb_indcs;
  int &ng = indcs.ng;
  auto &mb_bcs = pmy_pack->pmb->mb_bcs;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nangt1 = nang_tot - 1;
  int nang = pang->nang;
  auto ir_ = ir;

  if (pm->mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::reflect ||
      pm->mesh_bcs[BoundaryFace::outer_x1] == BoundaryFlag::reflect) {
    int &is = indcs.is;
    int &ie = indcs.ie;
    par_for("sc_ir_reflect_x1", DevExeSpace(), 0,nmb1, 0,nangt1, 0,(n3-1), 0,(n2-1),
    KOKKOS_LAMBDA(int m, int angg, int k, int j) {
      const int ma = MirrorRay(angg, nang, 1);
      if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::reflect) {
        for (int i = 0; i < ng; ++i) { ir_(m,angg,k,j,is-i-1) = ir_(m,ma,k,j,is+i); }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::reflect) {
        for (int i = 0; i < ng; ++i) { ir_(m,angg,k,j,ie+i+1) = ir_(m,ma,k,j,ie-i); }
      }
    });
  }
  if (pm->one_d) return;

  if (pm->mesh_bcs[BoundaryFace::inner_x2] == BoundaryFlag::reflect ||
      pm->mesh_bcs[BoundaryFace::outer_x2] == BoundaryFlag::reflect) {
    int &js = indcs.js;
    int &je = indcs.je;
    par_for("sc_ir_reflect_x2", DevExeSpace(), 0,nmb1, 0,nangt1, 0,(n3-1), 0,(n1-1),
    KOKKOS_LAMBDA(int m, int angg, int k, int i) {
      const int ma = MirrorRay(angg, nang, 2);
      if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::reflect) {
        for (int j = 0; j < ng; ++j) { ir_(m,angg,k,js-j-1,i) = ir_(m,ma,k,js+j,i); }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::reflect) {
        for (int j = 0; j < ng; ++j) { ir_(m,angg,k,je+j+1,i) = ir_(m,ma,k,je-j,i); }
      }
    });
  }
  if (!(pm->three_d)) return;

  if (pm->mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::reflect ||
      pm->mesh_bcs[BoundaryFace::outer_x3] == BoundaryFlag::reflect) {
    int &ks = indcs.ks;
    int &ke = indcs.ke;
    par_for("sc_ir_reflect_x3", DevExeSpace(), 0,nmb1, 0,nangt1, 0,(n2-1), 0,(n1-1),
    KOKKOS_LAMBDA(int m, int angg, int j, int i) {
      const int ma = MirrorRay(angg, nang, 4);
      if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::reflect) {
        for (int k = 0; k < ng; ++k) { ir_(m,angg,ks-k-1,j,i) = ir_(m,ma,ks+k,j,i); }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::reflect) {
        for (int k = 0; k < ng; ++k) { ir_(m,angg,ke+k+1,j,i) = ir_(m,ma,ke-k,j,i); }
      }
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn void SC::ApplyPhysicalBCsSource
//! \brief Nearest-active copy of S (srad) into the ghost layers of faces flagged outflow
//! or inflow. Other flags keep the srad that UpdateSource wrote from the ghost state.

void SC::ApplyPhysicalBCsSource() {
  auto &pm = pmy_pack->pmesh;
  if (pm->strictly_periodic) return;

  auto &indcs = pm->mb_indcs;
  int &ng = indcs.ng;
  auto &mb_bcs = pmy_pack->pmb->mb_bcs;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto srad_ = srad;

  if (pm->mesh_bcs[BoundaryFace::inner_x1] != BoundaryFlag::periodic) {
    int &is = indcs.is;
    int &ie = indcs.ie;
    par_for("sc_srad_bc_x1", DevExeSpace(), 0, nmb1, 0, (n3-1), 0, (n2-1),
    KOKKOS_LAMBDA(int m, int k, int j) {
      auto f_in = mb_bcs.d_view(m, BoundaryFace::inner_x1);
      auto f_ox = mb_bcs.d_view(m, BoundaryFace::outer_x1);
      if (f_in == BoundaryFlag::outflow || f_in == BoundaryFlag::inflow) {
        for (int i = 0; i < ng; ++i) {
          srad_(m,0,k,j,is-i-1) = srad_(m,0,k,j,is);
        }
      }
      if (f_ox == BoundaryFlag::outflow || f_ox == BoundaryFlag::inflow) {
        for (int i = 0; i < ng; ++i) {
          srad_(m,0,k,j,ie+i+1) = srad_(m,0,k,j,ie);
        }
      }
    });
  }
  if (pm->one_d) return;

  if (pm->mesh_bcs[BoundaryFace::inner_x2] != BoundaryFlag::periodic) {
    int &js = indcs.js;
    int &je = indcs.je;
    par_for("sc_srad_bc_x2", DevExeSpace(), 0, nmb1, 0, (n3-1), 0, (n1-1),
    KOKKOS_LAMBDA(int m, int k, int i) {
      auto f_in = mb_bcs.d_view(m, BoundaryFace::inner_x2);
      auto f_ox = mb_bcs.d_view(m, BoundaryFace::outer_x2);
      if (f_in == BoundaryFlag::outflow || f_in == BoundaryFlag::inflow) {
        for (int j = 0; j < ng; ++j) {
          srad_(m,0,k,js-j-1,i) = srad_(m,0,k,js,i);
        }
      }
      if (f_ox == BoundaryFlag::outflow || f_ox == BoundaryFlag::inflow) {
        for (int j = 0; j < ng; ++j) {
          srad_(m,0,k,je+j+1,i) = srad_(m,0,k,je,i);
        }
      }
    });
  }
  if (pm->two_d) return;

  if (pm->mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::periodic) return;
  int &ks = indcs.ks;
  int &ke = indcs.ke;
  par_for("sc_srad_bc_x3", DevExeSpace(), 0, nmb1, 0, (n2-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int j, int i) {
    auto f_in = mb_bcs.d_view(m, BoundaryFace::inner_x3);
    auto f_ox = mb_bcs.d_view(m, BoundaryFace::outer_x3);
    if (f_in == BoundaryFlag::outflow || f_in == BoundaryFlag::inflow) {
      for (int k = 0; k < ng; ++k) {
        srad_(m,0,ks-k-1,j,i) = srad_(m,0,ks,j,i);
      }
    }
    if (f_ox == BoundaryFlag::outflow || f_ox == BoundaryFlag::inflow) {
      for (int k = 0; k < ng; ++k) {
        srad_(m,0,ke+k+1,j,i) = srad_(m,0,ke,j,i);
      }
    }
  });
}

//----------------------------------------------------------------------------------------
//! \fn Real SC::ResidualJ
//! \brief Max over active cells of the symmetric relative change |Jn-Jo| / max(Jn,Jo)
//! between J and j_prev, allreduced across ranks. A cell lit from exactly zero reports
//! 1, so a front entering vacuum cannot stop the iteration early; the only zero-change
//! case is Jn == Jo == 0.

Real SC::ResidualJ() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nx1 = ie - is + 1, nx2 = je - js + 1, nx3 = ke - ks + 1;
  int nmkji = pmy_pack->nmb_thispack * nx3 * nx2 * nx1;
  auto jn_ = moments;
  auto jo_ = j_prev;
  Real rmax = 0.0;
  Kokkos::parallel_reduce("sc_residual_j", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int idx, Real &lmax) {
    int m = idx / (nx3*nx2*nx1);
    int kji = idx - m*(nx3*nx2*nx1);
    int k = kji / (nx2*nx1);
    int ji = kji - k*(nx2*nx1);
    int j = ji / nx1;
    int i = ji - j*nx1 + is;
    j += js;
    k += ks;
    Real jn = jn_(m,0,k,j,i), jo = jo_(m,0,k,j,i);
    Real smag = fmax(jn, jo);
    Real r = (smag > 0.0) ? fabs(jn - jo) / smag : 0.0;
    lmax = fmax(lmax, r);
  }, Kokkos::Max<Real>(rmax));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &rmax, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
  return rmax;
}

//----------------------------------------------------------------------------------------
//! \fn SC::SendHflx / RecvHflx / ClearSendHflx / ClearRecvHflx
//! \brief Coarse-fine correction of the face-centred H, mirroring Hydro's flux tasks:
//! the coarse face is replaced by the mean of the fine faces so both sides of a
//! refinement boundary difference the same number. The matching InitFluxRecv(1) is
//! posted in SolveTransfer before the iteration loop. No-ops on a uniform mesh.

TaskStatus SC::SendHflx(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  TaskStatus tstat = TaskStatus::complete;
  if (pbval_hflx != nullptr) {    // exists iff a differential form on a multilevel mesh
    tstat = pbval_hflx->PackAndSendFluxCC(hflx);
  }
  return tstat;
}

TaskStatus SC::RecvHflx(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  TaskStatus tstat = TaskStatus::complete;
  if (pbval_hflx != nullptr) {    // exists iff a differential form on a multilevel mesh
    tstat = pbval_hflx->RecvAndUnpackFluxCC(hflx);
  }
  return tstat;
}

TaskStatus SC::ClearSendHflx(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  TaskStatus tstat = TaskStatus::complete;
  if (pbval_hflx != nullptr) {    // exists iff a differential form on a multilevel mesh
    tstat = pbval_hflx->ClearFluxSend();
  }
  return tstat;
}

TaskStatus SC::ClearRecvHflx(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  TaskStatus tstat = TaskStatus::complete;
  if (pbval_hflx != nullptr) {    // exists iff a differential form on a multilevel mesh
    tstat = pbval_hflx->ClearFluxRecv();
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! Boundary-exchange wrapper tasks -- intensity (ir)

TaskStatus SC::InitRecvIr(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  return pbval_ir->InitRecv(nang_tot);
}

TaskStatus SC::RestrictIr(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictCC(ir, coarse_ir);
  }
  return TaskStatus::complete;
}

TaskStatus SC::SendIr(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  return pbval_ir->PackAndSendCC(ir, coarse_ir);
}

TaskStatus SC::RecvIr(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  return pbval_ir->RecvAndUnpackCC(ir, coarse_ir);
}

TaskStatus SC::ApplyPhysicalBCsIr(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  ApplyPhysicalBCs();
  return TaskStatus::complete;
}

TaskStatus SC::ProlongateIr(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  if (pmy_pack->pmesh->multilevel) {
    pbval_ir->FillCoarseInBndryCC(ir, coarse_ir);
    pbval_ir->ProlongateCC(ir, coarse_ir);
  }
  return TaskStatus::complete;
}

TaskStatus SC::ClearSendIr(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  return pbval_ir->ClearSend();
}

TaskStatus SC::ClearRecvIr(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  return pbval_ir->ClearRecv();
}

//----------------------------------------------------------------------------------------
//! Boundary-exchange wrapper tasks -- source S (srad)

TaskStatus SC::InitRecvSrad(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  return pbval_srad->InitRecv(1);
}

TaskStatus SC::RestrictSrad(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictCC(srad, coarse_srad);
  }
  return TaskStatus::complete;
}

TaskStatus SC::SendSrad(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  return pbval_srad->PackAndSendCC(srad, coarse_srad);
}

TaskStatus SC::RecvSrad(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  return pbval_srad->RecvAndUnpackCC(srad, coarse_srad);
}

TaskStatus SC::ApplyPhysicalBCsSrad(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  ApplyPhysicalBCsSource();
  return TaskStatus::complete;
}

TaskStatus SC::ProlongateSrad(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  if (pmy_pack->pmesh->multilevel) {
    pbval_srad->FillCoarseInBndryCC(srad, coarse_srad);
    pbval_srad->ProlongateCC(srad, coarse_srad);
  }
  return TaskStatus::complete;
}

TaskStatus SC::ClearSendSrad(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  return pbval_srad->ClearSend();
}

TaskStatus SC::ClearRecvSrad(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  return pbval_srad->ClearRecv();
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus SC::SolveTransfer
//! \brief One quasi-static solve per cycle: opacity/source -> [sweep -> exchange -> J ->
//! residual] until the per-sweep change of J is below iter_tol -> H, K and Q_rad.
//! moments slot 0 carries the previous solve's J as the first sweep's reference, so a
//! quasi-static field exits after one sweep (two on the first solve, where J == 0).

TaskStatus SC::SolveTransfer(Driver *pdrive, int stage) {
  (void)stage;
  UpdateOpacity();
  UpdateEmission();
  UpdateSource();

  // Post the flux-correction receives before the iteration, as hydro does (InitFluxRecv
  // in "before_stagen", SendFlux in "stagen"): the sends are iter_max sweeps away, and
  // every rank posts here before any rank sends, so the MPI_Wait in ClearSendHflx always
  // clears. The argument must be 1, hflx's nvar. Matched by ClearRecvHflx in "sc_flxcor".
  // nullptr unless a differential form runs on a multilevel mesh.
  if (pbval_hflx != nullptr) {
    (void)pbval_hflx->InitFluxRecv(1);
  }

  niter_last = 0;
  resid_last = 0.0;
  converged = false;
  Real max_rel = std::numeric_limits<Real>::max();

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  auto mom_ = moments;
  auto jprev_ = j_prev;

  // Ghost ir is still what the previous solve's post-sweep exchange left, unless this is
  // the first solve, a restart (flag false) or a remesh/load balance (counter moved),
  // each of which needs a full exchange. Otherwise only srad is refreshed, since the
  // fluid moved. ApplyPhysicalBCs runs either way so a time-dependent user BC is stamped
  // at this t.
  if (!ir_ghosts_fresh_ ||
      ir_ghosts_seq_ != pmy_pack->pmesh->GetAMRLoadBalanceUpdateSeq()) {
    pdrive->ExecuteTaskList(pmy_pack->pmesh, "sc_bvals", 0);
  } else {
    ApplyPhysicalBCs();
    pdrive->ExecuteTaskList(pmy_pack->pmesh, "sc_bvals_srad", 0);
  }

  for (int it = 0; it < iter_max; ++it) {
    // carry J forward as the residual's reference; ResidualJ reads only the active range
    par_for("sc_carry_j", DevExeSpace(), 0,nmb1, ks,ke, js,je, is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      jprev_(m,0,k,j,i) = mom_(m,0,k,j,i);
    });
    FormalSolution();
    // bval exchange placed after the sweep so the ghost ir that ComputeHK reads one cell
    // into the ghosts is current, and two blocks sharing a face form the same face flux.
    pdrive->ExecuteTaskList(pmy_pack->pmesh, "sc_bvals", 0);
    ir_ghosts_fresh_ = true;
    ir_ghosts_seq_ = pmy_pack->pmesh->GetAMRLoadBalanceUpdateSeq();
    ComputeJ();
    max_rel = ResidualJ();

    niter_last = it + 1;
    resid_last = max_rel;
    if (niter_last >= iter_min && max_rel <= iter_tol) {
      converged = true;
      break;
    }
  }
  // per-solve report, rank 0 only, at the driver's diagnostic cadence (<time>/ndiag, the
  // test Driver::OutputCycleDiagnostics uses) so it sits under the cycle line it belongs
  // to; the warning is unconditional
  if (global_variable::my_rank == 0) {
    if (pmy_pack->pmesh->ncycle % pdrive->ndiag == 0) {
      std::cout << "Iteration stops at niter: " << niter_last
                << " relative error: " << resid_last << std::endl;
    }
    if (!converged) {   // iter_max >= 1, so the loop ran and exhausted the cap
      std::cout << "### WARNING: SC iteration did not converge in " << iter_max
                << " sweeps (max residual = " << max_rel << ", iter_tol = " << iter_tol
                << ")" << std::endl;
    }
  }

  ComputeHK();

  if (qrad_form != QradForm::integral) {
    BuildFaceFlux(pdrive);
  }
  ComputeQrad();

  // Bound the gas-energy update this Q is about to drive; taken here rather than in
  // NewTimeStep so it constrains the very step that applies this Q. No-op if
  // cfl_qrad = 0.
  LimitDtByQrad();
  return TaskStatus::complete;
}

}  // namespace nr_radiation
