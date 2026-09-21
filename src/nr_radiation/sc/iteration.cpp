//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file iteration.cpp
//! \brief Task assembly, the per-solve opacity/source update, boundary-exchange tasks,
//! and SolveTransfer: the exchange-before-sweep loop iterated until J stops changing.
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
//! \fn void SC::AssembleTasks

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
  // InitRecv both from none → Restrict → Send → Recv → PhysBCs (after both Recvs)
  // → Prolong → Clear*
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
}

//----------------------------------------------------------------------------------------
//! \fn void SC::UpdateOpacity / SC::UpdateEmission
//! \brief The per-cell opacity and LTE emission of the CURRENT fluid state, over ALL
//! cells including ghosts (the sweep reads both at ghost footpoints):
//!   sigma_a:  the enrolled opacity hook, else kappa_a * rho of the fluid;  chi = sigma_a
//!   emission: the enrolled emission hook, else the Planck function T^4 of the fluid
//! Both run at the start of every solve. The opacity alone is also refreshed wherever the
//! relaxation timestep is evaluated (driver initialization, the end of each step, after a
//! remesh), so that dt always sees the opacity of the state it is timing -- which is what
//! makes a restart continue a run bitwise. Without a fluid both hooks are required
//! (checked after the pgen runs), so neither array is ever state: nothing about them is
//! packed by AMR or checkpointed. The gas emits whenever a fluid is present, independent
//! of affect_fluid (which gates only the reverse coupling).

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
//! \brief The source the sweep reads: S = brad in LTE. (With scattering S becomes the
//! iterate of the solve and this is where it is initialised; AMR packs srad so that an
//! accelerated iteration resumes from its exact iterate after a remesh.)

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
//! \brief Mirror the intensity into the ghost zones of reflecting physical boundaries.
//!
//! Two mirrors, both exact. A symmetry plane sits BETWEEN the last ghost and the first
//! active cell, so ghost is-i-1 takes its value from is+i, not from is. And the ray
//! mirrors: the ghost value of ray angg is the interior value of the ray with that
//! direction cosine negated (MirrorRay). Because the Carlson set is closed under these
//! reflections and equal-weight, a half-domain run with a reflecting face reproduces the
//! corresponding half of a full-domain run bit for bit.
//!
//! Only ghosts are written and only interior cells are read, so the two index ranges are
//! disjoint and the gather is race-free.
//!
//! This is not in bvals/RadiationBCs with the other flags because the ray mirror is a
//! property of THIS angular grid: an exact permutation for an octant-structured
//! quadrature, but needing angular interpolation for the geodesic mesh the GR radiation
//! module uses, which is a different and lossier operation.

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
//! \brief Nearest-active copy of S (srad) into physical-boundary ghost layers.

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
//! between J and j_prev, allreduced across ranks. Departs from the Athena-C LTE test
//! (|dJ|/J_old with dJ/0 := 0 plus an all-zero guard, jacobi_3d_linear.c): that form
//! cannot see a cell lit from exactly zero, so a front entering vacuum could stop the
//! iteration early whenever any other cell moved by a tiny nonzero amount. Here 0 -> J
//! reports 1; the only zero-change case is Jn == Jo == 0.

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
//! Boundary-exchange wrapper tasks — intensity (ir)

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
//! Boundary-exchange wrapper tasks — source S (srad)

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
//! \brief One quasi-static solve per cycle: opacity/source -> [exchange -> sweep -> J ->
//! residual] until the per-sweep change of J is below iter_tol -> H, K and Q_rad.
//! moments slot 0 (J) is NOT zeroed between solves: it carries the previous step's
//! converged J (or, after a remesh, J(ir) rebuilt by MeshRefinement) as the reference for
//! the first sweep, so a
//! quasi-static field exits after ONE sweep. At construction J == 0, so the first solve
//! (and a restart without a checkpointed field) takes two.

TaskStatus SC::SolveTransfer(Driver *pdrive, int stage) {
  (void)stage;
  UpdateOpacity();
  UpdateEmission();
  UpdateSource();

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

  for (int it = 0; it < iter_max; ++it) {
    // carry J forward as the residual's reference. A par_for rather than a deep_copy: the
    // source is slot 0 of the moments array, and ResidualJ reads only the active range,
    // so there is no point moving the ghosts the old whole-array copy did.
    par_for("sc_carry_j", DevExeSpace(), 0,nmb1, ks,ke, js,je, is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      jprev_(m,0,k,j,i) = mom_(m,0,k,j,i);
    });
    // Davis sec. 3.5 / Athena-C order: refresh ghosts before the formal solution
    pdrive->ExecuteTaskList(pmy_pack->pmesh, "sc_bvals", 0);
    FormalSolution();
    ComputeJ();
    max_rel = ResidualJ();

    niter_last = it + 1;
    resid_last = max_rel;
    if (niter_last >= iter_min && max_rel <= iter_tol) {
      converged = true;
      break;
    }
  }
  // per-solve report in Athena++'s static-RT format (static_iteration.cpp), rank 0 only,
  // at the driver's diagnostic cadence (<time>/ndiag, the same test that
  // Driver::OutputCycleDiagnostics uses) so it sits under the cycle line it belongs to;
  // the warning is unconditional
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
  ComputeQrad();
  return TaskStatus::complete;
}

}  // namespace nr_radiation
