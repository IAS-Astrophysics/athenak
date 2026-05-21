//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mg_task_list.cpp
//! \brief functions for MultigridTaskList class

// C headers

// C++ headers
#include <iostream>   // endl
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <iomanip>    // setprecision

// Athena++ headers
#include "../athena.hpp"
#include "../globals.hpp"
#include "../mesh/mesh.hpp"
#include "multigrid.hpp"

//namespace multigrid{ // NOLINT (build/namespace)

//----------------------------------------------------------------------------------------
//! \fn void MultigridTaskList::DoTaskListOneStage(MultigridDriver *pmd)
//! \brief completes all tasks in this list, will not return until all are tasks done
void MultigridDriver::DoTaskListOneStage() {
  return;
}

TaskStatus MultigridDriver::ClearRecv(Driver *pdrive, int stage) {
  TaskStatus tstat;
  tstat = pmg->pbval->ClearRecv();
  return tstat;
}

TaskStatus MultigridDriver::ClearSend(Driver *pdrive, int stage) {
  TaskStatus tstat;
  tstat = pmg->pbval->ClearSend();
  return tstat;
}

TaskStatus MultigridDriver::SendBoundary(Driver *pdrive, int stage) {
  TaskStatus tstat;
  DvceArray5D<Real> u = pmg->GetCurrentData();
  tstat = pmg->pbval->PackAndSendMG(u);
  return tstat;
}

TaskStatus MultigridDriver::RecvBoundary(Driver *pdrive, int stage) {
  TaskStatus tstat;
  DvceArray5D<Real> u = pmg->GetCurrentData();
  tstat = pmg->pbval->RecvAndUnpackMG(u);
  return tstat;
}

TaskStatus MultigridDriver::StartReceive(Driver *pdrive, int stage) {
  TaskStatus tstat;
  tstat = pmg->pbval->InitRecvMG(pmg->nvar_);
  return tstat;
}

TaskStatus MultigridDriver::SmoothRed(Driver *pdrive, int stage) {
  pmg->SmoothPack(coffset_);
  return TaskStatus::complete;
}

TaskStatus MultigridDriver::SmoothBlack(Driver *pdrive, int stage) {
  pmg->SmoothPack(1-coffset_);
  return TaskStatus::complete;
}

TaskStatus MultigridDriver::Restrict(Driver *pdrive, int stage) {
  pmg->RestrictPack();
  return TaskStatus::complete;
}

TaskStatus MultigridDriver::Prolongate(Driver *pdrive, int stage) {
  pmg->ProlongateAndCorrectPack();
  return TaskStatus::complete;
}

TaskStatus MultigridDriver::FMGProlongateTask(Driver *pdrive, int stage) {
  pmg->FMGProlongatePack();
  return TaskStatus::complete;
}

TaskStatus MultigridDriver::CalculateFASRHS(Driver *pdrive, int stage) {
  if (current_level_ < fmglevel_) {
    pmg->StoreOldData();
    pmg->CalculateFASRHSPack();
  }
  return TaskStatus::complete;
}

TaskStatus MultigridDriver::ProlongateBoundary(Driver *pdrive, int stage) {
  //pmg->pmgbval->ProlongateMultigridBoundariesFluxCons();
  return TaskStatus::complete;
}


TaskStatus MultigridDriver::ProlongateBoundaryForProlongation(Driver *pdrive, int stage) {
  //pmg->pmgbval->ProlongateMultigridBoundaries(true, false);
  return TaskStatus::complete;
}

TaskStatus MultigridDriver::FillFCBoundary(Driver *pdrive, int stage) {
  if (nreflevel_ == 0) return TaskStatus::complete;
  DvceArray5D<Real> u = pmg->GetCurrentData();
  pmg->pbval->FillFineCoarseMGGhosts(u);
  return TaskStatus::complete;
}

TaskStatus MultigridDriver::PhysicalBoundary(Driver *pdrive, int stage) {
  if (pmy_pack_->pmesh->strictly_periodic) return TaskStatus::complete;

  DvceArray5D<Real> u = pmg->GetCurrentData();
  int nvar = u.extent_int(1);
  int shift = pmg->GetLevelShift();
  int ngh = pmg->GetGhostCells();
  int ncells = pmg->GetSize() >> shift;
  if (ncells < 1) return TaskStatus::complete;

  int nmb = pmy_pack_->nmb_thispack;
  auto &mb_bcs = pmy_pack_->pmb->mb_bcs;

  BoundaryFlag bc_ix1 = mg_mesh_bcs_[BoundaryFace::inner_x1];
  BoundaryFlag bc_ox1 = mg_mesh_bcs_[BoundaryFace::outer_x1];
  BoundaryFlag bc_ix2 = mg_mesh_bcs_[BoundaryFace::inner_x2];
  BoundaryFlag bc_ox2 = mg_mesh_bcs_[BoundaryFace::outer_x2];
  BoundaryFlag bc_ix3 = mg_mesh_bcs_[BoundaryFace::inner_x3];
  BoundaryFlag bc_ox3 = mg_mesh_bcs_[BoundaryFace::outer_x3];

  bool has_multipole = (bc_ix1 == BoundaryFlag::mg_multipole ||
                        bc_ox1 == BoundaryFlag::mg_multipole ||
                        bc_ix2 == BoundaryFlag::mg_multipole ||
                        bc_ox2 == BoundaryFlag::mg_multipole ||
                        bc_ix3 == BoundaryFlag::mg_multipole ||
                        bc_ox3 == BoundaryFlag::mg_multipole);

  // Copy multipole coefficients to device if needed
  DvceArray1D<Real> d_mpc;
  Real d_xo = 0.0, d_yo = 0.0, d_zo = 0.0;
  int d_order = 0;
  auto &mb_size = pmy_pack_->pmb->mb_size;
  if (has_multipole && mporder_ > 0) {
    Kokkos::realloc(d_mpc, 25);
    auto h_mpc = Kokkos::create_mirror_view(d_mpc);
    for (int c = 0; c < 25; ++c) h_mpc(c) = mpcoeff_[c];
    Kokkos::deep_copy(d_mpc, h_mpc);
    d_xo = mpo_[0]; d_yo = mpo_[1]; d_zo = mpo_[2];
    d_order = mporder_;
  }

  int nx1 = pmg->indcs_.nx1 >> shift;
  int nx2 = pmg->indcs_.nx2 >> shift;
  int nx3 = pmg->indcs_.nx3 >> shift;

  Kokkos::parallel_for("MGPhysicalBoundary",
    Kokkos::RangePolicy<DevExeSpace>(0, nmb),
    KOKKOS_LAMBDA(const int m) {
      for (int v = 0; v < nvar; ++v) {
        // inner x1
        if (mb_bcs.d_view(m, BoundaryFace::inner_x1) != BoundaryFlag::block &&
            mb_bcs.d_view(m, BoundaryFace::inner_x1) != BoundaryFlag::periodic) {
          if (bc_ix1 == BoundaryFlag::mg_multipole && d_mpc.data() != nullptr) {
            Real dx1 = (mb_size.d_view(m).x1max - mb_size.d_view(m).x1min)
                       / static_cast<Real>(nx1);
            Real dx2_l = (mb_size.d_view(m).x2max - mb_size.d_view(m).x2min)
                       / static_cast<Real>(nx2);
            Real dx3_l = (mb_size.d_view(m).x3max - mb_size.d_view(m).x3min)
                       / static_cast<Real>(nx3);
            Real xf = mb_size.d_view(m).x1min - d_xo;
            for (int k = ngh; k < ngh + ncells; ++k) {
              Real zv = mb_size.d_view(m).x3min + (k-ngh+0.5)*dx3_l - d_zo;
              for (int j = ngh; j < ngh + ncells; ++j) {
                Real yv = mb_size.d_view(m).x2min + (j-ngh+0.5)*dx2_l - d_yo;
                Real phis = EvalMultipolePhi(xf, yv, zv, d_mpc.data(), d_order);
                for (int n = 0; n < ngh; ++n)
                  u(m,v,k,j,ngh-1-n) = 2.0*phis - u(m,v,k,j,ngh+n);
              }
            }
          } else {
            for (int k = 0; k < ncells + 2*ngh; ++k) {
              for (int j = 0; j < ncells + 2*ngh; ++j) {
                for (int n = 0; n < ngh; ++n) {
                  Real sign = (bc_ix1 == BoundaryFlag::mg_zerofixed) ? -1.0 : 1.0;
                  u(m, v, k, j, ngh - 1 - n) = sign * u(m, v, k, j, ngh + n);
                }
              }
            }
          }
        }
        // outer x1
        if (mb_bcs.d_view(m, BoundaryFace::outer_x1) != BoundaryFlag::block &&
            mb_bcs.d_view(m, BoundaryFace::outer_x1) != BoundaryFlag::periodic) {
          if (bc_ox1 == BoundaryFlag::mg_multipole && d_mpc.data() != nullptr) {
            Real dx2_l = (mb_size.d_view(m).x2max - mb_size.d_view(m).x2min)
                       / static_cast<Real>(nx2);
            Real dx3_l = (mb_size.d_view(m).x3max - mb_size.d_view(m).x3min)
                       / static_cast<Real>(nx3);
            Real xf = mb_size.d_view(m).x1max - d_xo;
            for (int k = ngh; k < ngh + ncells; ++k) {
              Real zv = mb_size.d_view(m).x3min + (k-ngh+0.5)*dx3_l - d_zo;
              for (int j = ngh; j < ngh + ncells; ++j) {
                Real yv = mb_size.d_view(m).x2min + (j-ngh+0.5)*dx2_l - d_yo;
                Real phis = EvalMultipolePhi(xf, yv, zv, d_mpc.data(), d_order);
                for (int n = 0; n < ngh; ++n)
                  u(m,v,k,j,ngh+ncells+n) = 2.0*phis - u(m,v,k,j,ngh+ncells-1-n);
              }
            }
          } else {
            for (int k = 0; k < ncells + 2*ngh; ++k) {
              for (int j = 0; j < ncells + 2*ngh; ++j) {
                for (int n = 0; n < ngh; ++n) {
                  Real sign = (bc_ox1 == BoundaryFlag::mg_zerofixed) ? -1.0 : 1.0;
                  u(m, v, k, j, ngh + ncells + n) = sign *
                      u(m, v, k, j, ngh + ncells - 1 - n);
                }
              }
            }
          }
        }
        // inner x2
        if (mb_bcs.d_view(m, BoundaryFace::inner_x2) != BoundaryFlag::block &&
            mb_bcs.d_view(m, BoundaryFace::inner_x2) != BoundaryFlag::periodic) {
          if (bc_ix2 == BoundaryFlag::mg_multipole && d_mpc.data() != nullptr) {
            Real dx1_l = (mb_size.d_view(m).x1max - mb_size.d_view(m).x1min)
                       / static_cast<Real>(nx1);
            Real dx3_l = (mb_size.d_view(m).x3max - mb_size.d_view(m).x3min)
                       / static_cast<Real>(nx3);
            Real yf = mb_size.d_view(m).x2min - d_yo;
            for (int k = ngh; k < ngh + ncells; ++k) {
              Real zv = mb_size.d_view(m).x3min + (k-ngh+0.5)*dx3_l - d_zo;
              for (int i = ngh; i < ngh + ncells; ++i) {
                Real xv = mb_size.d_view(m).x1min + (i-ngh+0.5)*dx1_l - d_xo;
                Real phis = EvalMultipolePhi(xv, yf, zv, d_mpc.data(), d_order);
                for (int n = 0; n < ngh; ++n)
                  u(m,v,k,ngh-1-n,i) = 2.0*phis - u(m,v,k,ngh+n,i);
              }
            }
          } else {
            for (int k = 0; k < ncells + 2*ngh; ++k) {
              for (int i = 0; i < ncells + 2*ngh; ++i) {
                for (int n = 0; n < ngh; ++n) {
                  Real sign = (bc_ix2 == BoundaryFlag::mg_zerofixed) ? -1.0 : 1.0;
                  u(m, v, k, ngh - 1 - n, i) = sign * u(m, v, k, ngh + n, i);
                }
              }
            }
          }
        }
        // outer x2
        if (mb_bcs.d_view(m, BoundaryFace::outer_x2) != BoundaryFlag::block &&
            mb_bcs.d_view(m, BoundaryFace::outer_x2) != BoundaryFlag::periodic) {
          if (bc_ox2 == BoundaryFlag::mg_multipole && d_mpc.data() != nullptr) {
            Real dx1_l = (mb_size.d_view(m).x1max - mb_size.d_view(m).x1min)
                       / static_cast<Real>(nx1);
            Real dx3_l = (mb_size.d_view(m).x3max - mb_size.d_view(m).x3min)
                       / static_cast<Real>(nx3);
            Real yf = mb_size.d_view(m).x2max - d_yo;
            for (int k = ngh; k < ngh + ncells; ++k) {
              Real zv = mb_size.d_view(m).x3min + (k-ngh+0.5)*dx3_l - d_zo;
              for (int i = ngh; i < ngh + ncells; ++i) {
                Real xv = mb_size.d_view(m).x1min + (i-ngh+0.5)*dx1_l - d_xo;
                Real phis = EvalMultipolePhi(xv, yf, zv, d_mpc.data(), d_order);
                for (int n = 0; n < ngh; ++n)
                  u(m,v,k,ngh+ncells+n,i) = 2.0*phis - u(m,v,k,ngh+ncells-1-n,i);
              }
            }
          } else {
            for (int k = 0; k < ncells + 2*ngh; ++k) {
              for (int i = 0; i < ncells + 2*ngh; ++i) {
                for (int n = 0; n < ngh; ++n) {
                  Real sign = (bc_ox2 == BoundaryFlag::mg_zerofixed) ? -1.0 : 1.0;
                  u(m, v, k, ngh + ncells + n, i) = sign *
                      u(m, v, k, ngh + ncells - 1 - n, i);
                }
              }
            }
          }
        }
        // inner x3
        if (mb_bcs.d_view(m, BoundaryFace::inner_x3) != BoundaryFlag::block &&
            mb_bcs.d_view(m, BoundaryFace::inner_x3) != BoundaryFlag::periodic) {
          if (bc_ix3 == BoundaryFlag::mg_multipole && d_mpc.data() != nullptr) {
            Real dx1_l = (mb_size.d_view(m).x1max - mb_size.d_view(m).x1min)
                       / static_cast<Real>(nx1);
            Real dx2_l = (mb_size.d_view(m).x2max - mb_size.d_view(m).x2min)
                       / static_cast<Real>(nx2);
            Real zf = mb_size.d_view(m).x3min - d_zo;
            for (int j = ngh; j < ngh + ncells; ++j) {
              Real yv = mb_size.d_view(m).x2min + (j-ngh+0.5)*dx2_l - d_yo;
              for (int i = ngh; i < ngh + ncells; ++i) {
                Real xv = mb_size.d_view(m).x1min + (i-ngh+0.5)*dx1_l - d_xo;
                Real phis = EvalMultipolePhi(xv, yv, zf, d_mpc.data(), d_order);
                for (int n = 0; n < ngh; ++n)
                  u(m,v,ngh-1-n,j,i) = 2.0*phis - u(m,v,ngh+n,j,i);
              }
            }
          } else {
            for (int j = 0; j < ncells + 2*ngh; ++j) {
              for (int i = 0; i < ncells + 2*ngh; ++i) {
                for (int n = 0; n < ngh; ++n) {
                  Real sign = (bc_ix3 == BoundaryFlag::mg_zerofixed) ? -1.0 : 1.0;
                  u(m, v, ngh - 1 - n, j, i) = sign * u(m, v, ngh + n, j, i);
                }
              }
            }
          }
        }
        // outer x3
        if (mb_bcs.d_view(m, BoundaryFace::outer_x3) != BoundaryFlag::block &&
            mb_bcs.d_view(m, BoundaryFace::outer_x3) != BoundaryFlag::periodic) {
          if (bc_ox3 == BoundaryFlag::mg_multipole && d_mpc.data() != nullptr) {
            Real dx1_l = (mb_size.d_view(m).x1max - mb_size.d_view(m).x1min)
                       / static_cast<Real>(nx1);
            Real dx2_l = (mb_size.d_view(m).x2max - mb_size.d_view(m).x2min)
                       / static_cast<Real>(nx2);
            Real zf = mb_size.d_view(m).x3max - d_zo;
            for (int j = ngh; j < ngh + ncells; ++j) {
              Real yv = mb_size.d_view(m).x2min + (j-ngh+0.5)*dx2_l - d_yo;
              for (int i = ngh; i < ngh + ncells; ++i) {
                Real xv = mb_size.d_view(m).x1min + (i-ngh+0.5)*dx1_l - d_xo;
                Real phis = EvalMultipolePhi(xv, yv, zf, d_mpc.data(), d_order);
                for (int n = 0; n < ngh; ++n)
                  u(m,v,ngh+ncells+n,j,i) = 2.0*phis - u(m,v,ngh+ncells-1-n,j,i);
              }
            }
          } else {
            for (int j = 0; j < ncells + 2*ngh; ++j) {
              for (int i = 0; i < ncells + 2*ngh; ++i) {
                for (int n = 0; n < ngh; ++n) {
                  Real sign = (bc_ox3 == BoundaryFlag::mg_zerofixed) ? -1.0 : 1.0;
                  u(m, v, ngh + ncells + n, j, i) = sign *
                      u(m, v, ngh + ncells - 1 - n, j, i);
                }
              }
            }
          }
        }
      }
    });

  return TaskStatus::complete;
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridTaskList::SetMGTaskListToFiner(int nsmooth, int ngh, int flag)
//! \brief Set the task list for prolongation and post smoothing

void MultigridDriver::SetMGTaskListToFiner(int nsmooth, int ngh, int flag) {
  auto &tl = pmy_pack_->tl_map;
  tl.erase("mg_to_finer");
  tl.emplace(std::make_pair("mg_to_finer", std::make_shared<TaskList>()));
  TaskID none(0);

  if (flag == 1) {
    // First time on meshblock levels: no boundary comm before prolongation
    id.prolongate = tl["mg_to_finer"]->AddTask(&MultigridDriver::Prolongate, this, none);
  } else {
    // Coarse-level boundary comm before prolongation
    id.ircv0    = tl["mg_to_finer"]->AddTask(&MultigridDriver::StartReceive, this, none);
    id.send0    = tl["mg_to_finer"]->AddTask(&MultigridDriver::SendBoundary, this, none);
    id.recv0    = tl["mg_to_finer"]->AddTask(&MultigridDriver::RecvBoundary, this, id.send0);
    id.physb0   = tl["mg_to_finer"]->AddTask(&MultigridDriver::PhysicalBoundary, this, id.recv0);
    id.fc_ghosts0 = tl["mg_to_finer"]->AddTask(&MultigridDriver::FillFCBoundary, this, id.physb0);
    id.prolongate = tl["mg_to_finer"]->AddTask(&MultigridDriver::Prolongate, this, id.fc_ghosts0);
  }

  TaskID last_smooth = id.prolongate;
  if (nsmooth > 0) {
    // Fine-level boundary comm after prolongation
    id.ircv1    = tl["mg_to_finer"]->AddTask(&MultigridDriver::StartReceive, this, id.prolongate);
    id.send1    = tl["mg_to_finer"]->AddTask(&MultigridDriver::SendBoundary, this, id.prolongate);
    id.recv1    = tl["mg_to_finer"]->AddTask(&MultigridDriver::RecvBoundary, this, id.send1);
    id.physb1   = tl["mg_to_finer"]->AddTask(&MultigridDriver::PhysicalBoundary, this, id.recv1);
    id.fc_ghosts_prol = tl["mg_to_finer"]->AddTask(&MultigridDriver::FillFCBoundary, this, id.physb1);

    // Post-smoothing (red-black)
    id.smoothR   = tl["mg_to_finer"]->AddTask(&MultigridDriver::SmoothRed, this, id.fc_ghosts_prol);
    id.ircvR    = tl["mg_to_finer"]->AddTask(&MultigridDriver::StartReceive, this, id.fc_ghosts_prol);
    id.sendR    = tl["mg_to_finer"]->AddTask(&MultigridDriver::SendBoundary, this, id.smoothR);
    id.recvR    = tl["mg_to_finer"]->AddTask(&MultigridDriver::RecvBoundary, this, id.sendR);
    id.physbR   = tl["mg_to_finer"]->AddTask(&MultigridDriver::PhysicalBoundary, this, id.recvR);
    id.fc_ghostsR = tl["mg_to_finer"]->AddTask(&MultigridDriver::FillFCBoundary, this, id.physbR);

    id.smoothB   = tl["mg_to_finer"]->AddTask(&MultigridDriver::SmoothBlack, this, id.fc_ghostsR);
    last_smooth = id.smoothB;

    if (nsmooth > 1) {
      id.ircvB    = tl["mg_to_finer"]->AddTask(&MultigridDriver::StartReceive, this, id.fc_ghostsR);
      id.sendB    = tl["mg_to_finer"]->AddTask(&MultigridDriver::SendBoundary, this, id.smoothB);
      id.recvB    = tl["mg_to_finer"]->AddTask(&MultigridDriver::RecvBoundary, this, id.ircvB);
      id.physbB   = tl["mg_to_finer"]->AddTask(&MultigridDriver::PhysicalBoundary, this, id.recvB);
      id.fc_ghostsB = tl["mg_to_finer"]->AddTask(&MultigridDriver::FillFCBoundary, this, id.physbB);

      id.smoothR2   = tl["mg_to_finer"]->AddTask(&MultigridDriver::SmoothRed, this, id.fc_ghostsB);

      id.ircvR2    = tl["mg_to_finer"]->AddTask(&MultigridDriver::StartReceive, this, id.fc_ghostsB);
      id.sendR2    = tl["mg_to_finer"]->AddTask(&MultigridDriver::SendBoundary, this, id.smoothR2);
      id.recvR2    = tl["mg_to_finer"]->AddTask(&MultigridDriver::RecvBoundary, this, id.ircvR2);
      id.physbR2   = tl["mg_to_finer"]->AddTask(&MultigridDriver::PhysicalBoundary, this, id.recvR2);
      id.fc_ghostsR2 = tl["mg_to_finer"]->AddTask(&MultigridDriver::FillFCBoundary, this, id.physbR2);

      id.smoothB2   = tl["mg_to_finer"]->AddTask(&MultigridDriver::SmoothBlack, this, id.fc_ghostsR2);
      last_smooth = id.smoothB2;
    }
  }

  // flag==2: last step to finest level -- add final boundary exchange so ghost cells
  // are up-to-date for CalculateDefectNorm (matching Athena++ MG_STARTRECVL etc.)
  if (flag == 2) {
    id.clear_sendB2 = tl["mg_to_finer"]->AddTask(&MultigridDriver::ClearSend, this, none);
    id.clear_recvB2 = tl["mg_to_finer"]->AddTask(&MultigridDriver::ClearRecv, this, id.clear_sendB2);
    TaskID last_clear = id.clear_recvB2;

    TaskID ircvL  = tl["mg_to_finer"]->AddTask(&MultigridDriver::StartReceive, this, last_clear);
    TaskID send_dep = last_smooth | last_clear;
    TaskID sendL  = tl["mg_to_finer"]->AddTask(&MultigridDriver::SendBoundary, this, send_dep);
    TaskID recvL  = tl["mg_to_finer"]->AddTask(&MultigridDriver::RecvBoundary, this, sendL);
    TaskID physL  = tl["mg_to_finer"]->AddTask(&MultigridDriver::PhysicalBoundary, this, recvL);
    TaskID fcL    = tl["mg_to_finer"]->AddTask(&MultigridDriver::FillFCBoundary, this, physL);
    id.clear_send0 = tl["mg_to_finer"]->AddTask(&MultigridDriver::ClearSend, this, fcL);
    id.clear_recv0 = tl["mg_to_finer"]->AddTask(&MultigridDriver::ClearRecv, this, id.clear_send0);
  } else {
    id.clear_send0 = tl["mg_to_finer"]->AddTask(&MultigridDriver::ClearSend, this, none);
    id.clear_recv0 = tl["mg_to_finer"]->AddTask(&MultigridDriver::ClearRecv, this, id.clear_send0);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SetMGTaskListFMGProlongate(int ngh)
//! \brief Set the task list for FMG prolongation only (no smoothing)

void MultigridDriver::SetMGTaskListFMGProlongate(int ngh) {
  auto &tl = pmy_pack_->tl_map;
  tl.erase("mg_fmg_prolongate");
  tl.emplace(std::make_pair("mg_fmg_prolongate", std::make_shared<TaskList>()));
  TaskID none(0);

  // Boundary comm before prolongation
  id.ircv0    = tl["mg_fmg_prolongate"]->AddTask(
                  &MultigridDriver::StartReceive, this, none);
  id.send0    = tl["mg_fmg_prolongate"]->AddTask(
                  &MultigridDriver::SendBoundary, this, none);
  id.recv0    = tl["mg_fmg_prolongate"]->AddTask(
                  &MultigridDriver::RecvBoundary, this, id.send0);
  id.physb0   = tl["mg_fmg_prolongate"]->AddTask(
                  &MultigridDriver::PhysicalBoundary, this, id.recv0);
  id.fc_ghosts0 = tl["mg_fmg_prolongate"]->AddTask(
                  &MultigridDriver::FillFCBoundary, this, id.physb0);

  // FMG prolongation (direct overwrite)
  id.fmg_prolongate = tl["mg_fmg_prolongate"]->AddTask(
                  &MultigridDriver::FMGProlongateTask, this, id.fc_ghosts0);

  id.clear_send0 = tl["mg_fmg_prolongate"]->AddTask(
                  &MultigridDriver::ClearSend, this, none);
  id.clear_recv0 = tl["mg_fmg_prolongate"]->AddTask(
                  &MultigridDriver::ClearRecv, this, id.clear_send0);
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridTaskList::SetMGTaskListToCoarser(int nsmooth, int ngh)
//! \brief Set the task list for pre smoothing and restriction

void MultigridDriver::SetMGTaskListToCoarser(int nsmooth, int cycle) {
  auto &tl = pmy_pack_->tl_map;
  tl.erase("mg_to_coarser");
  tl.emplace(std::make_pair("mg_to_coarser",std::make_shared<TaskList>()));
  TaskID none(0);

  id.ircv0    = tl["mg_to_coarser"]->AddTask(&MultigridDriver::StartReceive, this, none);
  id.send0      = tl["mg_to_coarser"]->AddTask(&MultigridDriver::SendBoundary, this, none);
  id.recv0      = tl["mg_to_coarser"]->AddTask(&MultigridDriver::RecvBoundary, this, id.send0);
  id.physb0     = tl["mg_to_coarser"]->AddTask(&MultigridDriver::PhysicalBoundary, this, id.recv0);
  id.fc_ghosts0 = tl["mg_to_coarser"]->AddTask(&MultigridDriver::FillFCBoundary, this, id.physb0);

  id.calc_rhs   = tl["mg_to_coarser"]->AddTask(&MultigridDriver::CalculateFASRHS, this, id.fc_ghosts0);

  if (nsmooth > 0) {
    id.ircvR      = tl["mg_to_coarser"]->AddTask(&MultigridDriver::StartReceive, this, id.fc_ghosts0);
    id.smoothR    = tl["mg_to_coarser"]->AddTask(&MultigridDriver::SmoothRed, this, id.calc_rhs);
    id.sendR      = tl["mg_to_coarser"]->AddTask(&MultigridDriver::SendBoundary, this, id.smoothR);
    id.recvR      = tl["mg_to_coarser"]->AddTask(&MultigridDriver::RecvBoundary, this, id.sendR);
    id.physbR     = tl["mg_to_coarser"]->AddTask(&MultigridDriver::PhysicalBoundary, this, id.recvR);
    id.fc_ghostsR = tl["mg_to_coarser"]->AddTask(&MultigridDriver::FillFCBoundary, this, id.physbR);

    id.ircvB      = tl["mg_to_coarser"]->AddTask(&MultigridDriver::StartReceive, this, id.fc_ghostsR);
    id.smoothB    = tl["mg_to_coarser"]->AddTask(&MultigridDriver::SmoothBlack, this, id.fc_ghostsR);
    id.sendB      = tl["mg_to_coarser"]->AddTask(&MultigridDriver::SendBoundary, this, id.smoothB);
    id.recvB      = tl["mg_to_coarser"]->AddTask(&MultigridDriver::RecvBoundary, this, id.sendB);
    id.physbB     = tl["mg_to_coarser"]->AddTask(&MultigridDriver::PhysicalBoundary, this, id.recvB);
    id.fc_ghostsB = tl["mg_to_coarser"]->AddTask(&MultigridDriver::FillFCBoundary, this, id.physbB);
    if (nsmooth > 1) {
      id.smoothR2   = tl["mg_to_coarser"]->AddTask(&MultigridDriver::SmoothRed, this, id.fc_ghostsB);

      id.ircvR2    = tl["mg_to_coarser"]->AddTask(&MultigridDriver::StartReceive, this, id.smoothR2);
      id.sendR2      = tl["mg_to_coarser"]->AddTask(&MultigridDriver::SendBoundary, this, id.smoothR2);
      id.recvR2     = tl["mg_to_coarser"]->AddTask(&MultigridDriver::RecvBoundary, this, id.sendR2);
      id.physbR2     = tl["mg_to_coarser"]->AddTask(&MultigridDriver::PhysicalBoundary, this, id.recvR2);
      id.fc_ghostsR2 = tl["mg_to_coarser"]->AddTask(&MultigridDriver::FillFCBoundary, this, id.physbR2);

      id.smoothB2   = tl["mg_to_coarser"]->AddTask(&MultigridDriver::SmoothBlack, this, id.fc_ghostsR2);

      id.sendB2      = tl["mg_to_coarser"]->AddTask(&MultigridDriver::SendBoundary, this, id.smoothB2);
      id.recvB2      = tl["mg_to_coarser"]->AddTask(&MultigridDriver::RecvBoundary, this, id.ircvB2);
      id.physbB2     = tl["mg_to_coarser"]->AddTask(&MultigridDriver::PhysicalBoundary, this, id.recvB2);
      id.fc_ghostsB2 = tl["mg_to_coarser"]->AddTask(&MultigridDriver::FillFCBoundary, this, id.physbB2);

      id.restrict_  = tl["mg_to_coarser"]->AddTask(&MultigridDriver::Restrict, this, id.fc_ghostsB2);
    } else {
      id.restrict_  = tl["mg_to_coarser"]->AddTask(&MultigridDriver::Restrict, this, id.fc_ghostsB);
    }
  } else {
    id.restrict_  = tl["mg_to_coarser"]->AddTask(&MultigridDriver::Restrict, this, id.calc_rhs);
  }
  id.clear_send0 = tl["mg_to_coarser"]->AddTask(&MultigridDriver::ClearSend, this, none);
  id.clear_recv0 = tl["mg_to_coarser"]->AddTask(&MultigridDriver::ClearRecv, this, id.clear_send0);

}