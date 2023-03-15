//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyngr.cpp
//  \brief implementation of functions for DynGR and DynGRPS controlling the task list

#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "bvals/bvals.hpp"
#include "hydro/hydro.hpp"
#include "z4c/z4c.hpp"
#include "adm/adm.hpp"
#include "dyngr.hpp"

#include "eos/primitive_solver_hyd.hpp"
#include "eos/primitive-solver/idealgas.hpp"
#include "eos/primitive-solver/reset_floor.hpp"

namespace dyngr {

// A dumb template function containing the switch statement needed to select an EOS.
template<class ErrorPolicy>
DynGR* SelectDynGREOS(MeshBlockPack *ppack, ParameterInput *pin, DynGR_EOS eos_policy) {
  DynGR* dyn_gr = nullptr;
  switch(eos_policy) {
    case DynGR_EOS::eos_ideal:
      dyn_gr = new DynGRPS<Primitive::IdealGas, ErrorPolicy>(ppack, pin);
      break;
  }
  return dyn_gr;
}

DynGR* BuildDynGR(MeshBlockPack *ppack, ParameterInput *pin) {
  std::string eos_string = pin->GetString("hydro", "dyn_eos");
  std::string error_string = pin->GetString("hydro", "dyn_error");
  DynGR_EOS eos_policy;
  DynGR_Error error_policy;

  if (eos_string.compare("ideal") == 0) {
    eos_policy = DynGR_EOS::eos_ideal;
  } else {
    std::cout << "### FATAL ERROR in " <<__FILE__ << " at line " << __LINE__
              << std::endl << "<hydro> dyn_eos = '" << eos_string
              << "' not implemented for GR dynamics" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (error_string.compare("reset_floor") == 0) {
    error_policy = DynGR_Error::reset_floor;
  } else {
    std::cout << "### FATAL ERROR in " <<__FILE__ << " at line " << __LINE__
              << std::endl << "<hydro> dyn_error = '" << error_string
              << "' not implemented for GR dynamics" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  DynGR* dyn_gr = nullptr;

  switch (error_policy) {
    case DynGR_Error::reset_floor:
      dyn_gr = SelectDynGREOS<Primitive::ResetFloor>(ppack, pin, eos_policy);
      break;
  }

  dyn_gr->eos_policy = eos_policy;
  dyn_gr->error_policy = error_policy;

  return dyn_gr;
}

DynGR::DynGR(MeshBlockPack *pp, ParameterInput *pin) : pmy_pack(pp) {
  std::string rsolver = pin->GetString("hydro", "rsolver");
  if (rsolver.compare("llf") == 0) {
    rsolver_method = DynGR_RSolver::llf_dyngr;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "<hydro> rsolver = '" << rsolver
              << "' not implemented for GR dynamics" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  scratch_level = pin->GetOrAddInteger("hydro", "dyn_scratch", 0);
}

DynGR::~DynGR() {

}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus DynGR::ADMMatterSource_(Driver *pdrive, int stage) {
//  \brief
TaskStatus DynGR::ADMMatterSource_(Driver *pdrive, int stage) {
  pmy_pack->pz4c->ADMMatterSource(pmy_pack);

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void DynGRPS::AssembleDynGRTasks
//  \brief Adds dyngr tasks to stage start/run/end task lists
//  Called by MeshBlockPack::AddPhysics() function directly after Hydro constrctr
//
//  Stage start tasks are those that must be cmpleted over all MeshBlocks before EACH
//  stage can be run (such as posting MPI receives, setting BoundaryCommStatus flags, etc)
//
//  Stage run tasks are those performed in EACH stage
//
//  Stage end tasks are those that can only be cmpleted after all the stage run tasks are
//  finished over all MeshBlocks for EACH stage, such as clearing all MPI non-blocking
//  sends, etc.
template<class EOSPolicy, class ErrorPolicy>
void DynGRPS<EOSPolicy, ErrorPolicy>::AssembleDynGRTasks(TaskList &start, 
    TaskList &run, TaskList &end) {
  TaskID none(0);
  auto &indcs = pmy_pack->pmesh->mb_indcs;

  using namespace hydro;  // NOLINT(build/namespaces)
  using namespace z4c;    // NOLINT(build/namespaces)
  Z4c *pz4c = pmy_pack->pz4c;
  Hydro *phyd = pmy_pack->phydro;

  // naming convention: hydro task id names are unchanged, all z4c tasks now have a z at the beginning

  // start task list
  // Hydro
  id.irecv = start.AddTask(&Hydro::InitRecv, phyd, none);
  // Z4c
  if (pz4c != nullptr) {
    id.zrecv = start.AddTask(&Z4c::InitRecv, pz4c, none);
  }

  // run task list
  id.copyu = run.AddTask(&Hydro::CopyCons, phyd, none);
  // select which calculate flux function to add based on rsolver_method.
  // Calc flux requires metric in flux - must happen before z4ctoadm updates the metric
  if (rsolver_method == DynGR_RSolver::llf_dyngr) {
    id.flux = run.AddTask(&DynGRPS<EOSPolicy, ErrorPolicy>::CalcFluxes<DynGR_RSolver::llf_dyngr>,this,id.copyu);
  } // put more rsolvers here


  // now the rest of the Hydro run tasks
  id.sendf = run.AddTask(&Hydro::SendFlux, phyd, id.flux);
  id.recvf = run.AddTask(&Hydro::RecvFlux, phyd, id.sendf);
  id.expl  = run.AddTask(&Hydro::ExpRKUpdate, phyd, id.recvf); // requires metric in geometric source terms - must happen before z4ctoadm
  id.restu = run.AddTask(&Hydro::RestrictU, phyd, id.expl);
  id.sendu = run.AddTask(&Hydro::SendU, phyd, id.restu);
  id.recvu = run.AddTask(&Hydro::RecvU, phyd, id.sendu);
  id.bcs   = run.AddTask(&Hydro::ApplyPhysicalBCs, phyd, id.recvu);

  // Z4c
  if (pz4c != nullptr) {
    id.zmattersrc = run.AddTask(&DynGR::ADMMatterSource_, this, none); // must happen before c2p, must happen before CalcRHS
    id.zcopyu = run.AddTask(&Z4c::CopyU, pz4c, none);
    id.zcrhsdep = id.zcopyu | id.zmattersrc;
    switch (indcs.ng) {
      case 2: id.zcrhs = run.AddTask(&Z4c::CalcRHS<2>, pz4c, id.zcrhsdep);
              break;
      case 3: id.zcrhs = run.AddTask(&Z4c::CalcRHS<3>, pz4c, id.zcrhsdep);
              break;
      case 4: id.zcrhs = run.AddTask(&Z4c::CalcRHS<4>, pz4c, id.zcrhsdep);
              break;
    }
    id.zsombc = run.AddTask(&Z4c::Z4cBoundaryRHS, pz4c, id.zcrhs);
    id.zexpl  = run.AddTask(&Z4c::ExpRKUpdate, pz4c, id.zsombc); // after this point z4c variables are timestepped
    id.zrestu = run.AddTask(&Z4c::RestrictU, pz4c, id.zexpl);
    id.zsendu = run.AddTask(&Z4c::SendU, pz4c, id.zrestu);
    id.zrecvu = run.AddTask(&Z4c::RecvU, pz4c, id.zsendu);
    id.zbcs   = run.AddTask(&Z4c::ApplyPhysicalBCs, pz4c, id.zrecvu);
    id.zalgc  = run.AddTask(&Z4c::EnforceAlgConstr, pz4c, id.zbcs);
    id.zadep  = id.zalgc | id.flux | id.expl;
    id.z4tad  = run.AddTask(&Z4c::Z4cToADM_, pz4c, id.zadep); // after this point ADM variables are updated - now cannot calcualte geometric source terms/fluxes etc.
    id.zadmc  = run.AddTask(&Z4c::ADMConstraints_, pz4c, id.z4tad);
    id.znewdt = run.AddTask(&Z4c::NewTimeStep, pz4c, id.zadmc); // only need 1 time step
  }
  // TODO MHD

  if (pz4c != nullptr) {
    id.c2pdep = id.bcs | id.z4tad;
  } else {
    id.c2pdep = id.bcs;
  }
  id.c2p = run.AddTask(&DynGRPS<EOSPolicy, ErrorPolicy>::ConToPrim, this, id.c2pdep);

  // c2p takes time-stepped cons -> time-stepped prims, should use time-stepped metric
  // should happen after z4ctoadm. Prims are now updated so can no longer calculate source
  // for Z4c equations, must happen after calcz4crhs (but this happens before z4ctoadm anyway)
  id.newdt = run.AddTask(&Hydro::NewTimeStep, phyd, id.c2p); // only need 1 timestep


  // end task list
  id.clear = end.AddTask(&Hydro::ClearSend, phyd, none);

  if (pz4c != nullptr) {
    id.zclear = end.AddTask(&Z4c::ClearSend, pz4c, none);
  }
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus DynGR::ADMMatterSource_(Driver *pdrive, int stage) {
//  \brief
template<class EOSPolicy, class ErrorPolicy>
void DynGRPS<EOSPolicy, ErrorPolicy>::PrimToConInit(int is, int ie, int js, int je, int ks, int ke) {
  eos.PrimToCons(pmy_pack->phydro->w0, pmy_pack->phydro->u0, is, ie, js, je, ks, ke);
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus DynGR::ADMMatterSource_(Driver *pdrive, int stage) {
//  \brief
template<class EOSPolicy, class ErrorPolicy>
TaskStatus DynGRPS<EOSPolicy, ErrorPolicy>::ConToPrim(Driver *pdrive, int stage) {
  // Extract the indices
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n1m1 = indcs.nx1 + 2*ng - 1;
  int n2m1 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng - 1) : 0;
  int n3m1 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng - 1) : 0;
  eos.ConsToPrim(pmy_pack->phydro->u0, pmy_pack->phydro->w0,
                 0, n1m1, 0, n2m1, 0, n3m1, false);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus DynGR::ADMMatterSource_(Driver *pdrive, int stage) {
//  \brief
template<class EOSPolicy, class ErrorPolicy>
void DynGRPS<EOSPolicy, ErrorPolicy>::AddCoordTerms(const DvceArray5D<Real> &prim,
    const Real dt, DvceArray5D<Real> &rhs, int nghost) {
  switch (nghost) {
    case 2: AddCoordTermsEOS<2>(prim, dt, rhs);
            break;
    case 3: AddCoordTermsEOS<3>(prim, dt, rhs);
            break;
    case 4: AddCoordTermsEOS<4>(prim, dt, rhs);
            break;
  }
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus DynGR::ADMMatterSource_(Driver *pdrive, int stage) {
//  \brief
template<class EOSPolicy, class ErrorPolicy> template<int NGHOST>
void DynGRPS<EOSPolicy, ErrorPolicy>::AddCoordTermsEOS(const DvceArray5D<Real> &prim, 
    const Real dt, DvceArray5D<Real> &rhs) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size  = pmy_pack->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  int ncells1 = indcs.nx1+indcs.ng; // Align scratch buffers with variables
  int nmb = pmy_pack->nmb_thispack;

  auto &adm = pmy_pack->padm->adm;
  auto &eos_ = eos.ps.GetEOS();

  int &nhyd = pmy_pack->phydro->nhydro;
  int &nscal = pmy_pack->phydro->nscalars;

  const Real mb = eos.ps.GetEOS().GetBaryonMass();

  int scr_level = scratch_level;
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1)*2       // scalars
                  + ScrArray2D<Real>::shmem_size(3, ncells1)*2    // vectors
                  + ScrArray2D<Real>::shmem_size(6, ncells1)*2    // symmetric 2 tensors
                  + ScrArray2D<Real>::shmem_size(9, ncells1)*1    // general 2 tensors
                  + ScrArray2D<Real>::shmem_size(18, ncells1)*1;  // symmetric 3 tensors
  par_for_outer("dyngr_coord_terms_loop",DevExeSpace(),scr_size,scr_level,0,nmb-1,ks,ke,js,je,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {
    // Scratch space
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> vol;      // sqrt of determinant of spatial metric
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> E;        // fluid energy density

    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> dalpha_d; // lapse 1st drvts
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> S_d;      // matter momentum

    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> g_uu;     // inverse metric
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> S_uu;     // spatial component of stress tensor

    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 2> dbeta_du; // derivatives of the shift
    
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 3> dg_ddd;    // metric 1st drvts

    vol.NewAthenaScratchTensor(member, scr_level, ncells1);
    E.NewAthenaScratchTensor(member, scr_level, ncells1);
    dalpha_d.NewAthenaScratchTensor(member, scr_level, ncells1);
    S_d.NewAthenaScratchTensor(member, scr_level, ncells1);
    dbeta_du.NewAthenaScratchTensor(member, scr_level, ncells1);
    g_uu.NewAthenaScratchTensor(member, scr_level, ncells1);
    S_uu.NewAthenaScratchTensor(member, scr_level, ncells1);
    dg_ddd.NewAthenaScratchTensor(member, scr_level, ncells1);

    par_for_inner(member, is, ie, [&](int const i) {
      Real detg = SpatialDet(adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i), adm.g_dd(m,0,2,k,j,i),
                             adm.g_dd(m,1,1,k,j,i), adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i));
      vol(i) = sqrt(detg);
    });
    member.team_barrier();

    par_for_inner(member, is, ie, [&](int const i) {
      SpatialInv(1.0/SQR(vol(i)),
                 adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i), adm.g_dd(m,0,2,k,j,i),
                 adm.g_dd(m,1,1,k,j,i), adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i),
                 &g_uu(0,0,i), &g_uu(0,1,i), &g_uu(0,2,i),
                 &g_uu(1,1,i), &g_uu(1,2,i), &g_uu(2,2,i));
    });
    member.team_barrier();

    // Metric derivatives
    Real idx[] = {size.d_view(m).idx1, size.d_view(m).idx2, size.d_view(m).idx3};
    for (int a =0; a < 3; ++a) {
      par_for_inner(member, is, ie, [&](int const i){
        dalpha_d(a, i) = Dx<NGHOST>(a, idx, adm.alpha, m, k, j, i);
      });
    }
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        par_for_inner(member, is, ie, [&](int const i){
          dbeta_du(b,a,i) = Dx<NGHOST>(b, idx, adm.beta_u, m, a, k, j, i);
        });
      }
    }
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        for (int c = 0; c < 3; ++c) {
          par_for_inner(member, is, ie, [&](int const i) {
            dg_ddd(c,a,b,i) = Dx<NGHOST>(c, idx, adm.g_dd, m, a, b, k, j, i);
          });
        }
      }
    }

    // Fluid quantities
    par_for_inner(member, is, ie, [&](int const i) {
      Real prim_pt[NPRIM] = {0.0};
      prim_pt[PRH] = prim(m, IDN, k, j, i)/mb;
      prim_pt[PVX] = prim(m, IVX, k, j, i);
      prim_pt[PVY] = prim(m, IVY, k, j, i);
      prim_pt[PVZ] = prim(m, IVZ, k, j, i);
      for (int i = 0; i < nscal; i++) {
        prim_pt[PYF + i] = prim(m, nhyd + i, k, j, i);
      }
      // FIXME: Go back and change to temperature after validating it all works.
      //Real e = prim(m, IEN, k, j, i) + prim(m, IDN, k, j, i);
      //prim_pt[PTM] = eos_.GetTemperatureFromE(prim_pt[PRH], e, &prim_pt[PYF]);
      //prim_pt[PPR] = eos_.GetPressure(prim_pt[PRH], prim_pt[PTM], &prim_pt[PYF]);
      prim_pt[PPR] = prim(m, IPR, k, j, i);
      prim_pt[PTM] = eos_.GetTemperatureFromP(prim_pt[PRH], prim_pt[PPR], &prim_pt[PYF]);

      if (!isfinite(prim_pt[PTM]) || !isfinite(prim_pt[PPR])) {
        printf("There's a problem with the temperature or pressure in the source terms!\n");
      }

      // Get the conserved variables. Note that we don't use PrimitiveSolver here -- that's
      // because we would need to recalculate quantities used in E and S_d in order to get S_dd.
      Real H = prim(m, IDN, k, j, i)*eos_.GetEnthalpy(prim_pt[PRH], prim_pt[PTM], &prim_pt[PYF]);
      Real usq = 0.0;
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          usq += adm.g_dd(m,a,b,k,j,i)*prim_pt[PVX + a]*prim_pt[PVX + b];
        }
      }
      Real const Wsq = 1.0 + usq;
      Real const W = std::sqrt(Wsq);

      E(i) = H*Wsq - prim_pt[PPR];

      for (int a = 0; a < 3; ++a) {
        S_d(a, i) = 0.0;
        for (int b = 0; b < 3; ++b) {
          S_d(a, i) += H*W*prim_pt[PVX + b]*adm.g_dd(m, a, b, k, j, i);
        }
      }

      for (int a = 0; a < 3; ++a) {
        for (int b = a; b < 3; ++b) {
          S_uu(a,b,i) = H*prim_pt[PVX + a]*prim_pt[PVX + b] + prim_pt[PPR]*g_uu(a,b,i);
        }
      }
    });
    member.team_barrier();

    // Assemble energy RHS
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        par_for_inner(member, is, ie, [&](int const i) {
          rhs(m, IEN, k, j, i) += dt * vol(i) * (
            adm.alpha(m, k, j, i) * adm.K_dd(m, a, b, k, j, i) * S_uu(a, b, i) -
            g_uu(a, b, i) * S_d(a, i) * dalpha_d(b, i));
        });
      }
    }

    // Assemble momentum RHS
    // TODO: Profile and check loop ordering here.
    /*for (int a = 0; a < 3; ++a) {
      par_for_inner(member, is, ie, [&](int const i) {
        for (int b = 0; b < 3; ++b) {
          for (int c = 0; c < 3; ++c) {
            rhs(m,IM1+a, k, j, i) += 0.5 * dt * adm.alpha(m,k,j,i) * vol(i) *
              S_uu(b, c, i) * dg_ddd(a,b,c,i);
          }
          rhs(m, IM1 + a, k, j, i) += dt * vol(i) * S_d(b, i) * dbeta_du(a, b, i);
        }
        rhs(m, IM1 + a, k, j, i) -= dt * vol(i) * E(i) * dalpha_d(a, i);
      });
    }*/
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        for (int c = 0; c < 3; ++c) {
          par_for_inner(member, is, ie, [&](int const i) {
            rhs(m,IM1+a, k, j, i) += 0.5 * dt * adm.alpha(m,k,j,i) * vol(i) *
              S_uu(b, c, i) * dg_ddd(a,b,c,i);
          });
        }
        par_for_inner(member, is, ie, [&](int const i) {
          rhs(m, IM1 + a, k, j, i) += dt * vol(i) *S_d(b, i) * dbeta_du(a, b, i);
        });
      }
      par_for_inner(member, is, ie, [&](int const i) {
        rhs(m, IM1 + a, k, j, i) -= dt * vol(i) * E(i) * dalpha_d(a, i);
      });
    }
  });
}

// Instantiated templates
template class DynGRPS<Primitive::IdealGas, Primitive::ResetFloor>;

// Macro for defining CoordTerms templates
#define INSTANTIATE_COORD_TERMS(EOSPolicy, ErrorPolicy) \
template \
void DynGRPS<EOSPolicy, ErrorPolicy>::AddCoordTermsEOS<2>(const DvceArray5D<Real> &prim, const Real dt, DvceArray5D<Real> &rhs); \
template \
void DynGRPS<EOSPolicy, ErrorPolicy>::AddCoordTermsEOS<3>(const DvceArray5D<Real> &prim, const Real dt, DvceArray5D<Real> &rhs); \
template \
void DynGRPS<EOSPolicy, ErrorPolicy>::AddCoordTermsEOS<4>(const DvceArray5D<Real> &prim, const Real dt, DvceArray5D<Real> &rhs);

INSTANTIATE_COORD_TERMS(Primitive::IdealGas, Primitive::ResetFloor);

#undef INSTANTIATE_COORD_TERMS

} // namespace dyngr
