//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mesh_refinement.cpp
//! \brief Implements constructor and functions in MeshRefinement class.
//! Note while restriction functions for CC and FC data are implemented here, prolongation
//! is part of BVals classes.

#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh.hpp"
#include "mesh_refinement.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

//----------------------------------------------------------------------------------------
// MeshRefinement constructor:

MeshRefinement::MeshRefinement(Mesh *pm, ParameterInput *pin) :
  pmy_mesh(pm),
  d_threshold_(0.0), dd_threshold_(0.0), dv_threshold_(0.0),
  check_cons_(false),
  refine_flag("refine_flag",pm->nmb_thisrank) {

  // read thresholds from <mesh_refinement> block in input file
  if (pin->DoesParameterExist("mesh_refinement", "dens_max")) {
    d_threshold_ = pin->GetReal("mesh_refinement", "dens_max");
    check_cons_ = true;
  }
  if (pin->DoesParameterExist("mesh_refinement", "ddens_max")) {
    dd_threshold_ = pin->GetReal("mesh_refinement", "ddens_max");
    check_cons_ = true;
  }
  if (pin->DoesParameterExist("mesh_refinement", "dvel_max")) {
    dd_threshold_ = pin->GetReal("mesh_refinement", "dvel_max");
    check_cons_ = true;
  }

  if (pm->adaptive) {  // allocate arrays for AMR
    nref = new int[global_variable::nranks];
    nderef = new int[global_variable::nranks];
/**
    rdisp = new int[global_variable::nranks];
    ddisp = new int[global_variable::nranks];
    bnref = new int[global_variable::nranks];
    bnderef = new int[global_variable::nranks];
    brdisp = new int[global_variable::nranks];
    bddisp = new int[global_variable::nranks];
**/
  }
}

//----------------------------------------------------------------------------------------
// destructor

MeshRefinement::~MeshRefinement() {
  if (pmy_mesh->adaptive) { // deallocate arrays for AMR
    delete [] nref;
    delete [] nderef;
/**
    delete [] rdisp;
    delete [] ddisp;
    delete [] bnref;
    delete [] bnderef;
    delete [] brdisp;
    delete [] bddisp;
**/
  }
}

//----------------------------------------------------------------------------------------
//! \fn bool MeshRefinement::CheckForRefinementCondition()
//! \brief Checks for refinement/de-refinement and sets refine_flag(m) for all MeshBlocks
//! within a MeshBlockPack.  Also returns true if any MeshBlock needs to be refined.
//! Default refinement conditions implemented are:
//!   (1) gradient of density above a threshold value (hydro/MHD)
//!   (2) shear of velocity above a threshold value (hydro/MHD)
//!   (3) density max above a threshold value (hydro/MHD)
//!   (4) current density above a threshold (MHD)
//! These are controlled by input parameters in the <mesh_refinement> block.
//! User-defined refinement conditions can also be enrolled in the problem generator
//! by calling the EnrollUserRefinementCondition() function.

bool MeshRefinement::CheckForRefinement(MeshBlockPack* pmbp) {
  bool return_flag = false;
  int nmb = pmbp->pmb->nmb;
  // zero refine_flag in host space and sync with device
  for (int m=0; m<nmb; ++m) {
    refine_flag.h_view(m) = 0;
  }
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();

  // capture variables for kernels
  auto &multi_d = pmy_mesh->multi_d;
  auto &three_d = pmy_mesh->three_d;
  auto &indcs = pmy_mesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie, nx1 = indcs.nx1;
  int &js = indcs.js, &je = indcs.je, nx2 = indcs.nx2;
  int &ks = indcs.ks, &ke = indcs.ke, nx3 = indcs.nx3;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;

  // check Hydro/MHD refinement conditions for cons vars over all MeshBlocks in parallel
  if (((pmbp->phydro != nullptr) || (pmbp->pmhd != nullptr)) && check_cons_) {
    auto &u0 = (pmbp->phydro != nullptr)? pmbp->phydro->u0 : pmbp->pmhd->u0;

    par_for_outer("HydroRefineCond",DevExeSpace(), 0, 0, 0, (nmb-1),
    KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
      // check density threshold
      if (d_threshold_ != 0.0) {
        Real team_dmax=0.0;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, nkji),
        [=](const int idx, Real& dmax) {
          int k = (idx)/nji;
          int j = (idx - k*nji)/nx1;
          int i = (idx - k*nji - j*nx1) + is;
          j += js;
          k += ks;
          dmax = fmax(u0(m,IDN,k,j,i), dmax);
        },Kokkos::Max<Real>(team_dmax));

        // set refinement flag on each MeshBlock if density threshold exceeded
        if (team_dmax > d_threshold_) {
          refine_flag.d_view(m) = 1;
        } else {
          refine_flag.d_view(m) = -1;
        }
      }

      // check density difference threshold
      if (dd_threshold_ != 0.0) {
        Real team_ddmax;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, nkji),
        [=](const int idx, Real& ddmax) {
          int k = (idx)/nji;
          int j = (idx - k*nji)/nx1;
          int i = (idx - k*nji - j*nx1) + is;
          j += js;
          k += ks;
          Real d2 = SQR(u0(m,IDN,k,j,i+1) - u0(m,IDN,k,j,i-1));
          if (multi_d) {d2 += SQR(u0(m,IDN,k,j+1,i) - u0(m,IDN,k,j-1,i));}
          if (three_d) {d2 += SQR(u0(m,IDN,k+1,j,i) - u0(m,IDN,k-1,j,i));}
          ddmax = fmax((sqrt(d2)/u0(m,IDN,k,j,i)), ddmax);
        },Kokkos::Max<Real>(team_ddmax));

        // set refinement flag on each MeshBlock if density threshold exceeded
        if (team_ddmax > dd_threshold_) {refine_flag.d_view(m) = 1;}
        if (team_ddmax < 0.25*dd_threshold_) {refine_flag.d_view(m) = -1;}
      }
    });
  }

  // Check user-defined refinement condition(s), if any
  if (pmy_mesh->pgen->user_ref_func != nullptr) {
    pmy_mesh->pgen->user_ref_func(pmbp);
  }

  // sync refine_flag and check if any MeshBlocks need to be refined/de-refined
  refine_flag.template modify<DevExeSpace>();
  refine_flag.template sync<HostMemSpace>();

  for (int m=0; m<nmb; ++m) {
    if (refine_flag.h_view(m) != 0) {return_flag = true;}
  }
  return return_flag;
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::AdaptiveMeshRefinement(ParameterInput *pin)
//! \brief Main function for adaptive mesh refinement

void MeshRefinement::AdaptiveMeshRefinement() {
  int nnew = 0, ndel = 0;

/***
  amr_updated = false;

  if (adaptive) {
    UpdateMeshBlockTree(nnew, ndel);
    nbnew += nnew; nbdel += ndel;
  }

  lb_flag_ |= lb_automatic_;

  UpdateCostList();

  if (nnew != 0 || ndel != 0) { // at least one (de)refinement happened
    amr_updated = true;
    GatherCostListAndCheckBalance();
    RedistributeAndRefineMeshBlocks(pin, nbtotal + nnew - ndel);
  } else if (lb_flag_ && step_since_lb >= lb_interval_) {
    if (!GatherCostListAndCheckBalance()) // load imbalance detected
      RedistributeAndRefineMeshBlocks(pin, nbtotal);
    lb_flag_ = false;
  }
***/

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::UpdateMeshBlockTree(int &nnew, int &ndel)
//! \brief collect refinement flags and manipulate the MeshBlockTree with AMR
//! Returns total number of MBs refined/derefined in arguments.

void MeshRefinement::UpdateMeshBlockTree(int &nnew, int &ndel) {
  // compute nleaf= number of leaf MeshBlocks per refined block
  int nleaf = 2, dim = 1;
  if (pmy_mesh->two_d) {nleaf = 4, dim = 2;}
  if (pmy_mesh->three_d) {nleaf = 8, dim = 3;}

  // count the number of the blocks to be (de)refined on this rank
  nref[global_variable::my_rank] = 0;
  nderef[global_variable::my_rank] = 0;
  for (int i=0; i<(pmy_mesh->nmb_thisrank); ++i) {
    if (refine_flag.h_view(i) ==  1) nref[global_variable::my_rank]++;
    if (refine_flag.h_view(i) == -1) nderef[global_variable::my_rank]++;
  }
#ifdef MPI_PARALLEL
  MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, nref,   1, MPI_INT, MPI_COMM_WORLD);
  MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, nderef, 1, MPI_INT, MPI_COMM_WORLD);
#endif

  // count the number of the blocks to be (de)refined over all ranks
  int tnref = 0, tnderef = 0;
  for (int n=0; n<global_variable::nranks; n++) {
    tnref  += nref[n];
    tnderef += nderef[n];
  }
  // nothing to do (only derefine if all MeshBlocks within a leaf are flagged)
  if (tnref == 0 && tnderef < nleaf) {
    return;
  }

/***
  // count displacement
  int rd = 0, dd = 0;
  for (int n=0; n<global_variable::nranks; n++) {
    rdisp[n] = rd;
    ddisp[n] = dd;
    // technically could overflow, since sizeof() operator returns
    // std::size_t = long unsigned int > int
    // on many platforms (LP64). However, these are used below in MPI calls for
    // integer arguments (recvcounts, displs). MPI does not support > 64-bit count ranges
    bnref[n] = static_cast<int>(nref[n]*sizeof(LogicalLocation));
    bnderef[n] = static_cast<int>(nderef[n]*sizeof(LogicalLocation));
    brdisp[n] = static_cast<int>(rd*sizeof(LogicalLocation));
    bddisp[n] = static_cast<int>(dd*sizeof(LogicalLocation));
    rd += nref[n];
    dd += nderef[n];
  }
****/

  // allocate memory for logical location arrays over total number MBs refined/derefined
  LogicalLocation *lref{}, *lderef{}, *clderef{};
  if (tnref > 0) {
    lref = new LogicalLocation[tnref];
  }
  if (tnderef >= nleaf) {
    lderef = new LogicalLocation[tnderef];
    clderef = new LogicalLocation[tnderef/nleaf];
  }

  // collect logical locations of MBs to be refined/derefined into arrays
  // calculate starting index in array for updated MBs on this rank (sum of number of
  // updated MBs on ranks < my_rank
  int iref = 0, ideref = 0;
  for (int n=0; n<global_variable::my_rank; n++) {
    iref += nref[n];
    ideref += nderef[n];
  }

  // load logical location arrays for updated MBs in this rank
  for (int i=0; i<(pmy_mesh->nmb_thisrank); ++i) {
    int gid = pmy_mesh->pmb_pack->pmb->mb_gid.h_view(i);
    if (refine_flag.h_view(i) ==  1) {
      lref[iref++] = pmy_mesh->lloclist[gid];;
    } else if (refine_flag.h_view(i) == -1 && tnderef >= nleaf) {
      lderef[ideref++] = pmy_mesh->lloclist[gid];
    }
  }
#ifdef MPI_PARALLEL
  // THIS WILL NOT COMPILE. Look into passing LogicalLocation array elements directly
  // as MPI_Datatype as opposed to sending bytes

  if (tnref > 0) {
    MPI_Allgatherv(MPI_IN_PLACE, bnref[global_variable::my_rank],   MPI_BYTE,
                   lref,   bnref,   brdisp, MPI_BYTE, MPI_COMM_WORLD);
  }
  if (tnderef >= nleaf) {
    MPI_Allgatherv(MPI_IN_PLACE, bnderef[global_variable::my_rank], MPI_BYTE,
                   lderef, bnderef, bddisp, MPI_BYTE, MPI_COMM_WORLD);
  }
#endif

  // calculate the list of the newly derefined blocks
  int ctnd = 0;
  if (tnderef >= nleaf) {
    int lk = 0, lj = 0;
    if (pmy_mesh->multi_d) lj = 1;
    if (pmy_mesh->three_d) lk = 1;
    for (int n=0; n<tnderef; n++) {
      if ((lderef[n].lx1 & 1) == 0 &&
          (lderef[n].lx2 & 1) == 0 &&
          (lderef[n].lx3 & 1) == 0) {
        int r = n, rr = 0;
        for (std::int32_t k=0; k<=lk; k++) {
          for (std::int32_t j=0; j<=lj; j++) {
            for (std::int32_t i=0; i<=1; i++) {
              if (r < tnderef) {
                if ((lderef[n].lx1+i) == lderef[r].lx1 &&
                    (lderef[n].lx2+j) == lderef[r].lx2 &&
                    (lderef[n].lx3+k) == lderef[r].lx3 &&
                     lderef[n].level  == lderef[r].level) {
                  rr++;
                }
                r++;
              }
            }
          }
        }
        if (rr == nleaf) {
          clderef[ctnd].lx1   = lderef[n].lx1 >> 1;
          clderef[ctnd].lx2   = lderef[n].lx2 >> 1;
          clderef[ctnd].lx3   = lderef[n].lx3 >> 1;
          clderef[ctnd].level = lderef[n].level - 1;
          ctnd++;
        }
      }
    }
  }
  // sort the lists by level
  if (ctnd > 1) {
    std::sort(clderef, &(clderef[ctnd-1]), LogicalLocation::Greater);
  }

  if (tnderef >= nleaf) {
    delete [] lderef;
  }

  // Now the lists of the blocks to be refined and derefined are completed
  // Start tree manipulation
  // Step 1. perform refinement
  for (int n=0; n<tnref; n++) {
    MeshBlockTree *bt = pmy_mesh->ptree->FindMeshBlock(lref[n]);
    bt->Refine(nnew);
  }
  if (tnref != 0) {
    delete [] lref;
  }

  // Step 2. perform derefinement
  for (int n=0; n<ctnd; n++) {
    MeshBlockTree *bt = pmy_mesh->ptree->FindMeshBlock(clderef[n]);
    bt->Derefine(ndel);
  }
  if (tnderef >= nleaf) {
    delete [] clderef;
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::RedistributeAndRefineMeshBlocks(int ntot)
//! \brief redistribute MeshBlocks according to the new load balance
//! Input argument is total number of MBs after refinement (current number +/- number of
//! MBs refined/derefined).

void MeshRefinement::RedistributeAndRefineMeshBlocks(int ntot) {
  // compute nleaf = number of leaf MeshBlocks per refined block
  int nleaf = 2;
  if (pmy_mesh->two_d) nleaf = 4;
  if (pmy_mesh->three_d) nleaf = 8;

  // Step 1. construct new lists
  LogicalLocation *newloc = new LogicalLocation[ntot];
  int *newrank = new int[ntot];
  double *newcost = new double[ntot];
  int *newtoold = new int[ntot];
  int *oldtonew = new int[nbtotal];
  int nbtold = nbtotal;
  pmy_mesh->ptree->GetMeshBlockList(newloc, newtoold, nbtotal);

  // create a list mapping the previous gid to the current one
  oldtonew[0] = 0;
  int mb_idx = 1;
  for (int n=1; n<ntot; n++) {
    if (newtoold[n] == newtoold[n-1] + 1) { // normal
      oldtonew[mb_idx++] = n;
    } else if (newtoold[n] == newtoold[n-1] + nleaf) { // derefined
      for (int j=0; j<nleaf-1; j++)
        oldtonew[mb_idx++] = n-1;
      oldtonew[mb_idx++] = n;
    }
  }
  // fill the last block
  for ( ; mb_idx<nbtold; mb_idx++)
    oldtonew[mb_idx] = ntot-1;

  current_level = 0;
  for (int n=0; n<ntot; n++) {
    // "on" = "old n" = "old gid" = "old global MeshBlock ID"
    int on = newtoold[n];
    if (newloc[n].level>current_level) // set the current max level
      current_level = newloc[n].level;
    if (newloc[n].level >= loclist[on].level) { // same or refined
      newcost[n] = costlist[on];
    } else {
      double acost = 0.0;
      for (int l=0; l<nleaf; l++)
        acost += costlist[on+l];
      newcost[n] = acost/nleaf;
    }
  }

  // Step 2. Calculate new load balance
  pmy_mesh->LoadBalance(newcost, newrank, nslist, nblist, ntot);

  int nbs = nslist[global_variable::my_rank];
  int nbe = nbs + nblist[global_variable::my_rank] - 1;

  int bnx1 = my_blocks(0)->block_size.nx1;
  int bnx2 = my_blocks(0)->block_size.nx2;
  int bnx3 = my_blocks(0)->block_size.nx3;

#ifdef MPI_PARALLEL
  // Step 3. count the number of the blocks to be sent / received

#endif // MPI_PARALLEL

/**
  // Step 7. construct a new MeshBlock list (moving the data within the MPI rank)
  AthenaArray<MeshBlock*> newlist;
  newlist.NewAthenaArray(nblist[Globals::my_rank]);
  RegionSize block_size = my_blocks(0)->block_size;

  for (int n=nbs; n<=nbe; n++) {
    int on = newtoold[n];
    if ((ranklist[on] == Globals::my_rank) && (loclist[on].level == newloc[n].level)) {
      // on the same MPI rank and same level -> just move it
      MeshBlock* pob = FindMeshBlock(on);
      pob->gid = n;
      pob->lid = n - nbs;
      newlist(n-nbs) = pob;
      my_blocks(on-gids_) = nullptr;
    } else {
      // on a different refinement level or MPI rank - create a new block
      BoundaryFlag block_bcs[6];
      SetBlockSizeAndBoundaries(newloc[n], block_size, block_bcs);
      newlist(n-nbs) = new MeshBlock(n, n-nbs, newloc[n], block_size, block_bcs, this,
                                     pin, gflag, true);
      // fill the conservative variables
      if ((loclist[on].level > newloc[n].level)) { // fine to coarse (f2c)
        for (int ll=0; ll<nleaf; ll++) {
          if (ranklist[on+ll] != Globals::my_rank) continue;
          // fine to coarse on the same MPI rank (different AMR level) - restriction
          MeshBlock* pob = FindMeshBlock(on+ll);
          FillSameRankFineToCoarseAMR(pob, newlist(n-nbs), loclist[on+ll]);
        }
      } else if ((loclist[on].level < newloc[n].level) && // coarse to fine (c2f)
                 (ranklist[on] == Globals::my_rank)) {
        // coarse to fine on the same MPI rank (different AMR level) - prolongation
        MeshBlock* pob = FindMeshBlock(on);
        FillSameRankCoarseToFineAMR(pob, newlist(n-nbs), newloc[n]);
      }
    }
  }

  // discard remaining MeshBlocks
  // they could be reused, but for the moment, just throw them away for simplicity
  for (int n = 0; n<nblocal; n++) {
    delete my_blocks(n); // OK to delete even if it is nullptr
    my_blocks(n) = nullptr;
  }

  // Replace the MeshBlock list
  my_blocks.ExchangeAthenaArray(newlist);
  nblocal = nblist[Globals::my_rank];
  gids_ = nbs;
  gide_ = nbe;

#ifdef MPI_PARALLEL
  // Step 8. Receive the data and load into MeshBlocks
  // This is a test: try MPI_Waitall later.
  if (nrecv != 0) {
    int rb_idx = 0;     // recv buffer index
    for (int n=nbs; n<=nbe; n++) {
      int on = newtoold[n];
      LogicalLocation &oloc = loclist[on];
      LogicalLocation &nloc = newloc[n];
      MeshBlock *pb = FindMeshBlock(n);
      if (oloc.level == nloc.level) { // same
        if (ranklist[on] == Globals::my_rank) continue;
        MPI_Wait(&(req_recv[rb_idx]), MPI_STATUS_IGNORE);
        FinishRecvSameLevel(pb, recvbuf[rb_idx]);
        rb_idx++;
      } else if (oloc.level > nloc.level) { // f2c
        for (int l=0; l<nleaf; l++) {
          if (ranklist[on+l] == Globals::my_rank) continue;
          MPI_Wait(&(req_recv[rb_idx]), MPI_STATUS_IGNORE);
          FinishRecvFineToCoarseAMR(pb, recvbuf[rb_idx], loclist[on+l]);
          rb_idx++;
        }
      } else { // c2f
        if (ranklist[on] == Globals::my_rank) continue;
        MPI_Wait(&(req_recv[rb_idx]), MPI_STATUS_IGNORE);
        FinishRecvCoarseToFineAMR(pb, recvbuf[rb_idx]);
        rb_idx++;
      }
    }
  }
#endif

  // deallocate arrays
  delete [] loclist;
  delete [] ranklist;
  delete [] costlist;
  delete [] newtoold;
  delete [] oldtonew;
#ifdef MPI_PARALLEL
  if (nsend != 0) {
    MPI_Waitall(nsend, req_send, MPI_STATUSES_IGNORE);
    for (int n=0; n<nsend; n++)
      delete [] sendbuf[n];
    delete [] sendbuf;
    delete [] req_send;
  }
  if (nrecv != 0) {
    for (int n=0; n<nrecv; n++)
      delete [] recvbuf[n];
    delete [] recvbuf;
    delete [] req_recv;
  }
#endif

  // update the lists
  loclist = newloc;
  ranklist = newrank;
  costlist = newcost;

  // re-initialize the MeshBlocks
  for (int i=0; i<nblocal; ++i)
    my_blocks(i)->pbval->SearchAndSetNeighbors(tree, ranklist, nslist);
  Initialize(2, pin);

  ResetLoadBalanceVariables();

***/
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::RestrictCC
//!  \brief Restricts cell-centered variables to coarse mesh

void MeshRefinement::RestrictCC(DvceArray5D<Real> &u, DvceArray5D<Real> &cu) {
  int nmb  = u.extent_int(0);  // TODO(@user): 1st index from L of in array must be NMB
  int nvar = u.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR

  auto &cis = pmy_mesh->mb_indcs.cis;
  auto &cie = pmy_mesh->mb_indcs.cie;
  auto &cjs = pmy_mesh->mb_indcs.cjs;
  auto &cje = pmy_mesh->mb_indcs.cje;
  auto &cks = pmy_mesh->mb_indcs.cks;
  auto &cke = pmy_mesh->mb_indcs.cke;

  // restrict in 1D
  if (pmy_mesh->one_d) {
    par_for("restrict3D",DevExeSpace(),0, nmb-1, 0, nvar-1, cis, cie,
    KOKKOS_LAMBDA(const int m, const int n, const int i) {
      int finei = 2*i - cis;  // correct when cis=is
      cu(m,n,cks,cjs,i) = 0.5*(u(m,n,cks,cjs,finei) + u(m,n,cks,cjs,finei+1));
    });
  // restrict in 2D
  } else if (pmy_mesh->two_d) {
    par_for("restrict3D",DevExeSpace(),0, nmb-1, 0, nvar-1, cjs, cje, cis, cie,
    KOKKOS_LAMBDA(const int m, const int n, const int j, const int i) {
      int finei = 2*i - cis;  // correct when cis=is
      int finej = 2*j - cjs;  // correct when cjs=js
      cu(m,n,cks,j,i) = 0.25*(u(m,n,cks,finej  ,finei) + u(m,n,cks,finej  ,finei+1)
                            + u(m,n,cks,finej+1,finei) + u(m,n,cks,finej+1,finei+1));
    });

  // restrict in 3D
  } else {
    par_for("restrict3D",DevExeSpace(),0, nmb-1, 0, nvar-1, cks, cke, cjs, cje, cis, cie,
    KOKKOS_LAMBDA(const int m, const int n, const int k, const int j, const int i) {
      int finei = 2*i - cis;  // correct when cis=is
      int finej = 2*j - cjs;  // correct when cjs=js
      int finek = 2*k - cks;  // correct when cks=ks
      cu(m,n,k,j,i) =
          0.125*(u(m,n,finek  ,finej  ,finei) + u(m,n,finek  ,finej  ,finei+1)
               + u(m,n,finek  ,finej+1,finei) + u(m,n,finek  ,finej+1,finei+1)
               + u(m,n,finek+1,finej,  finei) + u(m,n,finek+1,finej,  finei+1)
               + u(m,n,finek+1,finej+1,finei) + u(m,n,finek+1,finej+1,finei+1));
    });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::RestrictFC
//! \brief Restricts face-centered variables to coarse mesh

void MeshRefinement::RestrictFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb) {
  int nmb  = b.x1f.extent_int(0);  // TODO(@user): 1st idx from L of in array must be NMB

  auto &cis = pmy_mesh->mb_indcs.cis;
  auto &cie = pmy_mesh->mb_indcs.cie;
  auto &cjs = pmy_mesh->mb_indcs.cjs;
  auto &cje = pmy_mesh->mb_indcs.cje;
  auto &cks = pmy_mesh->mb_indcs.cks;
  auto &cke = pmy_mesh->mb_indcs.cke;

  // restrict in 1D
  if (pmy_mesh->one_d) {
    par_for("restrict3D",DevExeSpace(),0, nmb-1, cis, cie,
    KOKKOS_LAMBDA(const int m, const int i) {
      int finei = 2*i - cis;  // correct when cis=is
      // restrict B1
      cb.x1f(m,cks,cjs,i) = b.x1f(m,cks,cjs,finei);
      if (i==cie) {
        cb.x1f(m,cks,cjs,i+1) = b.x1f(m,cks,cjs,finei+2);
      }
      // restrict B2
      Real b2coarse = 0.5*(b.x2f(m,cks,cjs,finei) + b.x2f(m,cks,cjs,finei+1));
      cb.x2f(m,cks,cjs  ,i) = b2coarse;
      cb.x2f(m,cks,cjs+1,i) = b2coarse;
      // restrict B3
      Real b3coarse = 0.5*(b.x3f(m,cks,cjs,finei) + b.x3f(m,cks,cjs,finei+1));
      cb.x3f(m,cks  ,cjs,i) = b3coarse;
      cb.x3f(m,cks+1,cjs,i) = b3coarse;
    });

  // restrict in 2D
  } else if (pmy_mesh->two_d) {
    par_for("restrict3D",DevExeSpace(),0, nmb-1, cjs, cje, cis, cie,
    KOKKOS_LAMBDA(const int m, const int j, const int i) {
      int finei = 2*i - cis;  // correct when cis=is
      int finej = 2*j - cjs;  // correct when cjs=js
      // restrict B1
      cb.x1f(m,cks,j,i) = 0.5*(b.x1f(m,cks,finej,finei) + b.x1f(m,cks,finej+1,finei));
      if (i==cie) {
        cb.x1f(m,cks,j,i+1) =
          0.5*(b.x1f(m,cks,finej,finei+2) + b.x1f(m,cks,finej+1,finei+2));
      }
      // restrict B2
      cb.x2f(m,cks,j,i) = 0.5*(b.x2f(m,cks,finej,finei) + b.x2f(m,cks,finej,finei+1));
      if (j==cje) {
        cb.x2f(m,cks,j+1,i) =
          0.5*(b.x2f(m,cks,finej+2,finei) + b.x2f(m,cks,finej+2,finei+1));
      }
      // restrict B3
      Real b3coarse = 0.25*(b.x3f(m,cks,finej  ,finei) + b.x3f(m,cks,finej  ,finei+1)
                          + b.x3f(m,cks,finej+1,finei) + b.x3f(m,cks,finej+1,finei+1));
      cb.x3f(m,cks  ,j,i) = b3coarse;
      cb.x3f(m,cks+1,j,i) = b3coarse;
    });

  // restrict in 3D
  } else {
    par_for("restrict3D",DevExeSpace(),0, nmb-1, cks, cke, cjs, cje, cis, cie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      int finei = 2*i - cis;  // correct when cis=is
      int finej = 2*j - cjs;  // correct when cjs=js
      int finek = 2*k - cks;  // correct when cks=ks
      // restrict B1
      cb.x1f(m,k,j,i) =
        0.25*(b.x1f(m,finek  ,finej,finei) + b.x1f(m,finek  ,finej+1,finei)
            + b.x1f(m,finek+1,finej,finei) + b.x1f(m,finek+1,finej+1,finei));
      if (i==cie) {
        cb.x1f(m,k,j,i+1) =
          0.25*(b.x1f(m,finek  ,finej,finei+2) + b.x1f(m,finek  ,finej+1,finei+2)
              + b.x1f(m,finek+1,finej,finei+2) + b.x1f(m,finek+1,finej+1,finei+2));
      }
      // restrict B2
      cb.x2f(m,k,j,i) =
        0.25*(b.x2f(m,finek  ,finej,finei) + b.x2f(m,finek  ,finej,finei+1)
            + b.x2f(m,finek+1,finej,finei) + b.x2f(m,finek+1,finej,finei+1));
      if (j==cje) {
        cb.x2f(m,k,j+1,i) =
          0.25*(b.x2f(m,finek  ,finej+2,finei) + b.x2f(m,finek  ,finej+2,finei+1)
              + b.x2f(m,finek+1,finej+2,finei) + b.x2f(m,finek+1,finej+2,finei+1));
      }
      // restrict B3
      cb.x3f(m,k,j,i) =
        0.25*(b.x3f(m,finek,finej  ,finei) + b.x3f(m,finek,finej  ,finei+1)
            + b.x3f(m,finek,finej+1,finei) + b.x3f(m,finek,finej+1,finei+1));
      if (k==cke) {
        cb.x3f(m,k+1,j,i) =
          0.25*(b.x3f(m,finek+2,finej  ,finei) + b.x3f(m,finek+2,finej  ,finei+1)
              + b.x3f(m,finek+2,finej+1,finei) + b.x3f(m,finek+2,finej+1,finei+1));
      }
    });
  }
  return;
}
