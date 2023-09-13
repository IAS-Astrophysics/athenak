//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mesh_refinement.cpp
//! \brief Implements constructor and functions in MeshRefinement class.
//! Note while restriction functions for CC and FC data are implemented in this file,
//! prolongation operators are implemented as INLINE functions in prolongation.hpp (and
//! are used both here for AMR and in the BVals class at fine/coarse boundaries).

#include <cstdint>   // int32_t
#include <iostream>
#include <algorithm> // sort
#include <utility>   // pair

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh.hpp"
#include "mesh_refinement.hpp"

#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "prolongation.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

//----------------------------------------------------------------------------------------
// MeshRefinement constructor:
// called from Mesh::BuildTree (before physics modules are enrolled)

MeshRefinement::MeshRefinement(Mesh *pm, ParameterInput *pin) :
  pmy_mesh(pm),
  refine_flag("rflag",pm->nmb_total),
  ncyc_since_ref("cyc_since_ref",pm->nmb_total),
  nmb_created(0),
  nmb_deleted(0),
  nmb_sent_thisrank(0),
  ncyc_check_amr(1),
  refinement_interval(5),
  prolong_prims(false),
  d_threshold_(0.0),
  dd_threshold_(0.0),
  dp_threshold_(0.0),
  dv_threshold_(0.0),
  check_cons_(false) {
  if (pin->DoesBlockExist("mesh_refinement")) {
    // read interval (in cycles) between check of AMR and derefinement
    ncyc_check_amr = pin->GetOrAddReal("mesh_refinement", "ncycle_check", 1);
    refinement_interval = pin->GetOrAddReal("mesh_refinement", "refinement_interval", 5);
    // read prolongate primitives flag
    if (pin->DoesParameterExist("mesh_refinement", "prolong_primitives")) {
      prolong_prims = pin->GetBoolean("mesh_refinement", "prolong_primitives");
    }
    // read refinement criteria thresholds
    if (pin->DoesParameterExist("mesh_refinement", "dens_max")) {
      d_threshold_ = pin->GetReal("mesh_refinement", "dens_max");
      check_cons_ = true;
    }
    if (pin->DoesParameterExist("mesh_refinement", "ddens_max")) {
      dd_threshold_ = pin->GetReal("mesh_refinement", "ddens_max");
      check_cons_ = true;
    }
    if (pin->DoesParameterExist("mesh_refinement", "dpres_max")) {
      dp_threshold_ = pin->GetReal("mesh_refinement", "dpres_max");
      check_cons_ = true;
    }
    if (pin->DoesParameterExist("mesh_refinement", "dvel_max")) {
      dd_threshold_ = pin->GetReal("mesh_refinement", "dvel_max");
      check_cons_ = true;
    }
  }

  if (pm->adaptive) {  // allocate arrays for AMR
    nref_eachrank = new int[global_variable::nranks];
    nderef_eachrank = new int[global_variable::nranks];
    nref_rsum = new int[global_variable::nranks];
    nderef_rsum = new int[global_variable::nranks];
  }

  // be sure Views are initialized to zero
  for (int m=0; m<(pm->nmb_total); ++m) {
    refine_flag.h_view(m) = 0;
    ncyc_since_ref(m) = 0;
  }
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();

#if MPI_PARALLEL_ENABLED
  // create unique communicators for AMR
  MPI_Comm_dup(MPI_COMM_WORLD, &amr_comm);
#endif
}

//----------------------------------------------------------------------------------------
// destructor

MeshRefinement::~MeshRefinement() {
  if (pmy_mesh->adaptive) { // deallocate arrays for AMR
    delete [] nref_eachrank;
    delete [] nderef_eachrank;
    delete [] nref_rsum;
    delete [] nderef_rsum;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::AdaptiveMeshRefinement()
//! \brief Simple driver function for adaptive mesh refinement

void MeshRefinement::AdaptiveMeshRefinement(Driver *pdrive, ParameterInput *pin) {
  // first check refinement criteria
  MeshBlockPack* pmbp = pmy_mesh->pmb_pack;
  CheckForRefinement(pmbp);

  // then update mesh tree if MeshBlock anywhere (on any rank) is flagged for refinement
  int nnew = 0, ndel = 0;
  UpdateMeshBlockTree(nnew, ndel);

  // Refine/derefine mesh and eveolved data, set boundary conditions on new mesh
  if (nnew != 0 || ndel != 0) { // at least one (de)refinement flagged
    RedistAndRefineMeshBlocks(pin, nnew, ndel);
    pdrive->InitBoundaryValuesAndPrimitives(pmy_mesh);
    nmb_created += nnew;
    nmb_deleted += ndel;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn bool MeshRefinement::CheckForRefinement()
//! \brief Checks for refinement/de-refinement and sets refine_flag(m) for all
//! MeshBlocks within a MeshBlockPack. Returns true if any MeshBlock needs to be refined.
//! Default refinement conditions implemented are:
//!   (1) density max above a threshold value (hydro/MHD)
//!   (2) gradient of density above a threshold value (hydro/MHD)
//!   (3) gradient of pressure above a threshold value (hydro/MHD)
//!   TODO(@user) (4) shear of velocity above a threshold value (hydro/MHD)
//!   TODO(@user) (5) current density above a threshold (MHD)
//! These are controlled by input parameters in the <mesh_refinement> block.
//! User-defined refinement conditions can also be enrolled by setting the *usr_ref_func
//! pointer in the problem generator.

void MeshRefinement::CheckForRefinement(MeshBlockPack* pmbp) {
  // increment cycle counter for each MB
  for (int m=0; m<(pmy_mesh->nmb_total); ++m) {
    ncyc_since_ref(m) += 1;
  }
  if ((pmbp->pmesh->ncycle)%(ncyc_check_amr) != 0) {return;}  // not cycle to check

  // zero refine_flag in host space and sync with device
  for (int m=0; m<(pmy_mesh->nmb_total); ++m) {
    refine_flag.h_view(m) = 0;
  }
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();

  // capture variables for kernels
  auto &multi_d = pmy_mesh->multi_d;
  auto &three_d = pmy_mesh->three_d;
  auto &indcs = pmy_mesh->mb_indcs;
  int &is = indcs.is, nx1 = indcs.nx1;
  int &js = indcs.js, nx2 = indcs.nx2;
  int &ks = indcs.ks, nx3 = indcs.nx3;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;

  // check (on device) Hydro/MHD refinement conditions for cons vars over all MeshBlocks
  auto refine_flag_ = refine_flag;
  auto &dens_thresh  = d_threshold_;
  auto &ddens_thresh = dd_threshold_;
  auto &dpres_thresh = dp_threshold_;
  int nmb = pmbp->nmb_thispack;
  int mbs = pmy_mesh->gids_eachrank[global_variable::my_rank];
  if (((pmbp->phydro != nullptr) || (pmbp->pmhd != nullptr)) && check_cons_) {
    auto &u0 = (pmbp->phydro != nullptr)? pmbp->phydro->u0 : pmbp->pmhd->u0;
    auto &w0 = (pmbp->phydro != nullptr)? pmbp->phydro->w0 : pmbp->pmhd->w0;

    par_for_outer("ConsRefineCond",DevExeSpace(), 0, 0, 0, (nmb-1),
    KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
      // density threshold
      if (dens_thresh!= 0.0) {
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

        if (team_dmax > dens_thresh) {refine_flag_.d_view(m+mbs) = 1;}
        if (team_dmax < dens_thresh) {refine_flag_.d_view(m+mbs) = -1;}
      }

      // density gradient threshold
      if (ddens_thresh != 0.0) {
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

        if (team_ddmax > ddens_thresh) {refine_flag_.d_view(m+mbs) = 1;}
        if (team_ddmax < 0.25*ddens_thresh) {refine_flag_.d_view(m+mbs) = -1;}
      }

      // pressure gradient threshold
      if (dpres_thresh != 0.0) {
        Real team_dpmax;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, nkji),
        [=](const int idx, Real& dpmax) {
          int k = (idx)/nji;
          int j = (idx - k*nji)/nx1;
          int i = (idx - k*nji - j*nx1) + is;
          j += js;
          k += ks;
          Real d2 = SQR(w0(m,IEN,k,j,i+1) - w0(m,IEN,k,j,i-1));
          if (multi_d) {d2 += SQR(w0(m,IEN,k,j+1,i) - w0(m,IEN,k,j-1,i));}
          if (three_d) {d2 += SQR(w0(m,IEN,k+1,j,i) - w0(m,IEN,k-1,j,i));}
          dpmax = fmax((sqrt(d2)/w0(m,IEN,k,j,i)), dpmax);
        },Kokkos::Max<Real>(team_dpmax));

        if (team_dpmax > dpres_thresh) {refine_flag_.d_view(m+mbs) = 1;}
        if (team_dpmax < 0.25*dpres_thresh) {refine_flag_.d_view(m+mbs) = -1;}
      }
    });
  }

  // Check (on device) user-defined refinement condition(s), if any
  if (pmy_mesh->pgen->user_ref_func != nullptr) {
    pmy_mesh->pgen->user_ref_func(pmbp);
  }
  // sync device array with host
  refine_flag.template modify<DevExeSpace>();
  refine_flag.template sync<HostMemSpace>();

  // Check (on host) for MeshBlocks at max/root level flagged for refine/derefine
  for (int m=0; m<nmb; ++m) {
    if (pmy_mesh->lloc_eachmb[m+mbs].level == pmy_mesh->max_level) {
      if (refine_flag.h_view(m+mbs) > 0) {refine_flag.h_view(m+mbs) = 0;}
    }
    if (pmy_mesh->lloc_eachmb[m+mbs].level == pmy_mesh->root_level) {
      if (refine_flag.h_view(m+mbs) < 0) {refine_flag.h_view(m+mbs) = 0;}
    }
  }

  // Check (on host) that MB has not been recently refined
  for (int m=0; m<nmb; ++m) {
    if (ncyc_since_ref(m+mbs) < refinement_interval) {refine_flag.h_view(m+mbs) = 0;}
  }
#if MPI_PARALLEL_ENABLED
  // Pass refine_glag between all ranks
    MPI_Allgatherv(MPI_IN_PLACE, pmy_mesh->nmb_eachrank[global_variable::my_rank],
                   MPI_INT, refine_flag.h_view.data(), pmy_mesh->nmb_eachrank,
                   pmy_mesh->gids_eachrank, MPI_INT, MPI_COMM_WORLD);
#endif
  // sync host array with device
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::UpdateMeshBlockTree(int &nnew, int &ndel)
//! \brief collect refinement flags and manipulate the MeshBlockTree with AMR
//! Returns total number of MBs refined/derefined in arguments.

void MeshRefinement::UpdateMeshBlockTree(int &nnew, int &ndel) {
  // compute nleaf= number of leaf MeshBlocks per refined block
  int nleaf = 2;
  if (pmy_mesh->two_d) {nleaf = 4;}
  if (pmy_mesh->three_d) {nleaf = 8;}

  // count the number of the blocks to be (de)refined on this rank
  nref_eachrank[global_variable::my_rank] = 0;
  nderef_eachrank[global_variable::my_rank] = 0;
  int mbs = pmy_mesh->gids_eachrank[global_variable::my_rank];
  for (int i=0; i<(pmy_mesh->nmb_thisrank); ++i) {
    if (refine_flag.h_view(i+mbs) ==  1) nref_eachrank[global_variable::my_rank]++;
    if (refine_flag.h_view(i+mbs) == -1) nderef_eachrank[global_variable::my_rank]++;
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, nref_eachrank,   1, MPI_INT, MPI_COMM_WORLD);
  MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, nderef_eachrank, 1, MPI_INT, MPI_COMM_WORLD);
#endif

  // count the number of the blocks to be (de)refined over all ranks
  int tnref = 0, tnderef = 0;
  for (int n=0; n<global_variable::nranks; n++) {
    tnref  += nref_eachrank[n];
    tnderef += nderef_eachrank[n];
  }
  // nothing to do (only derefine if all MeshBlocks within a leaf are flagged)
  if (tnref == 0 && tnderef < nleaf) {
    return;
  }

  // allocate memory for logical location arrays over total number MBs refined/derefined
  LogicalLocation *llref, *llderef, *cllderef;
  if (tnref > 0) {
    llref = new LogicalLocation[tnref];
  }
  if (tnderef >= nleaf) {
    llderef = new LogicalLocation[tnderef];
    cllderef = new LogicalLocation[tnderef/nleaf];
  }

  // calculate running sum of number of MBs to be refined/de-refined
  nref_rsum[0] = 0;
  nderef_rsum[0] = 0;
  for (int n=1; n<global_variable::nranks; n++) {
    nref_rsum[n] = nref_rsum[n-1] + nref_eachrank[n-1];
    nderef_rsum[n] = nderef_rsum[n-1] + nderef_eachrank[n-1];
  }

  // collect logical locations of MBs to be refined/derefined into arrays
  {
    int iref = nref_rsum[global_variable::my_rank];
    int ideref = nderef_rsum[global_variable::my_rank];
    for (int i=0; i<(pmy_mesh->nmb_thisrank); ++i) {
      int gid = pmy_mesh->pmb_pack->pmb->mb_gid.h_view(i);
      if (refine_flag.h_view(i+mbs) ==  1) {
        llref[iref++] = pmy_mesh->lloc_eachmb[gid];;
      } else if (refine_flag.h_view(i+mbs) == -1 && tnderef >= nleaf) {
        llderef[ideref++] = pmy_mesh->lloc_eachmb[gid];
      }
    }
  }
#if MPI_PARALLEL_ENABLED
  // Now pass Logical Locations of MBs updated between all ranks.
  MPI_Datatype lloc_type;
  MPI_Type_contiguous(4, MPI_INT32_T, &lloc_type);
  MPI_Type_commit(&lloc_type);
  if (tnref > 0) {
    MPI_Allgatherv(MPI_IN_PLACE, nref_eachrank[global_variable::my_rank], lloc_type,
                   llref, nref_eachrank, nref_rsum, lloc_type, MPI_COMM_WORLD);
  }
  if (tnderef >= nleaf) {
    MPI_Allgatherv(MPI_IN_PLACE, nderef_eachrank[global_variable::my_rank], lloc_type,
                   llderef, nderef_eachrank, nderef_rsum, lloc_type, MPI_COMM_WORLD);
  }
  MPI_Type_free(&lloc_type);
#endif

  // Each rank now has a complete list of the LLs of MBs refined/derefined on other ranks
  // calculate the list of the newly derefined blocks
  int ctnd = 0;
  if (tnderef >= nleaf) {
    int lk = 0, lj = 0;
    if (pmy_mesh->multi_d) lj = 1;
    if (pmy_mesh->three_d) lk = 1;
    for (int n=0; n<tnderef; n++) {
      if ((llderef[n].lx1 & 1) == 0 &&
          (llderef[n].lx2 & 1) == 0 &&
          (llderef[n].lx3 & 1) == 0) {
        int r = n, rr = 0;
        for (std::int32_t k=0; k<=lk; k++) {
          for (std::int32_t j=0; j<=lj; j++) {
            for (std::int32_t i=0; i<=1; i++) {
              if (r < tnderef) {
                if ((llderef[n].lx1+i) == llderef[r].lx1 &&
                    (llderef[n].lx2+j) == llderef[r].lx2 &&
                    (llderef[n].lx3+k) == llderef[r].lx3 &&
                     llderef[n].level  == llderef[r].level) {
                  rr++;
                }
                r++;
              }
            }
          }
        }
        if (rr == nleaf) {
          cllderef[ctnd].lx1   = llderef[n].lx1 >> 1;
          cllderef[ctnd].lx2   = llderef[n].lx2 >> 1;
          cllderef[ctnd].lx3   = llderef[n].lx3 >> 1;
          cllderef[ctnd].level = llderef[n].level - 1;
          ctnd++;
        }
      }
    }
  }
  // sort the lists by level
  if (ctnd > 1) {
    std::sort(cllderef, &(cllderef[ctnd-1]), Mesh::GreaterLevel);
  }

  if (tnderef >= nleaf) {
    delete [] llderef;
  }

  // Now the lists of the blocks to be refined and derefined are completed
  // Start tree manipulation
  // Step 1. perform refinement
  for (int n=0; n<tnref; n++) {
    MeshBlockTree *bt = pmy_mesh->ptree->FindMeshBlock(llref[n]);
    bt->Refine(nnew);
  }
  if (tnref != 0) {
    delete [] llref;
  }

  // Step 2. perform derefinement
  for (int n=0; n<ctnd; n++) {
    MeshBlockTree *bt = pmy_mesh->ptree->FindMeshBlock(cllderef[n]);
    bt->Derefine(ndel);
    refine_flag.h_view(bt->GetGID()) = -nleaf; // flag root node of derefinements
  }
  // sync host view with device
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();

  if (tnderef >= nleaf) {
    delete [] cllderef;
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::RedistAndRefineMeshBlocks()
//! \brief redistribute MeshBlocks according to the new load balance
//! This requires moving data within the evolved variable arrays for each Physics (e.g.,
//! hydro, mhd, radiation) both within a rank (using deep copies) and potentially between
//! ranks (using MPI calls), and applying restriction and prolongation operators as
//! required. It also requires rebuilding the MB data arrays, coordinates, and neighbors.
//! Boundary values and primitives are set in calling function: AdaptiveMeshRefinement()

void MeshRefinement::RedistAndRefineMeshBlocks(ParameterInput *pin, int nnew, int ndel) {
  Mesh* pm = pmy_mesh;
  int old_nmb = pm->nmb_total;
  int new_nmb = old_nmb + nnew - ndel;
  // compute nleaf = number of leaf MeshBlocks per refined block
  int nleaf = 2;
  if (pm->two_d) nleaf = 4;
  if (pm->three_d) nleaf = 8;

  // Step 1. Create Z-ordered list of logical locations for new MBs, and newtoold list
  // mapping (new MB gid [n])-->(old gid) for all MBs. Index of array [n] is new gid,
  // value is old gid.
  new_lloc_eachmb = new LogicalLocation[new_nmb];
  newtoold = new int[new_nmb];
  int new_nmb_total;
  pm->ptree->CreateZOrderedLLList(new_lloc_eachmb, newtoold, new_nmb_total);
  if (new_nmb_total != new_nmb) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Number of MeshBlocks in new tree = " << new_nmb_total << " but expected "
        << "value = " << new_nmb << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Step 2.  Create oldtonew list mapping the previous gid to the current one for all MBs
  // Index of array is old gid, value is new gid.
  oldtonew = new int[old_nmb];
  oldtonew[0] = 0;
  int mb_idx = 1;
  for (int n=1; n<new_nmb; n++) {
    if (newtoold[n] == newtoold[n-1] + 1) { // normal
      oldtonew[mb_idx++] = n;
    } else if (newtoold[n] == newtoold[n-1] + nleaf) { // derefined
      for (int j=0; j<nleaf-1; j++) {
        oldtonew[mb_idx++] = n-1;
      }
      oldtonew[mb_idx++] = n;
    }
  }
  // fill the last block
  while (mb_idx < old_nmb) {
    oldtonew[mb_idx] = new_nmb-1;
    mb_idx++;
  }

  // Step 3.
  // Calculate new load balance. Initialize new cost array with the simplest estimate
  // possible: all the blocks are equal
  // TODO(@user): implement variable cost per MeshBlock as needed
  new_cost_eachmb = new float[new_nmb];
  new_rank_eachmb = new int[new_nmb];
  new_gids_eachrank = new int[global_variable::nranks];
  new_nmb_eachrank = new int[global_variable::nranks];

  for (int i=0; i<new_nmb; i++) {new_cost_eachmb[i] = 1.0;}
  pm->LoadBalance(new_cost_eachmb, new_rank_eachmb, new_gids_eachrank, new_nmb_eachrank,
                  new_nmb_total);
  if (new_nmb_eachrank[global_variable::my_rank] > pm->nmb_maxperrank) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Number of MeshBlocks in this rank on new tree = "
        << new_nmb_eachrank[global_variable::my_rank] << " on rank = "
        << global_variable::my_rank <<" exceeds <mesh_refinement>/max_nmb_per_rank = "
        << pm->nmb_maxperrank << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // UpdateMeshBlockTree function can refine/de-refine MBs to ensure resolution jump is
  // no more than 2x at boundaries.  Reset refine_flag for these MBs.
  for (int oldm=0; oldm<old_nmb; oldm++) {
    int newm = oldtonew[oldm];
    LogicalLocation &old_lloc = pmy_mesh->lloc_eachmb[oldm];
    LogicalLocation &new_lloc = new_lloc_eachmb[newm];
    if (old_lloc.level > new_lloc.level) {          // old MB was de-refined
      refine_flag.h_view(oldm) = -nleaf;
    } else if (old_lloc.level < new_lloc.level) {   // old MB was refined
      refine_flag.h_view(oldm) = 1;
    }
  }
  // sync host view with device
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();

  // Step 4.
  // Allocate send/recv buffers for load balancing, post receives.
  // Pack send buffers for load blancing and send data
#if MPI_PARALLEL_ENABLED
  InitRecvAMR(nleaf);
  PackAndSendAMR(nleaf);
  nmb_sent_thisrank += nmb_send;
#endif

  // Step 5.
  // De-refine (restrict) evolved physics variables for MeshBlocks within this rank.
  // Simply copies data from coarse arrays in source MBs to appropriate octant of fine
  // array in target MB.
  hydro::Hydro* phydro = pm->pmb_pack->phydro;
  mhd::MHD* pmhd = pm->pmb_pack->pmhd;
  // derefine (if needed)
  if (ndel > 0) {
    if (phydro != nullptr) {
      DerefineCCSameRank(phydro->u0, phydro->coarse_u0);
    }
    if (pmhd != nullptr) {
      DerefineCCSameRank(pmhd->u0, pmhd->coarse_u0);
      DerefineFCSameRank(pmhd->b0, pmhd->coarse_b0);
    }
  }

  // Step 6.
  // Copy evolved physics variables to new MB index within View for MeshBlocks that stay
  // within this rank
  if (phydro != nullptr) {
    CopyCC(phydro->u0);
  }
  if (pmhd != nullptr) {
    CopyCC(pmhd->u0);
    CopyFC(pmhd->b0);
  }

  // Step 7.
  // Copy evolved physics variables for MBs flagged for refinement from source fine array
  // to target coarse array, when both are on same rank.
  if (nnew > 0) {
    if (phydro != nullptr) {
      CopyForRefinementCC(phydro->u0, phydro->coarse_u0);
    }
    if (pmhd != nullptr) {
      CopyForRefinementCC(pmhd->u0, pmhd->coarse_u0);
      CopyForRefinementFC(pmhd->b0, pmhd->coarse_b0);
    }
  }

  // Step 8.
  // Wait for all MPI load balancing communications to finish.  Unpack data.
#if MPI_PARALLEL_ENABLED
  if (nmb_recv > 0) {RecvAndUnpackAMR();}
  if (nmb_send > 0) {ClearSendAMR();}
#endif

  // copy newtoold array to DualView so that it can be accessed in kernel
  DualArray1D<int> new_to_old("newtoold",new_nmb_total);
  for (int m=0; m<new_nmb_total; ++m) {
    new_to_old.h_view(m) = newtoold[m];
  }
  new_to_old.template modify<HostMemSpace>();
  new_to_old.template sync<DevExeSpace>();

  // Step 9.
  // Coarse arrays are now up-to-date, either through copies on same rank or MPI calls
  // So prolongate (refine) evolved physics variables for all MBs flagged for refinement.

  if (nnew > 0) {
    if (phydro != nullptr) {
      RefineCC(new_to_old, phydro->u0, phydro->coarse_u0);
    }
    if (pmhd != nullptr) {
      RefineCC(new_to_old, pmhd->u0, pmhd->coarse_u0);
      RefineFC(new_to_old, pmhd->b0, pmhd->coarse_b0);
    }
  }

  // Update new number of cycles since refinement
  HostArray1D<int> new_ncyc_since_ref("nnref",new_nmb_total);
  for (int m=0; m<(new_nmb_total); ++m) {
    int oldm = newtoold[m];
    if (refine_flag.h_view(oldm) != 0) {
      new_ncyc_since_ref(m) = 0;
    } else {
      new_ncyc_since_ref(m) = ncyc_since_ref(oldm);
    }
  }
  Kokkos::realloc(ncyc_since_ref, new_nmb_total);
  Kokkos::deep_copy(ncyc_since_ref, new_ncyc_since_ref);
  // reallocate refine_flag, will be zeroed out at start of CheckForRefinement
  Kokkos::realloc(refine_flag, new_nmb_total);

  // Step 10.
  // Update data in Mesh/MeshBlockPack/MeshBlock classes with new grid properties
  delete [] pm->lloc_eachmb;
  delete [] pm->rank_eachmb;
  delete [] pm->cost_eachmb;
  delete [] pm->gids_eachrank;
  delete [] pm->nmb_eachrank;
  pm->lloc_eachmb = new_lloc_eachmb;
  pm->rank_eachmb = new_rank_eachmb;
  pm->cost_eachmb = new_cost_eachmb;
  pm->gids_eachrank = new_gids_eachrank;
  pm->nmb_eachrank  = new_nmb_eachrank;
  pm->nmb_total = new_nmb_total;
  pm->nmb_thisrank = pm->nmb_eachrank[global_variable::my_rank];

  pm->pmb_pack->gids = pm->gids_eachrank[global_variable::my_rank];
  pm->pmb_pack->gide = pm->pmb_pack->gids + pm->nmb_eachrank[global_variable::my_rank]-1;
  pm->pmb_pack->nmb_thispack = pm->pmb_pack->gide - pm->pmb_pack->gids + 1;

  delete (pm->pmb_pack->pmb);
  delete (pm->pmb_pack->pcoord);
  pm->pmb_pack->AddMeshBlocks(pin);
  pm->pmb_pack->AddCoordinates(pin);
  pm->pmb_pack->pmb->SetNeighbors(pm->ptree, pm->rank_eachmb);

  // clean-up and return
  delete [] newtoold;
  delete [] oldtonew;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::DerefineCCSameRank
//! \brief For any MeshBlock m flagged for derefinment (refine_flag = -nleaf), copies
//! cell-centered variables in input coarse array for the nleaf MeshBlock indices that are
//! immediately following to the appropriate quadrant of the MeshBlock m in the input
//! fine array,overwriting any data located there.  Only operates on MBs on the same rank

void MeshRefinement::DerefineCCSameRank(DvceArray5D<Real> &a, DvceArray5D<Real> &ca) {
  // nleaf = number of leaf MeshBlocks per refined block
  int nleaf = 2;
  if (pmy_mesh->two_d) nleaf = 4;
  if (pmy_mesh->three_d) nleaf = 8;

  auto &indcs = pmy_mesh->mb_indcs;
  auto &is  = indcs.is,  &js  = indcs.js,  &ks  = indcs.ks;
  auto &cis = indcs.cis, &cjs = indcs.cjs, &cks = indcs.cks;
  auto &cie = indcs.cie, &cje = indcs.cje, &cke = indcs.cke;
  auto &cnx1 = indcs.cnx1, &cnx2 = indcs.cnx2, &cnx3 = indcs.cnx3;

  // Set indices of source (coarse) array
  std::pair<int,int> isrc = std::make_pair(cis,cie+1);
  std::pair<int,int> jsrc = std::make_pair(cjs,cje+1);
  std::pair<int,int> ksrc = std::make_pair(cks,cke+1);

  // loop over old MBs
  int ombs = pmy_mesh->gids_eachrank[global_variable::my_rank];
  int ombe = ombs + pmy_mesh->nmb_eachrank[global_variable::my_rank] - 1;
  for (int oldm=ombs; oldm<=ombe; ++oldm) {
    if (refine_flag.h_view(oldm) < -1) {  // only derefine if nleaf blocks flagged
      int newm = oldtonew[oldm];
      // only copy data if target MB stays on this rank
      if (new_rank_eachmb[newm] == global_variable::my_rank) {
        for (int l=0; l<nleaf; l++) {
          // only move if source array on this rank
          if ((oldm+l) <= ombe) {
            LogicalLocation &lloc = pmy_mesh->lloc_eachmb[oldm+l];
            int ox1 = ((lloc.lx1 & 1) == 1);
            int ox2 = ((lloc.lx2 & 1) == 1);
            int ox3 = ((lloc.lx3 & 1) == 1);
            std::pair<int,int> idst = std::make_pair((is+ox1*cnx1),(is+(ox1+1)*cnx1));
            std::pair<int,int> jdst = std::make_pair((js+ox2*cnx2),(js+(ox2+1)*cnx2));
            std::pair<int,int> kdst = std::make_pair((ks+ox3*cnx3),(ks+(ox3+1)*cnx3));

            // Copy data directly from coarse arrays in MBs to fine array in target MB
            // use indices of old MBs since this function called before CopyL/R
            auto src = Kokkos::subview(ca,(oldm-ombs+l),Kokkos::ALL,ksrc,jsrc,isrc);
            auto dst = Kokkos::subview( a,(oldm-ombs  ),Kokkos::ALL,kdst,jdst,idst);
            Kokkos::deep_copy(DevExeSpace(), dst, src);
          }
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::DerefineFCSameRank
//! \brief Same as DerefineCCSameRank, except for face-centered variables

void MeshRefinement::DerefineFCSameRank(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb) {
  // nleaf = number of leaf MeshBlocks per refined block
  int nleaf = 2;
  if (pmy_mesh->two_d) nleaf = 4;
  if (pmy_mesh->three_d) nleaf = 8;

  auto &indcs = pmy_mesh->mb_indcs;
  auto &is  = indcs.is,  &js  = indcs.js,  &ks  = indcs.ks;
  auto &cis = indcs.cis, &cjs = indcs.cjs, &cks = indcs.cks;
  auto &cie = indcs.cie, &cje = indcs.cje, &cke = indcs.cke;
  auto &cnx1 = indcs.cnx1, &cnx2 = indcs.cnx2, &cnx3 = indcs.cnx3;

  // Set indices of source (coarse) array
  std::pair<int,int> isrc  = std::make_pair(cis,cie+1);
  std::pair<int,int> isrc1 = std::make_pair(cis,cie+2);
  std::pair<int,int> jsrc  = std::make_pair(cjs,cje+1);
  std::pair<int,int> jsrc1 = std::make_pair(cjs,cje+2);
  std::pair<int,int> ksrc  = std::make_pair(cks,cke+1);
  std::pair<int,int> ksrc1 = std::make_pair(cks,cke+2);

  // loop over old MBs
  int ombs = pmy_mesh->gids_eachrank[global_variable::my_rank];
  int ombe = ombs + pmy_mesh->nmb_eachrank[global_variable::my_rank] - 1;
  for (int oldm=ombs; oldm<=ombe; ++oldm) {
    if (refine_flag.h_view(oldm) < -1) {  // only derefine if nleaf blocks flagged
      int newm = oldtonew[oldm];
      // only copy data if target MB stays on this rank
      if (new_rank_eachmb[newm] == global_variable::my_rank) {
        for (int l=0; l<nleaf; l++) {
          // only move if source array on this rank
          if ((oldm+l) <= ombe) {
            LogicalLocation &lloc = pmy_mesh->lloc_eachmb[oldm+l];
            int ox1 = ((lloc.lx1 & 1) == 1);
            int ox2 = ((lloc.lx2 & 1) == 1);
            int ox3 = ((lloc.lx3 & 1) == 1);
            std::pair<int,int> idst  = std::make_pair((is+ox1*cnx1),(is+(ox1+1)*cnx1  ));
            std::pair<int,int> idst1 = std::make_pair((is+ox1*cnx1),(is+(ox1+1)*cnx1+1));
            std::pair<int,int> jdst  = std::make_pair((js+ox2*cnx2),(js+(ox2+1)*cnx2  ));
            std::pair<int,int> jdst1 = std::make_pair((js+ox2*cnx2),(js+(ox2+1)*cnx2+1));
            std::pair<int,int> kdst  = std::make_pair((ks+ox3*cnx3),(ks+(ox3+1)*cnx3  ));
            std::pair<int,int> kdst1 = std::make_pair((ks+ox3*cnx3),(ks+(ox3+1)*cnx3+1));

            // Copy data directly from coarse arrays in MBs to fine array in target MB
            // use indices of old MBs since this function called before CopyL/R
            auto src1 = Kokkos::subview(cb.x1f,(oldm-ombs+l),ksrc,jsrc,isrc1);
            auto dst1 = Kokkos::subview( b.x1f,(oldm-ombs  ),kdst,jdst,idst1);
            Kokkos::deep_copy(DevExeSpace(), dst1, src1);
            auto src2 = Kokkos::subview(cb.x2f,(oldm-ombs+l),ksrc,jsrc1,isrc);
            auto dst2 = Kokkos::subview( b.x2f,(oldm-ombs  ),kdst,jdst1,idst);
            Kokkos::deep_copy(DevExeSpace(), dst2, src2);
            auto src3 = Kokkos::subview(cb.x3f,(oldm-ombs+l),ksrc1,jsrc,isrc);
            auto dst3 = Kokkos::subview( b.x3f,(oldm-ombs  ),kdst1,jdst,idst);
            Kokkos::deep_copy(DevExeSpace(), dst3, src3);
          }
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::CopyCC
//! \brief Copy cell-centered variables to new MB index within View for MeshBlocks that
//! stay within this rank

void MeshRefinement::CopyCC(DvceArray5D<Real> &a) {
  int ombs = pmy_mesh->gids_eachrank[global_variable::my_rank];
  int ombe = ombs + pmy_mesh->nmb_eachrank[global_variable::my_rank] - 1;

  // Copy MBs for which (new gid) < (old gid) [Move L]
  // loop over old MBs (note m=mbs cannot be moved)
  for (int oldm=ombs+1; oldm<=ombe; ++oldm) {
    int newm  = oldtonew[oldm];
    int newm1 = oldtonew[oldm-1];
    // only move data if target array on this rank
    if (new_rank_eachmb[newm] != global_variable::my_rank) continue;
    int msrc = oldm - ombs;
    int mdst = newm - new_gids_eachrank[global_variable::my_rank];
    // Only move MBs whose location in View moves L (mdst < msrc)
    // Do not move MBs that have been de-refined (for which new[m] = new[m-1]).
    if ( ((mdst - msrc) < 0) && (newm != newm1) ) {
      auto src = Kokkos::subview(a,msrc,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
      auto dst = Kokkos::subview(a,mdst,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
      Kokkos::deep_copy(DevExeSpace(), dst, src);
    }
  }

  // Copy MBs for which (new gid) > (old gid) [Move R]
  for (int oldm=ombe; oldm>=ombs; --oldm) {
    int newm  = oldtonew[oldm];
    // only move data if target array on this rank
    if (new_rank_eachmb[newm] != global_variable::my_rank) continue;
    int msrc = oldm - ombs;
    int mdst = newm - new_gids_eachrank[global_variable::my_rank];
    // Only move MBs whose location in View moves R (mdst > msrc)
    if ((mdst - msrc) > 0) {
      auto src = Kokkos::subview(a,msrc,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
      auto dst = Kokkos::subview(a,mdst,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
      Kokkos::deep_copy(DevExeSpace(), dst, src);
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::CopyFC
//! \brief Copy face-centered variables to new MB index within View for MeshBlocks that
//! stay within this rank

void MeshRefinement::CopyFC(DvceFaceFld4D<Real> &b) {
  int ombs = pmy_mesh->gids_eachrank[global_variable::my_rank];
  int ombe = ombs + pmy_mesh->nmb_eachrank[global_variable::my_rank] - 1;

  // Copy MBs for which (new gid) < (old gid) [Move L]
  // loop over old MBs (note m=mbs cannot be moved)
  for (int oldm=ombs+1; oldm<=ombe; ++oldm) {
    int newm  = oldtonew[oldm];
    int newm1 = oldtonew[oldm-1];
    // only move data if target array on this rank
    if (new_rank_eachmb[newm] != global_variable::my_rank) continue;
    int msrc = oldm - ombs;
    int mdst = newm - new_gids_eachrank[global_variable::my_rank];
    // Only move MBs whose location in View moves L (mdst < msrc)
    // Do not move MBs that have been de-refined (for which new[m] = new[m-1]).
    if ( ((mdst - msrc) < 0) && (newm != newm1) ) {
      auto src1 = Kokkos::subview(b.x1f,msrc,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
      auto dst1 = Kokkos::subview(b.x1f,mdst,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
      Kokkos::deep_copy(DevExeSpace(), dst1, src1);
      auto src2 = Kokkos::subview(b.x2f,msrc,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
      auto dst2 = Kokkos::subview(b.x2f,mdst,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
      Kokkos::deep_copy(DevExeSpace(), dst2, src2);
      auto src3 = Kokkos::subview(b.x3f,msrc,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
      auto dst3 = Kokkos::subview(b.x3f,mdst,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
      Kokkos::deep_copy(DevExeSpace(), dst3, src3);
    }
  }

  // Copy MBs for which (new gid) > (old gid) [Move R]
  for (int oldm=ombe; oldm>=ombs; --oldm) {
    int newm  = oldtonew[oldm];
    // only move data if target array on this rank
    if (new_rank_eachmb[newm] != global_variable::my_rank) continue;
    int msrc = oldm - ombs;
    int mdst = newm - new_gids_eachrank[global_variable::my_rank];
    // Only move MBs whose location in View moves R (mdst > msrc)
    if ((mdst - msrc) > 0) {
      auto src1 = Kokkos::subview(b.x1f,msrc,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
      auto dst1 = Kokkos::subview(b.x1f,mdst,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
      Kokkos::deep_copy(DevExeSpace(), dst1, src1);
      auto src2 = Kokkos::subview(b.x2f,msrc,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
      auto dst2 = Kokkos::subview(b.x2f,mdst,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
      Kokkos::deep_copy(DevExeSpace(), dst2, src2);
      auto src3 = Kokkos::subview(b.x3f,msrc,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
      auto dst3 = Kokkos::subview(b.x3f,mdst,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
      Kokkos::deep_copy(DevExeSpace(), dst3, src3);
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::CopyForRefinementCC
//! \brief For any MeshBlock m flagged for refinment (refine_flag = 1), copies
//! cell-centered variables in octants of input fine array to the input coarse arrays at
//! the nleaf-index locations that are immediately following (overwriting any data located
//! there).  Only operates on MBs on the same rank.

void MeshRefinement::CopyForRefinementCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca) {
  auto &indcs = pmy_mesh->mb_indcs;
  auto &ng = indcs.ng;
  int il = indcs.cis - ng, iu = indcs.cie + ng;
  int jl = indcs.cjs,      ju = indcs.cje;
  int kl = indcs.cks,      ku = indcs.cke;
  if (pmy_mesh->multi_d) {
    jl -= ng; ju += ng;
  }
  if (pmy_mesh->three_d) {
    kl -= ng; ku += ng;
  }
  auto &cnx1 = indcs.cnx1, &cnx2 = indcs.cnx2, &cnx3 = indcs.cnx3;

  // Set indices of destination (coarse) array
  std::pair<int,int> idst = std::make_pair(il,iu+1);
  std::pair<int,int> jdst = std::make_pair(jl,ju+1);
  std::pair<int,int> kdst = std::make_pair(kl,ku+1);

  // loop over new MBs on this rank
  int nmbs = new_gids_eachrank[global_variable::my_rank];
  int nmbe = nmbs + new_nmb_eachrank[global_variable::my_rank] - 1;
  for (int newm=nmbs; newm<=nmbe; ++newm) {
    int oldm = newtoold[newm];
    if (refine_flag.h_view(oldm) > 0) {
      // only copy if old and new location of MB on this rank
      if ((new_rank_eachmb[oldtonew[oldm]] == global_variable::my_rank) &&
          (new_rank_eachmb[newm] == global_variable::my_rank)) {
        int msrc = oldtonew[oldm] - nmbs;
        int mdst = newm - nmbs;
        LogicalLocation &lloc = new_lloc_eachmb[newm];
        int ox1 = ((lloc.lx1 & 1) == 1);
        int ox2 = ((lloc.lx2 & 1) == 1);
        int ox3 = ((lloc.lx3 & 1) == 1);
        std::pair<int,int> isrc = std::make_pair((il + ox1*cnx1),(iu+1 + ox1*cnx1));
        std::pair<int,int> jsrc = std::make_pair((jl + ox2*cnx2),(ju+1 + ox2*cnx2));
        std::pair<int,int> ksrc = std::make_pair((kl + ox3*cnx3),(ku+1 + ox3*cnx3));

        // copy data in MBs to be refined to coarse arrays in target MBs
        auto src = Kokkos::subview( a,msrc,Kokkos::ALL,ksrc,jsrc,isrc);
        auto dst = Kokkos::subview(ca,mdst,Kokkos::ALL,kdst,jdst,idst);
        Kokkos::deep_copy(DevExeSpace(), dst, src);
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::CopyForRefinementFC
//! \brief Same as CopyForRefinementCC, but for face-centered arrays

void MeshRefinement::CopyForRefinementFC(DvceFaceFld4D<Real> &b,DvceFaceFld4D<Real> &cb) {
  auto &indcs = pmy_mesh->mb_indcs;
  auto &ng = indcs.ng;
  int il = indcs.cis - ng, iu = indcs.cie + ng;
  int jl = indcs.cjs,      ju = indcs.cje;
  int kl = indcs.cks,      ku = indcs.cke;
  if (pmy_mesh->multi_d) {
    jl -= ng; ju += ng;
  }
  if (pmy_mesh->three_d) {
    kl -= ng; ku += ng;
  }
  auto &cnx1 = indcs.cnx1, &cnx2 = indcs.cnx2, &cnx3 = indcs.cnx3;

  // Set indices of destination (coarse) array
  std::pair<int,int> idst  = std::make_pair(il,iu+1);
  std::pair<int,int> idst1 = std::make_pair(il,iu+2);
  std::pair<int,int> jdst  = std::make_pair(jl,ju+1);
  std::pair<int,int> jdst1 = std::make_pair(jl,ju+2);
  std::pair<int,int> kdst  = std::make_pair(kl,ku+1);
  std::pair<int,int> kdst1 = std::make_pair(kl,ku+2);

  // loop over new MBs on this rank
  int nmbs = new_gids_eachrank[global_variable::my_rank];
  int nmbe = nmbs + new_nmb_eachrank[global_variable::my_rank] - 1;
  for (int newm=nmbs; newm<=nmbe; ++newm) {
    int oldm = newtoold[newm];
    if (refine_flag.h_view(oldm) > 0) {
      // only copy if old and new location of MB on this rank
      if ((new_rank_eachmb[oldtonew[oldm]] == global_variable::my_rank) &&
          (new_rank_eachmb[newm] == global_variable::my_rank)) {
        int msrc = oldtonew[oldm] - nmbs;
        int mdst = newm - nmbs;
        LogicalLocation &lloc = new_lloc_eachmb[newm];
        int ox1 = ((lloc.lx1 & 1) == 1);
        int ox2 = ((lloc.lx2 & 1) == 1);
        int ox3 = ((lloc.lx3 & 1) == 1);
        std::pair<int,int> isrc  = std::make_pair((il + ox1*cnx1),(iu+1 + ox1*cnx1  ));
        std::pair<int,int> isrc1 = std::make_pair((il + ox1*cnx1),(iu+1 + ox1*cnx1+1));
        std::pair<int,int> jsrc  = std::make_pair((jl + ox2*cnx2),(ju+1 + ox2*cnx2  ));
        std::pair<int,int> jsrc1 = std::make_pair((jl + ox2*cnx2),(ju+1 + ox2*cnx2+1));
        std::pair<int,int> ksrc  = std::make_pair((kl + ox3*cnx3),(ku+1 + ox3*cnx3  ));
        std::pair<int,int> ksrc1 = std::make_pair((kl + ox3*cnx3),(ku+1 + ox3*cnx3+1));

        // copy data in MBs to be refined to coarse arrays in target MBs
        auto src1 = Kokkos::subview( b.x1f,msrc,ksrc,jsrc,isrc1);
        auto dst1 = Kokkos::subview(cb.x1f,mdst,kdst,jdst,idst1);
        Kokkos::deep_copy(DevExeSpace(), dst1, src1);
        auto src2 = Kokkos::subview( b.x2f,msrc,ksrc,jsrc1,isrc);
        auto dst2 = Kokkos::subview(cb.x2f,mdst,kdst,jdst1,idst);
        Kokkos::deep_copy(DevExeSpace(), dst2, src2);
        auto src3 = Kokkos::subview( b.x3f,msrc,ksrc1,jsrc,isrc);
        auto dst3 = Kokkos::subview(cb.x3f,mdst,kdst1,jdst,idst);
        Kokkos::deep_copy(DevExeSpace(), dst3, src3);
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::RefineCC
//! \brief Refines cell-centered variables in input view at any MeshBlock index m that is
//! flagged for refinement to the m-index locations which are immediately following,
//! overwriting any data located there. The data in these locations must already have been
//! copied to another location or sent to another rank via MPI.

void MeshRefinement::RefineCC(DualArray1D<int> &n2o, DvceArray5D<Real> &a,
                              DvceArray5D<Real> &ca) {
  int nvar = a.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR
  auto &new_nmb = new_nmb_eachrank[global_variable::my_rank];;
  auto &indcs = pmy_mesh->mb_indcs;
  auto &cis = indcs.cis, &cie = indcs.cie;
  auto &cjs = indcs.cjs, &cje = indcs.cje;
  auto &cks = indcs.cks, &cke = indcs.cke;

  auto &refine_flag_ = refine_flag;
  bool &multi_d = pmy_mesh->multi_d;
  bool &three_d = pmy_mesh->three_d;
  auto &ngids_ = new_gids_eachrank[global_variable::my_rank];
  // Outer loop over (# of MeshBlocks sent)*(# of variables)
  int nmv = new_nmb*nvar;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmv, Kokkos::AUTO);
  Kokkos::parallel_for("SendBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/nvar;
    const int v = (tmember.league_rank() - m*nvar);

    if (refine_flag_.d_view(n2o.d_view(m+ngids_)) > 0) {
      const int ni = cie - cis + 1;
      const int nj = cje - cjs + 1;
      const int nk = cke - cks + 1;
      const int nkji = nk*nj*ni;
      const int nji  = nj*ni;

      // Middle loop over k,j,i
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
        int k = (idx)/nji;
        int j = (idx - k*nji)/ni;
        int i = (idx - k*nji - j*ni) + cis;
        k += cks;
        j += cjs;

        // fine indices refer to target array
        int fi = 2*i - cis;  // correct when cis=is
        int fj = 2*j - cjs;  // correct when cjs=js
        int fk = 2*k - cks;  // correct when cks=ks

        // call inlined prolongation operator for CC variables
        ProlongCC(m,v,k,j,i,fk,fj,fi,multi_d,three_d,ca,a);
      });
    }
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::RefineFC
//! \brief Same as RefineCC, except for face-centered arrays

void MeshRefinement::RefineFC(DualArray1D<int> &n2o, DvceFaceFld4D<Real> &b,
                              DvceFaceFld4D<Real> &cb) {
  auto &new_nmb = new_nmb_eachrank[global_variable::my_rank];;
  auto &indcs = pmy_mesh->mb_indcs;
  auto &is = indcs.is;
  auto &js = indcs.js;
  auto &ks = indcs.ks;
  auto &cis = indcs.cis, &cie = indcs.cie;
  auto &cjs = indcs.cjs, &cje = indcs.cje;
  auto &cks = indcs.cks, &cke = indcs.cke;

  // First prolongate face-centered fields at shared faces betwen fine and coarse cells
  auto &refine_flag_ = refine_flag;
  bool &multi_d = pmy_mesh->multi_d;
  bool &three_d = pmy_mesh->three_d;
  auto &ngids_ = new_gids_eachrank[global_variable::my_rank];

  // Prolongate x1f
  par_for("RefineFC1",DevExeSpace(), 0,(new_nmb-1), cks,cke, cjs,cje, cis,cie+1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    if (refine_flag_.d_view(n2o.d_view(m+ngids_)) > 0) {
      // fine indices refer to target array
      int fi = (i - cis)*2 + is;                   // fine i
      int fj = (multi_d)? ((j - cjs)*2 + js) : j;  // fine j
      int fk = (three_d)? ((k - cks)*2 + ks) : k;  // fine k
      ProlongFCSharedX1Face(m,k,j,i,fk,fj,fi,multi_d,three_d,cb.x1f,b.x1f);
    }
  });

  // Prolongate x2f
  par_for("RefineFC2",DevExeSpace(), 0,(new_nmb-1), cks,cke, cjs,cje+1, cis,cie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    if (refine_flag_.d_view(n2o.d_view(m+ngids_)) > 0) {
      // fine indices refer to target array
      int fi = (i - cis)*2 + is;                   // fine i
      int fj = (multi_d)? ((j - cjs)*2 + js) : j;  // fine j
      int fk = (three_d)? ((k - cks)*2 + ks) : k;  // fine k
      ProlongFCSharedX2Face(m,k,j,i,fk,fj,fi,three_d,cb.x2f,b.x2f);
    }
  });

  // Prolongate x3f
  par_for("RefineFC3",DevExeSpace(), 0,(new_nmb-1), cks,cke+1, cjs,cje, cis,cie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    if (refine_flag_.d_view(n2o.d_view(m+ngids_)) > 0) {
      // fine indices refer to target array
      int fi = (i - cis)*2 + is;                   // fine i
      int fj = (multi_d)? ((j - cjs)*2 + js) : j;  // fine j
      int fk = (three_d)? ((k - cks)*2 + ks) : k;  // fine k
      ProlongFCSharedX3Face(m,k,j,i,fk,fj,fi,multi_d,cb.x3f,b.x3f);
    }
  });

  // Second prolongate face-centered fields at internal faces of fine cells using
  // divergence-preserving operator of Toth & Roe (2002)
  bool &one_d = pmy_mesh->one_d;
  par_for("RefineFC-int",DevExeSpace(), 0,(new_nmb-1), cks,cke, cjs,cje, cis,cie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    if (refine_flag_.d_view(n2o.d_view(m+ngids_)) > 0) {
      // fine indices refer to target array
      int fi = (i - cis)*2 + is;   // fine i
      int fj = (j - cjs)*2 + js;   // fine j
      int fk = (k - cks)*2 + ks;   // fine k

      if (one_d) {
        // In 1D, interior face field is trivial
        b.x1f(m,fk,fj,fi+1) = 0.5*(b.x1f(m,fk,fj,fi) + b.x1f(m,fk,fj,fi+2));
      } else {
        // in multi-D call inlined prolongation operator for FC fields at internal faces
        ProlongFCInternal(m,fk,fj,fi,three_d,b);
      }
    }
  });

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
    par_for("restrictCC-1D",DevExeSpace(), 0,nmb-1, 0,nvar-1, cis,cie,
    KOKKOS_LAMBDA(const int m, const int n, const int i) {
      int finei = 2*i - cis;  // correct when cis=is
      cu(m,n,cks,cjs,i) = 0.5*(u(m,n,cks,cjs,finei) + u(m,n,cks,cjs,finei+1));
    });
  // restrict in 2D
  } else if (pmy_mesh->two_d) {
    par_for("restrictCC-2D",DevExeSpace(), 0,nmb-1, 0,nvar-1, cjs,cje, cis,cie,
    KOKKOS_LAMBDA(const int m, const int n, const int j, const int i) {
      int finei = 2*i - cis;  // correct when cis=is
      int finej = 2*j - cjs;  // correct when cjs=js
      cu(m,n,cks,j,i) = 0.25*(u(m,n,cks,finej  ,finei) + u(m,n,cks,finej  ,finei+1)
                            + u(m,n,cks,finej+1,finei) + u(m,n,cks,finej+1,finei+1));
    });

  // restrict in 3D
  } else {
    par_for("restrictCC-3D",DevExeSpace(), 0,nmb-1, 0,nvar-1, cks,cke, cjs,cje, cis,cie,
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
    par_for("restrictFC-1D",DevExeSpace(), 0,nmb-1, cis,cie,
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
    par_for("restrictFC-2D",DevExeSpace(), 0,nmb-1, cjs,cje, cis,cie,
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
    par_for("restrictFC-3D",DevExeSpace(), 0,nmb-1, cks,cke, cjs,cje, cis,cie,
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
