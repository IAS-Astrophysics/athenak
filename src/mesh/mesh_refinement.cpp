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
  refine_flag("rflag",pm->nmb_maxperrank),
  cyc_since_ref("cyc_since_ref",pm->nmb_maxperrank),
  nmb_created(0),
  nmb_deleted(0),
  ncycle_check_amr(1),
  ncycle_ref_inter(5),
  d_threshold_(0.0),
  dd_threshold_(0.0),
  dp_threshold_(0.0),
  dv_threshold_(0.0),
  check_cons_(false) {
  // read interval (in cycles) between check of AMR and derefinement
  if (pin->DoesBlockExist("mesh_refinement")) {
    ncycle_check_amr = pin->GetOrAddReal("mesh_refinement", "ncycle_check", 1);
    ncycle_ref_inter = pin->GetOrAddReal("mesh_refinement", "refine_interval", 5);
  }
  // read thresholds from <mesh_refinement> block in input file
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
  if (pm->adaptive) {  // allocate arrays for AMR
    nref_eachrank = new int[global_variable::nranks];
    nderef_eachrank = new int[global_variable::nranks];
    nref_rsum = new int[global_variable::nranks];
    nderef_rsum = new int[global_variable::nranks];
  }

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
  int nnew = 0, ndel = 0;
  UpdateMeshBlockTree(nnew, ndel);

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

bool MeshRefinement::CheckForRefinement(MeshBlockPack* pmbp) {
  bool return_flag = false;
  int nmb = pmbp->nmb_thispack;
  for (int m=0; m<nmb; ++m) {
    cyc_since_ref(m) += 1;
  }
  // Return if not correct cycle for checks
  if ((pmbp->pmesh->ncycle)%(ncycle_check_amr) != 0) {
    return return_flag;
  }

  // zero refine_flag in host space and sync with device
  for (int m=0; m<(pmbp->pmesh->nmb_maxperrank); ++m) {
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

        if (team_dmax > dens_thresh) {refine_flag_.d_view(m) = 1;}
        if (team_dmax < dens_thresh) {refine_flag_.d_view(m) = -1;}
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

        if (team_ddmax > ddens_thresh) {refine_flag_.d_view(m) = 1;}
        if (team_ddmax < 0.25*ddens_thresh) {refine_flag_.d_view(m) = -1;}
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

        if (team_dpmax > dpres_thresh) {refine_flag_.d_view(m) = 1;}
        if (team_dpmax < 0.25*dpres_thresh) {refine_flag_.d_view(m) = -1;}
      }
    });
  }

  // Check (on device) user-defined refinement condition(s), if any
  if (pmy_mesh->pgen->user_ref_func != nullptr) {
    pmy_mesh->pgen->user_ref_func(pmbp);
  }
  refine_flag.template modify<DevExeSpace>();
  refine_flag.template sync<HostMemSpace>();

  // Check (on host) for MeshBlocks at max/root level flagged for refine/derefine
  for (int m=0; m<nmb; ++m) {
    if (pmy_mesh->lloc_eachmb[m].level == pmy_mesh->max_level) {
      if (refine_flag.h_view(m) > 0) {refine_flag.h_view(m) = 0;}
    }
    if (pmy_mesh->lloc_eachmb[m].level == pmy_mesh->root_level) {
      if (refine_flag.h_view(m) < 0) {refine_flag.h_view(m) = 0;}
    }
  }

  // Check (on host) that MB has not been recently refined
  for (int m=0; m<nmb; ++m) {
    if (cyc_since_ref(m) < ncycle_ref_inter) {refine_flag.h_view(m) = 0;}
  }
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();

  // return true if refinement flag set on any MeshBlock
  for (int m=0; m<nmb; ++m) {
    if (refine_flag.h_view(m) != 0) {return_flag = true;}
  }
  return return_flag;
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
  for (int i=0; i<(pmy_mesh->nmb_thisrank); ++i) {
    if (refine_flag.h_view(i) ==  1) nref_eachrank[global_variable::my_rank]++;
    if (refine_flag.h_view(i) == -1) nderef_eachrank[global_variable::my_rank]++;
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
      if (refine_flag.h_view(i) ==  1) {
        llref[iref++] = pmy_mesh->lloc_eachmb[gid];;
      } else if (refine_flag.h_view(i) == -1 && tnderef >= nleaf) {
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
    refine_flag.h_view(bt->GetGID()) = -nleaf;  // flag root node of derefinement
  }
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();

  if (tnderef >= nleaf) {
    delete [] cllderef;
  }

/***/
for (int m=0; m<(pmy_mesh->nmb_thisrank); ++m) {
std::cout << "m=" << m << " flag="<< refine_flag.h_view(m) << std::endl;
}
/***/

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

  // Step 4.
  // Restrict evolved variables within each physics for MBs flagged for derefinement
  hydro::Hydro* phydro = pm->pmb_pack->phydro;
  mhd::MHD* pmhd = pm->pmb_pack->pmhd;
  auto &nmb = pm->pmb_pack->nmb_thispack;                           // old nmb
  int mbs = pmy_mesh->gids_eachrank[global_variable::my_rank];      // old gids
  // derefine (if needed)
  if (ndel > 0) {
    if (phydro != nullptr) {
      DerefineCC(phydro->u0, phydro->coarse_u0);
    }
    if (pmhd != nullptr) {
      DerefineCC(pmhd->u0, pmhd->coarse_u0);
      DerefineFC(pmhd->b0, pmhd->coarse_b0);
    }
  }

  // Step 5. Move evolved variables within view for any MB in which (new gid) > (old gid)
  // Start loop at one since first MB cannot be moved. Do not move MBs that have been
  // de-refined (for which n = nm1).
  for (int m=1; m<nmb; ++m) {
    int nm1 = oldtonew[mbs + m - 1] - mbs;
    int n   = oldtonew[mbs + m] - mbs;
    if ( ((n-m) < 0) && (n != nm1) ) {
      if (phydro != nullptr) {
        auto u0 = phydro->u0;
        auto src = Kokkos::subview(u0,m,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        auto dst = Kokkos::subview(u0,n,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        Kokkos::deep_copy(DevExeSpace(), dst, src);
      }
      if (pmhd != nullptr) {
        auto u0 = pmhd->u0;
        auto src = Kokkos::subview(u0,m,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        auto dst = Kokkos::subview(u0,n,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        Kokkos::deep_copy(DevExeSpace(), dst, src);
        auto b1 = pmhd->b0.x1f;
        auto src1 = Kokkos::subview(b1,m,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        auto dst1 = Kokkos::subview(b1,n,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        Kokkos::deep_copy(DevExeSpace(), dst1, src1);
        auto b2 = pmhd->b0.x2f;
        auto src2 = Kokkos::subview(b2,m,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        auto dst2 = Kokkos::subview(b2,n,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        Kokkos::deep_copy(DevExeSpace(), dst2, src2);
        auto b3 = pmhd->b0.x3f;
        auto src3 = Kokkos::subview(b3,m,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        auto dst3 = Kokkos::subview(b3,n,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        Kokkos::deep_copy(DevExeSpace(), dst3, src3);
      }
    }
  }

  // Step 6. Move evolved variables within view for any MB in which (new gid) < (old gid)
  for (int m=(nmb-1); m >= 0; --m) {
    int n = oldtonew[mbs + m] - mbs;
    if ((n-m) > 0) {
      if (phydro != nullptr) {
        auto u0 = phydro->u0;
        auto src = Kokkos::subview(u0,m,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        auto dst = Kokkos::subview(u0,n,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        Kokkos::deep_copy(DevExeSpace(), dst, src);
      }
      if (pmhd != nullptr) {
        auto u0 = pmhd->u0;
        auto src = Kokkos::subview(u0,m,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        auto dst = Kokkos::subview(u0,n,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        Kokkos::deep_copy(DevExeSpace(), dst, src);
        auto b1 = pmhd->b0.x1f;
        auto src1 = Kokkos::subview(b1,m,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        auto dst1 = Kokkos::subview(b1,n,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        Kokkos::deep_copy(DevExeSpace(), dst1, src1);
        auto b2 = pmhd->b0.x2f;
        auto src2 = Kokkos::subview(b2,m,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        auto dst2 = Kokkos::subview(b2,n,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        Kokkos::deep_copy(DevExeSpace(), dst2, src2);
        auto b3 = pmhd->b0.x3f;
        auto src3 = Kokkos::subview(b3,m,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        auto dst3 = Kokkos::subview(b3,n,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        Kokkos::deep_copy(DevExeSpace(), dst3, src3);
      }
    }
  }

  // Step 7. Prolongate evolved variables for MBs flagged for refinement.
  if (nnew > 0) {
    if (phydro != nullptr) {
      RefineCC(new_nmb_total, phydro->u0, phydro->coarse_u0);
    }
    if (pmhd != nullptr) {
      RefineCC(new_nmb_total, pmhd->u0, pmhd->coarse_u0);
      RefineFC(new_nmb_total, pmhd->b0, pmhd->coarse_b0);
    }
  }

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

  // Update new number of cycles since refinement
  HostArray1D<int> new_cyc_since_ref("new_ncyc_ref",pm->nmb_maxperrank);
  for (int m=0; m<(pm->pmb_pack->nmb_thispack); ++m) {
    if (refine_flag.h_view(newtoold[m]) != 0) {
      new_cyc_since_ref(m) = 0;
    } else {
      new_cyc_since_ref(m) = cyc_since_ref(newtoold[m]);
    }
  }
  Kokkos::deep_copy(cyc_since_ref, new_cyc_since_ref);

  delete [] newtoold;
  delete [] oldtonew;
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

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::RefineCC
//! \brief Refines cell-centered variables in input view at any MeshBlock index m that is
//! flagged for refinement to the m-index locations which are immediately following,
//! overwriting any data located there. The data in these locations must already have been
//! copied to another location or sent to another rank via MPI.

void MeshRefinement::RefineCC(int new_nmb, DvceArray5D<Real> &a, DvceArray5D<Real> &ca) {
  int nvar = a.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR
  auto &old_nmb = pmy_mesh->pmb_pack->nmb_thispack;
  auto &indcs = pmy_mesh->mb_indcs;
  auto &is = indcs.is, &ie = indcs.ie;
  auto &js = indcs.js, &je = indcs.je;
  auto &ks = indcs.ks, &ke = indcs.ke;
  auto &cis = indcs.cis, &cie = indcs.cie;
  auto &cjs = indcs.cjs, &cje = indcs.cje;
  auto &cks = indcs.cks, &cke = indcs.cke;
  auto &cnx1 = indcs.cnx1, &cnx2 = indcs.cnx2, &cnx3 = indcs.cnx3;

  // First copy data in MBs to be refined to coarse arrays in target MBs
  std::pair<int,int> kdst = std::make_pair(cks,cke+1);
  std::pair<int,int> jdst = std::make_pair(cjs,cje+1);
  std::pair<int,int> idst = std::make_pair(cis,cie+1);
  for (int m=0; m<old_nmb; ++m) {
    if (refine_flag.h_view(m) > 0) {
      int newm = oldtonew[m];
      int newn = newm;
      for (int k=ks; k<=ke; k += cnx3) {
        std::pair<int,int> ksrc = std::make_pair(k,k+cnx3);
        for (int j=js; j<=je; j += cnx2) {
          std::pair<int,int> jsrc = std::make_pair(j,j+cnx2);
          for (int i=is; i<=ie; i += cnx1) {
            std::pair<int,int> isrc = std::make_pair(i,i+cnx1);
            auto src = Kokkos::subview( a,newm,Kokkos::ALL,ksrc,jsrc,isrc);
            auto dst = Kokkos::subview(ca,newn,Kokkos::ALL,kdst,jdst,idst);
            Kokkos::deep_copy(DevExeSpace(), dst, src);
            ++newn;
          }
        }
      }
    }
  }

  DualArray1D<int> new_to_old("newtoold",new_nmb);
  for (int m=0; m<new_nmb; ++m) {
    new_to_old.h_view(m) = newtoold[m];
  }
  new_to_old.template modify<HostMemSpace>();
  new_to_old.template sync<DevExeSpace>();


  // Now prolongate data in coarse arrays to fine arrays for all MBs being refined
  auto &refine_flag_ = refine_flag;
  bool &multi_d = pmy_mesh->multi_d;
  bool &three_d = pmy_mesh->three_d;
  par_for("refineCC",DevExeSpace(), 0,(new_nmb-1), 0,nvar-1, cks,cke, cjs,cje, cis,cie,
  KOKKOS_LAMBDA(const int m, const int v, const int k, const int j, const int i) {
    if (refine_flag_.d_view(new_to_old.d_view(m)) > 0) {
      // fine indices refer to target array
      int fi = 2*i - cis;  // correct when cis=is
      int fj = 2*j - cjs;  // correct when cjs=js
      int fk = 2*k - cks;  // correct when cks=ks

      // call inlined prolongation operator for CC variables
      ProlongCC(m,v,k,j,i,fk,fj,fi,multi_d,three_d,ca,a);
    }
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::RefineFC
//! \brief Refines face-centered variables in input view at any MeshBlock index m that is
//! flagged for refinement to the m-index locations which are immediately following,
//! overwriting any data located there. Logic is identical to CC refinement.

void MeshRefinement::RefineFC(int new_nmb, DvceFaceFld4D<Real> &b,
                              DvceFaceFld4D<Real> &cb) {
  auto &old_nmb = pmy_mesh->pmb_pack->nmb_thispack;
  auto &indcs = pmy_mesh->mb_indcs;
  auto &is = indcs.is, &ie = indcs.ie;
  auto &js = indcs.js, &je = indcs.je;
  auto &ks = indcs.ks, &ke = indcs.ke;
  auto &cis = indcs.cis, &cie = indcs.cie;
  auto &cjs = indcs.cjs, &cje = indcs.cje;
  auto &cks = indcs.cks, &cke = indcs.cke;
  auto &cnx1 = indcs.cnx1, &cnx2 = indcs.cnx2, &cnx3 = indcs.cnx3;

  // First copy data in MBs to be refined to coarse arrays in target MBs
  std::pair<int,int> kdst  = std::make_pair(cks,cke+1);
  std::pair<int,int> kdst1 = std::make_pair(cks,cke+2);
  std::pair<int,int> jdst  = std::make_pair(cjs,cje+1);
  std::pair<int,int> jdst1 = std::make_pair(cjs,cje+2);
  std::pair<int,int> idst  = std::make_pair(cis,cie+1);
  std::pair<int,int> idst1 = std::make_pair(cis,cie+2);
  for (int m=0; m<old_nmb; ++m) {
    if (refine_flag.h_view(m) > 0) {
      int newm = oldtonew[m];
      int newn = newm;
      for (int k=ks; k<=ke; k += cnx3) {
        std::pair<int,int> ksrc  = std::make_pair(k,k+cnx3);
        std::pair<int,int> ksrc1 = std::make_pair(k,k+cnx3+1);
        for (int j=js; j<=je; j += cnx2) {
          std::pair<int,int> jsrc  = std::make_pair(j,j+cnx2);
          std::pair<int,int> jsrc1 = std::make_pair(j,j+cnx2+1);
          for (int i=is; i<=ie; i += cnx1) {
            std::pair<int,int> isrc  = std::make_pair(i,i+cnx1);
            std::pair<int,int> isrc1 = std::make_pair(i,i+cnx1+1);
            auto src1 = Kokkos::subview( b.x1f,newm,ksrc,jsrc,isrc1);
            auto dst1 = Kokkos::subview(cb.x1f,newn,kdst,jdst,idst1);
            Kokkos::deep_copy(DevExeSpace(), dst1, src1);
            auto src2 = Kokkos::subview( b.x2f,newm,ksrc,jsrc1,isrc);
            auto dst2 = Kokkos::subview(cb.x2f,newn,kdst,jdst1,idst);
            Kokkos::deep_copy(DevExeSpace(), dst2, src2);
            auto src3 = Kokkos::subview( b.x3f,newm,ksrc1,jsrc,isrc);
            auto dst3 = Kokkos::subview(cb.x3f,newn,kdst1,jdst,idst);
            Kokkos::deep_copy(DevExeSpace(), dst3, src3);
            ++newn;
          }
        }
      }
    }
  }

  DualArray1D<int> new_to_old("newtoold",new_nmb);
  for (int m=0; m<new_nmb; ++m) {
    new_to_old.h_view(m) = newtoold[m];
  }
  new_to_old.template modify<HostMemSpace>();
  new_to_old.template sync<DevExeSpace>();

  // Now prolongate data in coarse arrays to fine arrays for all MBs being refined
  // First prolongate face-centered fields at shared faces betwen fine and coarse cells
  auto &refine_flag_ = refine_flag;
  bool &multi_d = pmy_mesh->multi_d;
  bool &three_d = pmy_mesh->three_d;

  // Prolongate x1f
  par_for("RefineFC1",DevExeSpace(), 0,(new_nmb-1), cks,cke, cjs,cje, cis,cie+1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    if (refine_flag_.d_view(new_to_old.d_view(m)) > 0) {
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
    if (refine_flag_.d_view(new_to_old.d_view(m)) > 0) {
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
    if (refine_flag_.d_view(new_to_old.d_view(m)) > 0) {
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
    if (refine_flag_.d_view(new_to_old.d_view(m)) > 0) {
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
//! \fn void MeshRefinement::DerefineCC
//! \brief Derefines cell-centered variables in input view at any MeshBlock index m that
//! is flagged for derefinement to the m-index locations which are immediately following,
//! overwriting any data located there.

void MeshRefinement::DerefineCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca) {
  int nvar = a.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR
  auto &old_nmb = pmy_mesh->pmb_pack->nmb_thispack;
  auto &indcs = pmy_mesh->mb_indcs;
  auto &is = indcs.is, &ie = indcs.ie;
  auto &js = indcs.js, &je = indcs.je;
  auto &ks = indcs.ks, &ke = indcs.ke;
  auto &cis = indcs.cis, &cie = indcs.cie;
  auto &cjs = indcs.cjs, &cje = indcs.cje;
  auto &cks = indcs.cks, &cke = indcs.cke;
  auto &cnx1 = indcs.cnx1, &cnx2 = indcs.cnx2, &cnx3 = indcs.cnx3;

  // Copy data directly from coarse arrays in MBs to be refined to fine array in target MB
  std::pair<int,int> ksrc = std::make_pair(cks,cke+1);
  std::pair<int,int> jsrc = std::make_pair(cjs,cje+1);
  std::pair<int,int> isrc = std::make_pair(cis,cie+1);
  for (int m=0; m<old_nmb; ++m) {
    if (refine_flag.h_view(m) < -1) {  // only derefine if nleaf blocks flagged
      int srcm = m;
      for (int k=ks; k<=ke; k += cnx3) {
        std::pair<int,int> kdst = std::make_pair(k,k+cnx3);
        for (int j=js; j<=je; j += cnx2) {
          std::pair<int,int> jdst = std::make_pair(j,j+cnx2);
          for (int i=is; i<=ie; i += cnx1) {
            std::pair<int,int> idst = std::make_pair(i,i+cnx1);
            auto src = Kokkos::subview(ca,srcm,Kokkos::ALL,ksrc,jsrc,isrc);
            auto dst = Kokkos::subview( a,m   ,Kokkos::ALL,kdst,jdst,idst);
            Kokkos::deep_copy(DevExeSpace(), dst, src);
            ++srcm;
          }
        }
      }
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::DerefineFC
//! \brief Derefines face-centered variables in input view at any MeshBlock index m that
//! is flagged for derefinement to the m-index locations which are immediately following,
//! overwriting any data located there.  Similar to CC case.

void MeshRefinement::DerefineFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb) {
  auto &old_nmb = pmy_mesh->pmb_pack->nmb_thispack;
  auto &indcs = pmy_mesh->mb_indcs;
  auto &is = indcs.is, &ie = indcs.ie;
  auto &js = indcs.js, &je = indcs.je;
  auto &ks = indcs.ks, &ke = indcs.ke;
  auto &cis = indcs.cis, &cie = indcs.cie;
  auto &cjs = indcs.cjs, &cje = indcs.cje;
  auto &cks = indcs.cks, &cke = indcs.cke;
  auto &cnx1 = indcs.cnx1, &cnx2 = indcs.cnx2, &cnx3 = indcs.cnx3;

  // Copy data directly from coarse arrays in MBs to be refined to fine array in target MB
  std::pair<int,int> ksrc  = std::make_pair(cks,cke+1);
  std::pair<int,int> ksrc1 = std::make_pair(cks,cke+2);
  std::pair<int,int> jsrc  = std::make_pair(cjs,cje+1);
  std::pair<int,int> jsrc1 = std::make_pair(cjs,cje+2);
  std::pair<int,int> isrc  = std::make_pair(cis,cie+1);
  std::pair<int,int> isrc1 = std::make_pair(cis,cie+2);
  for (int m=0; m<old_nmb; ++m) {
    if (refine_flag.h_view(m) < -1) {  // only derefine if nleaf blocks flagged
      int srcm = m;
      for (int k=ks; k<=ke; k += cnx3) {
        std::pair<int,int> kdst  = std::make_pair(k,k+cnx3);
        std::pair<int,int> kdst1 = std::make_pair(k,k+cnx3+1);
        for (int j=js; j<=je; j += cnx2) {
          std::pair<int,int> jdst  = std::make_pair(j,j+cnx2);
          std::pair<int,int> jdst1 = std::make_pair(j,j+cnx2+1);
          for (int i=is; i<=ie; i += cnx1) {
            std::pair<int,int> idst  = std::make_pair(i,i+cnx1);
            std::pair<int,int> idst1 = std::make_pair(i,i+cnx1+1);
            auto src1 = Kokkos::subview(cb.x1f,srcm,ksrc,jsrc,isrc1);
            auto dst1 = Kokkos::subview( b.x1f,m   ,kdst,jdst,idst1);
            Kokkos::deep_copy(DevExeSpace(), dst1, src1);
            auto src2 = Kokkos::subview(cb.x2f,srcm,ksrc,jsrc1,isrc);
            auto dst2 = Kokkos::subview( b.x2f,m   ,kdst,jdst1,idst);
            Kokkos::deep_copy(DevExeSpace(), dst2, src2);
            auto src3 = Kokkos::subview(cb.x3f,srcm,ksrc1,jsrc,isrc);
            auto dst3 = Kokkos::subview( b.x3f,m   ,kdst1,jdst,idst);
            Kokkos::deep_copy(DevExeSpace(), dst3, src3);
            ++srcm;
          }
        }
      }
    }
  }

  return;
}

