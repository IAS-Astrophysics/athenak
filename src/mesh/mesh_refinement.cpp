//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mesh_refinement.cpp
//! \brief Implements constructor and functions in MeshRefinement class.
//! Note while restriction functions for CC and FC data are implemented here, prolongation
//! is part of BVals classes.

#include <cstdint>  // int32_t
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
  refine_flag("rflag",pm->nmb_max),
  cyc_since_ref("cyc_since_ref",pm->nmb_max),
  nmb_created(0), nmb_deleted(0),
  ncycle_check_amr(1), ncycle_ref_inter(5),
  d_threshold_(0.0), dd_threshold_(0.0), dv_threshold_(0.0),
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
  if (pin->DoesParameterExist("mesh_refinement", "dvel_max")) {
    dd_threshold_ = pin->GetReal("mesh_refinement", "dvel_max");
    check_cons_ = true;
  }

  if (pm->adaptive) {  // allocate arrays for AMR
    nref = new int[global_variable::nranks];
    nderef = new int[global_variable::nranks];
  }
}

//----------------------------------------------------------------------------------------
// destructor

MeshRefinement::~MeshRefinement() {
  if (pmy_mesh->adaptive) { // deallocate arrays for AMR
    delete [] nref;
    delete [] nderef;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::AdaptiveMeshRefinement()
//! \brief Simple driver function for adaptive mesh refinement

void MeshRefinement::AdaptiveMeshRefinement(Driver *pdrive, ParameterInput *pin) {
  int nnew = 0, ndel = 0;
  UpdateMeshBlockTree(nnew, ndel);

  if (nnew != 0 || ndel != 0) { // at least one (de)refinement flagged
    RedistributeAndRefineMeshBlocks(pin, pmy_mesh->nmb_total + nnew - ndel);
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
//!   (1) gradient of density above a threshold value (hydro/MHD)
//!   (2) shear of velocity above a threshold value (hydro/MHD)
//!   (3) density max above a threshold value (hydro/MHD)
//!   (4) current density above a threshold (MHD)
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
  for (int m=0; m<(pmbp->pmesh->nmb_max); ++m) {
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

  // check Hydro/MHD refinement conditions for cons vars over all MeshBlocks (on device)
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

  // Check user-defined refinement condition(s), if any (on device)
  if (pmy_mesh->pgen->user_ref_func != nullptr) {
    pmy_mesh->pgen->user_ref_func(pmbp);
  }

  // sync refine_flag on device to host
  refine_flag.template modify<DevExeSpace>();
  refine_flag.template sync<HostMemSpace>();

  // Check (on host) for MeshBlocks at max/root level flagged for refine/derefine
  for (int m=0; m<nmb; ++m) {
    if (pmy_mesh->lloclist[m].level == pmy_mesh->max_level) {
      if (refine_flag.h_view(m) > 0) {refine_flag.h_view(m) = 0;}
    }
    if (pmy_mesh->lloclist[m].level == pmy_mesh->root_level) {
      if (refine_flag.h_view(m) < 0) {refine_flag.h_view(m) = 0;}
    }
  }

  // Check (on host) that MB has not been recently refined
  for (int m=0; m<nmb; ++m) {
    if (cyc_since_ref(m) < ncycle_ref_inter) {refine_flag.h_view(m) = 0;}
  }

  // sync refine_flag on host to device
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();

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

void MeshRefinement::RedistributeAndRefineMeshBlocks(ParameterInput *pin, int nmb_new) {
  // compute nleaf = number of leaf MeshBlocks per refined block
  int nleaf = 2;
  Mesh* pm = pmy_mesh;
  if (pm->two_d) nleaf = 4;
  if (pm->three_d) nleaf = 8;

  // Step 1. Create list of logical locations for new MBs, and newtoold list mapping
  // (new MB gid)-->(old gid) for all MBs.  Index of array is new gid, value is old gid.
  int nmb_old = pm->nmb_total;
  LogicalLocation *new_lloclist = new LogicalLocation[nmb_new];
  newtoold = new int[nmb_new];
  int new_nmb_total;
  pm->ptree->CreateMeshBlockList(new_lloclist, newtoold, new_nmb_total);
  if (new_nmb_total != nmb_new) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Number of MeshBlocks in new tree = " << new_nmb_total << " but expected "
        << "value = " << nmb_new << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Step 2.  Create oldtonew list mapping the previous gid to the current one for all MBs
  // Index of array is old gid, value os new gid.
  oldtonew = new int[nmb_old];
  oldtonew[0] = 0;
  int mb_idx = 1;
  for (int n=1; n<nmb_new; n++) {
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
  while (mb_idx < nmb_old) {
    oldtonew[mb_idx] = nmb_new-1;
    mb_idx++;
  }

  // Step 3. Calculate new load balance.  Initialize new cost array with the simplest
  // estimate possible: all the blocks are equal
  // TODO (@user): implement variable cost per MeshBlock as needed
  float *new_costlist = new float[nmb_new];
  int *new_ranklist = new int[nmb_new];
  int *new_gidslist = new int[global_variable::nranks];
  int *new_nmblist = new int[global_variable::nranks];

  for (int i=0; i<nmb_new; i++) {new_costlist[i] = 1.0;}
  // Return arguments over-write old values of [gidslist, nmblist] with values for new
  // Mesh. These arrays are dimensioned [nranks] so re-dimensioning not needed
  pm->LoadBalance(new_costlist, new_ranklist, new_gidslist, new_nmblist, new_nmb_total);


/***/
std::cout << std::endl << "nmb_old=" << nmb_old << "  nmb_new=" << nmb_new << "  Old to new:" << std::endl;
for (int n=0; n<nmb_old; ++n) {
std::cout << "n=" << n << "  new=" << oldtonew[n] << std::endl;
}
/***/


/***/
std::cout << std::endl << "New to old:" << std::endl;
for (int n=0; n<nmb_new; ++n) {
std::cout << "n=" << n << "  old=" << newtoold[n] << std::endl;
}
/***/

  hydro::Hydro* phydro = pm->pmb_pack->phydro;
  mhd::MHD* pmhd = pm->pmb_pack->pmhd;
  auto &nmb = pm->pmb_pack->nmb_thispack;                           // old nmb
  int mbs = pmy_mesh->gidslist[global_variable::my_rank];           // old gids
  int mbe = mbs + pmy_mesh->nmblist[global_variable::my_rank] - 1;  // old gide

  // Step 4. Restrict evolved variables for MBs flagged for derefinement
/**
  if (phydro != nullptr) {
    DerefineCC(phydro->u0, phydro->coarse_u0);
  }
***/

  // Step 5. Move evolved variables within view for any MB in which (new gid) > (old gid)
  for (int m=0; m<nmb; ++m) {
    int n = oldtonew[mbs + m] - mbs;
    if ((n-m) < 0) {
// std::cout << "Copy L: (m,n) = "<<m<<" "<<n<<std::endl;
      if (phydro != nullptr) {
        auto u0 = phydro->u0;
        auto src = Kokkos::subview(u0,m,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        auto dst = Kokkos::subview(u0,n,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        Kokkos::deep_copy(DevExeSpace(), dst, src);
      }
    }
  }

  // Step 6. Move evolved variables within view for any MB in which (new gid) < (old gid)
  for (int m=(nmb-1); m >= 0; --m) {
    int n = oldtonew[mbs + m] - mbs;
    if ((n-m) > 0) {
// std::cout << "Copy R: (m,n) = "<<m<<" "<<n<<std::endl;
      if (phydro != nullptr) {
        auto u0 = phydro->u0;
        auto src = Kokkos::subview(u0,m,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        auto dst = Kokkos::subview(u0,n,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        Kokkos::deep_copy(DevExeSpace(), dst, src);
      }
    }
  }

  // Step 7. Prolongate evolved variables for MBs flagged for refinement.
  if (phydro != nullptr) {
    RefineCC(new_nmb_total, phydro->u0, phydro->coarse_u0);
  }

  // Update data in Mesh/MeshBlockPack/MeshBlock classes with new grid properties
  delete [] pm->lloclist;
  delete [] pm->ranklist;
  delete [] pm->costlist;
  delete [] pm->gidslist;
  delete [] pm->nmblist;
  pm->lloclist = new_lloclist;
  pm->ranklist = new_ranklist;
  pm->costlist = new_costlist;
  pm->gidslist = new_gidslist;
  pm->nmblist  = new_nmblist;
  pm->nmb_total = new_nmb_total;
  pm->nmb_thisrank = pm->nmblist[global_variable::my_rank];

  pm->pmb_pack->gids = pm->gidslist[global_variable::my_rank];
  pm->pmb_pack->gide = pm->pmb_pack->gids + pm->nmblist[global_variable::my_rank] - 1;
  pm->pmb_pack->nmb_thispack = pm->pmb_pack->gide - pm->pmb_pack->gids + 1;

  delete (pm->pmb_pack->pmb);
  delete (pm->pmb_pack->pcoord);
  pm->pmb_pack->AddMeshBlocksAndCoordinates(pin, pm->mb_indcs);
  pm->pmb_pack->pmb->SetNeighbors(pm->ptree, pm->ranklist);

  // Update new number of cycles since refinement
  HostArray1D<int> new_cyc_since_ref("new_ncyc_ref",pm->nmb_max);
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

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::RefineCC
//! \brief Refines cell-centered variables in input view at any MeshBlock index m that is
//! flagged for refinement to the m-index locations which are immediately following,
//! overwriting any data located there. The data in these locations must already have been
//! copied to another location or sent to another rank via MPI.

void MeshRefinement::RefineCC(int nmb_new, DvceArray5D<Real> &a, DvceArray5D<Real> &ca) {
  int nvar = a.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR
  auto &nmb_old = pmy_mesh->pmb_pack->nmb_thispack;
  auto &indcs = pmy_mesh->mb_indcs;
  auto &is = indcs.is, &ie = indcs.ie;
  auto &js = indcs.js, &je = indcs.je;
  auto &ks = indcs.ks, &ke = indcs.ke;
  auto &cis = indcs.cis, &cie = indcs.cie;
  auto &cjs = indcs.cjs, &cje = indcs.cje;
  auto &cks = indcs.cks, &cke = indcs.cke;
  auto &cnx1 = indcs.cnx1, &cnx2 = indcs.cnx2, &cnx3 = indcs.cnx3;
  auto &multi_d = pmy_mesh->multi_d;
  auto &three_d = pmy_mesh->three_d;

  // First copy data in MBs to be refined to coarse arrays in target MBs
  std::pair kdst = std::make_pair(cks,cke+1);
  std::pair jdst = std::make_pair(cjs,cje+1);
  std::pair idst = std::make_pair(cis,cie+1);
/**
std::cout << "kdst = "<< kdst.first<<" "<<kdst.second << "  jdst = "<< jdst.first<<" "<<jdst.second <<"  idst = "<< idst.first<<" "<<idst.second << std::endl;
**/
  for (int m=0; m<nmb_old; ++m) {
    if (refine_flag.h_view(m) > 0) {
      int klim=2, jlim=2;
      if (!three_d) {klim=1;}
      if (!multi_d) {jlim=1;}

      int newm = oldtonew[m];
      int newn = newm;
      for (int k=0; k<klim; ++k) {
        std::pair ksrc = std::make_pair((ks+k*cnx3), (ks+(k+1)*cnx3));
        for (int j=0; j<jlim; ++j) {
          std::pair jsrc = std::make_pair((js+j*cnx2), (js+(j+1)*cnx2));
          for (int i=0; i<2; ++i) {
            std::pair isrc = std::make_pair((is+i*cnx1), (is+(i+1)*cnx1));
/**
std::cout << "(m,n)= "<<newm <<" "<<newn<<"  ksrc = "<< ksrc.first<<" "<<ksrc.second << "  jsrc = "<< jsrc.first<<" "<<jsrc.second <<"  isrc = "<< isrc.first<<" "<<isrc.second << std::endl;
**/
            auto src = Kokkos::subview(a, newm,Kokkos::ALL,ksrc,jsrc,isrc);
            auto dst = Kokkos::subview(ca,newn,Kokkos::ALL,kdst,jdst,idst);
            Kokkos::deep_copy(dst, src);
            ++newn;
          }
        }
      }
    }
  }

  // Now prolongate data in coarse arrays to fine arrays for all MBs being refined
  par_for("prolongCC",DevExeSpace(), 0,(nmb_new-1), 0,nvar-1, cks,cke, cjs,cje, cis,cie,
  KOKKOS_LAMBDA(const int m, const int v, const int k, const int j, const int i) {
    if (refine_flag.d_view(newtoold[m]) > 0) {
      // calculate x1-gradient using the min-mod limiter
      Real dl = ca(m,v,k,j,i  ) - ca(m,v,k,j,i-1);
      Real dr = ca(m,v,k,j,i+1) - ca(m,v,k,j,i  );
      Real dvar1 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));

      // calculate x2-gradient using the min-mod limiter
      Real dvar2 = 0.0;
      if (multi_d) {
        dl = ca(m,v,k,j  ,i) - ca(m,v,k,j-1,i);
        dr = ca(m,v,k,j+1,i) - ca(m,v,k,j  ,i);
        dvar2 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
      }

      // calculate x3-gradient using the min-mod limiter
      Real dvar3 = 0.0;
      if (three_d) {
        dl = ca(m,v,k  ,j,i) - ca(m,v,k-1,j,i);
        dr = ca(m,v,k+1,j,i) - ca(m,v,k  ,j,i);
        dvar3 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
      }

      // fine indices refer to target array
      int finei = 2*i - cis;  // correct when cis=is
      int finej = 2*j - cjs;  // correct when cjs=js
      int finek = 2*k - cks;  // correct when cks=ks

      a(m,v,finek,finej,finei  ) = ca(m,v,k,j,i) - dvar1 - dvar2 - dvar3;
      a(m,v,finek,finej,finei+1) = ca(m,v,k,j,i) + dvar1 - dvar2 - dvar3;
      if (multi_d) {
        a(m,v,finek,finej+1,finei  ) = ca(m,v,k,j,i) - dvar1 + dvar2 - dvar3;
        a(m,v,finek,finej+1,finei+1) = ca(m,v,k,j,i) + dvar1 + dvar2 - dvar3;
      }
      if (three_d) {
        a(m,v,finek+1,finej  ,finei  ) = ca(m,v,k,j,i) - dvar1 - dvar2 + dvar3;
        a(m,v,finek+1,finej  ,finei+1) = ca(m,v,k,j,i) + dvar1 - dvar2 + dvar3;
        a(m,v,finek+1,finej+1,finei  ) = ca(m,v,k,j,i) - dvar1 + dvar2 + dvar3;
        a(m,v,finek+1,finej+1,finei+1) = ca(m,v,k,j,i) + dvar1 + dvar2 + dvar3;
      }
    }
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::DerefineCC
//! \brief Derefines cell-centered variables in input view at any MeshBlock index m that
//! is flagged for derefinement to the m-index locations which are immediately following,
//! overwriting any data located there. The data in these locations must already have been
//! copied to another location or sent to another rank via MPI.

void MeshRefinement::DerefineCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca) {
  int nvar = a.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR
  auto &nmb_old = pmy_mesh->pmb_pack->nmb_thispack;
  auto &indcs = pmy_mesh->mb_indcs;
  auto &is = indcs.is, &ie = indcs.ie;
  auto &js = indcs.js, &je = indcs.je;
  auto &ks = indcs.ks, &ke = indcs.ke;
  auto &cis = indcs.cis, &cie = indcs.cie;
  auto &cjs = indcs.cjs, &cje = indcs.cje;
  auto &cks = indcs.cks, &cke = indcs.cke;
  auto &cnx1 = indcs.cnx1, &cnx2 = indcs.cnx2, &cnx3 = indcs.cnx3;
  auto &multi_d = pmy_mesh->multi_d;
  auto &three_d = pmy_mesh->three_d;

  // Copy data directly from coarse arrays in MBs to be refined to fine array in target MB
  std::pair ksrc = std::make_pair(cks,cke+1);
  std::pair jsrc = std::make_pair(cjs,cje+1);
  std::pair isrc = std::make_pair(cis,cie+1);
  for (int m=0; m<nmb_old; ++m) {
    if (refine_flag.h_view(m) < 0) {
std::cout << "Derefine: m = "<<m<<" flag = "<<refine_flag.h_view(m) << std::endl;
      int klim=2, jlim=2;
      if (!three_d) {klim=1;}
      if (!multi_d) {jlim=1;}

      int newm = oldtonew[m];
      for (int k=0; k<klim; ++k) {
        std::pair kdst = std::make_pair((ks+k*cnx3), (ks+(k+1)*cnx3));
        for (int j=0; j<jlim; ++j) {
          std::pair jdst = std::make_pair((js+j*cnx2), (js+(j+1)*cnx2));
          for (int i=0; i<2; ++i) {
            std::pair idst = std::make_pair((is+i*cnx1), (is+(i+1)*cnx1));
/**
std::cout << "(m,n)= "<<newm <<" "<<newn<<"  ksrc = "<< ksrc.first<<" "<<ksrc.second << "  jsrc = "<< jsrc.first<<" "<<jsrc.second <<"  isrc = "<< isrc.first<<" "<<isrc.second << std::endl;
**/
            auto src = Kokkos::subview(ca,m,   Kokkos::ALL,ksrc,jsrc,isrc);
            auto dst = Kokkos::subview(a, newm,Kokkos::ALL,kdst,jdst,idst);
            Kokkos::deep_copy(dst, src);
            ++m;
          }
        }
      }
    }
  }

  return;
}
