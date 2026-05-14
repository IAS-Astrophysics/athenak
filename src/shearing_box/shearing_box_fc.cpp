//========================================================================================
// AthenaK astrophysical fluid dynamics code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file shearing_box_fc.cpp
//! \brief functions to pack/send and recv/unpack boundary values for face-centered (FC)
//! variables (magnetic fields) with shearing box boundaries.

#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <utility>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "bvals/bvals.hpp"
#include "shearing_box.hpp"
#include "remap_fluxes.hpp"

// Helpers for the shearing-box EMF correction below.  They reuse the integer x2 shear
// case logic from PackAndSendFC(), but index separate E2/E3 messages and request slots.
static constexpr int sbox_emf_nreq = 6;       // 3 y-segments x 2 EMF components
static constexpr int sbox_emf_tag_offset = 8; // avoid shearing-box vars tags

// Return which integer-shift case applies for the current x2 shear offset.
static int ShearCase(const int jr, const int nx2, const int ng) {
  if (jr < ng) {
    return 1;
  } else if (jr < (nx2-ng)) {
    return 2;
  }
  return 3;
}

// CASE 2 overlaps two x2 segments; CASE 1/3 overlap three.
static int NumShearBuffers(const int scase) {
  return (scase == 2) ? 2 : 3;
}

// Offset of the target MB for send-side copies/sends in each shear case.
static int SendShearJShift(const int n, const int ji, const int l, const int scase) {
  if (scase == 1) {
    return (n==0) ? (ji+l-1) : (l-1-ji);
  } else if (scase == 2) {
    return (n==0) ? (ji+l) : (l-1-ji);
  }
  return (n==0) ? (ji+l) : (l-2-ji);
}

// Receive/test/wait logic uses the inverse offset to find the sender MB.
static int RecvShearJShift(const int n, const int ji, const int l, const int scase) {
  return -SendShearJShift(n,ji,l,scase);
}

// Flatten (boundary MB, x2 segment, EMF component) into the EMF request arrays.
static int EMFReqIndex(const int m, const int l, const int c) {
  return sbox_emf_nreq*m + 2*l + c;
}

// Buffer ID used in MPI tags; offset keeps EMF messages distinct from vars messages.
static int EMFTagBuffer(const int n, const int l, const int c) {
  return sbox_emf_tag_offset + ((n<<3) | (l<<1) | c);
}

// Source/destination j-ranges for the integer shear shift.  With edge=true, the upper
// range is extended by one for E3, whose x2 extent is js..je+1 rather than js..je.
static void SetShearJRanges(const int n, const int scase, const int js, const int je,
                            const int ng, const int nx2, const int jr, const bool edge,
                            std::pair<int,int> jsrc[3], std::pair<int,int> jdst[3]) {
  const int e = edge ? 1 : 0;
  if (scase == 1) {
    if (n==0) {
      jsrc[0] = std::make_pair(js,js+ng-jr);
      jsrc[1] = std::make_pair(js,je+1+e);
      jsrc[2] = std::make_pair(je-(ng-1)-jr+e,je+1+e);
      jdst[0] = std::make_pair(je+1+jr+e,je+ng+1+e);
      jdst[1] = std::make_pair(js+jr,je+jr+1+e);
      jdst[2] = std::make_pair(js-ng,js+jr);
    } else {
      jsrc[0] = std::make_pair(js,js+ng+jr);
      jsrc[1] = std::make_pair(js,je+1+e);
      jsrc[2] = std::make_pair(je-(ng-1)+jr+e,je+1+e);
      jdst[0] = std::make_pair(je+1-jr+e,je+ng+1+e);
      jdst[1] = std::make_pair(js-jr,je-jr+1+e);
      jdst[2] = std::make_pair(js-ng,js-jr);
    }
  } else if (scase == 2) {
    if (n==0) {
      jsrc[0] = std::make_pair(js,je+ng-jr+1+e);
      jsrc[1] = std::make_pair(je-(ng-1)-jr+e,je+1+e);
      jdst[0] = std::make_pair(js+jr,je+ng+1+e);
      jdst[1] = std::make_pair(js-ng,js+jr);
    } else {
      jsrc[0] = std::make_pair(js,js+ng+jr);
      jsrc[1] = std::make_pair(js-ng+jr,je+1+e);
      jdst[0] = std::make_pair(je-jr+1+e,je+ng+1+e);
      jdst[1] = std::make_pair(js-ng,je-jr+1+e);
    }
  } else {
    const int dj = nx2-jr;
    if (n==0) {
      jsrc[0] = std::make_pair(js,js+ng+dj);
      jsrc[1] = std::make_pair(js,je+1+e);
      jsrc[2] = std::make_pair(je-(ng-1)+dj+e,je+1+e);
      jdst[0] = std::make_pair(je+1-dj+e,je+ng+1+e);
      jdst[1] = std::make_pair(js-dj,je-dj+1+e);
      jdst[2] = std::make_pair(js-ng,js-dj);
    } else {
      jsrc[0] = std::make_pair(js,js+ng-dj);
      jsrc[1] = std::make_pair(js,je+1+e);
      jsrc[2] = std::make_pair(je-(ng-1)-dj+e,je+1+e);
      jdst[0] = std::make_pair(je+1+dj+e,je+ng+1+e);
      jdst[1] = std::make_pair(js+dj,je+dj+1+e);
      jdst[2] = std::make_pair(js-ng,js+dj);
    }
  }
}

#if MPI_PARALLEL_ENABLED
static int EMFMessageSize(const std::pair<int,int> &jrng,
                          const std::pair<int,int> &krng) {
  return (jrng.second - jrng.first)*(krng.second - krng.first);
}

static void PackEMFMPIBuffer(const DvceArray5D<Real> &flux,
                             DvceArray2D<Real> &flux_mpi, const int req,
                             const int m, const int c,
                             const std::pair<int,int> &jrng,
                             const std::pair<int,int> &krng) {
  const int jl = jrng.first;
  const int kl = krng.first;
  const int nj = jrng.second - jrng.first;
  const int data_size = EMFMessageSize(jrng, krng);
  par_for("shemf_mpi_pack", DevExeSpace(), 0, data_size-1,
  KOKKOS_LAMBDA(const int p) {
    const int k = p/nj + kl;
    const int j = p - (k - kl)*nj + jl;
    flux_mpi(req,p) = flux(m,j,c,k,0);
  });
}

static void UnpackEMFMPIBuffer(const DvceArray2D<Real> &flux_mpi,
                               DvceArray5D<Real> &flux, const int req,
                               const int m, const int c,
                               const std::pair<int,int> &jrng,
                               const std::pair<int,int> &krng) {
  const int jl = jrng.first;
  const int kl = krng.first;
  const int nj = jrng.second - jrng.first;
  const int data_size = EMFMessageSize(jrng, krng);
  par_for("shemf_mpi_unpack", DevExeSpace(), 0, data_size-1,
  KOKKOS_LAMBDA(const int p) {
    const int k = p/nj + kl;
    const int j = p - (k - kl)*nj + jl;
    flux(m,j,c,k,0) = flux_mpi(req,p);
  });
}
#endif

//----------------------------------------------------------------------------------------
// ShearingBoxFC derived class constructor:

ShearingBoxFC::ShearingBoxFC(MeshBlockPack *pp, ParameterInput *pin) :
    ShearingBox(pp, pin) {
  // Allocate boundary buffers
  auto &indcs = pp->pmesh->mb_indcs;
  int ncells3 = indcs.nx3 + 2*indcs.ng;
  int ncells2 = indcs.nx2 + 2*indcs.ng;
  int ncells1 = indcs.ng;
  for (int n=0; n<2; ++n) {
    int nmb = std::max(1,nmb_x1bndry(n));
    Kokkos::realloc(sendbuf[n].vars,nmb,ncells2,3,ncells3,ncells1);
    Kokkos::realloc(recvbuf[n].vars,nmb,ncells2,3,ncells3,ncells1);
    Kokkos::realloc(sendbuf[n].flux,nmb,ncells2+1,2,ncells3+1,1);
    Kokkos::realloc(recvbuf[n].flux,nmb,ncells2+1,2,ncells3+1,1);
#if MPI_PARALLEL_ENABLED
    int max_emf_msg_size = (ncells2 + 1)*(indcs.nx3 + 1);
    Kokkos::realloc(sendbuf[n].flux_mpi,sbox_emf_nreq*nmb,max_emf_msg_size);
    Kokkos::realloc(recvbuf[n].flux_mpi,sbox_emf_nreq*nmb,max_emf_msg_size);
    if (nmb_x1bndry(n) > 0) {
      sendbuf[n].flux_req = new MPI_Request[sbox_emf_nreq*nmb_x1bndry(n)];
      recvbuf[n].flux_req = new MPI_Request[sbox_emf_nreq*nmb_x1bndry(n)];
      for (int m=0; m<sbox_emf_nreq*nmb_x1bndry(n); ++m) {
        sendbuf[n].flux_req[m] = MPI_REQUEST_NULL;
        recvbuf[n].flux_req[m] = MPI_REQUEST_NULL;
      }
    }
#endif
  }
}

//----------------------------------------------------------------------------------------
// ShearingBoxFC derived class destructor:

ShearingBoxFC::~ShearingBoxFC() {
#if MPI_PARALLEL_ENABLED
  for (int n=0; n<2; ++n) {
    if (nmb_x1bndry(n) > 0) {
      delete [] sendbuf[n].flux_req;
      delete [] recvbuf[n].flux_req;
    }
  }
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void ShearingBox::PackAndSendFC()
//! \brief Apply shearing sheet BCs to cell-centered variables, including MPI
//! MPI communications. Both the inner_x1 and outer_x1 boundaries are updated.
//! Called on the physics_bcs task after purely periodic BC communication is finished.

TaskStatus ShearingBoxFC::PackAndSendFC(DvceFaceFld4D<Real> &b,
                                        ReconstructionMethod rcon) {
  const auto &indcs = pmy_pack->pmesh->mb_indcs;
  const auto &ie = indcs.ie;
  const auto &js = indcs.js, &je = indcs.je;
  const auto &ks = indcs.ks, &ke = indcs.ke;
  const auto &ng = indcs.ng;

  // copy ghost zones at x1-faces into send buffer view
  // apply fractional cell offset to data in send buffers using conservative remap
  const auto &mbsize = pmy_pack->pmb->mb_size;
  int kl=ks, ku=ke;
  if (pmy_pack->pmesh->three_d) {kl -= ng; ku += ng;}
  int nj = indcs.nx2 + 2*ng;
  const int &gids_ = pmy_pack->gids;
  const Real &yshear_ = yshear;
  const auto &x1bndry_mbgid_ = x1bndry_mbgid;
  auto &sbuf = sendbuf;
  int scr_lvl=0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(nj) * 2;
  for (int n=0; n<2; ++n) {
    int nmb1 = nmb_x1bndry(n) - 1;
    par_for_outer("shrcc",DevExeSpace(),scr_size,scr_lvl,0,nmb1,0,2,kl,ku,0,(ng-1),
    KOKKOS_LAMBDA(TeamMember_t member,const int m,const int v,const int k,const int i) {
      ScrArray1D<Real> a_(member.team_scratch(scr_lvl), nj); // 1D slice of data
      ScrArray1D<Real> flx(member.team_scratch(scr_lvl), nj); // "flux" at faces
      int mm = x1bndry_mbgid_.d_view(n,m) - gids_;

      // Load scratch array
      if (n==0) {
        if (v==0) {
          par_for_inner(member, 0, nj-1, [&](const int j) {
            a_(j) = b.x1f(mm,k,j,i);
          });
        } else if (v==1) {
          par_for_inner(member, 0, nj-1, [&](const int j) {
            a_(j) = b.x2f(mm,k,j,i);
          });
        } else if (v==2) {
          par_for_inner(member, 0, nj-1, [&](const int j) {
            a_(j) = b.x3f(mm,k,j,i);
          });
        }
      } else if (n==1) {
        if (v==0) {
          par_for_inner(member, 0, nj-1, [&](const int j) {
            a_(j) = b.x1f(mm,k,j,(ie+2)+i);
          });
        } else if (v==1) {
          par_for_inner(member, 0, nj-1, [&](const int j) {
            a_(j) = b.x2f(mm,k,j,(ie+1)+i);
          });
        } else if (v==2) {
          par_for_inner(member, 0, nj-1, [&](const int j) {
            a_(j) = b.x3f(mm,k,j,(ie+1)+i);
          });
        }
      }
      member.team_barrier();

      // compute fractional offset
      Real eps = fmod(yshear_,(mbsize.d_view(mm).dx2))/(mbsize.d_view(mm).dx2);
      if (n == 1) {eps *= -1.0;}

      // Compute "fluxes" at shifted cell faces
      switch (rcon) {
        case ReconstructionMethod::dc:
          DC_RemapFlx(member, js, (je+1), eps, a_, flx);
          break;
        case ReconstructionMethod::plm:
          PLM_RemapFlx(member, js, (je+1), eps, a_, flx);
          break;
        case ReconstructionMethod::ppm4:
        case ReconstructionMethod::ppmx:
        case ReconstructionMethod::wenoz:
          PPMX_RemapFlx(member, js, (je+1), eps, a_, flx);
          break;
        default:
          break;
      }
      member.team_barrier();

      // update data in send buffer with fractional shift
      par_for_inner(member, js, je, [&](const int j) {
        sbuf[n].vars(m,j,v,k,i) = a_(j) - (flx(j+1) - flx(j));
      });
    });
  }

  // shift data at x1 boundaries by integer number of cells.
  // Algorithm is broken into three steps: case1/2/3.
  //  * Case1 and case3 are when the integer shift (jr<ng), so that the sending MB
  //    overlaps the ghost cells of the two neighbors, and so requires copy/send
  //    to three separate target MBs.
  //  * Case2 is when the sending MB straddles the boundary between MBs, and so requires
  //    copy/send to only two target MBs.
  // Use deep copy if target MB on same rank, or MPI sends if not
  Kokkos::fence();
  const int &nx2 = indcs.nx2;
  bool no_errors=true;
  for (int n=0; n<2; ++n) {
    for (int m=0; m<nmb_x1bndry(n); ++m) {
      int gid = x1bndry_mbgid.h_view(n,m);
      int mm = gid - pmy_pack->gids;
      // Find integer and fractional number of grids over which offset extends.
      // This assumes every grid has same number of cells in x2-direction!
      int joffset  = static_cast<int>(yshear/(mbsize.h_view(mm).dx2));
      int ji = joffset/nx2;
      int jr = joffset - ji*nx2;

      if (jr < ng) {               //--- CASE 1 (in my nomenclature)
        int tgid, trank;
        std::pair<int,int> jsrc[3],jdst[3];
        if (n==0) {
          jsrc[0] = std::make_pair(js,js+ng-jr);
          jsrc[1] = std::make_pair(js,je+1);
          jsrc[2] = std::make_pair(je-(ng-1)-jr,je+1);
          jdst[0] = std::make_pair(je+1+jr,je+ng+1);
          jdst[1] = std::make_pair(js+jr,je+jr+1);
          jdst[2] = std::make_pair(js-ng,js+jr);
        } else {
          jsrc[0] = std::make_pair(js,js+ng+jr);
          jsrc[1] = std::make_pair(js,je+1);
          jsrc[2] = std::make_pair(je-(ng-1)+jr,je+1);
          jdst[0] = std::make_pair(je+1-jr,je+ng+1);
          jdst[1] = std::make_pair(js-jr,je-jr+1);
          jdst[2] = std::make_pair(js-ng,js-jr);
        }
        // ix1 boundary: send to (target-1) through (target+1)
        // ox1 boundary: send to (target-1) through (target+1)
        for (int l=0; l<3; ++l) {
          int jshift;
          if (n==0) {jshift = ji+l-1;} else {jshift = l-1-ji;} // offset of target
          FindTargetMB(gid,jshift,tgid,trank);
          if (trank == global_variable::my_rank) {
            int tm = TargetIndex(n,tgid);
            using Kokkos::ALL;
            auto src = subview(sendbuf[n].vars,m, jsrc[l],ALL,ALL,ALL);
            auto dst = subview(recvbuf[n].vars,tm,jdst[l],ALL,ALL,ALL);
            deep_copy(DevExeSpace(), dst, src);
#if MPI_PARALLEL_ENABLED
          } else {
            using Kokkos::ALL;
            auto send_ptr = subview(sendbuf[n].vars,m,jsrc[l],ALL,ALL,ALL);
            // create tag using GID of *receiving* MeshBlock
            int tag = CreateBvals_MPI_Tag(tgid, ((n<<2) | l));
            int data_size = send_ptr.size();
            int ierr = MPI_Isend(send_ptr.data(), data_size, MPI_ATHENA_REAL, trank, tag,
                                 comm_sbox, &(sendbuf[n].vars_req[3*m + l]));
            if (ierr != MPI_SUCCESS) {no_errors=false;}
#endif
          }
        }
      } else if (jr < (nx2-ng)) {  //--- CASE 2
        int tgid, trank;
        std::pair<int,int> jsrc[2],jdst[2];
        if (n==0) {
          jsrc[0] = std::make_pair(js,je+ng-jr+1);
          jsrc[1] = std::make_pair(je-(ng-1)-jr,je+1);
          jdst[0] = std::make_pair(js+jr,je+ng+1);
          jdst[1] = std::make_pair(js-ng,js+jr);
        } else {
          jsrc[0] = std::make_pair(js,js+ng+jr);
          jsrc[1] = std::make_pair(js-ng+jr,je+1);
          jdst[0] = std::make_pair(je-jr+1,je+ng+1);
          jdst[1] = std::make_pair(js-ng,je-jr+1);
        }
        // ix1 boundary: send to (target  ) through (target+1)
        // ox1 boundary: send to (target-1) through (target  )
        for (int l=0; l<2; ++l) {
          int jshift;
          if (n==0) {jshift = ji+l;} else {jshift = l-1-ji;}
          FindTargetMB(gid,jshift,tgid,trank);
          if (trank == global_variable::my_rank) {
            int tm = TargetIndex(n,tgid);
            using Kokkos::ALL;
            auto src = subview(sendbuf[n].vars,m, jsrc[l],ALL,ALL,ALL);
            auto dst = subview(recvbuf[n].vars,tm,jdst[l],ALL,ALL,ALL);
            deep_copy(DevExeSpace(), dst, src);
#if MPI_PARALLEL_ENABLED
          } else {
            using Kokkos::ALL;
            auto send_ptr = subview(sendbuf[n].vars,m,jsrc[l],ALL,ALL,ALL);
            // create tag using GID of *receiving* MeshBlock
            int tag = CreateBvals_MPI_Tag(tgid, ((n<<2) | l));
            int data_size = send_ptr.size();
            int ierr = MPI_Isend(send_ptr.data(), data_size, MPI_ATHENA_REAL, trank, tag,
                                 comm_sbox, &(sendbuf[n].vars_req[3*m + l]));
            if (ierr != MPI_SUCCESS) {no_errors=false;}
#endif
          }
        }
      } else {                     //--- CASE 3
        int tgid, trank;
        std::pair<int,int> jsrc[3],jdst[3];
        if (n==0) {
          jsrc[0] = std::make_pair(js,js+ng+(nx2-jr));
          jsrc[1] = std::make_pair(js,je+1);
          jsrc[2] = std::make_pair(je-(ng-1)+(nx2-jr),je+1);
          jdst[0] = std::make_pair(je+1-(nx2-jr),je+ng+1);
          jdst[1] = std::make_pair(js-(nx2-jr),je-(nx2-jr)+1);
          jdst[2] = std::make_pair(js-ng,js-(nx2-jr));
        } else {
          jsrc[0] = std::make_pair(js,js+ng-(nx2-jr));
          jsrc[1] = std::make_pair(js,je+1);
          jsrc[2] = std::make_pair(je-(ng-1)-(nx2-jr),je+1);
          jdst[0] = std::make_pair(je+1+(nx2-jr),je+ng+1);
          jdst[1] = std::make_pair(js+(nx2-jr),je+(nx2-jr)+1);
          jdst[2] = std::make_pair(js-ng,js+(nx2-jr));
        }
        // ix1 boundary: send to (target  ) through (target+2)
        // ox1 boundary: send to (target-2) through (target  )
        for (int l=0; l<3; ++l) {
          int jshift;
          if (n==0) {jshift = ji+l;} else {jshift = l-2-ji;}
          FindTargetMB(gid,jshift,tgid,trank);
          if (trank == global_variable::my_rank) {
            int tm = TargetIndex(n,tgid);
            using Kokkos::ALL;
            auto src = subview(sendbuf[n].vars,m, jsrc[l],ALL,ALL,ALL);
            auto dst = subview(recvbuf[n].vars,tm,jdst[l],ALL,ALL,ALL);
            deep_copy(DevExeSpace(), dst, src);
#if MPI_PARALLEL_ENABLED
          } else {
            using Kokkos::ALL;
            auto send_ptr = subview(sendbuf[n].vars,m,jsrc[l],ALL,ALL,ALL);
            // create tag using GID of *receiving* MeshBlock
            int tag = CreateBvals_MPI_Tag(tgid, ((n<<2) | l));
            int data_size = send_ptr.size();
            int ierr = MPI_Isend(send_ptr.data(), data_size, MPI_ATHENA_REAL, trank, tag,
                                 comm_sbox, &(sendbuf[n].vars_req[3*m + l]));
#endif
          }
        }
      }
    }
  }
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in posting sends" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \!fn void ShearingBoxFC::RecvAndUnpackFC()
//! \brief Check MPI communication of boundary buffers for FC variables have finished,
//! then copy buffers into ghost zones. Shift has already been performed in
//! PackAndSendFC() function

TaskStatus ShearingBoxFC::RecvAndUnpackFC(DvceFaceFld4D<Real> &b) {
  // create local references for variables in kernel
  const auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int &ng = indcs.ng;
#if MPI_PARALLEL_ENABLED
  //----- STEP 1: check that recv boundary buffer communications have all completed
  const int &nx2 = indcs.nx2;
  bool bflag = false;
  bool no_errors=true;
  for (int n=0; n<2; ++n) {
    for (int m=0; m<nmb_x1bndry(n); ++m) {
      int gid = x1bndry_mbgid.h_view(n,m);
      int mm = gid - pmy_pack->gids;
      // Find integer and fractional number of grids over which offset extends.
      // This assumes every grid has same number of cells in x2-direction!
      int joffset  = static_cast<int>(yshear/(pmy_pack->pmb->mb_size.h_view(mm).dx2));
      int ji = joffset/nx2;
      int jr = joffset - ji*nx2;

      if (jr < ng) {               //--- CASE 1 (in my nomenclature)
        // ix1 boundary: receive from (target+1) through (target-1)
        // ox1 boundary: receive from (target+1) through (target-1)
        for (int l=0; l<3; ++l) {
          int jshift;
          if (n==0) {jshift = -(ji+l-1);} else {jshift = -(l-1-ji);} // offset of sender
          int sgid, srank;
          FindTargetMB(gid,jshift,sgid,srank);
          if (srank != global_variable::my_rank) {
            int test;
            int ierr = MPI_Test(&(recvbuf[n].vars_req[3*m + l]),&test,MPI_STATUS_IGNORE);
            if (ierr != MPI_SUCCESS) {no_errors=false;}
            if (!(static_cast<bool>(test))) {bflag = true;}
          }
        }
      } else if (jr < (nx2-ng)) {  //--- CASE 2
        // ix1 boundary: receive from (target  ) through (target-1)
        // ox1 boundary: receive from (target+1) through (target  )
        for (int l=0; l<2; ++l) {
          int jshift;
          if (n==0) {jshift = -(ji+l);} else {jshift = -(l-1-ji);} // offset of sender
          int sgid, srank;
          FindTargetMB(gid,jshift,sgid,srank);
          if (srank != global_variable::my_rank) {
            int test;
            int ierr = MPI_Test(&(recvbuf[n].vars_req[3*m + l]),&test,MPI_STATUS_IGNORE);
            if (ierr != MPI_SUCCESS) {no_errors=false;}
            if (!(static_cast<bool>(test))) {bflag = true;}
          }
        }
      } else {                     //--- CASE 3
        // ix1 boundary: send to (target  ) through (target+2)
        // ox1 boundary: send to (target-2) through (target  )
        for (int l=0; l<3; ++l) {
          int jshift;
          if (n==0) {jshift = -(ji+l);} else {jshift = -(l-2-ji);} // offset of sender
          int sgid, srank;
          FindTargetMB(gid,jshift,sgid,srank);
          if (srank != global_variable::my_rank) {
            int test;
            int ierr = MPI_Test(&(recvbuf[n].vars_req[3*m + l]),&test,MPI_STATUS_IGNORE);
            if (ierr != MPI_SUCCESS) {no_errors=false;}
            if (!(static_cast<bool>(test))) {bflag = true;}
          }
        }
      }
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in testing non-blocking receives"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // exit if recv boundary buffer communications have not completed
  if (bflag) {return TaskStatus::incomplete;}
#endif

  //----- STEP 2: communications have all completed, so unpack and apply shift
  // copy recv buffer view into ghost zones at x1-faces
  const int &ie = indcs.ie;
  int kl=indcs.ks, ku=indcs.ke;
  if (pmy_pack->pmesh->three_d) {kl -= ng; ku += ng;}
  int nj = indcs.nx2 + 2*ng;
  const int &gids_ = pmy_pack->gids;
  const auto &x1bndry_mbgid_ = x1bndry_mbgid;
  auto &rbuf = recvbuf;
  int scr_lvl=0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(nj) * 3;
  for (int n=0; n<2; ++n) {
    int nmb1 = nmb_x1bndry(n) - 1;
    par_for_outer("shrcc",DevExeSpace(),scr_size,scr_lvl,0,nmb1,0,2,kl,ku,0,(ng-1),
    KOKKOS_LAMBDA(TeamMember_t member,const int m,const int v,const int k,const int i) {
      int mm = x1bndry_mbgid_.d_view(n,m) - gids_;
      if (n==0) {
        if (v==0) {
          par_for_inner(member, 0, nj-1, [&](const int j) {
            b.x1f(mm,k,j,i) = rbuf[n].vars(m,j,v,k,i);
          });
          member.team_barrier();
        } else if (v==1) {
          par_for_inner(member, 0, nj-1, [&](const int j) {
            b.x2f(mm,k,j,i) = rbuf[n].vars(m,j,v,k,i);
          });
          member.team_barrier();
        } else if (v==2) {
          par_for_inner(member, 0, nj-1, [&](const int j) {
            b.x3f(mm,k,j,i) = rbuf[n].vars(m,j,v,k,i);
          });
          member.team_barrier();
        }
      } else {
        if (v==0) {
          par_for_inner(member, 0, nj-1, [&](const int j) {
            b.x1f(mm,k,j,(ie+2)+i) = rbuf[n].vars(m,j,v,k,i);
          });
          member.team_barrier();
        } else if (v==1) {
          par_for_inner(member, 0, nj-1, [&](const int j) {
            b.x2f(mm,k,j,(ie+1)+i) = rbuf[n].vars(m,j,v,k,i);
          });
          member.team_barrier();
        } else if (v==2) {
          par_for_inner(member, 0, nj-1, [&](const int j) {
            b.x3f(mm,k,j,(ie+1)+i) = rbuf[n].vars(m,j,v,k,i);
          });
          member.team_barrier();
        }
      }
    });
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void ShearingBoxFC::InitEMFRecv()
//! \brief With MPI, post receives for shearing-box EMF correction before constrained
//! transport. Receives use the same shearing offsets as FC magnetic boundary data, but
//! E2 and E3 are posted separately because their x2 edge extents differ.

TaskStatus ShearingBoxFC::InitEMFRecv() {
#if MPI_PARALLEL_ENABLED
  bool no_errors=true;
  const auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int &js = indcs.js, &je = indcs.je;
  const int &ks = indcs.ks, &ke = indcs.ke;
  const int &ng = indcs.ng;
  const int &nx2 = indcs.nx2;
  for (int n=0; n<2; ++n) {
    for (int m=0; m<nmb_x1bndry(n); ++m) {
      int gid = x1bndry_mbgid.h_view(n,m);
      int mm = gid - pmy_pack->gids;
      int joffset  = static_cast<int>(yshear/(pmy_pack->pmb->mb_size.h_view(mm).dx2));
      int ji = joffset/nx2;
      int jr = joffset - ji*nx2;
      int scase = ShearCase(jr, nx2, ng);
      int nbuff = NumShearBuffers(scase);

      for (int c=0; c<2; ++c) {
        std::pair<int,int> jsrc[3],jdst[3];
        SetShearJRanges(n,scase,js,je,ng,nx2,jr,(c==1),jsrc,jdst);
        std::pair<int,int> krng = (c==0) ? std::make_pair(ks,ke+2) :
                                           std::make_pair(ks,ke+1);
        for (int l=0; l<nbuff; ++l) {
          int jshift = RecvShearJShift(n,ji,l,scase);
          int sgid, srank;
          FindTargetMB(gid,jshift,sgid,srank);
          if (srank != global_variable::my_rank) {
            int req = EMFReqIndex(m,l,c);
            int tag = CreateBvals_MPI_Tag(gid, EMFTagBuffer(n,l,c));
            int data_size = EMFMessageSize(jdst[l], krng);
            auto recv_ptr = subview(recvbuf[n].flux_mpi, req,
                                    std::make_pair(0,data_size));
            int ierr = MPI_Irecv(recv_ptr.data(), data_size, MPI_ATHENA_REAL, srank, tag,
                                 comm_sbox, &(recvbuf[n].flux_req[req]));
            if (ierr != MPI_SUCCESS) {no_errors=false;}
          }
        }
      }
    }
  }
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in posting EMF receives" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void ShearingBoxFC::PackAndSendEMF()
//! \brief Pack, shift, and communicate E2/E3 on shearing x1 faces for EMF correction.

TaskStatus ShearingBoxFC::PackAndSendEMF(DvceEdgeFld4D<Real> &efld) {
  const auto &indcs = pmy_pack->pmesh->mb_indcs;
  const auto &is = indcs.is, &ie = indcs.ie;
  const auto &js = indcs.js, &je = indcs.je;
  const auto &ks = indcs.ks, &ke = indcs.ke;
  const auto &ng = indcs.ng;

  const auto &mbsize = pmy_pack->pmb->mb_size;
  const int &gids_ = pmy_pack->gids;
  const auto &x1bndry_mbgid_ = x1bndry_mbgid;
  auto &sbuf = sendbuf;
  for (int n=0; n<2; ++n) {
    int nmb1 = nmb_x1bndry(n) - 1;
    int i = (n==0) ? is : (ie+1);
    par_for("shemf_pack_e2", DevExeSpace(), 0, nmb1, ks, ke+1, js, je,
    KOKKOS_LAMBDA(const int m, const int k, const int j) {
      int mm = x1bndry_mbgid_.d_view(n,m) - gids_;
      sbuf[n].flux(m,j,0,k,0) = efld.x2e(mm,k,j,i);
    });
    par_for("shemf_pack_e3", DevExeSpace(), 0, nmb1, ks, ke, js, je+1,
    KOKKOS_LAMBDA(const int m, const int k, const int j) {
      int mm = x1bndry_mbgid_.d_view(n,m) - gids_;
      sbuf[n].flux(m,j,1,k,0) = efld.x3e(mm,k,j,i);
    });
  }

  Kokkos::fence();
  const int &nx2 = indcs.nx2;
#if MPI_PARALLEL_ENABLED
  bool no_errors=true;
#endif
  for (int n=0; n<2; ++n) {
    for (int m=0; m<nmb_x1bndry(n); ++m) {
      int gid = x1bndry_mbgid.h_view(n,m);
      int mm = gid - pmy_pack->gids;
      int joffset  = static_cast<int>(yshear/(mbsize.h_view(mm).dx2));
      int ji = joffset/nx2;
      int jr = joffset - ji*nx2;
      int scase = ShearCase(jr, nx2, ng);
      int nbuff = NumShearBuffers(scase);

      for (int c=0; c<2; ++c) {
        std::pair<int,int> jsrc[3],jdst[3];
        SetShearJRanges(n,scase,js,je,ng,nx2,jr,(c==1),jsrc,jdst);
        std::pair<int,int> crng = std::make_pair(c,c+1);
        std::pair<int,int> krng = (c==0) ? std::make_pair(ks,ke+2) :
                                           std::make_pair(ks,ke+1);
        for (int l=0; l<nbuff; ++l) {
          int jshift = SendShearJShift(n,ji,l,scase);
          int tgid, trank;
          FindTargetMB(gid,jshift,tgid,trank);
          if (trank == global_variable::my_rank) {
            int tm = TargetIndex(n,tgid);
            using Kokkos::ALL;
            auto src = subview(sendbuf[n].flux,m,jsrc[l],crng,krng,ALL);
            auto dst = subview(recvbuf[n].flux,tm,jdst[l],crng,krng,ALL);
            deep_copy(DevExeSpace(), dst, src);
#if MPI_PARALLEL_ENABLED
          } else {
            PackEMFMPIBuffer(sendbuf[n].flux, sendbuf[n].flux_mpi, EMFReqIndex(m,l,c),
                             m, c, jsrc[l], krng);
#endif
          }
        }
      }
    }
  }
#if MPI_PARALLEL_ENABLED
  Kokkos::fence();
  for (int n=0; n<2; ++n) {
    for (int m=0; m<nmb_x1bndry(n); ++m) {
      int gid = x1bndry_mbgid.h_view(n,m);
      int mm = gid - pmy_pack->gids;
      int joffset  = static_cast<int>(yshear/(mbsize.h_view(mm).dx2));
      int ji = joffset/nx2;
      int jr = joffset - ji*nx2;
      int scase = ShearCase(jr, nx2, ng);
      int nbuff = NumShearBuffers(scase);

      for (int c=0; c<2; ++c) {
        std::pair<int,int> jsrc[3],jdst[3];
        SetShearJRanges(n,scase,js,je,ng,nx2,jr,(c==1),jsrc,jdst);
        std::pair<int,int> krng = (c==0) ? std::make_pair(ks,ke+2) :
                                           std::make_pair(ks,ke+1);
        for (int l=0; l<nbuff; ++l) {
          int jshift = SendShearJShift(n,ji,l,scase);
          int tgid, trank;
          FindTargetMB(gid,jshift,tgid,trank);
          if (trank != global_variable::my_rank) {
            int req = EMFReqIndex(m,l,c);
            int tag = CreateBvals_MPI_Tag(tgid, EMFTagBuffer(n,l,c));
            int data_size = EMFMessageSize(jsrc[l], krng);
            auto send_ptr = subview(sendbuf[n].flux_mpi, req,
                                    std::make_pair(0,data_size));
            int ierr = MPI_Isend(send_ptr.data(), data_size, MPI_ATHENA_REAL, trank, tag,
                                 comm_sbox, &(sendbuf[n].flux_req[req]));
            if (ierr != MPI_SUCCESS) {no_errors=false;}
          }
        }
      }
    }
  }
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in posting EMF sends" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void ShearingBoxFC::RecvAndCorrectEMF()
//! \brief Wait for shearing-box EMF buffers, remap them, and average into E2/E3.

TaskStatus ShearingBoxFC::RecvAndCorrectEMF(DvceEdgeFld4D<Real> &efld,
                                            ReconstructionMethod rcon) {
  const auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int &is = indcs.is, &ie = indcs.ie;
  const int &js = indcs.js, &je = indcs.je;
  const int &ks = indcs.ks, &ke = indcs.ke;
  const int &ng = indcs.ng;
#if MPI_PARALLEL_ENABLED
  const int &nx2 = indcs.nx2;
  bool bflag = false;
  bool no_errors=true;
  for (int n=0; n<2; ++n) {
    for (int m=0; m<nmb_x1bndry(n); ++m) {
      int gid = x1bndry_mbgid.h_view(n,m);
      int mm = gid - pmy_pack->gids;
      int joffset  = static_cast<int>(yshear/(pmy_pack->pmb->mb_size.h_view(mm).dx2));
      int ji = joffset/nx2;
      int jr = joffset - ji*nx2;
      int scase = ShearCase(jr, nx2, ng);
      int nbuff = NumShearBuffers(scase);
      for (int c=0; c<2; ++c) {
        for (int l=0; l<nbuff; ++l) {
          int jshift = RecvShearJShift(n,ji,l,scase);
          int sgid, srank;
          FindTargetMB(gid,jshift,sgid,srank);
          if (srank != global_variable::my_rank) {
            int test;
            int ierr = MPI_Test(&(recvbuf[n].flux_req[EMFReqIndex(m,l,c)]),
                                &test, MPI_STATUS_IGNORE);
            if (ierr != MPI_SUCCESS) {no_errors=false;}
            if (!(static_cast<bool>(test))) {bflag = true;}
          }
        }
      }
    }
  }
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in testing EMF receives" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (bflag) {return TaskStatus::incomplete;}

  for (int n=0; n<2; ++n) {
    for (int m=0; m<nmb_x1bndry(n); ++m) {
      int gid = x1bndry_mbgid.h_view(n,m);
      int mm = gid - pmy_pack->gids;
      int joffset  = static_cast<int>(yshear/(pmy_pack->pmb->mb_size.h_view(mm).dx2));
      int ji = joffset/nx2;
      int jr = joffset - ji*nx2;
      int scase = ShearCase(jr, nx2, ng);
      int nbuff = NumShearBuffers(scase);
      for (int c=0; c<2; ++c) {
        std::pair<int,int> jsrc[3],jdst[3];
        SetShearJRanges(n,scase,js,je,ng,nx2,jr,(c==1),jsrc,jdst);
        std::pair<int,int> krng = (c==0) ? std::make_pair(ks,ke+2) :
                                           std::make_pair(ks,ke+1);
        for (int l=0; l<nbuff; ++l) {
          int jshift = RecvShearJShift(n,ji,l,scase);
          int sgid, srank;
          FindTargetMB(gid,jshift,sgid,srank);
          if (srank != global_variable::my_rank) {
            UnpackEMFMPIBuffer(recvbuf[n].flux_mpi, recvbuf[n].flux,
                               EMFReqIndex(m,l,c), m, c, jdst[l], krng);
          }
        }
      }
    }
  }
  Kokkos::fence();
#endif

  const int nj = indcs.nx2 + 2*ng;
  const int nj1 = nj+1;
  const int &gids_ = pmy_pack->gids;
  const auto &x1bndry_mbgid_ = x1bndry_mbgid;
  const auto &mbsize = pmy_pack->pmb->mb_size;
  const Real &yshear_ = yshear;
  auto &rbuf = recvbuf;
  int scr_lvl=0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(nj1) * 2;
  for (int n=0; n<2; ++n) {
    int nmb1 = nmb_x1bndry(n) - 1;
    par_for_outer("shemf_remap",DevExeSpace(),scr_size,scr_lvl,0,nmb1,0,1,ks,ke+1,
    KOKKOS_LAMBDA(TeamMember_t member,const int m,const int c,const int k) {
      if ((c==1) && (k > ke)) {return;}
      ScrArray1D<Real> a_(member.team_scratch(scr_lvl), nj1);
      ScrArray1D<Real> flx(member.team_scratch(scr_lvl), nj1);
      int mm = x1bndry_mbgid_.d_view(n,m) - gids_;

      if (c==0) {
        par_for_inner(member, 0, nj-1, [&](const int j) {
          a_(j) = rbuf[n].flux(m,j,c,k,0);
        });
      } else {
        par_for_inner(member, 0, nj, [&](const int j) {
          a_(j) = rbuf[n].flux(m,j,c,k,0);
        });
      }
      member.team_barrier();

      Real eps = fmod(yshear_,(mbsize.d_view(mm).dx2))/(mbsize.d_view(mm).dx2);
      if (n==1) {eps *= -1.0;}

      if (c==0) {
        switch (rcon) {
          case ReconstructionMethod::dc:
            DC_RemapFlx(member, js, (je+1), eps, a_, flx);
            break;
          case ReconstructionMethod::plm:
            PLM_RemapFlx(member, js, (je+1), eps, a_, flx);
            break;
          case ReconstructionMethod::ppm4:
          case ReconstructionMethod::ppmx:
          case ReconstructionMethod::wenoz:
            PPMX_RemapFlx(member, js, (je+1), eps, a_, flx);
            break;
          default:
            break;
        }
      } else {
        switch (rcon) {
          case ReconstructionMethod::dc:
            DC_RemapFlx(member, js, (je+2), eps, a_, flx);
            break;
          case ReconstructionMethod::plm:
            PLM_RemapFlx(member, js, (je+2), eps, a_, flx);
            break;
          case ReconstructionMethod::ppm4:
          case ReconstructionMethod::ppmx:
          case ReconstructionMethod::wenoz:
            PPMX_RemapFlx(member, js, (je+2), eps, a_, flx);
            break;
          default:
            break;
        }
      }
      member.team_barrier();

      if (c==0) {
        par_for_inner(member, js, je, [&](const int j) {
          rbuf[n].flux(m,j,c,k,0) = a_(j) - (flx(j+1) - flx(j));
        });
      } else {
        par_for_inner(member, js, je+1, [&](const int j) {
          rbuf[n].flux(m,j,c,k,0) = a_(j) - (flx(j+1) - flx(j));
        });
      }
    });
  }
  Kokkos::fence();

  for (int n=0; n<2; ++n) {
    int nmb1 = nmb_x1bndry(n) - 1;
    int i = (n==0) ? is : (ie+1);
    par_for("shemf_corr_e2", DevExeSpace(), 0, nmb1, ks, ke+1, js, je,
    KOKKOS_LAMBDA(const int m, const int k, const int j) {
      int mm = x1bndry_mbgid_.d_view(n,m) - gids_;
      efld.x2e(mm,k,j,i) = 0.5*(efld.x2e(mm,k,j,i) + rbuf[n].flux(m,j,0,k,0));
    });
    par_for("shemf_corr_e3", DevExeSpace(), 0, nmb1, ks, ke, js, je+1,
    KOKKOS_LAMBDA(const int m, const int k, const int j) {
      int mm = x1bndry_mbgid_.d_view(n,m) - gids_;
      efld.x3e(mm,k,j,i) = 0.5*(efld.x3e(mm,k,j,i) + rbuf[n].flux(m,j,1,k,0));
    });
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void ShearingBoxFC::ClearEMFRecv()
//! \brief Wait for all MPI receives associated with shearing-box EMF correction.

TaskStatus ShearingBoxFC::ClearEMFRecv() {
#if MPI_PARALLEL_ENABLED
  bool no_errors=true;
  const auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int &ng = indcs.ng;
  const int &nx2 = indcs.nx2;
  for (int n=0; n<2; ++n) {
    for (int m=0; m<nmb_x1bndry(n); ++m) {
      int gid = x1bndry_mbgid.h_view(n,m);
      int mm = gid - pmy_pack->gids;
      int joffset  = static_cast<int>(yshear/(pmy_pack->pmb->mb_size.h_view(mm).dx2));
      int ji = joffset/nx2;
      int jr = joffset - ji*nx2;
      int scase = ShearCase(jr, nx2, ng);
      int nbuff = NumShearBuffers(scase);
      for (int c=0; c<2; ++c) {
        for (int l=0; l<nbuff; ++l) {
          int jshift = RecvShearJShift(n,ji,l,scase);
          int sgid, srank;
          FindTargetMB(gid,jshift,sgid,srank);
          if (srank != global_variable::my_rank) {
            int ierr = MPI_Wait(&(recvbuf[n].flux_req[EMFReqIndex(m,l,c)]),
                                MPI_STATUS_IGNORE);
            if (ierr != MPI_SUCCESS) {no_errors=false;}
          }
        }
      }
    }
  }
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in waiting on EMF receives" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void ShearingBoxFC::ClearEMFSend()
//! \brief Wait for all MPI sends associated with shearing-box EMF correction.

TaskStatus ShearingBoxFC::ClearEMFSend() {
#if MPI_PARALLEL_ENABLED
  bool no_errors=true;
  const auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int &ng = indcs.ng;
  const int &nx2 = indcs.nx2;
  for (int n=0; n<2; ++n) {
    for (int m=0; m<nmb_x1bndry(n); ++m) {
      int gid = x1bndry_mbgid.h_view(n,m);
      int mm = gid - pmy_pack->gids;
      int joffset  = static_cast<int>(yshear/(pmy_pack->pmb->mb_size.h_view(mm).dx2));
      int ji = joffset/nx2;
      int jr = joffset - ji*nx2;
      int scase = ShearCase(jr, nx2, ng);
      int nbuff = NumShearBuffers(scase);
      for (int c=0; c<2; ++c) {
        for (int l=0; l<nbuff; ++l) {
          int jshift = SendShearJShift(n,ji,l,scase);
          int tgid, trank;
          FindTargetMB(gid,jshift,tgid,trank);
          if (trank != global_variable::my_rank) {
            int ierr = MPI_Wait(&(sendbuf[n].flux_req[EMFReqIndex(m,l,c)]),
                                MPI_STATUS_IGNORE);
            if (ierr != MPI_SUCCESS) {no_errors=false;}
          }
        }
      }
    }
  }
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in waiting on EMF sends" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}
