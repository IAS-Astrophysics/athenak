//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file zoom_data.cpp
//! \brief Implementation of constructor and basic functions in ZoomData class

#include <iostream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "radiation/radiation.hpp"
#include "cyclic_zoom/cyclic_zoom.hpp"

//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

ZoomData::ZoomData(CyclicZoom *pz, ParameterInput *pin) :
    pzoom(pz),
    u0("zcons",1,1,1,1,1),
    w0("zprim",1,1,1,1,1),
    coarse_u0("zccons",1,1,1,1,1),
    coarse_w0("zcprim",1,1,1,1,1),
    efld_pre("zefldp",1,1,1,1),
    efld_aft("zeflda",1,1,1,1),
    delta_efld("zdelta_efld",1,1,1,1),
    efld_buf("zefld_buf",1,1,1,1),
    i0("zi0",1,1,1,1,1),
    coarse_i0("zci0",1,1,1,1,1),
    zbuf("z_buffer",1),
    zdata("z_data",1) {
  // allocate memory for primitive variables
  pzmesh = pzoom->pzmesh;
  auto &indcs = pzoom->pmesh->mb_indcs;
  int &nzmb = pzmesh->nzmb_max_perdvce;
  auto pmbp = pzoom->pmesh->pmb_pack;
  nvars = 0;
  if (pmbp->phydro != nullptr) {
    nvars = pmbp->phydro->nhydro + pmbp->phydro->nscalars;
  } else if (pmbp->pmhd != nullptr) {
    nvars = pmbp->pmhd->nmhd + pmbp->pmhd->nscalars;
  }
  nangles = 0;
  if (pmbp->prad != nullptr) {
    nangles = pmbp->prad->prgeo->nangles;
  }
  // compute size of data per Zoom MeshBlock
  zmb_data_cnt = 0;
  MeshBlockDataSize();

  d_zoom = pin->GetOrAddReal(pzoom->block_name,"d_zoom",(FLT_MIN));
  p_zoom = pin->GetOrAddReal(pzoom->block_name,"p_zoom",(FLT_MIN));
  // allocate ZoomData arrays
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  int nccells1 = indcs.cnx1 + 2*(indcs.ng);
  int nccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
  int nccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;
  if (pmbp->phydro != nullptr || pmbp->pmhd != nullptr) {
    Kokkos::realloc(u0, nzmb, nvars, ncells3, ncells2, ncells1);
    Kokkos::realloc(w0, nzmb, nvars, ncells3, ncells2, ncells1);
    Kokkos::realloc(coarse_u0, nzmb, nvars, nccells3, nccells2, nccells1);
    Kokkos::realloc(coarse_w0, nzmb, nvars, nccells3, nccells2, nccells1);
  }

  if (pmbp->pmhd != nullptr) {
    // allocate electric fields
    Kokkos::realloc(efld_pre.x1e, nzmb, nccells3+1, nccells2+1, nccells1);
    Kokkos::realloc(efld_pre.x2e, nzmb, nccells3+1, nccells2, nccells1+1);
    Kokkos::realloc(efld_pre.x3e, nzmb, nccells3, nccells2+1, nccells1+1);

    // allocate electric fields just after zoom
    Kokkos::realloc(efld_aft.x1e, nzmb, nccells3+1, nccells2+1, nccells1);
    Kokkos::realloc(efld_aft.x2e, nzmb, nccells3+1, nccells2, nccells1+1);
    Kokkos::realloc(efld_aft.x3e, nzmb, nccells3, nccells2+1, nccells1+1);

    // allocate delta electric fields
    Kokkos::realloc(delta_efld.x1e, nzmb, nccells3+1, nccells2+1, nccells1);
    Kokkos::realloc(delta_efld.x2e, nzmb, nccells3+1, nccells2, nccells1+1);
    Kokkos::realloc(delta_efld.x3e, nzmb, nccells3, nccells2+1, nccells1+1);

    // allocate buffer for electric fields during zoom
    Kokkos::realloc(efld_buf.x1e, nzmb, nccells3+1, nccells2+1, nccells1);
    Kokkos::realloc(efld_buf.x2e, nzmb, nccells3+1, nccells2, nccells1+1);
    Kokkos::realloc(efld_buf.x3e, nzmb, nccells3, nccells2+1, nccells1+1);
  }

  // allocate memory for radiation
  if (pmbp->prad != nullptr) {
    Kokkos::realloc(i0,nzmb,nangles,ncells3,ncells2,ncells1);
    Kokkos::realloc(coarse_i0,nzmb,nangles,nccells3,nccells2,nccells1);
  }

  // allocate device and host arrays for data transfer and storage
  // DualView buffer: device side for packing, host side (pinned) for MPI send
  // Stores ZMBs currently owned by this rank before redistribution
  Kokkos::realloc(zbuf, nzmb * zmb_data_cnt);
  // Host data buffer: stores ZMBs for load balancing and restart
  // Note: zdata may contain completely different ZMBs than zbuf due to load balancing
  Kokkos::realloc(zdata, nzmb * zmb_data_cnt);

  Initialize();

#if MPI_PARALLEL_ENABLED
  // create unique communicators for AMR
  MPI_Comm_dup(MPI_COMM_WORLD, &zoom_comm);
#endif

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ZoomData::Initialize()
//! \brief Initialize ZoomData variables

void ZoomData::Initialize() {
  auto &indcs = pzoom->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 0;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 0;
  int nc1 = indcs.cnx1 + 2*ng;
  int nc2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*ng) : 0;
  int nc3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*ng) : 0;
  int &nzmb = pzmesh->nzmb_max_perdvce;

  auto &u0_ = u0;
  auto &w0_ = w0;
  auto &cu0 = coarse_u0;
  auto &cw0 = coarse_w0;
  auto e1 = efld_pre.x1e;
  auto e2 = efld_pre.x2e;
  auto e3 = efld_pre.x3e;
  auto e01 = efld_aft.x1e;
  auto e02 = efld_aft.x2e;
  auto e03 = efld_aft.x3e;
  auto de1 = delta_efld.x1e;
  auto de2 = delta_efld.x2e;
  auto de3 = delta_efld.x3e;
  auto ebuf1 = efld_buf.x1e;
  auto ebuf2 = efld_buf.x2e;
  auto ebuf3 = efld_buf.x3e;
  auto &i0_ = i0;
  auto &ci0 = coarse_i0;

  auto pmbp = pzoom->pmesh->pmb_pack;
  // initialize primitive and conserved variables
  if (pmbp->phydro != nullptr || pmbp->pmhd != nullptr) {
    auto peos = (pmbp->pmhd != nullptr)? pmbp->pmhd->peos : pmbp->phydro->peos;
    Real gm1 = peos->eos_data.gamma - 1.0;
    Real d0 = d_zoom;
    Real p0 = p_zoom;

    par_for("zoom_init", DevExeSpace(),0,nzmb-1,0,n3-1,0,n2-1,0,n1-1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      w0_(m,IDN,k,j,i) = d0;
      w0_(m,IM1,k,j,i) = 0.0;
      w0_(m,IM2,k,j,i) = 0.0;
      w0_(m,IM3,k,j,i) = 0.0;
      w0_(m,IEN,k,j,i) = p0/gm1;
    });

    par_for("zoom_init_c",DevExeSpace(),0,nzmb-1,0,nc3-1,0,nc2-1,0,nc1-1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      cw0(m,IDN,k,j,i) = d0;
      cw0(m,IM1,k,j,i) = 0.0;
      cw0(m,IM2,k,j,i) = 0.0;
      cw0(m,IM3,k,j,i) = 0.0;
      cw0(m,IEN,k,j,i) = p0/gm1;
    });

    // convert primitive to conserved variables
    peos->PrimToCons(w0_,u0_,0,n3-1,0,n2-1,0,n1-1);
    peos->PrimToCons(cw0,cu0,0,nc3-1,0,nc2-1,0,nc1-1);
  }

  // initialize electric fields to zero
  if (pmbp->pmhd != nullptr) {
    par_for("zoom_init_e1",DevExeSpace(),0,nzmb-1,0,nc3,0,nc2,0,nc1-1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      e1(m,k,j,i) = 0.0;
      e01(m,k,j,i) = 0.0;
      de1(m,k,j,i) = 0.0;
      ebuf1(m,k,j,i) = 0.0;
    });
    par_for("zoom_init_e2",DevExeSpace(),0,nzmb-1,0,nc3,0,nc2-1,0,nc1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      e2(m,k,j,i) = 0.0;
      e02(m,k,j,i) = 0.0;
      de2(m,k,j,i) = 0.0;
      ebuf2(m,k,j,i) = 0.0;
    });
    par_for("zoom_init_e3",DevExeSpace(),0,nzmb-1,0,nc3-1,0,nc2,0,nc1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      e3(m,k,j,i) = 0.0;
      e03(m,k,j,i) = 0.0;
      de3(m,k,j,i) = 0.0;
      ebuf3(m,k,j,i) = 0.0;
    });
  }

  // initialize intensity to zero
  if (pmbp->prad != nullptr) {
    int &nangles_ = nangles;
    par_for("zoom_init_i0", DevExeSpace(),0,nzmb-1,0,n3-1,0,n2-1,0,n1-1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      // Go through each angle
      for (int n=0; n<nangles_; ++n) {
        i0_(m,n,k,j,i) = 0.0;
      }
    });
    par_for("zoom_init_ci0",DevExeSpace(),0,nzmb-1,0,nc3-1,0,nc2-1,0,nc1-1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      // Go through each angle
      for (int n=0; n<nangles_; ++n) {
        ci0(m,n,k,j,i) = 0.0;
      }
    });
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ZoomData::MeshBlockDataSize()
//! \brief Calculate the count of data elements per MeshBlock needed for zooming

void ZoomData::MeshBlockDataSize() {
  int &cnt = zmb_data_cnt;
  cnt = 0;
  auto pmbp = pzoom->pmesh->pmb_pack;
  auto &indcs = pzoom->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  int nccells1 = indcs.cnx1 + 2*(indcs.ng);
  int nccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
  int nccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;
  if (pmbp->phydro != nullptr || pmbp->pmhd != nullptr) {
    cnt += 2 * nvars * ncells3 * ncells2 * ncells1; // u0 and w0
    cnt += 2 * nvars * nccells3 * nccells2 * nccells1; // coarse u0 and coarse w0
  }
  if (pmbp->pmhd != nullptr) {
    cnt += 3 * (nccells3+1) * (nccells2+1) * nccells1; // efld x1e
    cnt += 3 * (nccells3+1) * nccells2 * (nccells1+1); // efld x2e
    cnt += 3 * nccells3 * (nccells2+1) * (nccells1+1); // efld x3e
  }
  if (pmbp->prad != nullptr) {
    cnt += nangles * ncells3 * ncells2 * ncells1; // i0
    cnt += nangles * nccells3 * nccells2 * nccells1; // coarse i0
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ZoomData::ResetDataEC()
//! \brief Reset edge-centered data

void ZoomData::ResetDataEC(DvceEdgeFld4D<Real> ec) {
  int &nzmb = pzmesh->nzmb_thisdvce;
  auto e1 = ec.x1e;
  auto e2 = ec.x2e;
  auto e3 = ec.x3e;

  int nk = e1.extent_int(1), nj = e1.extent_int(2), ni = e1.extent_int(3);
  par_for("zoom_clear_e1",DevExeSpace(),0,nzmb-1,0,nk-1,0,nj-1,0,ni-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    e1(m,k,j,i) = 0.0;
  });
  nk = e2.extent_int(1), nj = e2.extent_int(2), ni = e2.extent_int(3);
  par_for("zoom_clear_e2",DevExeSpace(),0,nzmb-1,0,nk-1,0,nj-1,0,ni-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    e2(m,k,j,i) = 0.0;
  });
  nk = e3.extent_int(1), nj = e3.extent_int(2), ni = e3.extent_int(3);
  par_for("zoom_clear_e3",DevExeSpace(),0,nzmb-1,0,nk-1,0,nj-1,0,ni-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    e3(m,k,j,i) = 0.0;
  });

  return;
}
