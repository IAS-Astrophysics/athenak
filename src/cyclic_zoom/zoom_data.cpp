//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file zoom_data.cpp
//  \brief implementation of constructor and functions in CyclicZoom class

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "eos/ideal_c2p_hyd.hpp"
#include "eos/ideal_c2p_mhd.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "cyclic_zoom/cyclic_zoom.hpp"

//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

ZoomData::ZoomData(CyclicZoom *pz, ParameterInput *pin) :
    pzoom(pz),
    u0("zcons",1,1,1,1,1),
    w0("zprim",1,1,1,1,1),
    coarse_u0("czcons",1,1,1,1,1),
    coarse_w0("czprim",1,1,1,1,1),
    efld_pre("zefldp",1,1,1,1),
    efld_aft("zeflda",1,1,1,1),
    delta_efld("zdelta_efld",1,1,1,1),
    efld_buf("zefld_buf",1,1,1,1),
    zbuf("z_buffer",1),
    zdata("z_data",1)
  {
  // allocate memory for primitive variables
  pzmesh = pzoom->pzmesh;
  auto &indcs = pzoom->pmesh->mb_indcs;
  int &nzmb = pzmesh->nzmb_max_perdvce;
  int &nlevels = pzmesh->nlevels;
  auto pmbp = pzoom->pmesh->pmb_pack;
  bool is_mhd = (pmbp->pmhd != nullptr);
  nvars = 0;
  if (!is_mhd) {
    nvars = pmbp->phydro->nhydro + pmbp->phydro->nscalars;
  } else {
    nvars = pmbp->pmhd->nmhd + pmbp->pmhd->nscalars;
  }
  d_zoom = pin->GetOrAddReal(pzoom->block_name,"d_zoom",(FLT_MIN));
  p_zoom = pin->GetOrAddReal(pzoom->block_name,"p_zoom",(FLT_MIN));
  // compute size of data per Zoom MeshBlock
  MeshBlockDataSize();
  // allocate ZoomData arrays
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  Kokkos::realloc(u0, nzmb, nvars, ncells3, ncells2, ncells1);
  Kokkos::realloc(w0, nzmb, nvars, ncells3, ncells2, ncells1);
  int nccells1 = indcs.cnx1 + 2*(indcs.ng);
  int nccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
  int nccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;
  Kokkos::realloc(coarse_u0, nzmb, nvars, nccells3, nccells2, nccells1);
  Kokkos::realloc(coarse_w0, nzmb, nvars, nccells3, nccells2, nccells1);

  if (pzoom->pmesh->pmb_pack->pmhd != nullptr) {
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

  // allocate device and host arrays for data transfer and storage
  // DualView buffer: device side for packing, host side (pinned) for MPI send
  // Stores ZMBs currently owned by this rank before redistribution
  // zbuf.resize(nzmb * zmb_data_cnt);
  Kokkos::realloc(zbuf, nzmb * zmb_data_cnt);
  // Host receive buffer: stores ZMBs that will be owned by this rank after AMR/redistribution
  // Note: zdata may contain completely different ZMBs than zbuf due to load balancing
  Kokkos::realloc(zdata, nzmb * zmb_data_cnt);
  // ndata = pzdata->zmb_data_cnt * pzmesh->nzmb_max_perhost;
  // Kokkos::realloc(send_data, ndata);
  // Kokkos::realloc(recv_data, ndata);

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

void ZoomData::Initialize()
{
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
  bool is_mhd = (pzoom->pmesh->pmb_pack->pmhd != nullptr);
  auto peos = (is_mhd)? pzoom->pmesh->pmb_pack->pmhd->peos : pzoom->pmesh->pmb_pack->phydro->peos;
  Real gm1 = peos->eos_data.gamma - 1.0;

  Real d0 = d_zoom;
  Real p0 = p_zoom;

  par_for("zoom_init", DevExeSpace(),0,nzmb-1,0,n3-1,0,n2-1,0,n1-1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
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

  // initialize electric fields to zero
  if (pzoom->pmesh->pmb_pack->pmhd != nullptr) {
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

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ZoomData::MeshBlockDataSize()
//! \brief Calculate the count of data elements per MeshBlock needed for zooming

//TODO(@mhguo): consider magnetic fields, think if int is enough, maybe IOWrapperSizeT?
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
  cnt += 2 * nvars * ncells3 * ncells2 * ncells1; // u0 and w0
  cnt += 2 * nvars * nccells3 * nccells2 * nccells1; // coarse u0 and coarse w0
  if (pmbp->pmhd != nullptr) {
    cnt += 3 * (nccells3+1) * (nccells2+1) * nccells1; // efld x1e
    cnt += 3 * (nccells3+1) * nccells2 * (nccells1+1); // efld x2e
    cnt += 3 * nccells3 * (nccells2+1) * (nccells1+1); // efld x3e
  }
  // TODO(@mhguo): add radiation variables later
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

//----------------------------------------------------------------------------------------
//! \fn void ZoomData::DumpData()
//! \brief dump zoom data to file

// TODO: dumping on a single rank now, should consider parallel dumping
void ZoomData::DumpData() {
  if (global_variable::my_rank == 0) {
    std::cout << "CyclicZoom: Dumping data" << std::endl;
    auto pm = pzoom->pmesh;
    auto &indcs = pm->mb_indcs;
    int nccells1 = indcs.cnx1 + 2*(indcs.ng);
    int nccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
    int nccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;
    int nzmb = pzmesh->nzmb_thisdvce;  // Use actual count, not max

    std::string fname;
    fname.assign("CyclicZoom");
    // add pmesh ncycles
    fname.append(".");
    fname.append(std::to_string(pm->ncycle));
    fname.append(".dat");
    FILE *pfile;
    if ((pfile = std::fopen(fname.c_str(), "wb")) == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Error output file could not be opened" <<std::endl;
      std::exit(EXIT_FAILURE);
    }
    int datasize = sizeof(Real);
    
    // Create host mirrors and copy data from device
    auto h_coarse_w0 = Kokkos::create_mirror_view(coarse_w0);
    Kokkos::deep_copy(h_coarse_w0, coarse_w0);
    
    auto h_efld_pre_x1e = Kokkos::create_mirror_view(efld_pre.x1e);
    auto h_efld_pre_x2e = Kokkos::create_mirror_view(efld_pre.x2e);
    auto h_efld_pre_x3e = Kokkos::create_mirror_view(efld_pre.x3e);
    Kokkos::deep_copy(h_efld_pre_x1e, efld_pre.x1e);
    Kokkos::deep_copy(h_efld_pre_x2e, efld_pre.x2e);
    Kokkos::deep_copy(h_efld_pre_x3e, efld_pre.x3e);
    
    auto h_efld_aft_x1e = Kokkos::create_mirror_view(efld_aft.x1e);
    auto h_efld_aft_x2e = Kokkos::create_mirror_view(efld_aft.x2e);
    auto h_efld_aft_x3e = Kokkos::create_mirror_view(efld_aft.x3e);
    Kokkos::deep_copy(h_efld_aft_x1e, efld_aft.x1e);
    Kokkos::deep_copy(h_efld_aft_x2e, efld_aft.x2e);
    Kokkos::deep_copy(h_efld_aft_x3e, efld_aft.x3e);
    
    // Write host data to file
    IOWrapperSizeT cnt = nzmb*nvars*(nccells3)*(nccells2)*(nccells1);
    std::fwrite(h_coarse_w0.data(),datasize,cnt,pfile);
    
    cnt = nzmb*(nccells3+1)*(nccells2+1)*(nccells1);
    std::fwrite(h_efld_pre_x1e.data(),datasize,cnt,pfile);
    cnt = nzmb*(nccells3+1)*(nccells2)*(nccells1+1);
    std::fwrite(h_efld_pre_x2e.data(),datasize,cnt,pfile);
    cnt = nzmb*(nccells3)*(nccells2+1)*(nccells1+1);
    std::fwrite(h_efld_pre_x3e.data(),datasize,cnt,pfile);
    
    cnt = nzmb*(nccells3+1)*(nccells2+1)*(nccells1);
    std::fwrite(h_efld_aft_x1e.data(),datasize,cnt,pfile);
    cnt = nzmb*(nccells3+1)*(nccells2)*(nccells1+1);
    std::fwrite(h_efld_aft_x2e.data(),datasize,cnt,pfile);
    cnt = nzmb*(nccells3)*(nccells2+1)*(nccells1+1);
    std::fwrite(h_efld_aft_x3e.data(),datasize,cnt,pfile);
    
    std::fclose(pfile);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ZoomData::StoreDataToZoomData()
//! \brief Store data from MeshBlock m to zoom data zm

void ZoomData::StoreDataToZoomData(int zm, int m) {
  auto pmesh = pzoom->pmesh;
  DvceArray5D<Real> u, w;
  if (pmesh->pmb_pack->phydro != nullptr) {
    u = pmesh->pmb_pack->phydro->u0;
    w = pmesh->pmb_pack->phydro->w0;
    StoreCCData(zm, m, u, w);
  }
  if (pmesh->pmb_pack->pmhd != nullptr) {
    u = pmesh->pmb_pack->pmhd->u0;
    w = pmesh->pmb_pack->pmhd->w0;
    StoreCCData(zm, m, u, w);
    StoreCoarseHydroData(zm, m, u, w);
    DvceEdgeFld4D<Real> efld = pmesh->pmb_pack->pmhd->efld;
    StoreEFieldsBeforeAMR(zm, m, efld);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ZoomData::StoreCCData()
//! \brief Store cell-centered data from MeshBlock m to zoom meshblock zm

void ZoomData::StoreCCData(int zm, int m, DvceArray5D<Real> u, DvceArray5D<Real> w) {
  auto pmesh = pzoom->pmesh;
  auto des_slice = Kokkos::subview(u0, Kokkos::make_pair(zm,zm+1),
                                   Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
  auto src_slice = Kokkos::subview(u, Kokkos::make_pair(m,m+1),
                                   Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
  Kokkos::deep_copy(des_slice, src_slice);
  des_slice = Kokkos::subview(w0, Kokkos::make_pair(zm,zm+1),
                              Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
  src_slice = Kokkos::subview(w, Kokkos::make_pair(m,m+1),
                              Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
  Kokkos::deep_copy(des_slice, src_slice);
  // now do coarse data by averaging fine data
  auto &indcs = pzoom->pmesh->mb_indcs;
  int &cis = indcs.cis;  int &cie  = indcs.cie;
  int &cjs = indcs.cjs;  int &cje  = indcs.cje;
  int &cks = indcs.cks;  int &cke  = indcs.cke;
  int nvar = nvars;
  auto cu = coarse_u0, cw = coarse_w0;
  int hng = indcs.ng / 2;
  // TODO(@mhguo): may think whether we need to include ghost zones
  // TODO(@mhguo): 1D and 2D cases are not tested yet!
  // restrict in 1D
  if (pmesh->one_d) {
    par_for("zoom-restrictCC-1D",DevExeSpace(), 0, nvar-1, cis-hng, cie+hng,
    KOKKOS_LAMBDA(const int n, const int i) {
      int finei = 2*i - cis;  // correct when cis=is
      cu(zm,n,cks,cjs,i) = 0.5*(u(m,n,cks,cjs,finei) + u(m,n,cks,cjs,finei+1));
      cw(zm,n,cks,cjs,i) = 0.5*(w(m,n,cks,cjs,finei) + w(m,n,cks,cjs,finei+1));
    });
  // restrict in 2D
  } else if (pmesh->two_d) {
    par_for("zoom-restrictCC-2D",DevExeSpace(), 0, nvar-1,
            cjs-hng, cje+hng, cis-hng, cie+hng,
    KOKKOS_LAMBDA(const int n, const int j, const int i) {
      int finei = 2*i - cis;  // correct when cis=is
      int finej = 2*j - cjs;  // correct when cjs=js
      cu(zm,n,cks,j,i) = 0.25*(u(m,n,cks,finej  ,finei) + u(m,n,cks,finej  ,finei+1)
                             + u(m,n,cks,finej+1,finei) + u(m,n,cks,finej+1,finei+1));
      cw(zm,n,cks,j,i) = 0.25*(w(m,n,cks,finej  ,finei) + w(m,n,cks,finej  ,finei+1)
                             + w(m,n,cks,finej+1,finei) + w(m,n,cks,finej+1,finei+1));
    });
  // restrict in 3D
  } else {
    par_for("zoom-restrictCC-3D",DevExeSpace(), 0, nvar-1, cks-hng, cke+hng,
            cjs-hng, cje+hng, cis-hng, cie+hng,
    KOKKOS_LAMBDA(const int n, const int k, const int j, const int i) {
      int finei = 2*i - cis;  // correct if cis = is
      int finej = 2*j - cjs;  // correct if cjs = js
      int finek = 2*k - cks;  // correct if cks = ks
      cu(zm,n,k,j,i) =
                 0.125*(u(m,n,finek  ,finej  ,finei) + u(m,n,finek  ,finej  ,finei+1)
                      + u(m,n,finek  ,finej+1,finei) + u(m,n,finek  ,finej+1,finei+1)
                      + u(m,n,finek+1,finej,  finei) + u(m,n,finek+1,finej,  finei+1)
                      + u(m,n,finek+1,finej+1,finei) + u(m,n,finek+1,finej+1,finei+1));
      cw(zm,n,k,j,i) =
                 0.125*(w(m,n,finek  ,finej  ,finei) + w(m,n,finek  ,finej  ,finei+1)
                      + w(m,n,finek  ,finej+1,finei) + w(m,n,finek  ,finej+1,finei+1)
                      + w(m,n,finek+1,finej,  finei) + w(m,n,finek+1,finej,  finei+1)
                      + w(m,n,finek+1,finej+1,finei) + w(m,n,finek+1,finej+1,finei+1));
    });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ZoomData::StoreCoarseHydroData()
//! \brief Store coarse-grained hydro conserved variables from mb m to zoom mb zm
//! only for mhd case

void ZoomData::StoreCoarseHydroData(int zm, int m, DvceArray5D<Real> u0_, DvceArray5D<Real> w0_) {
  auto pmbp = pzoom->pmesh->pmb_pack;
  if (pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "StoreCoarseHydroData only works for MHD case" <<std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto &indcs = pzoom->pmesh->mb_indcs;
  auto &size = pzoom->pmesh->pmb_pack->pmb->mb_size;
  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;
  int &cis = indcs.cis;  int &cie  = indcs.cie;
  int &cjs = indcs.cjs;  int &cje  = indcs.cje;
  int &cks = indcs.cks;  int &cke  = indcs.cke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int cnx1 = indcs.cnx1, cnx2 = indcs.cnx2, cnx3 = indcs.cnx3;
  // DvceArray5D<Real> u0_, w0_;
  bool is_gr = pmbp->pcoord->is_general_relativistic;
  auto eos = pmbp->pmhd->peos->eos_data;
  auto cw = coarse_w0;
  Real &x1min = size.h_view(m).x1min;
  Real &x1max = size.h_view(m).x1max;
  Real &x2min = size.h_view(m).x2min;
  Real &x2max = size.h_view(m).x2max;
  Real &x3min = size.h_view(m).x3min;
  Real &x3max = size.h_view(m).x3max;
  bool flat = true;
  Real spin = 0.0;
  if (is_gr) {
    flat = pmbp->pcoord->coord_data.is_minkowski;
    spin = pmbp->pcoord->coord_data.bh_spin;
  }
  // TODO(@mhguo): should we consider 2D and 1D cases?
  // TODO(@mhguo): may think whether we need to include ghost zones
  int hng = indcs.ng / 2;
  par_for("zoom-update-cwu",DevExeSpace(), cks-hng,cke+hng, cjs-hng,cje+hng, cis-hng,cie+hng,
  KOKKOS_LAMBDA(const int ck, const int cj, const int ci) {
    int fi = 2*ci - cis;  // correct when cis=is
    int fj = 2*cj - cjs;  // correct when cjs=js
    int fk = 2*ck - cks;  // correct when cks=ks
    cw(zm,IDN,ck,cj,ci) = 0.0;
    cw(zm,IM1,ck,cj,ci) = 0.0;
    cw(zm,IM2,ck,cj,ci) = 0.0;
    cw(zm,IM3,ck,cj,ci) = 0.0;
    cw(zm,IEN,ck,cj,ci) = 0.0;
    Real glower[4][4], gupper[4][4];
    // Step 1: compute coarse-grained hydro conserved variables
    for (int ii=0; ii<2; ++ii) {
      for (int jj=0; jj<2; ++jj) {
        for (int kk=0; kk<2; ++kk) {
          // Load single state of primitive variables
          HydPrim1D w;
          w.d  = w0_(m,IDN,fk+kk,fj+jj,fi+ii);
          w.vx = w0_(m,IVX,fk+kk,fj+jj,fi+ii);
          w.vy = w0_(m,IVY,fk+kk,fj+jj,fi+ii);
          w.vz = w0_(m,IVZ,fk+kk,fj+jj,fi+ii);
          w.e  = w0_(m,IEN,fk+kk,fj+jj,fi+ii);

          // call p2c function
          HydCons1D u;
          if (is_gr) {
            Real x1v = CellCenterX(fi+ii-is, nx1, x1min, x1max);
            Real x2v = CellCenterX(fj+jj-js, nx2, x2min, x2max);
            Real x3v = CellCenterX(fk+kk-ks, nx3, x3min, x3max);
            ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);
            SingleP2C_IdealGRHyd(glower, gupper, w, eos.gamma, u);
          } else {
            SingleP2C_IdealHyd(w, u);
          }

          // store conserved quantities using cw
          cw(zm,IDN,ck,cj,ci) += 0.125*u.d;
          cw(zm,IM1,ck,cj,ci) += 0.125*u.mx;
          cw(zm,IM2,ck,cj,ci) += 0.125*u.my;
          cw(zm,IM3,ck,cj,ci) += 0.125*u.mz;
          cw(zm,IEN,ck,cj,ci) += 0.125*u.e;
        }
      }
    }
    // Step 2: convert coarse-grained hydro conserved variables to primitive variables
    // Shall we add excision?
    // load single state conserved variables
    HydCons1D u;
    u.d  = cw(zm,IDN,ck,cj,ci);
    u.mx = cw(zm,IM1,ck,cj,ci);
    u.my = cw(zm,IM2,ck,cj,ci);
    u.mz = cw(zm,IM3,ck,cj,ci);
    u.e  = cw(zm,IEN,ck,cj,ci);

    HydPrim1D w;
    if (is_gr) {
      // Extract components of metric
      Real x1v = CellCenterX(ci-cis, cnx1, x1min, x1max);
      Real x2v = CellCenterX(cj-cjs, cnx2, x2min, x2max);
      Real x3v = CellCenterX(ck-cks, cnx3, x3min, x3max);
      ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);

      HydCons1D u_sr;
      Real s2;
      TransformToSRHyd(u,glower,gupper,s2,u_sr);
      bool dfloor_used=false, efloor_used=false;
      bool c2p_failure=false;
      int iter_used=0;
      SingleC2P_IdealSRHyd(u_sr, eos, s2, w,
                        dfloor_used, efloor_used, c2p_failure, iter_used);
      // apply velocity ceiling if necessary
      Real tmp = glower[1][1]*SQR(w.vx)
                + glower[2][2]*SQR(w.vy)
                + glower[3][3]*SQR(w.vz)
                + 2.0*glower[1][2]*w.vx*w.vy + 2.0*glower[1][3]*w.vx*w.vz
                + 2.0*glower[2][3]*w.vy*w.vz;
      Real lor = sqrt(1.0+tmp);
      if (lor > eos.gamma_max) {
        Real factor = sqrt((SQR(eos.gamma_max)-1.0)/(SQR(lor)-1.0));
        w.vx *= factor;
        w.vy *= factor;
        w.vz *= factor;
      }
    } else {
      bool dfloor_used=false, efloor_used=false, tfloor_used=false;
      SingleC2P_IdealHyd(u, eos, w, dfloor_used, efloor_used, tfloor_used);
    }
    cw(zm,IDN,ck,cj,ci) = w.d;
    cw(zm,IVX,ck,cj,ci) = w.vx;
    cw(zm,IVY,ck,cj,ci) = w.vy;
    cw(zm,IVZ,ck,cj,ci) = w.vz;
    cw(zm,IEN,ck,cj,ci) = w.e;
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn ZoomData::LoadDataFromZoomData()
//! \brief Load data from zoom data zm to MeshBlock m

// TODO(@mhguo): is this what you want?
void ZoomData::LoadDataFromZoomData(int m, int zm) {
  LoadCCData(m, zm);
  LoadHydroData(m, zm);
  // TODO(@mhguo): shall we load magnetic fields too?
  // UpdateBFields(m, zm);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ZoomData::LoadCCData()
//! \brief Load cell-centered data from zoom data zm to MeshBlock m

// TODO(@mhguo): implement this function
void ZoomData::LoadCCData(int m, int zm) {
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ZoomData::LoadHydroData()
//! \brief Load hydro data from zoom data zm to MeshBlock m

// TODO(@mhguo): implement this function
void ZoomData::LoadHydroData(int m, int zm) {
  auto &indcs = pzoom->pmesh->mb_indcs;
  auto pmbp = pzoom->pmesh->pmb_pack;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  DvceArray5D<Real> u_, w_;
  if (pmbp->phydro != nullptr) {
    u_ = pmbp->phydro->u0;
    w_ = pmbp->phydro->w0;
  } else if (pmbp->pmhd != nullptr) {
    u_ = pmbp->pmhd->u0;
    w_ = pmbp->pmhd->w0;
  }
  auto u0_ = u0, w0_ = w0;
  auto peos = (pmbp->pmhd != nullptr)? pmbp->pmhd->peos : pmbp->phydro->peos;
  auto eos = peos->eos_data;
  Real gamma = eos.gamma;
  bool is_gr = pmbp->pcoord->is_general_relativistic;
  bool flat = true;
  Real spin = 0.0;
  if (is_gr) {
    flat = pmbp->pcoord->coord_data.is_minkowski;
    spin = pmbp->pcoord->coord_data.bh_spin;
  }
  Real &x1min = size.h_view(m).x1min;
  Real &x1max = size.h_view(m).x1max;
  Real &x2min = size.h_view(m).x2min;
  Real &x2max = size.h_view(m).x2max;
  Real &x3min = size.h_view(m).x3min;
  Real &x3max = size.h_view(m).x3max;
  auto ozregion = pzoom->old_zregion;
  if (global_variable::my_rank == 0) {
    std::cout << " Old zoom region radius: " << ozregion.radius << std::endl;
  }
  if (pmbp->phydro != nullptr) {
    // par_for("zoom_reinit", DevExeSpace(),ks-ng,ke+ng,js-ng,je+ng,is-ng,ie+ng,
    par_for("zoom_reinit", DevExeSpace(),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int k, int j, int i) {
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
      if (ozregion.IsInZoomRegion(x1v, x2v, x3v)) { // apply to old zoom region
        // simply copy primitive and conserved variables
        w_(m,IDN,k,j,i) = w0_(zm,IDN,k,j,i);
        w_(m,IVX,k,j,i) = w0_(zm,IVX,k,j,i);
        w_(m,IVY,k,j,i) = w0_(zm,IVY,k,j,i);
        w_(m,IVZ,k,j,i) = w0_(zm,IVZ,k,j,i);
        w_(m,IEN,k,j,i) = w0_(zm,IEN,k,j,i);
        u_(m,IDN,k,j,i) = u0_(zm,IDN,k,j,i);
        u_(m,IM1,k,j,i) = u0_(zm,IM1,k,j,i);
        u_(m,IM2,k,j,i) = u0_(zm,IM2,k,j,i);
        u_(m,IM3,k,j,i) = u0_(zm,IM3,k,j,i);
        u_(m,IEN,k,j,i) = u0_(zm,IEN,k,j,i);
      }
    });
  } else if (pmbp->pmhd != nullptr) {
    auto b = pmbp->pmhd->b0;
    // par_for("zoom_reinit", DevExeSpace(),ks-ng,ke+ng,js-ng,je+ng,is-ng,ie+ng,
    par_for("zoom_reinit", DevExeSpace(),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int k, int j, int i) {
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
      if (ozregion.IsInZoomRegion(x1v, x2v, x3v)) { // apply to old zoom region
        // convert primitive variables to conserved variables
        // load primitive variables from 3D array
        w_(m,IDN,k,j,i) = w0_(zm,IDN,k,j,i);
        w_(m,IVX,k,j,i) = w0_(zm,IVX,k,j,i);
        w_(m,IVY,k,j,i) = w0_(zm,IVY,k,j,i);
        w_(m,IVZ,k,j,i) = w0_(zm,IVZ,k,j,i);
        w_(m,IEN,k,j,i) = w0_(zm,IEN,k,j,i);

        // Load single state of primitive variables
        MHDPrim1D w;
        w.d  = w_(m,IDN,k,j,i);
        w.vx = w_(m,IVX,k,j,i);
        w.vy = w_(m,IVY,k,j,i);
        w.vz = w_(m,IVZ,k,j,i);
        w.e  = w_(m,IEN,k,j,i);

        // load cell-centered fields into primitive state
        // use simple linear average of face-centered fields as bcc is not updated
        w.bx = 0.5*(b.x1f(m,k,j,i) + b.x1f(m,k,j,i+1));
        w.by = 0.5*(b.x2f(m,k,j,i) + b.x2f(m,k,j+1,i));
        w.bz = 0.5*(b.x3f(m,k,j,i) + b.x3f(m,k+1,j,i));

        // call p2c function
        HydCons1D u;
        if (is_gr) {
          Real glower[4][4], gupper[4][4];
          ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);
          SingleP2C_IdealGRMHD(glower, gupper, w, gamma, u);
        } else {
          SingleP2C_IdealMHD(w, u);
        }

        // store conserved quantities in 3D array
        u_(m,IDN,k,j,i) = u.d;
        u_(m,IM1,k,j,i) = u.mx;
        u_(m,IM2,k,j,i) = u.my;
        u_(m,IM3,k,j,i) = u.mz;
        u_(m,IEN,k,j,i) = u.e;
      }
    });
  }
  std::cout << "  Rank " << global_variable::my_rank 
            << " Reinitialized variables in meshblock " << m
            << " using zoom meshblock " << zm << std::endl;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ZoomData::MaskDataInZoomRegion()
//! \brief Mask data in zoom region in MeshBlock m

void ZoomData::MaskDataInZoomRegion(int m, int zm) {
  auto &indcs = pzoom->pmesh->mb_indcs;
  auto pmbp = pzoom->pmesh->pmb_pack;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is, &js = indcs.js, &ks = indcs.ks;
  // int &ie = indcs.ie, &je = indcs.je, &ke = indcs.ke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int &cis = indcs.cis;  int &cie  = indcs.cie;
  int &cjs = indcs.cjs;  int &cje  = indcs.cje;
  int &cks = indcs.cks;  int &cke  = indcs.cke;
  int cnx1 = indcs.cnx1, cnx2 = indcs.cnx2, cnx3 = indcs.cnx3;
  DvceArray5D<Real> u_, w_;
  if (pmbp->phydro != nullptr) {
    u_ = pmbp->phydro->u0;
    w_ = pmbp->phydro->w0;
  } else if (pmbp->pmhd != nullptr) {
    u_ = pmbp->pmhd->u0;
    w_ = pmbp->pmhd->w0;
  }
  auto cu0 = coarse_u0, cw0 = coarse_w0;
  auto peos = (pmbp->pmhd != nullptr)? pmbp->pmhd->peos : pmbp->phydro->peos;
  auto eos = peos->eos_data;
  Real gamma = eos.gamma;
  bool is_gr = pmbp->pcoord->is_general_relativistic;
  bool flat = true;
  Real spin = 0.0;
  if (is_gr) {
    flat = pmbp->pcoord->coord_data.is_minkowski;
    spin = pmbp->pcoord->coord_data.bh_spin;
  }
  Real &x1min = size.h_view(m).x1min;
  Real &x1max = size.h_view(m).x1max;
  Real &x2min = size.h_view(m).x2min;
  Real &x2max = size.h_view(m).x2max;
  Real &x3min = size.h_view(m).x3min;
  Real &x3max = size.h_view(m).x3max;
  auto zregion = pzoom->zregion;
  // eachlevel[pzoom->zstate.zone-1]; // starting gid of zoom MBs on previous level
  int zmbs = pzmesh->gzms_eachdvce[global_variable::my_rank]; // global id start of dvce
  auto &zlloc = pzmesh->lloc_eachzmb[zm+zmbs];
  int ox1 = ((zlloc.lx1 & 1) == 1);
  int ox2 = ((zlloc.lx2 & 1) == 1);
  int ox3 = ((zlloc.lx3 & 1) == 1);
  // std::cout << "  Rank " << global_variable::my_rank 
  //           << " Masking variables in meshblock " << m
  //           << " using zoom meshblock " << zm 
  //           << " with offsets (" << ox1 << "," << ox2 << "," << ox3 << ")"
  //           << std::endl;
  // return;
  if (pmbp->phydro != nullptr) {
    // TODO(@mhguo): probably don't have to mask the ghost zones?
    // par_for("zoom_mask", DevExeSpace(),ks-ng,ke+ng,js-ng,je+ng,is-ng,ie+ng,
    par_for("zoom_mask", DevExeSpace(),cks,cke,cjs,cje,cis,cie,
    KOKKOS_LAMBDA(int ck, int cj, int ci) {
      int i = ci + ox1 * cnx1;
      int j = cj + ox2 * cnx2;
      int k = ck + ox3 * cnx3;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
      if (zregion.IsInZoomRegion(x1v, x2v, x3v)) { // apply to zoom region
        // simply copy primitive and conserved variables
        w_(m,IDN,k,j,i) = cw0(zm,IDN,ck,cj,ci);
        w_(m,IVX,k,j,i) = cw0(zm,IVX,ck,cj,ci);
        w_(m,IVY,k,j,i) = cw0(zm,IVY,ck,cj,ci);
        w_(m,IVZ,k,j,i) = cw0(zm,IVZ,ck,cj,ci);
        w_(m,IEN,k,j,i) = cw0(zm,IEN,ck,cj,ci);
        u_(m,IDN,k,j,i) = cu0(zm,IDN,ck,cj,ci);
        u_(m,IM1,k,j,i) = cu0(zm,IM1,ck,cj,ci);
        u_(m,IM2,k,j,i) = cu0(zm,IM2,ck,cj,ci);
        u_(m,IM3,k,j,i) = cu0(zm,IM3,ck,cj,ci);
        u_(m,IEN,k,j,i) = cu0(zm,IEN,ck,cj,ci);
      }
    });
  } else if (pmbp->pmhd != nullptr) {
    auto b = pmbp->pmhd->b0;
    // TODO(@mhguo): probably don't have to mask the ghost zones?
    // par_for("zoom_mask", DevExeSpace(),ks-ng,ke+ng,js-ng,je+ng,is-ng,ie+ng,
    par_for("zoom_mask", DevExeSpace(),cks,cke,cjs,cje,cis,cie,
    KOKKOS_LAMBDA(int ck, int cj, int ci) {
      int i = ci + ox1 * cnx1;
      int j = cj + ox2 * cnx2;
      int k = ck + ox3 * cnx3;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
      if (zregion.IsInZoomRegion(x1v, x2v, x3v)) { // apply to zoom region
        // convert primitive variables to conserved variables
        // load primitive variables from 3D array
        w_(m,IDN,k,j,i) = cw0(zm,IDN,ck,cj,ci);
        w_(m,IM1,k,j,i) = cw0(zm,IM1,ck,cj,ci);
        w_(m,IM2,k,j,i) = cw0(zm,IM2,ck,cj,ci);
        w_(m,IM3,k,j,i) = cw0(zm,IM3,ck,cj,ci);
        w_(m,IEN,k,j,i) = cw0(zm,IEN,ck,cj,ci);

        // Load single state of primitive variables
        MHDPrim1D w;
        w.d  = w_(m,IDN,k,j,i);
        w.vx = w_(m,IVX,k,j,i);
        w.vy = w_(m,IVY,k,j,i);
        w.vz = w_(m,IVZ,k,j,i);
        w.e  = w_(m,IEN,k,j,i);

        // load cell-centered fields into primitive state
        // TODO(@mhguo): use bcc if available?
        // use simple linear average of face-centered fields as bcc is not updated
        w.bx = 0.5*(b.x1f(m,k,j,i) + b.x1f(m,k,j,i+1));
        w.by = 0.5*(b.x2f(m,k,j,i) + b.x2f(m,k,j+1,i));
        w.bz = 0.5*(b.x3f(m,k,j,i) + b.x3f(m,k+1,j,i));

        // call p2c function
        HydCons1D u;
        if (is_gr) {
          Real glower[4][4], gupper[4][4];
          ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);
          SingleP2C_IdealGRMHD(glower, gupper, w, gamma, u);
        } else {
          SingleP2C_IdealMHD(w, u);
        }

        // store conserved quantities in 3D array
        u_(m,IDN,k,j,i) = u.d;
        u_(m,IM1,k,j,i) = u.mx;
        u_(m,IM2,k,j,i) = u.my;
        u_(m,IM3,k,j,i) = u.mz;
        u_(m,IEN,k,j,i) = u.e;
      }
    });
  }
  return;
}
