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
#include "mesh/cyclic_zoom.hpp"

//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

ZoomData::ZoomData(CyclicZoom *pz, ParameterInput *pin) :
    pzoom(pz),
    u0("zcons",1,1,1,1,1),
    w0("zprim",1,1,1,1,1),
    coarse_u0("czcons",1,1,1,1,1),
    coarse_w0("czprim",1,1,1,1,1),
    efld("zefld",1,1,1,1),
    emf0("zemf0",1,1,1,1),
    delta_efld("zdelta_efld",1,1,1,1),
    max_emf0("zmax_emf0",1,1),
    dzbuf("dz_buffer",1),
    hzbuf("hz_buffer",1),
    hzdata("hz_data",1)
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
    Kokkos::realloc(efld.x1e, nzmb, nccells3+1, nccells2+1, nccells1);
    Kokkos::realloc(efld.x2e, nzmb, nccells3+1, nccells2, nccells1+1);
    Kokkos::realloc(efld.x3e, nzmb, nccells3, nccells2+1, nccells1+1);

    // allocate electric fields just after zoom
    Kokkos::realloc(emf0.x1e, nzmb, nccells3+1, nccells2+1, nccells1);
    Kokkos::realloc(emf0.x2e, nzmb, nccells3+1, nccells2, nccells1+1);
    Kokkos::realloc(emf0.x3e, nzmb, nccells3, nccells2+1, nccells1+1);

    // allocate delta electric fields
    Kokkos::realloc(delta_efld.x1e, nzmb, nccells3+1, nccells2+1, nccells1);
    Kokkos::realloc(delta_efld.x2e, nzmb, nccells3+1, nccells2, nccells1+1);
    Kokkos::realloc(delta_efld.x3e, nzmb, nccells3, nccells2+1, nccells1+1);

    Kokkos::realloc(max_emf0, nlevels, 3);
    for (int i = 0; i < nlevels; i++) {
      for (int j = 0; j < 3; j++) {
        max_emf0(i,j) = 0.0;
      }
    }
  }

  // allocate device and host arrays for data transfer and storage
  Kokkos::realloc(dzbuf, nzmb * zmb_data_cnt);
  Kokkos::realloc(hzbuf, nzmb * zmb_data_cnt);
  Kokkos::realloc(hzdata, nzmb * zmb_data_cnt);
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
  auto e1 = efld.x1e;
  auto e2 = efld.x2e;
  auto e3 = efld.x3e;
  auto e01 = emf0.x1e;
  auto e02 = emf0.x2e;
  auto e03 = emf0.x3e;
  auto de1 = delta_efld.x1e;
  auto de2 = delta_efld.x2e;
  auto de3 = delta_efld.x3e;
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

  // In MHD, we don't use conserved variables so no need to convert
  // if (!is_mhd) {
  //   peos->PrimToCons(w0_,u0_,0,n3-1,0,n2-1,0,n1-1);
  //   peos->PrimToCons(cw0,cu0,0,nc3-1,0,nc2-1,0,nc1-1);
  // }

  if (pzoom->pmesh->pmb_pack->pmhd != nullptr) {
    par_for("zoom_init_e1",DevExeSpace(),0,nzmb-1,0,nc3,0,nc2,0,nc1-1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      e1(m,k,j,i) = 0.0;
      e01(m,k,j,i) = 0.0;
      de1(m,k,j,i) = 0.0;
    });
    par_for("zoom_init_e2",DevExeSpace(),0,nzmb-1,0,nc3,0,nc2-1,0,nc1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      e2(m,k,j,i) = 0.0;
      e02(m,k,j,i) = 0.0;
      de2(m,k,j,i) = 0.0;
    });
    par_for("zoom_init_e3",DevExeSpace(),0,nzmb-1,0,nc3-1,0,nc2,0,nc1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      e3(m,k,j,i) = 0.0;
      e03(m,k,j,i) = 0.0;
      de3(m,k,j,i) = 0.0;
    });
  }

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
    int &nzmb = pzmesh->nzmb_max_perdvce;

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
    // xyz? bcc?
    IOWrapperSizeT cnt = nzmb*nvars*(nccells3)*(nccells2)*(nccells1);
    std::fwrite(coarse_w0.data(),datasize,cnt,pfile);
    auto mbptr = efld.x1e;
    cnt = nzmb*(nccells3+1)*(nccells2+1)*(nccells1);
    std::fwrite(mbptr.data(),datasize,cnt,pfile);
    mbptr = efld.x2e;
    cnt = nzmb*(nccells3+1)*(nccells2)*(nccells1+1);
    std::fwrite(mbptr.data(),datasize,cnt,pfile);
    mbptr = efld.x3e;
    cnt = nzmb*(nccells3)*(nccells2+1)*(nccells1+1);
    std::fwrite(mbptr.data(),datasize,cnt,pfile);
    mbptr = emf0.x1e;
    cnt = nzmb*(nccells3+1)*(nccells2+1)*(nccells1);
    std::fwrite(mbptr.data(),datasize,cnt,pfile);
    mbptr = emf0.x2e;
    cnt = nzmb*(nccells3+1)*(nccells2)*(nccells1+1);
    std::fwrite(mbptr.data(),datasize,cnt,pfile);
    mbptr = emf0.x3e;
    cnt = nzmb*(nccells3)*(nccells2+1)*(nccells1+1);
    std::fwrite(mbptr.data(),datasize,cnt,pfile);
    std::fclose(pfile);
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
//! \fn void ZoomData::StoreDataToZoomData()
//! \brief Store data from MeshBlock m to zoom data zm

void ZoomData::StoreDataToZoomData(int zm, int m) {
  StoreCCData(zm, m);
  if (pzoom->pmesh->pmb_pack->pmhd != nullptr) {
    StoreHydroData(zm, m);
    UpdateElectricFields(zm, m);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ZoomData::StoreCCData()
//! \brief Store cell-centered data from MeshBlock m to zoom data zm

void ZoomData::StoreCCData(int zm, int m) {
  DvceArray5D<Real> u, w;
  auto pmesh = pzoom->pmesh;
  if (pmesh->pmb_pack->phydro != nullptr) {
    u = pmesh->pmb_pack->phydro->u0;
    w = pmesh->pmb_pack->phydro->w0;
  } else if (pmesh->pmb_pack->pmhd != nullptr) {
    u = pmesh->pmb_pack->pmhd->u0;
    w = pmesh->pmb_pack->pmhd->w0;
  }
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
//! \fn void ZoomData::StoreHydroData()
//! \brief Store data from MeshBlock m to zoom data zm

// TODO(@mhguo): this may only apply to MHD as HD can use copy
void ZoomData::StoreHydroData(int zm, int m) {
  auto &indcs = pzoom->pmesh->mb_indcs;
  auto pmbp = pzoom->pmesh->pmb_pack;
  auto &size = pzoom->pmesh->pmb_pack->pmb->mb_size;
  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;
  int &cis = indcs.cis;  int &cie  = indcs.cie;
  int &cjs = indcs.cjs;  int &cje  = indcs.cje;
  int &cks = indcs.cks;  int &cke  = indcs.cke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int cnx1 = indcs.cnx1, cnx2 = indcs.cnx2, cnx3 = indcs.cnx3;
  DvceArray5D<Real> u0_, w0_;
  bool is_gr = pmbp->pcoord->is_general_relativistic;
  auto peos = (pmbp->pmhd != nullptr)? pmbp->pmhd->peos : pmbp->phydro->peos;
  auto eos = peos->eos_data;
  if (pmbp->phydro != nullptr) {
    u0_ = pmbp->phydro->u0;
    w0_ = pmbp->phydro->w0;
  } else if (pmbp->pmhd != nullptr) {
    u0_ = pmbp->pmhd->u0;
    w0_ = pmbp->pmhd->w0;
  }
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
//! \fn void ZoomData::UpdateElectricFields()
//! \brief Update electric fields in zoom data zm from MeshBlock m

void ZoomData::UpdateElectricFields(int zm, int m) {
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ZoomData::PackBuffer()
//! \brief Packs data into AMR communication buffers for all MBs being sent

void ZoomData::PackBuffer() {
  if (global_variable::my_rank == 0) {
    std::cout << "CyclicZoom: Packing data into communication buffer" << std::endl;
  }
  // pack data for all zmbs on this device
  auto &pmhd = pzoom->pmesh->pmb_pack->pmhd;
  // use size_t for offset to avoid overflow
  size_t offset = 0;
  size_t cc_cnt = u0.extent(1) * u0.extent(2) * u0.extent(3) * u0.extent(4);
  size_t ccc_cnt = coarse_u0.extent(1) * coarse_u0.extent(2) * coarse_u0.extent(3) * coarse_u0.extent(4);
  size_t ec_cnt = 0;
  if (pmhd != nullptr) {
    ec_cnt = efld.x1e.extent(1) * efld.x1e.extent(2) * efld.x1e.extent(3);
    ec_cnt += efld.x2e.extent(1) * efld.x2e.extent(2) * efld.x2e.extent(3);
    ec_cnt += efld.x3e.extent(1) * efld.x3e.extent(2) * efld.x3e.extent(3);
  }
  for (int zm = 0; zm < pzmesh->nzmb_thisdvce; ++zm) {
    // offset = zm * zmb_data_cnt;
    // pack conserved variables
    PackBuffersCC(dzbuf, u0, offset, zm);
    offset += cc_cnt;
    // pack primitive variables
    PackBuffersCC(dzbuf, w0, offset, zm);
    offset += cc_cnt;
    // pack coarse conserved variables
    PackBuffersCC(dzbuf, coarse_u0, offset, zm);
    offset += ccc_cnt;
    // pack coarse primitive variables
    PackBuffersCC(dzbuf, coarse_w0, offset, zm);
    offset += ccc_cnt;
    // pack magnetic fields and/or electric fields if MHD
    if (pmhd != nullptr) {
      PackBuffersEC(dzbuf, efld, offset, zm);
      offset += ec_cnt;
      PackBuffersEC(dzbuf, emf0, offset, zm);
      offset += ec_cnt;
      PackBuffersEC(dzbuf, delta_efld, offset, zm);
      offset += ec_cnt;
    }
  }
  // Single copy: device buffer -> host buffer
  // Only copy the portion that's actually used
  size_t used_size = pzmesh->nzmb_thisdvce * zmb_data_cnt;
  Kokkos::deep_copy(
    Kokkos::subview(hzbuf, Kokkos::make_pair(size_t(0), used_size)),
    Kokkos::subview(dzbuf, Kokkos::make_pair(size_t(0), used_size))
  );
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ZoomData::PackBuffersCC()
//! \brief Packs data into AMR communication buffers for all MBs being sent

void ZoomData::PackBuffersCC(DvceArray1D<Real> packed_data, DvceArray5D<Real> a0,
                             size_t offset_a0, const int m) {
  // Pack array a0 at MeshBlock m into packed_data starting from offset_a0
  int nv = a0.extent_int(1);
  int nk = a0.extent_int(2);
  int nj = a0.extent_int(3);
  int ni = a0.extent_int(4);
  // Pack using parallel kernel on device
  par_for("pack_cc", DevExeSpace(), 0, nv-1, 0, nk-1, 0, nj-1, 0, ni-1,
  KOKKOS_LAMBDA(const int v, const int k, const int j, const int i) {
    packed_data(offset_a0 + (((v*nk + k)*nj + j)*ni + i)) = a0(m,v,k,j,i);
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ZoomData::PackBuffersFC()
//! \brief Packs face-centered data into AMR communication buffers for all MBs being sent

void ZoomData::PackBuffersFC(DvceArray1D<Real> packed_data, DvceFaceFld4D<Real> fc,
                             size_t offset_fc, const int m) {
  // Pack face field fc at MeshBlock m into packed_data starting from offset_fc
  // Pack f1
  int nk = fc.x1f.extent_int(1);
  int nj = fc.x1f.extent_int(2);
  int ni = fc.x1f.extent_int(3);
  par_for("pack_f1", DevExeSpace(), 0, nk-1, 0, nj-1, 0, ni-1,
  KOKKOS_LAMBDA(const int k, const int j, const int i) {
    packed_data(offset_fc + (k*nj + j)*ni + i) = fc.x1f(m,k,j,i);
  });
  offset_fc += nk*nj*ni;
  // Pack f2
  nk = fc.x2f.extent_int(1);
  nj = fc.x2f.extent_int(2);
  ni = fc.x2f.extent_int(3);
  par_for("pack_f2", DevExeSpace(), 0, nk-1, 0, nj-1, 0, ni-1,
  KOKKOS_LAMBDA(const int k, const int j, const int i) {
    packed_data(offset_fc + (k*nj + j)*ni + i) = fc.x2f(m,k,j,i);
  });
  offset_fc += nk*nj*ni;
  // Pack f3
  nk = fc.x3f.extent_int(1);
  nj = fc.x3f.extent_int(2);
  ni = fc.x3f.extent_int(3);
  par_for("pack_f3", DevExeSpace(), 0, nk-1, 0, nj-1, 0, ni-1,
  KOKKOS_LAMBDA(const int k, const int j, const int i) {
    packed_data(offset_fc + (k*nj + j)*ni + i) = fc.x3f(m,k,j,i);
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ZoomData::PackBuffersEC()
//! \brief Packs edge-centered data into AMR communication buffers for all MBs being sent

void ZoomData::PackBuffersEC(DvceArray1D<Real> packed_data, DvceEdgeFld4D<Real> ec,
                             size_t offset_ec, const int m) {
  // Pack edge field ec at MeshBlock m into packed_data starting from offset_ec
  // Pack e1
  int nk = ec.x1e.extent_int(1);
  int nj = ec.x1e.extent_int(2);
  int ni = ec.x1e.extent_int(3);
  par_for("pack_e1", DevExeSpace(), 0, nk-1, 0, nj-1, 0, ni-1,
  KOKKOS_LAMBDA(const int k, const int j, const int i) {
    packed_data(offset_ec + (k*nj + j)*ni + i) = ec.x1e(m,k,j,i);
  });
  offset_ec += nk*nj*ni;
  // Pack e2
  nk = ec.x2e.extent_int(1);
  nj = ec.x2e.extent_int(2);
  ni = ec.x2e.extent_int(3);
  par_for("pack_e2", DevExeSpace(), 0, nk-1, 0, nj-1, 0, ni-1,
  KOKKOS_LAMBDA(const int k, const int j, const int i) {
    packed_data(offset_ec + (k*nj + j)*ni + i) = ec.x2e(m,k,j,i);
  });
  offset_ec += nk*nj*ni;
  // Pack e3
  nk = ec.x3e.extent_int(1);
  nj = ec.x3e.extent_int(2);
  ni = ec.x3e.extent_int(3);
  par_for("pack_e3", DevExeSpace(), 0, nk-1, 0, nj-1, 0, ni-1,
  KOKKOS_LAMBDA(const int k, const int j, const int i) {
    packed_data(offset_ec + (k*nj + j)*ni + i) = ec.x3e(m,k,j,i);
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ZoomData::SyncBufferToHost()
//! \brief Sync zoom data buffer to host array

// TODO(@mhguo): need to move to different ranks...
// TODO(@mhguo): need some sync even if mpi is not enabled
void ZoomData::SyncBufferToHost(int zone) {
#if MPI_PARALLEL_ENABLED
  auto rank_send = pzmesh->rank_eachmb;
  auto rank_recv = pzmesh->rank_eachzmb;
  int nlmb = pzmesh->nzmb_eachlevel[zone];
  int lmbs = pzmesh->gids_eachlevel[zone];
  size_t data_per_zmb = zmb_data_cnt;

  int nsend = 0, nrecv = 0, ncopy = 0;
  int my_rank = global_variable::my_rank;
  std::vector<MPI_Request> requests;

  int zm = 0;
  for (int lm = 0; lm < nlmb; ++lm) {
    int src_rank = rank_send[lm+lmbs];
    int dst_rank = rank_recv[lm+lmbs];
    
    // Post receives first
    if (dst_rank == my_rank && src_rank != my_rank) {
      // Receive from src_rank
      MPI_Request req;
      size_t offset_dst = pzmesh->lid_eachzmb[lm+lmbs] * data_per_zmb;
      MPI_Irecv(hzdata.data() + offset_dst, data_per_zmb, 
                MPI_ATHENA_REAL, src_rank, lm, zoom_comm, &req);
      requests.push_back(req);
      std::cout << "  Rank " << global_variable::my_rank 
                << " Irecv to host for zmb " << lm+lmbs 
                << " from rank " << src_rank << " to offset " << offset_dst 
                << " with local id " << pzmesh->lid_eachzmb[lm+lmbs]
                << std::endl;
      ++nrecv;
    }

    // Post sends
    if (src_rank == my_rank && dst_rank != my_rank) {
      // Send to dst_rank
      MPI_Request req;
      size_t offset_src = zm * data_per_zmb;
      MPI_Isend(hzbuf.data() + offset_src, data_per_zmb,
                MPI_ATHENA_REAL, dst_rank, lm, zoom_comm, &req);
      requests.push_back(req);
      std::cout << "  Rank " << global_variable::my_rank 
                << " Isend to host for zmb " << lm+lmbs 
                << " to rank " << dst_rank << " from offset " << offset_src
                << " with local id " << zm
                << std::endl;
      ++nsend;
      ++zm;
    } 

    // Local copy
    if (src_rank == my_rank && dst_rank == my_rank) {
      size_t offset_src = zm * data_per_zmb;
      size_t offset_dst = pzmesh->lid_eachzmb[lm+lmbs] * data_per_zmb;
      Kokkos::deep_copy(
        Kokkos::subview(hzdata, Kokkos::make_pair(offset_dst, offset_dst + data_per_zmb)),
        Kokkos::subview(hzbuf, Kokkos::make_pair(offset_src, offset_src + data_per_zmb))
      );
      std::cout << "  Rank " << global_variable::my_rank 
                << " local copy to host for zmb " << lm+lmbs
                << " from offset " << offset_src << " to offset " << offset_dst
                << " from local id " << zm << " to local id " << pzmesh->lid_eachzmb[lm+lmbs]
                << std::endl;
      ++ncopy;
      ++zm;
    }
  }

  // Wait for all communications to complete
  if (!requests.empty()) {
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
  }

  std::cout << "SyncBufferToHost: Rank " << global_variable::my_rank << " completed "
            << requests.size() << " MPI operations"
            << " (sends: " << nsend << ", receives: " << nrecv
            << ", local copies: " << ncopy << ")" << std::endl;

#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ZoomData::SyncHostToBuffer()
//! \brief Sync zoom data from host array to buffer

// TODO(@mhguo): need to move to different ranks...
void ZoomData::SyncHostToBuffer(int zone) {
#if MPI_PARALLEL_ENABLED
  auto rank_send = pzmesh->rank_eachzmb;
  auto rank_recv = pzmesh->rank_eachmb;
  int nlmb = pzmesh->nzmb_eachlevel[zone];
  int lmbs = pzmesh->gids_eachlevel[zone];
  size_t data_per_zmb = zmb_data_cnt;

  int nsend = 0, nrecv = 0, ncopy = 0;
  int my_rank = global_variable::my_rank;
  std::vector<MPI_Request> requests;

  int zm = 0;
  for (int lm = 0; lm < nlmb; ++lm) {
    int src_rank = rank_send[lm+lmbs];
    int dst_rank = rank_recv[lm+lmbs];
    
    // Post receives first
    if (dst_rank == my_rank && src_rank != my_rank) {
      // Receive from src_rank
      MPI_Request req;
      size_t offset_dst = zm * data_per_zmb;
      MPI_Irecv(hzbuf.data() + offset_dst, data_per_zmb, 
                MPI_ATHENA_REAL, src_rank, lm, zoom_comm, &req);
      requests.push_back(req);
      std::cout << "  Rank " << global_variable::my_rank 
                << " Irecv to buffer for zmb " << lm+lmbs
                << " from rank " << src_rank << " to offset " << offset_dst 
                << " with local id " << zm
                << std::endl;
      ++nrecv;
      ++zm;
    }

    // Post sends
    if (src_rank == my_rank && dst_rank != my_rank) {
      // Send to dst_rank
      MPI_Request req;
      size_t offset_src = pzmesh->lid_eachzmb[lm+lmbs] * data_per_zmb;
      MPI_Isend(hzdata.data() + offset_src, data_per_zmb,
                MPI_ATHENA_REAL, dst_rank, lm, zoom_comm, &req);
      requests.push_back(req);
      std::cout << "  Rank " << global_variable::my_rank 
                << " Isend to buffer for zmb " << lm+lmbs 
                << " to rank " << dst_rank << " from offset " << offset_src
                << " with local id " << pzmesh->lid_eachzmb[lm+lmbs]
                << std::endl;
      ++nsend;
    } 

    // Local copy
    if (src_rank == my_rank && dst_rank == my_rank) {
      size_t offset_src = pzmesh->lid_eachzmb[lm+lmbs] * data_per_zmb;
      size_t offset_dst = zm * data_per_zmb;
      Kokkos::deep_copy(
        Kokkos::subview(hzbuf, Kokkos::make_pair(offset_dst, offset_dst + data_per_zmb)),
        Kokkos::subview(hzdata, Kokkos::make_pair(offset_src, offset_src + data_per_zmb))
      );
      std::cout << "  Rank " << global_variable::my_rank 
                << " local copy to buffer for zmb " << lm+lmbs
                << " from offset " << offset_src << " to offset " << offset_dst
                << " from local id " << pzmesh->lid_eachzmb[lm+lmbs ] << " to local id " << zm
                << std::endl;
      ++ncopy;
      ++zm;
    }
  }
  
  // Wait for all communications to complete
  if (!requests.empty()) {
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
  }
  
  std::cout << "SyncHostToBuffer: Rank " << global_variable::my_rank << " completed "
            << requests.size() << " MPI operations"
            << " (sends: " << nsend << ", receives: " << nrecv
            << ", local copies: " << ncopy << ")" << std::endl;

#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ZoomData::UnpackBuffer()
//! \brief Unpacks data from AMR communication buffers for all MBs being received

void ZoomData::UnpackBuffer() {
  // Single copy: host buffer -> device buffer
  // Only copy the portion that's actually used
  size_t used_size = pzmesh->nzmb_thisdvce * zmb_data_cnt;
  Kokkos::deep_copy(
    Kokkos::subview(dzbuf, Kokkos::make_pair(size_t(0), used_size)),
    Kokkos::subview(hzbuf, Kokkos::make_pair(size_t(0), used_size))
  );
  // Unpack data from dzbuf to zoom data arrays
  auto &pmhd = pzoom->pmesh->pmb_pack->pmhd;
  // use size_t for offset to avoid overflow
  size_t offset = 0;
  size_t cc_cnt = u0.extent(1) * u0.extent(2) * u0.extent(3) * u0.extent(4);
  size_t ccc_cnt = coarse_u0.extent(1) * coarse_u0.extent(2) * coarse_u0.extent(3) * coarse_u0.extent(4);
  size_t ec_cnt = 0;
  if (pmhd != nullptr) {
    ec_cnt = efld.x1e.extent(1) * efld.x1e.extent(2) * efld.x1e.extent(3);
    ec_cnt += efld.x2e.extent(1) * efld.x2e.extent(2) * efld.x2e.extent(3);
    ec_cnt += efld.x3e.extent(1) * efld.x3e.extent(2) * efld.x3e.extent(3);
  }
  for (int zm = 0; zm < pzmesh->nzmb_thisdvce; ++zm) {
    std::cout << " Rank " << global_variable::my_rank 
              << " Unpacking buffer for zmb " << zm << std::endl;
    // offset = zm * zmb_data_cnt;
    // unpack conserved variables
    UnpackBuffersCC(dzbuf, u0, offset, zm);
    offset += cc_cnt;
    // unpack primitive variables
    UnpackBuffersCC(dzbuf, w0, offset, zm);
    offset += cc_cnt;
    // unpack coarse conserved variables
    UnpackBuffersCC(dzbuf, coarse_u0, offset, zm);
    offset += ccc_cnt;
    // unpack coarse primitive variables
    UnpackBuffersCC(dzbuf, coarse_w0, offset, zm);
    offset += ccc_cnt;
    // unpack magnetic fields and/or electric fields if MHD
    if (pmhd != nullptr) {
      UnpackBuffersEC(dzbuf, efld, offset, zm);
      offset += ec_cnt;
      UnpackBuffersEC(dzbuf, emf0, offset, zm);
      offset += ec_cnt;
      UnpackBuffersEC(dzbuf, delta_efld, offset, zm);
      offset += ec_cnt;
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ZoomData::UnpackBuffersCC()
//! \brief Unpacks cell-centered data from AMR communication buffers

void ZoomData::UnpackBuffersCC(DvceArray1D<Real> packed_data, DvceArray5D<Real> a0,
                               size_t offset_a0, const int m) {
  // Unpack array a0 at MeshBlock m from packed_data starting from offset_a0
  int nv = a0.extent_int(1);
  int nk = a0.extent_int(2);
  int nj = a0.extent_int(3);
  int ni = a0.extent_int(4);
  // Unpack using parallel kernel on device
  par_for("unpack_cc", DevExeSpace(), 0, nv-1, 0, nk-1, 0, nj-1, 0, ni-1,
  KOKKOS_LAMBDA(const int v, const int k, const int j, const int i) {
    a0(m,v,k,j,i) = packed_data(offset_a0 + (((v*nk + k)*nj + j)*ni + i));
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ZoomData::UnpackBuffersFC()
//! \brief Unpacks face-centered data from AMR communication buffers

void ZoomData::UnpackBuffersFC(DvceArray1D<Real> packed_data, DvceFaceFld4D<Real> fc,
                               size_t offset_fc, const int m) {
  // Unpack face field fc at MeshBlock m from packed_data starting from offset_fc
  // Unpack f1
  int nk = fc.x1f.extent_int(1);
  int nj = fc.x1f.extent_int(2);
  int ni = fc.x1f.extent_int(3);
  par_for("unpack_f1", DevExeSpace(), 0, nk-1, 0, nj-1, 0, ni-1,
  KOKKOS_LAMBDA(const int k, const int j, const int i) {
    fc.x1f(m,k,j,i) = packed_data(offset_fc + (k*nj + j)*ni + i);
  });
  offset_fc += nk*nj*ni;
  // Unpack f2
  nk = fc.x2f.extent_int(1);
  nj = fc.x2f.extent_int(2);
  ni = fc.x2f.extent_int(3);
  par_for("unpack_f2", DevExeSpace(), 0, nk-1, 0, nj-1, 0, ni-1,
  KOKKOS_LAMBDA(const int k, const int j, const int i) {
    fc.x2f(m,k,j,i) = packed_data(offset_fc + (k*nj + j)*ni + i);
  });
  offset_fc += nk*nj*ni;
  // Unpack f3
  nk = fc.x3f.extent_int(1);
  nj = fc.x3f.extent_int(2);
  ni = fc.x3f.extent_int(3);
  par_for("unpack_f3", DevExeSpace(), 0, nk-1, 0, nj-1, 0, ni-1,
  KOKKOS_LAMBDA(const int k, const int j, const int i) {
    fc.x3f(m,k,j,i) = packed_data(offset_fc + (k*nj + j)*ni + i);
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ZoomData::UnpackBuffersEC()
//! \brief Unpacks edge-centered data from AMR communication buffers

void ZoomData::UnpackBuffersEC(DvceArray1D<Real> packed_data, DvceEdgeFld4D<Real> ec,
                               size_t offset_ec, const int m) {
  // Unpack edge field ec at MeshBlock m from packed_data starting from offset_ec
  // Unpack e1
  int nk = ec.x1e.extent_int(1);
  int nj = ec.x1e.extent_int(2);
  int ni = ec.x1e.extent_int(3);
  par_for("unpack_e1", DevExeSpace(), 0, nk-1, 0, nj-1, 0, ni-1,
  KOKKOS_LAMBDA(const int k, const int j, const int i) {
    ec.x1e(m,k,j,i) = packed_data(offset_ec + (k*nj + j)*ni + i);
  });
  offset_ec += nk*nj*ni;
  // Unpack e2
  nk = ec.x2e.extent_int(1);
  nj = ec.x2e.extent_int(2);
  ni = ec.x2e.extent_int(3);
  par_for("unpack_e2", DevExeSpace(), 0, nk-1, 0, nj-1, 0, ni-1,
  KOKKOS_LAMBDA(const int k, const int j, const int i) {
    ec.x2e(m,k,j,i) = packed_data(offset_ec + (k*nj + j)*ni + i);
  });
  offset_ec += nk*nj*ni;
  // Unpack e3
  nk = ec.x3e.extent_int(1);
  nj = ec.x3e.extent_int(2);
  ni = ec.x3e.extent_int(3);
  par_for("unpack_e3", DevExeSpace(), 0, nk-1, 0, nj-1, 0, ni-1,
  KOKKOS_LAMBDA(const int k, const int j, const int i) {
    ec.x3e(m,k,j,i) = packed_data(offset_ec + (k*nj + j)*ni + i);
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
  int &ng = indcs.ng;
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
        w_(m,IM1,k,j,i) = w0_(zm,IM1,k,j,i);
        w_(m,IM2,k,j,i) = w0_(zm,IM2,k,j,i);
        w_(m,IM3,k,j,i) = w0_(zm,IM3,k,j,i);
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
  int &ng = indcs.ng;
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
  int zmbs = pzmesh->gids_eachdvce[global_variable::my_rank]; // global id start of dvce
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
      Real x1v = CellCenterX(ci-cis, cnx1, x1min, x1max);
      Real x2v = CellCenterX(cj-cjs, cnx2, x2min, x2max);
      Real x3v = CellCenterX(ck-cks, cnx3, x3min, x3max);
      if (zregion.IsInZoomRegion(x1v, x2v, x3v)) { // apply to old zoom region
        // convert primitive variables to conserved variables
        // load primitive variables from 3D array
        int i = ci + ox1 * cnx1;
        int j = cj + ox2 * cnx2;
        int k = ck + ox3 * cnx3;
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