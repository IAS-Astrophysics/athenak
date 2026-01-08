//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file zoom_restart.cpp
//! \brief Functions to write and read cyclic zoom restart data

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "cyclic_zoom/cyclic_zoom.hpp"

//----------------------------------------------------------------------------------------
//! \fn IOWrapperSizeT CyclicZoom::RestartFileSize()
//! \brief Get size of restart file for zoom

IOWrapperSizeT CyclicZoom::RestartFileSize() {
  IOWrapperSizeT res_size = 0;
  // size of zoom parameters
  res_size += sizeof(ZoomState);
  // size of cell-centered data
  auto &indcs = pmesh->mb_indcs;
  int &mzoom = pzmesh->nzmb_max_perdvce;
  int &nzoom = pzdata->nvars;
  int nout1 = indcs.nx1 + 2*(indcs.ng);
  int nout2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int nout3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  int n_ccells1 = indcs.cnx1 + 2*(indcs.ng);
  int n_ccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
  int n_ccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;
  int nfld = 3; // number of fields: efld, emf0, delta_efld
  res_size += mzoom * nzoom * nout3 * nout2 * nout1 * sizeof(Real); // u0
  res_size += mzoom * nzoom * nout3 * nout2 * nout1 * sizeof(Real); // w0
  res_size += mzoom * nzoom * n_ccells3 * n_ccells2 * n_ccells1 * sizeof(Real); // coarse_u0
  res_size += mzoom * nzoom * n_ccells3 * n_ccells2 * n_ccells1 * sizeof(Real); // coarse_w0
  // size of edge-centered emf data, using face-centered field data for simplicity
  res_size += nfld * mzoom * (n_ccells3+1) * (n_ccells2+1) * (n_ccells1)   * sizeof(Real);
  res_size += nfld * mzoom * (n_ccells3+1) * (n_ccells2)   * (n_ccells1+1) * sizeof(Real);
  res_size += nfld * mzoom * (n_ccells3)   * (n_ccells2+1) * (n_ccells1+1) * sizeof(Real);
  res_size += mzoom * 3 * sizeof(Real); // max_emf0
  return res_size;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::WriteRestartFile(IOWrapper resfile)
//! \brief Write restart file for zoom, only the root process writes

void CyclicZoom::WriteRestartFile(IOWrapper &resfile) {
  // write zoom parameters
  resfile.Write_any_type(&(zstate), sizeof(ZoomState), "byte");
  auto &indcs = pmesh->mb_indcs;
  int &mzoom = pzmesh->nzmb_max_perdvce;
  int &nzoom = pzdata->nvars;
  int nout1 = indcs.nx1 + 2*(indcs.ng);
  int nout2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int nout3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  int n_ccells1 = indcs.cnx1 + 2*(indcs.ng);
  int n_ccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
  int n_ccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;
  // write cell-centered data
  // get ptr to cell-centered MeshBlock data
  HostArray5D<Real> outarray_zoom("rst-zc-out", 1, 1, 1, 1, 1);
  Kokkos::realloc(outarray_zoom, mzoom, nzoom, nout3, nout2, nout1);
  auto mbptr = outarray_zoom;
  Kokkos::deep_copy(outarray_zoom, pzdata->u0);
  resfile.Write_any_type(mbptr.data(),mbptr.size(),"Real");
  Kokkos::deep_copy(outarray_zoom, pzdata->w0);
  resfile.Write_any_type(mbptr.data(),mbptr.size(),"Real");
  Kokkos::realloc(outarray_zoom, mzoom, nzoom, n_ccells3, n_ccells2, n_ccells1);
  mbptr = outarray_zoom;
  Kokkos::deep_copy(outarray_zoom, pzdata->coarse_u0);
  resfile.Write_any_type(mbptr.data(),mbptr.size(),"Real");
  Kokkos::deep_copy(outarray_zoom, pzdata->coarse_w0);
  resfile.Write_any_type(mbptr.data(),mbptr.size(),"Real");

  // write edge-centered emf data, using face-centered field data for simplicity
  HostFaceFld4D<Real> outefld_zoom("ec_outvar", 1, 1, 1, 1);  // edge-centered output field on host
  Kokkos::realloc(outefld_zoom.x1f, mzoom, n_ccells3+1, n_ccells2+1, n_ccells1);
  Kokkos::realloc(outefld_zoom.x2f, mzoom, n_ccells3+1, n_ccells2, n_ccells1+1);
  Kokkos::realloc(outefld_zoom.x3f, mzoom, n_ccells3, n_ccells2+1, n_ccells1+1);
  auto e1ptr = outefld_zoom.x1f;
  auto e2ptr = outefld_zoom.x2f;
  auto e3ptr = outefld_zoom.x3f;
  Kokkos::deep_copy(outefld_zoom.x1f, pzdata->efld.x1e);
  Kokkos::deep_copy(outefld_zoom.x2f, pzdata->efld.x2e);
  Kokkos::deep_copy(outefld_zoom.x3f, pzdata->efld.x3e);
  resfile.Write_any_type(e1ptr.data(),e1ptr.size(),"Real");
  resfile.Write_any_type(e2ptr.data(),e2ptr.size(),"Real");
  resfile.Write_any_type(e3ptr.data(),e3ptr.size(),"Real");
  Kokkos::deep_copy(outefld_zoom.x1f, pzdata->emf0.x1e);
  Kokkos::deep_copy(outefld_zoom.x2f, pzdata->emf0.x2e);
  Kokkos::deep_copy(outefld_zoom.x3f, pzdata->emf0.x3e);
  resfile.Write_any_type(e1ptr.data(),e1ptr.size(),"Real");
  resfile.Write_any_type(e2ptr.data(),e2ptr.size(),"Real");
  resfile.Write_any_type(e3ptr.data(),e3ptr.size(),"Real");
  Kokkos::deep_copy(outefld_zoom.x1f, pzdata->delta_efld.x1e);
  Kokkos::deep_copy(outefld_zoom.x2f, pzdata->delta_efld.x2e);
  Kokkos::deep_copy(outefld_zoom.x3f, pzdata->delta_efld.x3e);
  resfile.Write_any_type(e1ptr.data(),e1ptr.size(),"Real");
  resfile.Write_any_type(e2ptr.data(),e2ptr.size(),"Real");
  resfile.Write_any_type(e3ptr.data(),e3ptr.size(),"Real");
  // write max_emf0
  auto maxptr = pzdata->max_emf0;
  resfile.Write_any_type(maxptr.data(),maxptr.size(),"Real");

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::ReadRestartFile(IOWrapper resfile)
//! \brief Read restart file for zoom

void CyclicZoom::ReadRestartFile(IOWrapper &resfile) {
  // TODO(@mhguo): read all data, then broadcast to all ranks
  
  // Read pzoom w0 data
  auto &indcs = pmesh->mb_indcs;
  int &mzoom = pzmesh->nzmb_max_perdvce;
  int &nzoom = pzdata->nvars;
  int nout1 = indcs.nx1 + 2*(indcs.ng);
  int nout2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int nout3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  int n_ccells1 = indcs.cnx1 + 2*(indcs.ng);
  int n_ccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
  int n_ccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;
  
  //---- STEP 1: read zoom parameters

  // root process reads zoom data
  char *zrdata = new char[sizeof(ZoomState)];
  if (global_variable::my_rank == 0) { // the master process reads the variables data
    if (resfile.Read_bytes(zrdata, 1, sizeof(ZoomState)) != sizeof(ZoomState)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "ZoomState data read from restart file is incorrect, "
                << "restart file is broken." << std::endl;
      exit(EXIT_FAILURE);
    }

    HostArray5D<Real> zcin("rst-zc-in", 1, 1, 1, 1, 1);
    Kokkos::realloc(zcin, mzoom, nzoom, nout3, nout2, nout1);
    resfile.Read_Reals(zcin.data(), zcin.size());
    Kokkos::deep_copy(pzdata->u0, zcin);
    resfile.Read_Reals(zcin.data(), zcin.size());
    Kokkos::deep_copy(pzdata->w0, zcin);
    // Read pzoom coarse u0 data
    Kokkos::realloc(zcin, mzoom, nzoom, n_ccells3, n_ccells2, n_ccells1);
    resfile.Read_Reals(zcin.data(), zcin.size());
    Kokkos::deep_copy(pzdata->coarse_u0, zcin);
    resfile.Read_Reals(zcin.data(), zcin.size());
    Kokkos::deep_copy(pzdata->coarse_w0, zcin);

    // Read pzoom edge-centered electric fields
    HostFaceFld4D<Real> zein("rst-ze-in", 1, 1, 1, 1);  // edge-centered output field on host
    Kokkos::realloc(zein.x1f, mzoom, n_ccells3+1, n_ccells2+1, n_ccells1); // x1e
    Kokkos::realloc(zein.x2f, mzoom, n_ccells3+1, n_ccells2, n_ccells1+1); // x2e
    Kokkos::realloc(zein.x3f, mzoom, n_ccells3, n_ccells2+1, n_ccells1+1); // x3e
    // assign pointer after reallocating
    auto e1ptr = zein.x1f;
    auto e2ptr = zein.x2f;
    auto e3ptr = zein.x3f;
    resfile.Read_Reals(e1ptr.data(), e1ptr.size());
    resfile.Read_Reals(e2ptr.data(), e2ptr.size());
    resfile.Read_Reals(e3ptr.data(), e3ptr.size());
    Kokkos::deep_copy(zein.x1f, pzdata->efld.x1e);
    Kokkos::deep_copy(zein.x2f, pzdata->efld.x2e);
    Kokkos::deep_copy(zein.x3f, pzdata->efld.x3e);
    resfile.Read_Reals(e1ptr.data(), e1ptr.size());
    resfile.Read_Reals(e2ptr.data(), e2ptr.size());
    resfile.Read_Reals(e3ptr.data(), e3ptr.size());
    Kokkos::deep_copy(zein.x1f, pzdata->emf0.x1e);
    Kokkos::deep_copy(zein.x2f, pzdata->emf0.x2e);
    Kokkos::deep_copy(zein.x3f, pzdata->emf0.x3e);
    resfile.Read_Reals(e1ptr.data(), e1ptr.size());
    resfile.Read_Reals(e2ptr.data(), e2ptr.size());
    resfile.Read_Reals(e3ptr.data(), e3ptr.size());
    Kokkos::deep_copy(zein.x1f, pzdata->delta_efld.x1e);
    Kokkos::deep_copy(zein.x2f, pzdata->delta_efld.x2e);
    Kokkos::deep_copy(zein.x3f, pzdata->delta_efld.x3e);

    // Read pzoom max emf0 data
    auto maxptr = pzdata->max_emf0;
    resfile.Read_Reals(maxptr.data(), maxptr.size());
  }

  // STEP 2: broadcast to all ranks
  // broadcast the zoom parameters
#if MPI_PARALLEL_ENABLED
  // then broadcast the ZoomState information
  MPI_Bcast(zrdata, sizeof(ZoomState), MPI_CHAR, 0, MPI_COMM_WORLD);
#endif
  std::memcpy(&(zstate), &(zrdata[0]), sizeof(ZoomState));

  // broadcast the zoom data array
#if MPI_PARALLEL_ENABLED
  // It looks device to device communication is not supported, so copy to host first
  //TODO(@mhguo): check whether this auto works as intended
  auto harr_5d = pzdata->harr_5d;
  Kokkos::realloc(harr_5d, mzoom, nzoom, nout3, nout2, nout1);
  Kokkos::deep_copy(harr_5d, pzdata->u0);
  MPI_Bcast(harr_5d.data(), harr_5d.size(), MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
  Kokkos::deep_copy(pzdata->u0, harr_5d);
  Kokkos::deep_copy(harr_5d, pzdata->w0);
  MPI_Bcast(harr_5d.data(), harr_5d.size(), MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
  Kokkos::deep_copy(pzdata->w0, harr_5d);
  Kokkos::realloc(harr_5d, mzoom, nzoom, n_ccells3, n_ccells2, n_ccells1);
  Kokkos::deep_copy(harr_5d, pzdata->coarse_u0);
  MPI_Bcast(harr_5d.data(), harr_5d.size(), MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
  Kokkos::deep_copy(pzdata->coarse_u0, harr_5d);
  Kokkos::deep_copy(harr_5d , pzdata->coarse_w0);
  MPI_Bcast(harr_5d.data(), harr_5d.size(), MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
  Kokkos::deep_copy(pzdata->coarse_w0, harr_5d);
  HostFaceFld4D<Real> hfld("host-ze", 1, 1, 1, 1);  // edge-centered output field on host
  auto e1ptr = hfld.x1f;
  auto e2ptr = hfld.x2f;
  auto e3ptr = hfld.x3f;
  Kokkos::realloc(hfld.x1f, mzoom, n_ccells3+1, n_ccells2+1, n_ccells1); // x1e
  Kokkos::realloc(hfld.x2f, mzoom, n_ccells3+1, n_ccells2, n_ccells1+1); // x2e 
  Kokkos::realloc(hfld.x3f, mzoom, n_ccells3, n_ccells2+1, n_ccells1+1); // x3e
  Kokkos::deep_copy(hfld.x1f, pzdata->efld.x1e);
  Kokkos::deep_copy(hfld.x2f, pzdata->efld.x2e);
  Kokkos::deep_copy(hfld.x3f, pzdata->efld.x3e);
  MPI_Bcast(hfld.x1f.data(), hfld.x1f.size(), MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
  MPI_Bcast(hfld.x2f.data(), hfld.x2f.size(), MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
  MPI_Bcast(hfld.x3f.data(), hfld.x3f.size(), MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
  Kokkos::deep_copy(pzdata->efld.x1e, hfld.x1f);
  Kokkos::deep_copy(pzdata->efld.x2e, hfld.x2f);
  Kokkos::deep_copy(pzdata->efld.x3e, hfld.x3f);
  Kokkos::deep_copy(hfld.x1f, pzdata->emf0.x1e);
  Kokkos::deep_copy(hfld.x2f, pzdata->emf0.x2e);
  Kokkos::deep_copy(hfld.x3f, pzdata->emf0.x3e);
  MPI_Bcast(hfld.x1f.data(), hfld.x1f.size(), MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
  MPI_Bcast(hfld.x2f.data(), hfld.x2f.size(), MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
  MPI_Bcast(hfld.x3f.data(), hfld.x3f.size(), MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
  Kokkos::deep_copy(pzdata->emf0.x1e, hfld.x1f);
  Kokkos::deep_copy(pzdata->emf0.x2e, hfld.x2f);
  Kokkos::deep_copy(pzdata->emf0.x3e, hfld.x3f);
  Kokkos::deep_copy(hfld.x1f, pzdata->delta_efld.x1e);
  Kokkos::deep_copy(hfld.x2f, pzdata->delta_efld.x2e);
  Kokkos::deep_copy(hfld.x3f, pzdata->delta_efld.x3e);
  MPI_Bcast(hfld.x1f.data(), hfld.x1f.size(), MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
  MPI_Bcast(hfld.x2f.data(), hfld.x2f.size(), MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
  MPI_Bcast(hfld.x3f.data(), hfld.x3f.size(), MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
  Kokkos::deep_copy(pzdata->delta_efld.x1e, hfld.x1f);
  Kokkos::deep_copy(pzdata->delta_efld.x2e, hfld.x2f);
  Kokkos::deep_copy(pzdata->delta_efld.x3e, hfld.x3f);
  // Read pzoom max emf0 data
  auto maxptr = pzdata->max_emf0;
  MPI_Bcast(maxptr.data(), maxptr.size(), MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
#endif

  return;
}
