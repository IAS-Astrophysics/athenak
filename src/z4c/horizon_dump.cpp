//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================

#include <assert.h>
#include <unistd.h>
#include <sys/stat.h>  // mkdir

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <cstdio>
#include <utility>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include "horizon_dump.hpp"
#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "utils/cart_grid.hpp"
#include "coordinates/adm.hpp"
#include "mhd/mhd.hpp"
#include "z4c/z4c.hpp"

//----------------------------------------------------------------------------------------
HorizonDump::HorizonDump(MeshBlockPack *pmbp, ParameterInput *pin, int n, int is_common):
              common_horizon{is_common}, pos{NAN, NAN, NAN},
              pmbp{pmbp}, horizon_ind{n} {
  std::string nstr = std::to_string(n);

  pos[0] = pin->GetOrAddReal("z4c", "co_" + nstr + "_x", 0.0);
  pos[1] = pin->GetOrAddReal("z4c", "co_" + nstr + "_y", 0.0);
  pos[2] = pin->GetOrAddReal("z4c", "co_" + nstr + "_z", 0.0);

  horizon_extent = pin->GetOrAddReal("z4c", "co_" + nstr + "_dump_radius", 2.0);
  horizon_nx = pin->GetOrAddInteger("z4c", "horizon_"
                              + nstr+"_Nx",10);
  horizon_dt = pin->GetOrAddReal("z4c", "horizon_dt", 1.0);
  r_guess = pin->GetOrAddReal("z4c", "horizon" + nstr + "r_guess", 0.5);
  output_count = 0;

  Real extend[3] = {horizon_extent,horizon_extent,horizon_extent};
  int Nx[3] = {horizon_nx,horizon_nx,horizon_nx};
  pcat_grid = new CartesianGrid(pmbp, pos, extend, Nx);

  // Initializing variables that will be dumped
  // The order is alpha, betax, betay, betaz,
  // gxx, gxy, gxz, gyy, gyz, gzz
  // Kxx, Kxy, Kxz, Kyy, Kyz, Kzz
  variable_to_dump.push_back(std::make_pair(pmbp->pz4c->I_Z4C_ALPHA, true));
  variable_to_dump.push_back(std::make_pair(pmbp->pz4c->I_Z4C_BETAX, true));
  variable_to_dump.push_back(std::make_pair(pmbp->pz4c->I_Z4C_BETAY, true));
  variable_to_dump.push_back(std::make_pair(pmbp->pz4c->I_Z4C_BETAZ, true));
  variable_to_dump.push_back(std::make_pair(pmbp->padm->I_ADM_GXX, false));
  variable_to_dump.push_back(std::make_pair(pmbp->padm->I_ADM_GXY, false));
  variable_to_dump.push_back(std::make_pair(pmbp->padm->I_ADM_GXZ, false));
  variable_to_dump.push_back(std::make_pair(pmbp->padm->I_ADM_GYY, false));
  variable_to_dump.push_back(std::make_pair(pmbp->padm->I_ADM_GYZ, false));
  variable_to_dump.push_back(std::make_pair(pmbp->padm->I_ADM_GZZ, false));
  variable_to_dump.push_back(std::make_pair(pmbp->padm->I_ADM_KXX, false));
  variable_to_dump.push_back(std::make_pair(pmbp->padm->I_ADM_KXY, false));
  variable_to_dump.push_back(std::make_pair(pmbp->padm->I_ADM_KXZ, false));
  variable_to_dump.push_back(std::make_pair(pmbp->padm->I_ADM_KYY, false));
  variable_to_dump.push_back(std::make_pair(pmbp->padm->I_ADM_KYZ, false));
  variable_to_dump.push_back(std::make_pair(pmbp->padm->I_ADM_KZZ, false));
}

//----------------------------------------------------------------------------------------
HorizonDump::~HorizonDump() {
  delete pcat_grid;
}

void HorizonDump::SetGridAndInterpolate(Real center[NDIM]) {
  // update center location
  pos[0] = center[0];
  pos[1] = center[1];
  pos[2] = center[2];

  pcat_grid->ResetCenter(pos);
  // Real* data_out = new Real [];
  // swap out to 1d array
  // Real data_out[horizon_nx][horizon_nx][horizon_nx][16];

  // Define the size of each dimension
  int count = horizon_nx * horizon_nx * horizon_nx * 16;

  // Dynamically allocate memory for the 4D array flattened into 1D
  Real* data_out = new Real[count];

  for(int nvar=0; nvar<16; nvar++) {
    // Interpolate here
    if (variable_to_dump[nvar].second) {
      pcat_grid->InterpolateToGrid(variable_to_dump[nvar].first,pmbp->pz4c->u0);
    } else {
      pcat_grid->InterpolateToGrid(variable_to_dump[nvar].first,pmbp->padm->u_adm);
    }
    for (int nx = 0; nx < horizon_nx; nx ++)
    for (int ny = 0; ny < horizon_nx; ny ++)
    for (int nz = 0; nz < horizon_nx; nz ++) {
      data_out[nvar * horizon_nx * horizon_nx * horizon_nx +  // Section for nvar
        nx * horizon_nx * horizon_nx +                 // Slice for nx
        ny * horizon_nx +                              // Row for ny
        nz]                                            // Column for nz
        = pcat_grid->interp_vals.h_view(nx, ny, nz);        // Value being assigned
    }
  }

  // MPI reduce here
  // Reduction to the master rank for data_out
  #if MPI_PARALLEL_ENABLED
  if (0 == global_variable::my_rank) {
    MPI_Reduce(MPI_IN_PLACE, data_out, count,
              MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(data_out, data_out, count, MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
  }
  #endif
  // Then write output file
  // Open the file in binary write mode
  std::string foldername = "horizon_"+std::to_string(horizon_ind)
                        +"/output_"+std::to_string(output_count);
  mkdir(foldername.c_str(),0775);

  if (0 == global_variable::my_rank) {
    std::string fname = foldername + "/etk_output_file.dat";
    FILE* etk_output_file = fopen(fname.c_str(), "wb");
    if (etk_output_file == nullptr) {
      perror("Error opening file");
      return;
    }
    fwrite(&common_horizon, sizeof(int), 1, etk_output_file);
    fwrite(&pmbp->pmesh->time, sizeof(Real), 1, etk_output_file);
    // Write the 4D array to the binary file
    size_t elementsWritten = fwrite(data_out, sizeof(Real), count, etk_output_file);
    if (elementsWritten != count) {
      perror("Error writing to file");
    }
    // Close the file
    fclose(etk_output_file);

    // Write input script for Einstein Toolkit
    ETK_setup_parfile();
    output_count++;
    // delete dataout
    delete[] data_out;
  }
}

void HorizonDump::ETK_setup_parfile() {
  std::string foldername;
  if (common_horizon == 0) {
    foldername = "horizon_"+std::to_string(horizon_ind)+
                "/output_"+std::to_string(output_count);
  } else {
    foldername = "horizon_common/output_"+std::to_string(output_count);
  }
  std::string fname = foldername + "/ET_analyze_BHaH_data_horizon.par";

  FILE *etk_parfile = fopen(fname.c_str(), "w");
  fprintf(etk_parfile,
          "ActiveThorns = \"PUGH SymBase CartGrid3D\"\n"
          "cactus::cctk_itlast = 0\n"
          "#cactus::cctk_show_schedule = \"yes\" # //"
          "Disables initial scheduler printout.\n"
          "cactus::cctk_show_schedule = \"no\" # //"
          "Disables initial scheduler printout.\n"
          "cactus::cctk_show_banners  = \"no\" # // Disables banners.\n"
          "Driver::ghost_size = 0\n"
          "Driver::global_nsize = %d\n"
          "Driver::info = load\n"
          "grid::type = byrange\n"
          "\n"
          "grid::xmin = %e\n"
          "grid::xmax = %e\n"
          "grid::ymin = %e\n"
          "grid::ymax = %e\n"
          "grid::zmin = %e\n"
          "grid::zmax = %e\n"
          "ActiveThorns = ADMBase\n"
          "#ActiveThorns = \"AHFinderDirect SphericalSurface SpaceMask StaticConformal"
          " IOUtil AEILocalInterp PUGHInterp PUGHReduce QuasiLocalMeasures IOBasic"
          " TmunuBase ADMCoupling ADMMacros LocalReduce\"\n"
          "ActiveThorns = \"AHFinderDirect SphericalSurface SpaceMask StaticConformal"
          " IOUtil AEILocalInterp  PUGHInterp PUGHReduce QuasiLocalMeasures IOBasic"
          " TmunuBase LocalReduce\"\n"
          "ActiveThorns = \"readBHaHdata\"\n"
          "ADMBase::metric_type = \"physical\"\n"
          "AHFinderDirect::find_every                             = 1\n"
          "AHFinderDirect::geometry_interpolator_name             ="
          " \"Hermite polynomial interpolation\"\n"
          "AHFinderDirect::geometry_interpolator_pars             = \"order=3\"\n"
          "AHFinderDirect::max_Newton_iterations__initial         = 100\n"
          "AHFinderDirect::max_Newton_iterations__subsequent      = 10\n"
          "AHFinderDirect::N_horizons                             = 1\n"
          "AHFinderDirect::output_BH_diagnostics                  = \"yes\"\n"
          "AHFinderDirect::reset_horizon_after_not_finding[1]     = \"no\"\n"
          "AHFinderDirect::set_mask_for_individual_horizon[1]     = \"no\"\n"
          "AHFinderDirect::surface_interpolator_name              ="
          " \"Hermite polynomial interpolation\"\n"
          "AHFinderDirect::surface_interpolator_pars              = \"order=3\"\n"
          "AHFinderDirect::verbose_level                          = \"physics details\"\n"
          "#AHFinderDirect::verbose_level                         ="
          " \"algorithm details\"\n"
          "AHFinderDirect::which_surface_to_store_info[1]         = 0\n"
          "AHFinderDirect::run_at_CCTK_POSTSTEP = false\n"
          "AHFinderDirect::run_at_CCTK_ANALYSIS = true\n"
          "\n"
          "# Parameters of thorn QuasiLocalMeasures (implementing QuasiLocalMeasures)\n"
          "#QuasiLocalMeasures::interpolator         ="
          " \"Lagrange polynomial interpolation\"\n"
          "#QuasiLocalMeasures::interpolator_options = \"order=4\"\n"
          "QuasiLocalMeasures::interpolator         ="
          " \"Hermite polynomial interpolation\"\n"
          "QuasiLocalMeasures::interpolator_options = \"order=3\"\n"
          "QuasiLocalMeasures::killing_vector_method = axial \n"
          "QuasiLocalMeasures::num_surfaces         = 1\n"
          "QuasiLocalMeasures::spatial_order        = 2\n"
          "QuasiLocalMeasures::surface_index[0]     = 0\n"
          "QuasiLocalMeasures::verbose              = yes\n"
          "#QuasiLocalMeasures::veryverbose          = yes\n"
          "SphericalSurface::nsurfaces       = 1\n"
          "# You may find benefit using super high SphericalSurface "
          "resolutions with very high spin BHs\n"
          "# SphericalSurface::maxntheta       = 301\n"
          "# SphericalSurface::maxnphi         = 504\n"
          "# SphericalSurface::ntheta      [0] = 301\n"
          "# SphericalSurface::nphi        [0] = 504\n"
          "SphericalSurface::maxntheta       = 161\n"
          "SphericalSurface::maxnphi         = 324\n"
          "SphericalSurface::ntheta      [0] = 161\n"
          "SphericalSurface::nphi        [0] = 324\n"
          "SphericalSurface::nghoststheta[0] = 2\n"
          "SphericalSurface::nghostsphi  [0] = 2\n"
          "# SphericalSurface::set_spherical[1]= yes\n"
          "# SphericalSurface::radius       [1]= 40\n"
          "# SphericalSurface::radius       [2]= 80\n"
          "IOBasic::outInfo_every          = 1\n"
          "IOBasic::outInfo_vars           = \"\n"
          "        QuasiLocalMeasures::qlm_scalars\n"
          "        QuasiLocalMeasures::qlm_spin[0]\n"
          "        QuasiLocalMeasures::qlm_radius[0]\n"
          "        QuasiLocalMeasures::qlm_mass[0]\n"
          "        QuasiLocalMeasures::qlm_3det[0] \"\n", horizon_nx,
          -horizon_extent, horizon_extent, -horizon_extent, horizon_extent,
          -horizon_extent, horizon_extent);

  if(horizon_ind==0)
    fprintf(etk_parfile,
            "IOUtil::out_dir = \"AHET_out_horizon_BH_0_ahf_ihf_diags\"\n"
            "readBHaHdata::outfilename = \"horizon_BH_0_ahf_ihf_diags.txt\"\n"
            "readBHaHdata::recent_ah_radius_max_filename = \"ah_radius_max_BH_0.txt\"\n");
  if(horizon_ind==1)
    fprintf(etk_parfile,
            "IOUtil::out_dir = \"AHET_out_horizon_BH_1_ahf_ihf_diags\"\n"
            "readBHaHdata::outfilename = \"horizon_BH_1_ahf_ihf_diags.txt\"\n"
            "readBHaHdata::recent_ah_radius_max_filename = \"ah_radius_max_BH_1.txt\"\n");

  //if(commondata->time == 0) {
    fprintf(etk_parfile,
            "AHFinderDirect::initial_guess_method[1]"
            "                = \"coordinate sphere\"\n"
            "AHFinderDirect::initial_guess__coord_sphere__radius[1] = %e\n",
            r_guess);
  /*} else {
    if(BH==LESSMASSIVE_BH_PREMERGER)
      fprintf(etk_parfile,
              "AHFinderDirect::initial_guess_method[1]                = \"read from named file\"\n"
              "AHFinderDirect::initial_guess__read_from_named_file__file_name[1] = \"AHET_out_horizon_BH_m_ahf_ihf_diags/latest_ah_surface.gp\"\n");
    if(BH==MOREMASSIVE_BH_PREMERGER || BH==BH_POSTMERGER)
      fprintf(etk_parfile,
              "AHFinderDirect::initial_guess_method[1]                = \"read from named file\"\n"
              "AHFinderDirect::initial_guess__read_from_named_file__file_name[1] = \"AHET_out_horizon_BH_M_ahf_ihf_diags/latest_ah_surface.gp\"\n");
  }*/
  fclose(etk_parfile);
}
