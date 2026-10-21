//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file zoom_restart.cpp
//! \brief Functions to write and read cyclic zoom restart data

#include <cmath>
#include <cstring>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "cyclic_zoom/cyclic_zoom.hpp"

// Zoom metadata format, written as a negative int before nzmb_total so that it cannot be
// mistaken for nzmb_total (>= 0), which is the first int of the original format (1).
// Format 2 adds zmb_data_cnt, checked on reading.
static constexpr int kZoomRstFormat = 2;

//----------------------------------------------------------------------------------------
//! \fn IOWrapperSizeT CyclicZoom::RestartFileSize()
//! \brief Calculate the size of restart file data for cyclic zoom
//! \details Computes total size including ZoomState, ZoomMesh metadata arrays
//!          (format, zmb_data_cnt, nzmb_total, nzmb_eachlevel, lloc_eachzmb), and ZMB
//!          data.

IOWrapperSizeT CyclicZoom::RestartFileSize() {
  IOWrapperSizeT res_size = 0;

  // Size of zoom state parameters
  res_size += sizeof(ZoomState);

  // Size of zoom mesh metadata
  res_size += sizeof(int);                              // -format
  res_size += sizeof(int);                              // zmb_data_cnt
  res_size += sizeof(int);                              // nzmb_total
  res_size += pzmesh->nlevels * sizeof(int);            // nzmb_eachlevel
  res_size += pzmesh->nzmb_total * sizeof(LogicalLocation);  // lloc_eachzmb

  // Size of ZMB data
  res_size += pzmesh->nzmb_total * pzdata->zmb_data_cnt * sizeof(Real);

  return res_size;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::WriteRestartFile()
//! \brief Write cyclic zoom restart data to file
//! \details Writes ZoomState and ZoomMesh metadata, then each rank writes its owned ZMBs.
//!          Metadata includes: nzmb_total, nzmb_eachlevel, and lloc_eachzmb arrays.
//!          Derived arrays are rebuilt during restart.

void CyclicZoom::WriteRestartFile(IOWrapper &resfile, IOWrapperSizeT offset_zoom,
                                  bool single_file_per_rank) {
  int &zmb_size = pzdata->zmb_data_cnt;
  int my_rank = global_variable::my_rank;

  if (verbose && my_rank == 0) {
    std::cout << "CyclicZoom: Writing cyclic zoom restart data at offset "
              << offset_zoom << std::endl;
  }

  // STEP 1: Use the passed offset_zoom which points to start of cyclic zoom data
  IOWrapperSizeT current_offset = offset_zoom;

  // STEP 2: Root process writes cyclic zoom metadata
  if (my_rank == 0) {
    // Write ZoomState parameters
    resfile.Write_any_type_at(&(zstate), sizeof(ZoomState), current_offset, "byte");
    current_offset += sizeof(ZoomState);
  }

  // If zone is 0, skip writing zoom mesh metadata and ZMB data
  if (zstate.zone == 0) {
    return;
  }

  if (my_rank == 0) {
    // Write format tag and ZMB layout (it changes with the physics stored)
    int tag = -kZoomRstFormat;
    resfile.Write_any_type_at(&tag, sizeof(int), current_offset, "byte");
    current_offset += sizeof(int);
    resfile.Write_any_type_at(&zmb_size, sizeof(int), current_offset, "byte");
    current_offset += sizeof(int);
    // Write ZoomMesh metadata
    resfile.Write_any_type_at(&(pzmesh->nzmb_total), sizeof(int), current_offset, "byte");
    current_offset += sizeof(int);
    resfile.Write_any_type_at(pzmesh->nzmb_eachlevel, pzmesh->nlevels*sizeof(int),
                              current_offset, "byte");
    current_offset += pzmesh->nlevels * sizeof(int);
    resfile.Write_any_type_at(pzmesh->lloc_eachzmb.data(),
                              pzmesh->nzmb_total*sizeof(LogicalLocation),
                              current_offset, "byte");
    current_offset += pzmesh->nzmb_total * sizeof(LogicalLocation);
  }

  // STEP 3: Calculate base offset for ZMB data (after metadata)
  IOWrapperSizeT base_offset = offset_zoom + sizeof(ZoomState) + 3 * sizeof(int)
                             + pzmesh->nlevels * sizeof(int)
                             + pzmesh->nzmb_total * sizeof(LogicalLocation);

  // STEP 4: Each rank writes its owned ZMBs to their global logical positions
  for (int gzm = 0; gzm < pzmesh->nzmb_total; ++gzm) {
    // Check if this ZMB is owned by this rank
    if (pzmesh->rank_eachzmb[gzm] == my_rank) {
      // Get local index in zdata for this ZMB
      int zm = pzmesh->lid_eachzmb[gzm];

      // Calculate offset in file for this ZMB based on its global index
      IOWrapperSizeT file_offset = base_offset
                                 + static_cast<IOWrapperSizeT>(gzm)*zmb_size*sizeof(Real);

      // Calculate offset in local zdata array
      size_t data_offset = static_cast<size_t>(zm) * zmb_size;
      if (resfile.Write_any_type_at(pzdata->zdata.data() + data_offset,
                                    zmb_size, file_offset, "Real") != zmb_size) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "cyclic zoom data not written correctly to rst file, "
                  << "restart file is broken." << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
#if MPI_PARALLEL_ENABLED
    // Avoid concurrent non-contiguous writes to the same restart file
    MPI_Barrier(MPI_COMM_WORLD);
#endif
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::ReadRestartFile()
//! \brief Read cyclic zoom restart data from file
//! \details Reads and broadcasts ZoomState and ZoomMesh metadata (nzmb_total,
//!          nzmb_eachlevel, lloc_eachzmb), then calls RebuildMeshStructure() to
//!          reconstruct derived arrays. Each rank reads its owned ZMBs,
//!          loads to zbuf, and unpacks to zoom arrays.

void CyclicZoom::ReadRestartFile(IOWrapper &resfile, IOWrapperSizeT offset_zoom,
                                 bool single_file_per_rank) {
  int my_rank = global_variable::my_rank;

  // STEP 1: Read and broadcast ZoomState parameters. The zoom data is the last section
  // of the file, so a restart file written without cyclic zoom ends at offset_zoom: keep
  // the zoom state set from the input file (zone, direction, next_time = time + runtime).
  char *zrdata = new char[sizeof(ZoomState)];
  IOWrapperSizeT current_offset = offset_zoom;
  int nread = 0;
  if (my_rank == 0) {
    nread = resfile.Read_bytes_at(zrdata, 1, sizeof(ZoomState), current_offset);
    current_offset += sizeof(ZoomState);
  }
#if MPI_PARALLEL_ENABLED
  MPI_Bcast(&nread, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
  if (nread == 0) {
    delete[] zrdata;
    if (my_rank == 0) {
      std::cout << "CyclicZoom: restart file has no cyclic zoom data, starting at zone "
                << zstate.zone << " with next zoom at t = " << zstate.next_time
                << std::endl;
    }
    PrintCyclicZoomDiagnostics();
    return;
  }
#if MPI_PARALLEL_ENABLED
  MPI_Bcast(zrdata, sizeof(ZoomState), MPI_CHAR, 0, MPI_COMM_WORLD);
#endif
  ZoomState zread;
  std::memcpy(&zread, &(zrdata[0]), sizeof(ZoomState));
  delete[] zrdata;
  if (nread != static_cast<int>(sizeof(ZoomState)) ||
      zread.zone < 0 || zread.zone >= zamr.nlevels ||
      zread.last_zone < 0 || zread.last_zone >= zamr.nlevels ||
      (zread.direction != 1 && zread.direction != -1) ||
      !std::isfinite(zread.next_time)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "CyclicZoom: invalid zoom state in restart file (zone=" << zread.zone
              << ", direction=" << zread.direction << ", " << nread << " bytes read); "
              << "was it written with a different nlevels or physics?" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  zstate = zread;

  // Update CyclicZoom runtime parameters from restart data
  UpdateAMRFromRestart();

  // If zone is 0, skip reading zoom mesh metadata and ZMB data
  if (zstate.zone == 0) {
    PrintCyclicZoomDiagnostics();
    return;
  }

  // STEP 2: Read and broadcast zoom mesh metadata. The first int is nzmb_total in the
  // original format and -format otherwise.
  int nzmb_total_read;
  int zmb_size_read = -1;  // not stored in the original format
  int format = 1;
  if (my_rank == 0) {
    int first;
    resfile.Read_bytes_at(&first, 1, sizeof(int), current_offset);
    current_offset += sizeof(int);
    if (first < 0) {
      format = -first;
      resfile.Read_bytes_at(&zmb_size_read, 1, sizeof(int), current_offset);
      current_offset += sizeof(int);
      resfile.Read_bytes_at(&nzmb_total_read, 1, sizeof(int), current_offset);
      current_offset += sizeof(int);
    } else {
      nzmb_total_read = first;
    }
    resfile.Read_bytes_at(pzmesh->nzmb_eachlevel, pzmesh->nlevels, sizeof(int),
                          current_offset);
    current_offset += pzmesh->nlevels * sizeof(int);
  }
#if MPI_PARALLEL_ENABLED
  MPI_Bcast(&format, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nzmb_total_read, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&zmb_size_read, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(pzmesh->nzmb_eachlevel, pzmesh->nlevels, MPI_INT, 0, MPI_COMM_WORLD);
#endif
  if (format != 1 && format != kZoomRstFormat) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "CyclicZoom: unknown zoom restart format " << format << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (format == 1 && my_rank == 0) {
    std::cout << "CyclicZoom: original zoom restart format, the zoom data layout "
              << "cannot be checked" << std::endl;
  }
  if (format != 1 && zmb_size_read != pzdata->zmb_data_cnt) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "CyclicZoom: restart file stores " << zmb_size_read << " values per "
              << "zoom MeshBlock, this build expects " << pzdata->zmb_data_cnt
              << " (different physics or code version)" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Resize arrays and read lloc data
  pzmesh->nzmb_total = nzmb_total_read;
  // Rebuild derived mesh structure arrays
  pzmesh->RebuildMeshStructure();

  if (my_rank == 0 && nzmb_total_read > 0) {
    resfile.Read_bytes_at(pzmesh->lloc_eachzmb.data(), nzmb_total_read,
                          sizeof(LogicalLocation), current_offset);
    current_offset += nzmb_total_read * sizeof(LogicalLocation);
  }
#if MPI_PARALLEL_ENABLED
  // Broadcast LogicalLocation array (4 int32_t per element)
  MPI_Bcast(pzmesh->lloc_eachzmb.data(), nzmb_total_read * 4, MPI_INT32_T, 0,
            MPI_COMM_WORLD);
#endif

  // STEP 3: Calculate base offset for ZMB data (after metadata)
  int nmeta = (format == 1) ? 1 : 3;  // ints before nzmb_eachlevel
  IOWrapperSizeT base_offset = offset_zoom + sizeof(ZoomState) + nmeta * sizeof(int)
                             + pzmesh->nlevels * sizeof(int)
                             + nzmb_total_read * sizeof(LogicalLocation);

  // STEP 4: Each rank reads its owned ZMBs from their global logical positions
  int &zmb_size = pzdata->zmb_data_cnt;

  for (int gzm = 0; gzm < nzmb_total_read; ++gzm) {
    // Check if this ZMB is owned by this rank
    if (pzmesh->rank_eachzmb[gzm] == my_rank) {
      // Get local index in zdata for this ZMB
      int zm = pzmesh->lid_eachzmb[gzm];

      // Calculate offset in file for this ZMB based on its global index
      IOWrapperSizeT file_offset = base_offset
                                 + static_cast<IOWrapperSizeT>(gzm)*zmb_size*sizeof(Real);

      // Calculate offset in local zdata array
      size_t data_offset = static_cast<size_t>(zm) * zmb_size;

      // Read this ZMB's data from file
      if (resfile.Read_Reals_at(pzdata->zdata.data() + data_offset, zmb_size,
                                file_offset) != static_cast<std::size_t>(zmb_size)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "CyclicZoom: zoom data not read correctly from rst "
                  << "file, restart file is broken." << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
  }

  // STEP 5: Load zoom data from host storage to device if needed
  if (zstate.zone > 0) {
    LoadZoomData(zstate.zone-1);
  }

  PrintCyclicZoomDiagnostics();

  return;
}
