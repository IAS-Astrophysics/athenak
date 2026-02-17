//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file zoom_restart.cpp
//! \brief Functions to write and read cyclic zoom restart data

#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "cyclic_zoom/cyclic_zoom.hpp"

//----------------------------------------------------------------------------------------
//! \fn IOWrapperSizeT CyclicZoom::RestartFileSize()
//! \brief Calculate the size of restart file data for cyclic zoom
//! \details Computes total size including ZoomState, ZoomMesh metadata arrays
//!          (nzmb_total, nzmb_eachlevel, lloc_eachzmb), and ZMB data arrays.

IOWrapperSizeT CyclicZoom::RestartFileSize() {
  IOWrapperSizeT res_size = 0;

  // Size of zoom state parameters
  res_size += sizeof(ZoomState);

  // Size of zoom mesh metadata
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
              << offset_zoom << " from rank " << my_rank << std::endl;
  }

  // STEP 1: Use the passed offset_zoom which points to start of cyclic zoom data
  IOWrapperSizeT current_offset = offset_zoom;

  // STEP 2: Root process writes cyclic zoom metadata
  if (my_rank == 0) {
    // Write ZoomState parameters
    resfile.Write_any_type_at(&(zstate), sizeof(ZoomState), current_offset, "byte");
    current_offset += sizeof(ZoomState);

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
  IOWrapperSizeT base_offset = offset_zoom + sizeof(ZoomState) + sizeof(int)
                             + pzmesh->nlevels * sizeof(int)
                             + pzmesh->nzmb_total * sizeof(LogicalLocation);

  // STEP 4: Each rank writes its owned ZMBs to their global logical positions
  for (int gzm = 0; gzm < pzmesh->nzmb_total; ++gzm) {
    // Check if this ZMB is owned by this rank
    if (pzmesh->rank_eachzmb[gzm] == my_rank) {
      // Get local index in zdata for this ZMB
      int zm = pzmesh->lid_eachzmb[gzm];

      // Calculate offset in file for this ZMB based on its global index
      IOWrapperSizeT file_offset = base_offset + gzm * zmb_size * sizeof(Real);

      // Calculate offset in local zdata array
      size_t data_offset = zm * zmb_size;
      resfile.Write_any_type_at(pzdata->zdata.data() + data_offset,
                                zmb_size, file_offset, "Real");
    }
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

  // STEP 1: Read and broadcast ZoomState parameters
  char *zrdata = new char[sizeof(ZoomState)];
  IOWrapperSizeT current_offset = offset_zoom;
  if (my_rank == 0) {
    resfile.Read_bytes_at(zrdata, 1, sizeof(ZoomState), current_offset);
    current_offset += sizeof(ZoomState);
  }
#if MPI_PARALLEL_ENABLED
  MPI_Bcast(zrdata, sizeof(ZoomState), MPI_CHAR, 0, MPI_COMM_WORLD);
#endif
  std::memcpy(&(zstate), &(zrdata[0]), sizeof(ZoomState));
  delete[] zrdata;

  // Update CyclicZoom runtime parameters from restart data
  UpdateAMRFromRestart();

  // STEP 2: Read and broadcast zoom mesh metadata
  int nzmb_total_read;
  if (my_rank == 0) {
    resfile.Read_bytes_at(&nzmb_total_read, 1, sizeof(int), current_offset);
    current_offset += sizeof(int);

    resfile.Read_bytes_at(pzmesh->nzmb_eachlevel, pzmesh->nlevels, sizeof(int),
                          current_offset);
    current_offset += pzmesh->nlevels * sizeof(int);
  }
#if MPI_PARALLEL_ENABLED
  MPI_Bcast(&nzmb_total_read, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(pzmesh->nzmb_eachlevel, pzmesh->nlevels, MPI_INT, 0, MPI_COMM_WORLD);
#endif

  // Resize arrays and read lloc data
  pzmesh->nzmb_total = nzmb_total_read;
  // Rebuild derived mesh structure arrays
  pzmesh->RebuildMeshStructure();

  if (my_rank == 0) {
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
  IOWrapperSizeT base_offset = offset_zoom + sizeof(ZoomState) + sizeof(int)
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
      IOWrapperSizeT file_offset = base_offset + gzm * zmb_size * sizeof(Real);

      // Calculate offset in local zdata array
      size_t data_offset = zm * zmb_size;

      // Read this ZMB's data from file
      resfile.Read_Reals_at(pzdata->zdata.data() + data_offset, zmb_size, file_offset);
    }
  }

  if (zstate.zone > 0) {
    pzmesh->FindRegion(zstate.zone-1);
    pzmesh->SyncMBLists();
    pzdata->LoadFromStorage(zstate.zone-1);
    pzdata->UnpackBuffer();
    // TODO(@mhguo): do you need to remask variables here?
    // MaskVariables();
  }

  PrintCyclicZoomDiagnostics();

  return;
}
