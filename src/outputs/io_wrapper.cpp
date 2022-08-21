//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file io_wrapper.cpp
//! \brief functions that provide wrapper for MPI-IO versus serial input/output

#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "athena.hpp"
#include "io_wrapper.hpp"

//----------------------------------------------------------------------------------------
//! \fn int IOWrapper::Open(const char* fname, FileMode rw)
//! \brief wrapper for {MPI_File_open} versus {std::fopen} including error check
//! This function must not be called by multiple threads in shared memory parallel regions

int IOWrapper::Open(const char* fname, FileMode rw) {
  // open file for reads
  if (rw == FileMode::read) {
#if MPI_PARALLEL_ENABLED
    int errcode = MPI_File_open(comm_, fname, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh_);
    if (errcode != MPI_SUCCESS) {
      char msg[MPI_MAX_ERROR_STRING];
      int resultlen;
      MPI_Error_string(errcode, msg, &resultlen);
      printf("%.*s\n", resultlen, msg);
      MPI_Abort(MPI_COMM_WORLD, 1);
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Input file '" << fname << "' could not be opened"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
#else
    if ((fh_ = std::fopen(fname,"rb")) == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Input file '" << fname << "' could not be opened"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
#endif

  // open file for writes
  } else if (rw == FileMode::write) {
#if MPI_PARALLEL_ENABLED
    MPI_File_delete(fname, MPI_INFO_NULL); // truncation
    int errcode = MPI_File_open(comm_, fname, MPI_MODE_WRONLY | MPI_MODE_CREATE,
                                MPI_INFO_NULL, &fh_);
    if (errcode != MPI_SUCCESS) {
      char msg[MPI_MAX_ERROR_STRING];
      int resultlen;
      MPI_Error_string(errcode, msg, &resultlen);
      printf("%.*s\n", resultlen, msg);
      MPI_Abort(MPI_COMM_WORLD, 1);
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Input file '" << fname << "' could not be opened"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
#else
    if ((fh_ = std::fopen(fname,"wb")) == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Output file '" << fname << "' could not be opened"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
#endif
  } else {
    return false;
  }

  return true;
}

//----------------------------------------------------------------------------------------
//! \fn int IOWrapper::Read(void *buf, IOWrapperSizeT size, IOWrapperSizeT count)
//! \brief wrapper for {MPI_File_read} versus {std::fread}.  Returns number of byte-blocks//! of given "size" actually read.

std::size_t IOWrapper::Read(void *buf, IOWrapperSizeT size, IOWrapperSizeT count) {
#if MPI_PARALLEL_ENABLED
  MPI_Status status;
  int errcode = MPI_File_read(fh_, buf, count*size, MPI_BYTE, &status);
  if (errcode != MPI_SUCCESS) {
    char msg[MPI_MAX_ERROR_STRING];
    int resultlen;
    MPI_Error_string(errcode, msg, &resultlen);
    printf("%.*s\n", resultlen, msg);
    return 0;
  }
  int nread;
  errcode = MPI_Get_count(&status,MPI_BYTE,&nread);
  if (errcode == MPI_UNDEFINED) {
    char msg[MPI_MAX_ERROR_STRING];
    int resultlen;
    MPI_Error_string(errcode, msg, &resultlen);
    printf("%.*s\n", resultlen, msg);
    return 0;
  }
  return nread/size;
#else
  return std::fread(buf,size,count,fh_);
#endif
}

//----------------------------------------------------------------------------------------
//! \fn int IOWrapper::Read_all(void *buf, IOWrapperSizeT size, IOWrapperSizeT count)
//! \brief wrapper for {MPI_File_read_all} versus {std::fread}.  Returns number of
//! byte-blocks of given "size" actually read.

std::size_t IOWrapper::Read_all(void *buf, IOWrapperSizeT size, IOWrapperSizeT count) {
#if MPI_PARALLEL_ENABLED
  MPI_Status status;
  int errcode = MPI_File_read_all(fh_, buf, count*size, MPI_BYTE, &status);
  if (errcode != MPI_SUCCESS) {
    char msg[MPI_MAX_ERROR_STRING];
    int resultlen;
    MPI_Error_string(errcode, msg, &resultlen);
    printf("%.*s\n", resultlen, msg);
    return 0;
  }
  int nread;
  errcode = MPI_Get_count(&status,MPI_BYTE,&nread);
  if (errcode == MPI_UNDEFINED) {
    char msg[MPI_MAX_ERROR_STRING];
    int resultlen;
    MPI_Error_string(errcode, msg, &resultlen);
    printf("%.*s\n", resultlen, msg);
    return 0;
  }
  return nread/size;
#else
  return std::fread(buf,size,count,fh_);
#endif
}

//----------------------------------------------------------------------------------------
//! \fn int IOWrapper::Read_at_all(void *buf, IOWrapperSizeT size,
//!                             IOWrapperSizeT count, IOWrapperSizeT offset)
//! \brief wrapper for {MPI_File_read_at_all} versus {std::fseek+std::fread}
//! Returns number of byte-blocks of given "size" actually read.

std::size_t IOWrapper::Read_at_all(void *buf, IOWrapperSizeT size,
                                   IOWrapperSizeT count, IOWrapperSizeT offset) {
#if MPI_PARALLEL_ENABLED
  MPI_Status status;
  int errcode = MPI_File_read_at_all(fh_, offset, buf, count*size, MPI_BYTE, &status);
  if (errcode != MPI_SUCCESS) {
    char msg[MPI_MAX_ERROR_STRING];
    int resultlen;
    MPI_Error_string(errcode, msg, &resultlen);
    printf("%.*s\n", resultlen, msg);
    return 0;
  }
  int nread;
  errcode = MPI_Get_count(&status,MPI_BYTE,&nread);
  if (errcode == MPI_UNDEFINED) {
    char msg[MPI_MAX_ERROR_STRING];
    int resultlen;
    MPI_Error_string(errcode, msg, &resultlen);
    printf("%.*s\n", resultlen, msg);
    return 0;
  }
  return nread/size;
#else
  std::fseek(fh_, offset, SEEK_SET);
  return std::fread(buf,size,count,fh_);
#endif
}

//----------------------------------------------------------------------------------------
//! \fn int IOWrapper::Write(const void *buf, IOWrapperSizeT size, IOWrapperSizeT cnt)
//! \brief wrapper for {MPI_File_write} versus {std::fwrite}
//! Returns number of byte-blocks of given "size" actually written.

std::size_t IOWrapper::Write(const void *buf, IOWrapperSizeT size, IOWrapperSizeT cnt) {
#if MPI_PARALLEL_ENABLED
  MPI_Status status;
  int errcode = MPI_File_write(fh_, buf, cnt*size, MPI_BYTE, &status);
  if (errcode != MPI_SUCCESS) {
    char msg[MPI_MAX_ERROR_STRING];
    int resultlen;
    MPI_Error_string(errcode, msg, &resultlen);
    printf("%.*s\n", resultlen, msg);
    return 0;
  }
  int nwrite;
  errcode = MPI_Get_count(&status,MPI_BYTE,&nwrite);
  if (errcode == MPI_UNDEFINED) {
    char msg[MPI_MAX_ERROR_STRING];
    int resultlen;
    MPI_Error_string(errcode, msg, &resultlen);
    printf("%.*s\n", resultlen, msg);
    return 0;
  }
  return nwrite/size;
#else
  return std::fwrite(buf,size,cnt,fh_);
#endif
}

//----------------------------------------------------------------------------------------
//! \fn int IOWrapper::Write_at_all(const void *buf, IOWrapperSizeT size,
//!                                 IOWrapperSizeT cnt, IOWrapperSizeT offset)
//! \brief wrapper for {MPI_File_write_at_all} versus {std::fseek+std::fwrite}.
//! Returns number of byte-blocks of given "size" actually written.

std::size_t IOWrapper::Write_at_all(const void *buf, IOWrapperSizeT size,
                                    IOWrapperSizeT cnt, IOWrapperSizeT offset) {
#if MPI_PARALLEL_ENABLED
  // create new MPI datatype to avoid exceeding limit of 2^31 elements
  MPI_Status status;
  int errcode = MPI_File_write_at_all(fh_, offset, buf, cnt*size, MPI_BYTE, &status);
  if (errcode != MPI_SUCCESS) {
    char msg[MPI_MAX_ERROR_STRING];
    int resultlen;
    MPI_Error_string(errcode, msg, &resultlen);
    printf("%.*s\n", resultlen, msg);
    return 0;
  }
  int nwrite;
  errcode = MPI_Get_count(&status,MPI_BYTE,&nwrite);
std::cout << "amount written = " << nwrite << std::endl;
  if (errcode == MPI_UNDEFINED) {
    char msg[MPI_MAX_ERROR_STRING];
    int resultlen;
    MPI_Error_string(errcode, msg, &resultlen);
    printf("%.*s\n", resultlen, msg);
    return 0;
  }
  return nwrite/size;
#else
  std::fseek(fh_, offset, SEEK_SET);
  return std::fwrite(buf,size,cnt,fh_);
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void IOWrapper::Close()
//  \brief wrapper for {MPI_File_close} versus {std::fclose}

int IOWrapper::Close() {
#if MPI_PARALLEL_ENABLED
  return MPI_File_close(&fh_);
#else
  return std::fclose(fh_);
#endif
}

//----------------------------------------------------------------------------------------
//! \fn int IOWrapper::Seek(IOWrapperSizeT offset)
//  \brief wrapper for {MPI_File_seek} versus {std::fseek}

int IOWrapper::Seek(IOWrapperSizeT offset) {
#if MPI_PARALLEL_ENABLED
  return MPI_File_seek(fh_,offset,MPI_SEEK_SET);
#else
  return std::fseek(fh_, offset, SEEK_SET);
#endif
}

//----------------------------------------------------------------------------------------
//! \fn IOWrapperSizeT IOWrapper::GetPosition()
//  \brief wrapper for {MPI_File_get_position} versus {ftell}

IOWrapperSizeT IOWrapper::GetPosition() {
#if MPI_PARALLEL_ENABLED
  MPI_Offset position;
  MPI_File_get_position(fh_,&position);
  return position;
#else
  return ftell(fh_);
#endif
}
