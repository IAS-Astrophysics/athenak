#ifndef OUTPUTS_IO_WRAPPER_HPP_
#define OUTPUTS_IO_WRAPPER_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file io_wrapper.hpp
//  \brief defines a set of small wrapper functions for MPI versus serial outputs.

#include <string>
#include <cstdio>
#include "athena.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
using  IOWrapperFile = MPI_File;
#else
using  IOWrapperFile = FILE*;
#endif

using IOWrapperSizeT = std::uint64_t;

class IOWrapper {
 public:
#if MPI_PARALLEL_ENABLED
  IOWrapper() : fh_(nullptr), comm_(MPI_COMM_WORLD) {}
  void SetCommunicator(MPI_Comm scomm) { comm_=scomm;}
#else
  IOWrapper() {fh_=nullptr;}
#endif
  ~IOWrapper() {}
  // nested type definition of strongly typed/scoped enum in class definition
  enum class FileMode {read, write, append};

  // wrapper functions for basic I/O tasks
  int Open(const char* fname, FileMode rw, bool single_file_per_rank = false);
  std::size_t Read_bytes(void *buf, IOWrapperSizeT size, IOWrapperSizeT count,
                         bool single_file_per_rank = false);
  std::size_t Read_bytes_at(void *buf, IOWrapperSizeT size, IOWrapperSizeT count,
                            IOWrapperSizeT offset, bool single_file_per_rank = false);
  std::size_t Read_bytes_at_all(void *buf, IOWrapperSizeT size, IOWrapperSizeT count,
                                IOWrapperSizeT offset, bool single_file_per_rank = false);
  std::size_t Write_any_type(const void *buf, IOWrapperSizeT count, std::string type,
                             bool single_file_per_rank = false);
  std::size_t Write_any_type_at(const void *buf, IOWrapperSizeT cnt,IOWrapperSizeT offset,
                                std::string datatype, bool single_file_per_rank = false);
  std::size_t Write_any_type_at_all(const void *buf, IOWrapperSizeT cnt,
                                    IOWrapperSizeT offset, std::string datatype,
                                    bool single_file_per_rank = false);
  std::size_t Read_Reals(void *buf, IOWrapperSizeT count,
                         bool single_file_per_rank = false);
  std::size_t Read_Reals_at(void *buf, IOWrapperSizeT count, IOWrapperSizeT offset,
                            bool single_file_per_rank = false);
  std::size_t Read_Reals_at_all(void *buf, IOWrapperSizeT count, IOWrapperSizeT offset,
                                bool single_file_per_rank = false);
  int Close(bool single_file_per_rank = false);
  int Seek(IOWrapperSizeT offset, bool single_file_per_rank = false);
  IOWrapperSizeT GetPosition(bool single_file_per_rank = false);

 private:
  IOWrapperFile fh_;
#if MPI_PARALLEL_ENABLED
  MPI_Comm comm_;
#endif
};
#endif // OUTPUTS_IO_WRAPPER_HPP_
