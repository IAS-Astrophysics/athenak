//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file io_wrapper.cpp
//! \brief functions that provide wrapper for MPI-IO versus serial input/output

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include "athena.hpp"
#include "io_wrapper.hpp"

namespace {

constexpr const char *kTestMaxChunkEnv = "ATHENAK_TEST_MAX_MPI_BYTES";

IOWrapperSizeT SafeByteCount(IOWrapperSizeT size, IOWrapperSizeT count,
                             const char *context) {
  if (size == 0 || count == 0) {
    return 0;
  }
  if (count > std::numeric_limits<IOWrapperSizeT>::max() / size) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << context << " byte count overflow." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return size*count;
}

#if MPI_PARALLEL_ENABLED
IOWrapperSizeT GetConfiguredMaxChunkedByteCount() {
  static const IOWrapperSizeT max_chunk_bytes = []() {
    IOWrapperSizeT max_bytes = static_cast<IOWrapperSizeT>(std::numeric_limits<int>::max());
    const char *env_value = std::getenv(kTestMaxChunkEnv);
    if (env_value == nullptr || *env_value == '\0') {
      return max_bytes;
    }

    char *parse_end = nullptr;
    unsigned long long parsed = std::strtoull(env_value, &parse_end, 10);
    if (parse_end == env_value || *parse_end != '\0' || parsed == 0) {
      return max_bytes;
    }

    IOWrapperSizeT requested = static_cast<IOWrapperSizeT>(parsed);
    return std::min(requested, max_bytes);
  }();
  return max_chunk_bytes;
}

int CheckedMpiFileRead(IOWrapperFile fh, void *buf, int count, MPI_Datatype type,
                       MPI_Status *status) {
  int errcode = MPI_File_read(fh, buf, count, type, status);
  if (errcode != MPI_SUCCESS) {
    char msg[MPI_MAX_ERROR_STRING];
    int resultlen;
    MPI_Error_string(errcode, msg, &resultlen);
    Kokkos::printf("%.*s\n", resultlen, msg);
  }
  return errcode;
}

int CheckedMpiFileReadAt(IOWrapperFile fh, IOWrapperSizeT offset, void *buf, int count,
                         MPI_Datatype type, MPI_Status *status) {
  int errcode = MPI_File_read_at(fh, offset, buf, count, type, status);
  if (errcode != MPI_SUCCESS) {
    char msg[MPI_MAX_ERROR_STRING];
    int resultlen;
    MPI_Error_string(errcode, msg, &resultlen);
    Kokkos::printf("%.*s\n", resultlen, msg);
  }
  return errcode;
}

int CheckedMpiFileReadAtAll(IOWrapperFile fh, IOWrapperSizeT offset, void *buf, int count,
                            MPI_Datatype type, MPI_Status *status) {
  int errcode = MPI_File_read_at_all(fh, offset, buf, count, type, status);
  if (errcode != MPI_SUCCESS) {
    char msg[MPI_MAX_ERROR_STRING];
    int resultlen;
    MPI_Error_string(errcode, msg, &resultlen);
    Kokkos::printf("%.*s\n", resultlen, msg);
  }
  return errcode;
}

int CheckedMpiFileWrite(IOWrapperFile fh, const void *buf, int count, MPI_Datatype type,
                        MPI_Status *status) {
  int errcode = MPI_File_write(fh, const_cast<void*>(buf), count, type, status);
  if (errcode != MPI_SUCCESS) {
    char msg[MPI_MAX_ERROR_STRING];
    int resultlen;
    MPI_Error_string(errcode, msg, &resultlen);
    Kokkos::printf("%.*s\n", resultlen, msg);
  }
  return errcode;
}

int CheckedMpiFileWriteAt(IOWrapperFile fh, IOWrapperSizeT offset, const void *buf, int count,
                          MPI_Datatype type, MPI_Status *status) {
  int errcode = MPI_File_write_at(fh, offset, const_cast<void*>(buf), count, type, status);
  if (errcode != MPI_SUCCESS) {
    char msg[MPI_MAX_ERROR_STRING];
    int resultlen;
    MPI_Error_string(errcode, msg, &resultlen);
    Kokkos::printf("%.*s\n", resultlen, msg);
  }
  return errcode;
}

int CheckedMpiFileWriteAtAll(IOWrapperFile fh, IOWrapperSizeT offset, const void *buf,
                             int count, MPI_Datatype type, MPI_Status *status) {
  int errcode = MPI_File_write_at_all(fh, offset, const_cast<void*>(buf), count, type,
                                      status);
  if (errcode != MPI_SUCCESS) {
    char msg[MPI_MAX_ERROR_STRING];
    int resultlen;
    MPI_Error_string(errcode, msg, &resultlen);
    Kokkos::printf("%.*s\n", resultlen, msg);
  }
  return errcode;
}

std::size_t ChunkedMpiByteRead(IOWrapperFile fh, char *buf, IOWrapperSizeT total_bytes) {
  IOWrapperSizeT bytes_read = 0;
  IOWrapperSizeT max_chunk = GetConfiguredMaxChunkedByteCount();
  while (bytes_read < total_bytes) {
    IOWrapperSizeT chunk_bytes = std::min(max_chunk, total_bytes - bytes_read);
    MPI_Status status;
    if (CheckedMpiFileRead(fh, buf + bytes_read, static_cast<int>(chunk_bytes), MPI_BYTE,
                           &status) != MPI_SUCCESS) {
      return 0;
    }
    int nread = 0;
    if (MPI_Get_count(&status, MPI_BYTE, &nread) == MPI_UNDEFINED) {
      return 0;
    }
    bytes_read += static_cast<IOWrapperSizeT>(nread);
    if (static_cast<IOWrapperSizeT>(nread) != chunk_bytes) {
      break;
    }
  }
  return bytes_read;
}

std::size_t ChunkedMpiByteReadAt(IOWrapperFile fh, char *buf, IOWrapperSizeT total_bytes,
                                 IOWrapperSizeT offset) {
  IOWrapperSizeT bytes_read = 0;
  IOWrapperSizeT max_chunk = GetConfiguredMaxChunkedByteCount();
  while (bytes_read < total_bytes) {
    IOWrapperSizeT chunk_bytes = std::min(max_chunk, total_bytes - bytes_read);
    MPI_Status status;
    if (CheckedMpiFileReadAt(fh, offset + bytes_read, buf + bytes_read,
                             static_cast<int>(chunk_bytes), MPI_BYTE,
                             &status) != MPI_SUCCESS) {
      return 0;
    }
    int nread = 0;
    if (MPI_Get_count(&status, MPI_BYTE, &nread) == MPI_UNDEFINED) {
      return 0;
    }
    bytes_read += static_cast<IOWrapperSizeT>(nread);
    if (static_cast<IOWrapperSizeT>(nread) != chunk_bytes) {
      break;
    }
  }
  return bytes_read;
}

std::size_t ChunkedMpiByteReadAtAll(IOWrapperFile fh, MPI_Comm comm, char *buf,
                                    IOWrapperSizeT total_bytes, IOWrapperSizeT offset) {
  IOWrapperSizeT max_total_bytes = total_bytes;
  MPI_Allreduce(MPI_IN_PLACE, &max_total_bytes, 1, MPI_UINT64_T, MPI_MAX, comm);

  char dummy = '\0';
  IOWrapperSizeT bytes_read = 0;
  IOWrapperSizeT max_chunk = GetConfiguredMaxChunkedByteCount();
  for (IOWrapperSizeT chunk_begin = 0; chunk_begin < max_total_bytes; chunk_begin += max_chunk) {
    IOWrapperSizeT local_chunk_bytes = 0;
    if (chunk_begin < total_bytes) {
      local_chunk_bytes = std::min(max_chunk, total_bytes - chunk_begin);
    }
    char *local_buf = (local_chunk_bytes > 0) ? (buf + chunk_begin) : &dummy;
    MPI_Status status;
    if (CheckedMpiFileReadAtAll(fh, offset + chunk_begin, local_buf,
                                static_cast<int>(local_chunk_bytes), MPI_BYTE,
                                &status) != MPI_SUCCESS) {
      return 0;
    }
    int nread = 0;
    if (MPI_Get_count(&status, MPI_BYTE, &nread) == MPI_UNDEFINED) {
      return 0;
    }
    bytes_read += static_cast<IOWrapperSizeT>(nread);
  }
  return bytes_read;
}

std::size_t ChunkedMpiByteWrite(IOWrapperFile fh, const char *buf, IOWrapperSizeT total_bytes) {
  IOWrapperSizeT bytes_written = 0;
  IOWrapperSizeT max_chunk = GetConfiguredMaxChunkedByteCount();
  while (bytes_written < total_bytes) {
    IOWrapperSizeT chunk_bytes = std::min(max_chunk, total_bytes - bytes_written);
    MPI_Status status;
    if (CheckedMpiFileWrite(fh, buf + bytes_written, static_cast<int>(chunk_bytes), MPI_BYTE,
                            &status) != MPI_SUCCESS) {
      return 0;
    }
    int nwrite = 0;
    if (MPI_Get_count(&status, MPI_BYTE, &nwrite) == MPI_UNDEFINED) {
      return 0;
    }
    bytes_written += static_cast<IOWrapperSizeT>(nwrite);
    if (static_cast<IOWrapperSizeT>(nwrite) != chunk_bytes) {
      break;
    }
  }
  return bytes_written;
}

std::size_t ChunkedMpiByteWriteAt(IOWrapperFile fh, const char *buf, IOWrapperSizeT total_bytes,
                                  IOWrapperSizeT offset) {
  IOWrapperSizeT bytes_written = 0;
  IOWrapperSizeT max_chunk = GetConfiguredMaxChunkedByteCount();
  while (bytes_written < total_bytes) {
    IOWrapperSizeT chunk_bytes = std::min(max_chunk, total_bytes - bytes_written);
    MPI_Status status;
    if (CheckedMpiFileWriteAt(fh, offset + bytes_written, buf + bytes_written,
                              static_cast<int>(chunk_bytes), MPI_BYTE,
                              &status) != MPI_SUCCESS) {
      return 0;
    }
    int nwrite = 0;
    if (MPI_Get_count(&status, MPI_BYTE, &nwrite) == MPI_UNDEFINED) {
      return 0;
    }
    bytes_written += static_cast<IOWrapperSizeT>(nwrite);
    if (static_cast<IOWrapperSizeT>(nwrite) != chunk_bytes) {
      break;
    }
  }
  return bytes_written;
}

std::size_t ChunkedMpiByteWriteAtAll(IOWrapperFile fh, MPI_Comm comm, const char *buf,
                                     IOWrapperSizeT total_bytes, IOWrapperSizeT offset) {
  IOWrapperSizeT max_total_bytes = total_bytes;
  MPI_Allreduce(MPI_IN_PLACE, &max_total_bytes, 1, MPI_UINT64_T, MPI_MAX, comm);

  char dummy = '\0';
  IOWrapperSizeT bytes_written = 0;
  IOWrapperSizeT max_chunk = GetConfiguredMaxChunkedByteCount();
  for (IOWrapperSizeT chunk_begin = 0; chunk_begin < max_total_bytes; chunk_begin += max_chunk) {
    IOWrapperSizeT local_chunk_bytes = 0;
    if (chunk_begin < total_bytes) {
      local_chunk_bytes = std::min(max_chunk, total_bytes - chunk_begin);
    }
    const char *local_buf = (local_chunk_bytes > 0) ? (buf + chunk_begin) : &dummy;
    MPI_Status status;
    if (CheckedMpiFileWriteAtAll(fh, offset + chunk_begin, local_buf,
                                 static_cast<int>(local_chunk_bytes), MPI_BYTE,
                                 &status) != MPI_SUCCESS) {
      return 0;
    }
    int nwrite = 0;
    if (MPI_Get_count(&status, MPI_BYTE, &nwrite) == MPI_UNDEFINED) {
      return 0;
    }
    bytes_written += static_cast<IOWrapperSizeT>(nwrite);
  }
  return bytes_written;
}
#endif

}  // namespace

namespace io_wrapper {

IOWrapperSizeT GetMaxChunkedByteCount() {
#if MPI_PARALLEL_ENABLED
  return GetConfiguredMaxChunkedByteCount();
#else
  return std::numeric_limits<IOWrapperSizeT>::max();
#endif
}

#if MPI_PARALLEL_ENABLED
void BroadcastBytes(void *buf, IOWrapperSizeT count, int root, MPI_Comm comm) {
  char *byte_buf = reinterpret_cast<char*>(buf);
  IOWrapperSizeT max_chunk = GetConfiguredMaxChunkedByteCount();
  for (IOWrapperSizeT offset = 0; offset < count; offset += max_chunk) {
    IOWrapperSizeT chunk_bytes = std::min(max_chunk, count - offset);
    int mpi_err = MPI_Bcast(byte_buf + offset, static_cast<int>(chunk_bytes), MPI_BYTE,
                            root, comm);
    if (mpi_err != MPI_SUCCESS) {
      char error_string[1024];
      int length_of_error_string;
      MPI_Error_string(mpi_err, error_string, &length_of_error_string);
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "MPI_Bcast failed with error: "
                << std::string(error_string, length_of_error_string) << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
}
#endif

}  // namespace io_wrapper

//----------------------------------------------------------------------------------------
//! \fn int IOWrapper::Open(const char* fname, FileMode rw)
//! \brief wrapper for {MPI_File_open} versus {std::fopen} including error check
//! This function must not be called by multiple threads in shared memory parallel regions

int IOWrapper::Open(const char* fname, FileMode rw, bool use_serial_io) {
  const char* mode;
  switch (rw) {
    case FileMode::read:
      mode = "rb";
      break;
    case FileMode::write:
      mode = "wb";
      break;
    case FileMode::append:
      mode = "ab";
      break;
    default:
      return false;
  }

#if MPI_PARALLEL_ENABLED
  if (!use_serial_io) {
    int mpi_mode;
    switch (rw) {
      case FileMode::read:
        mpi_mode = MPI_MODE_RDONLY;
        break;
      case FileMode::write:
        mpi_mode = MPI_MODE_WRONLY | MPI_MODE_CREATE;
        MPI_File_delete(fname, MPI_INFO_NULL); // truncation
        break;
      case FileMode::append:
        mpi_mode = MPI_MODE_WRONLY | MPI_MODE_APPEND;
        break;
      default:
        return false;
    }

    int errcode = MPI_File_open(comm_, fname, mpi_mode, MPI_INFO_NULL, &fh_);
    if (errcode != MPI_SUCCESS) {
      char msg[MPI_MAX_ERROR_STRING];
      int resultlen;
      MPI_Error_string(errcode, msg, &resultlen);
      Kokkos::printf("%.*s\n", resultlen, msg);
      MPI_Abort(MPI_COMM_WORLD, 1);
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "File '" << fname << "' could not be opened"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  } else {
    FILE* local_fh;
    if ((local_fh = std::fopen(fname, mode)) == nullptr) {
      perror("Error opening file");
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "File '" << fname << "' could not be opened"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    fh_ = reinterpret_cast<IOWrapperFile>(local_fh);
  }
#else
  FILE* local_fh;
  if ((local_fh = std::fopen(fname, mode)) == nullptr) {
    perror("Error opening file");
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "File '" << fname << "' could not be opened"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  fh_ = local_fh;
#endif

  return true;
}

//----------------------------------------------------------------------------------------
//! \fn int IOWrapper::Read_bytes(void *buf, IOWrapperSizeT size, IOWrapperSizeT cnt
//!                                  , bool single_file_per_rank)
//! \brief wrapper for {MPI_File_read} versus {std::fread}.  Returns number of byte-blocks
//! of given "size" actually read.

std::size_t IOWrapper::Read_bytes(void *buf, IOWrapperSizeT size, IOWrapperSizeT cnt,
                                  bool use_serial_io) {
  if (size == 0 || cnt == 0) {
    return 0;
  }
#if MPI_PARALLEL_ENABLED
  if (!use_serial_io) {
    IOWrapperSizeT total_bytes = SafeByteCount(size, cnt, "Read_bytes");
    return ChunkedMpiByteRead(fh_, reinterpret_cast<char*>(buf), total_bytes)/size;
  } else {
    return std::fread(buf, size, cnt, reinterpret_cast<FILE*>(fh_));
  }
#else
  return std::fread(buf, size, cnt, fh_);
#endif
}

//----------------------------------------------------------------------------------------
//! \fn int IOWrapper::Read_bytes_at(void *buf, IOWrapperSizeT size,
//!                                  IOWrapperSizeT cnt, IOWrapperSizeT offset,
//!                                  bool single_file_per_rank)
//! \brief wrapper for {MPI_File_read_at} versus {std::fseek+std::fread}
//! Returns number of byte-blocks of given "size" actually read.

std::size_t IOWrapper::Read_bytes_at(void *buf, IOWrapperSizeT size,
                                     IOWrapperSizeT cnt, IOWrapperSizeT offset,
                                     bool use_serial_io) {
  if (size == 0 || cnt == 0) {
    return 0;
  }
#if MPI_PARALLEL_ENABLED
  if (!use_serial_io) {
    IOWrapperSizeT total_bytes = SafeByteCount(size, cnt, "Read_bytes_at");
    return ChunkedMpiByteReadAt(fh_, reinterpret_cast<char*>(buf), total_bytes, offset)/size;
  } else {
    std::fseek(reinterpret_cast<FILE*>(fh_), offset, SEEK_SET);
    return std::fread(buf, size, cnt, reinterpret_cast<FILE*>(fh_));
  }
#else
  std::fseek(fh_, offset, SEEK_SET);
  return std::fread(buf, size, cnt, fh_);
#endif
}

//----------------------------------------------------------------------------------------
//! \fn int IOWrapper::Read_bytes_at_all(void *buf, IOWrapperSizeT size,
//!                                      IOWrapperSizeT cnt, IOWrapperSizeT offset,
//!                                      bool single_file_per_rank)
//! \brief wrapper for {MPI_File_read_at_all} versus {std::fseek+std::fread}
//! Returns number of byte-blocks of given "size" actually read.

std::size_t IOWrapper::Read_bytes_at_all(void *buf, IOWrapperSizeT size,
                                         IOWrapperSizeT cnt, IOWrapperSizeT offset,
                                         bool use_serial_io) {
  if (size == 0 || cnt == 0) {
#if MPI_PARALLEL_ENABLED
    if (!use_serial_io) {
      char dummy = '\0';
      return ChunkedMpiByteReadAtAll(fh_, comm_, &dummy, 0, offset);
    }
#endif
    return 0;
  }
#if MPI_PARALLEL_ENABLED
  if (!use_serial_io) {
    IOWrapperSizeT total_bytes = SafeByteCount(size, cnt, "Read_bytes_at_all");
    return ChunkedMpiByteReadAtAll(fh_, comm_, reinterpret_cast<char*>(buf), total_bytes,
                                   offset)/size;
  } else {
    std::fseek(reinterpret_cast<FILE*>(fh_), offset, SEEK_SET);
    return std::fread(buf, size, cnt, reinterpret_cast<FILE*>(fh_));
  }
#else
  std::fseek(fh_, offset, SEEK_SET);
  return std::fread(buf, size, cnt, fh_);
#endif
}

//----------------------------------------------------------------------------------------
//! \fn int IOWrapper::Read_Reals(void *buf, IOWrapperSizeT cnt,
//!                               bool single_file_per_rank)
//! \brief wrapper for {MPI_File_read} versus {std::fread} for reading Athena Reals.
//! Returns number of Reals actually read.

std::size_t IOWrapper::Read_Reals(void *buf, IOWrapperSizeT cnt,
                                  bool use_serial_io) {
#if MPI_PARALLEL_ENABLED
  if (!use_serial_io) {
    MPI_Status status;
    int errcode = MPI_File_read(fh_, buf, cnt, MPI_ATHENA_REAL, &status);
    if (errcode != MPI_SUCCESS) {
      char msg[MPI_MAX_ERROR_STRING];
      int resultlen;
      MPI_Error_string(errcode, msg, &resultlen);
      Kokkos::printf("%.*s\n", resultlen, msg);
      return 0;
    }
    int nread;
    if (MPI_Get_count(&status,MPI_ATHENA_REAL,&nread) == MPI_UNDEFINED) {return 0;}
    return nread;
  } else {
    return std::fread(buf, sizeof(Real), cnt, reinterpret_cast<FILE*>(fh_));
  }
#else
  return std::fread(buf, sizeof(Real), cnt, fh_);
#endif
}

//----------------------------------------------------------------------------------------
//! \fn int IOWrapper::Read_Reals_at(void *buf, IOWrapperSizeT cnt,
//!                                  IOWrapperSizeT offset, bool single_file_per_rank)
//! \brief wrapper for {MPI_File_read_at} versus {std::fseek+std::fread} for reading
//!  Athena Reals in parallel.  Returns number of Reals actually read.

std::size_t IOWrapper::Read_Reals_at(void *buf, IOWrapperSizeT cnt,
                                     IOWrapperSizeT offset, bool use_serial_io) {
#if MPI_PARALLEL_ENABLED
  if (!use_serial_io) {
    MPI_Status status;
    int errcode = MPI_File_read_at(fh_, offset, buf, cnt, MPI_ATHENA_REAL, &status);
    if (errcode != MPI_SUCCESS) {
      char msg[MPI_MAX_ERROR_STRING];
      int resultlen;
      MPI_Error_string(errcode, msg, &resultlen);
      Kokkos::printf("%.*s\n", resultlen, msg);
      return 0;
    }
    int nread;
    if (MPI_Get_count(&status,MPI_ATHENA_REAL,&nread) == MPI_UNDEFINED) {return 0;}
    return nread;
  } else {
    std::fseek(reinterpret_cast<FILE*>(fh_), offset, SEEK_SET);
    return std::fread(buf, sizeof(Real), cnt, reinterpret_cast<FILE*>(fh_));
  }
#else
  std::fseek(fh_, offset, SEEK_SET);
  return std::fread(buf, sizeof(Real), cnt, fh_);
#endif
}

//----------------------------------------------------------------------------------------
//! \fn int IOWrapper::Read_Reals_at_all(void *buf, IOWrapperSizeT cnt,
//!                                      IOWrapperSizeT offset, bool single_file_per_rank)
//! \brief wrapper for {MPI_File_read_at_all} versus {std::fseek+std::fread} for reading
//!  Athena Reals in parallel.  Returns number of Reals actually read.

std::size_t IOWrapper::Read_Reals_at_all(void *buf, IOWrapperSizeT cnt,
                                         IOWrapperSizeT offset,
                                         bool use_serial_io) {
#if MPI_PARALLEL_ENABLED
  if (!use_serial_io) {
    MPI_Status status;
    int errcode = MPI_File_read_at_all(fh_, offset, buf, cnt, MPI_ATHENA_REAL, &status);
    if (errcode != MPI_SUCCESS) {
      char msg[MPI_MAX_ERROR_STRING];
      int resultlen;
      MPI_Error_string(errcode, msg, &resultlen);
      Kokkos::printf("%.*s\n", resultlen, msg);
      return 0;
    }
    int nread;
    if (MPI_Get_count(&status,MPI_ATHENA_REAL,&nread) == MPI_UNDEFINED) {return 0;}
    return nread;
  } else {
    std::fseek(reinterpret_cast<FILE*>(fh_), offset, SEEK_SET);
    return std::fread(buf, sizeof(Real), cnt, reinterpret_cast<FILE*>(fh_));
  }
#else
  std::fseek(fh_, offset, SEEK_SET);
  return std::fread(buf, sizeof(Real), cnt, fh_);
#endif
}

//----------------------------------------------------------------------------------------
//! \fn int IOWrapper::Write_any_type()
//! \brief wrapper for {MPI_File_write} versus {std::fwrite} for writing any type of
//! data, specified by the mpitype argument. Returns number of data elements of given
//! "type" actually written.

std::size_t IOWrapper::Write_any_type(const void *buf, IOWrapperSizeT cnt,
                                      std::string datatype, bool use_serial_io) {
#if MPI_PARALLEL_ENABLED
  if (use_serial_io) {
    // Use standard C file handling
    std::size_t datasize;
    if (datatype == "byte") {
      datasize = sizeof(char);
    } else if (datatype == "int") {
      datasize = sizeof(int);
    } else if (datatype == "float") {
      datasize = sizeof(float);
    } else if (datatype == "double") {
      datasize = sizeof(double);
    } else if (datatype == "Real") {
      datasize = sizeof(Real);
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Unrecognized datatype '" << datatype << "'" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    std::size_t written = std::fwrite(buf, datasize, cnt, reinterpret_cast<FILE*>(fh_));
    if (written != cnt) {
      std::cerr << "Error writing data. Expected to write " << cnt
                << " elements, but wrote " << written << std::endl;
    }
    return written;
  } else {
    if (cnt == 0) {
      return 0;
    }
    // set appropriate MPI_Datatype
    MPI_Datatype mpitype;
    if (datatype.compare("byte") == 0) {
      mpitype = MPI_BYTE;
    } else if (datatype.compare("int") == 0) {
      mpitype = MPI_INT;
    } else if (datatype.compare("float") == 0) {
      mpitype = MPI_FLOAT;
    } else if (datatype.compare("double") == 0) {
      mpitype = MPI_DOUBLE;
    } else if (datatype.compare("Real") == 0) {
      mpitype = MPI_ATHENA_REAL;
    } else {
      MPI_Abort(MPI_COMM_WORLD, 1);
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Unrecognized datatype '" << datatype << "'" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (datatype.compare("byte") == 0) {
      return ChunkedMpiByteWrite(fh_, reinterpret_cast<const char*>(buf), cnt);
    }
    // Now write data using MPI-IO
    MPI_Status status;
    int errcode = MPI_File_write(fh_, buf, cnt, mpitype, &status);
    if (errcode != MPI_SUCCESS) {
      char msg[MPI_MAX_ERROR_STRING];
      int resultlen;
      MPI_Error_string(errcode, msg, &resultlen);
      Kokkos::printf("%.*s\n", resultlen, msg);
      return 0;
    }
    int nwrite;
    if (MPI_Get_count(&status, mpitype, &nwrite) == MPI_UNDEFINED) {return 0;}
    return nwrite;
  }
#else
  // set appropriate datasize
  std::size_t datasize;
  if (datatype.compare("byte") == 0) {
    datasize = sizeof(char);
  } else if (datatype.compare("int") == 0) {
    datasize = sizeof(int);
  } else if (datatype.compare("float") == 0) {
    datasize = sizeof(float);
  } else if (datatype.compare("double") == 0) {
    datasize = sizeof(double);
  } else if (datatype.compare("Real") == 0) {
    datasize = sizeof(Real);
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Unrecognized datatype '" << datatype << "'" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // Write data using standard C functions
  std::size_t written = std::fwrite(buf, datasize, cnt, fh_);
  if (written != cnt) {
    std::cerr << "Error writing data. Expected to write " << cnt
              << " elements, but wrote " << written << std::endl;
  }
  return written;
#endif
}

//----------------------------------------------------------------------------------------
//! \fn int IOWrapper::Write_any_type_at()
//! \brief wrapper for {MPI_File_write_at} versus {std::fseek+std::fwrite} for writing any
//! type of data, specified by the datatype argument. Returns number of data elements of
//! given "type" actually written.

std::size_t IOWrapper::Write_any_type_at(const void *buf, IOWrapperSizeT cnt,
                                         IOWrapperSizeT offset, std::string datatype,
                                         bool use_serial_io) {
#if MPI_PARALLEL_ENABLED
  if (use_serial_io) {
    // set appropriate datasize
    std::size_t datasize;
    if (datatype.compare("byte") == 0) {
      datasize = sizeof(char);
    } else if (datatype.compare("int") == 0) {
      datasize = sizeof(int);
    } else if (datatype.compare("float") == 0) {
      datasize = sizeof(float);
    } else if (datatype.compare("double") == 0) {
      datasize = sizeof(double);
    } else if (datatype.compare("Real") == 0) {
      datasize = sizeof(Real);
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Unrecognized datatype '" << datatype << "'" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    // Write data using standard C functions
    std::fseek(reinterpret_cast<FILE*>(fh_), offset, SEEK_SET);
    std::size_t written = std::fwrite(buf, datasize, cnt, reinterpret_cast<FILE*>(fh_));
    if (written != cnt) {
      std::cerr << "Error writing data. Expected to write " << cnt
                << " elements, but wrote " << written << std::endl;
    }
    return written;
  } else {
    if (cnt == 0) {
      return 0;
    }
    // set appropriate MPI_Datatype
    MPI_Datatype mpitype;
    if (datatype.compare("byte") == 0) {
      mpitype = MPI_BYTE;
    } else if (datatype.compare("int") == 0) {
      mpitype = MPI_INT;
    } else if (datatype.compare("float") == 0) {
      mpitype = MPI_FLOAT;
    } else if (datatype.compare("double") == 0) {
      mpitype = MPI_DOUBLE;
    } else if (datatype.compare("Real") == 0) {
      mpitype = MPI_ATHENA_REAL;
    } else {
      MPI_Abort(MPI_COMM_WORLD, 1);
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Unrecognized datatype '" << datatype << "'" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (datatype.compare("byte") == 0) {
      return ChunkedMpiByteWriteAt(fh_, reinterpret_cast<const char*>(buf), cnt, offset);
    }
    // Now write data using MPI-IO
    MPI_Status status;
    int errcode = MPI_File_write_at(fh_, offset, buf, cnt, mpitype, &status);
    if (errcode != MPI_SUCCESS) {
      char msg[MPI_MAX_ERROR_STRING];
      int resultlen;
      MPI_Error_string(errcode, msg, &resultlen);
      Kokkos::printf("%.*s\n", resultlen, msg);
      return 0;
    }
    int nwrite;
    if (MPI_Get_count(&status, mpitype, &nwrite) == MPI_UNDEFINED) {return 0;}
    return nwrite;
  }
#else
  // set appropriate datasize
  std::size_t datasize;
  if (datatype.compare("byte") == 0) {
    datasize = sizeof(char);
  } else if (datatype.compare("int") == 0) {
    datasize = sizeof(int);
  } else if (datatype.compare("float") == 0) {
    datasize = sizeof(float);
  } else if (datatype.compare("double") == 0) {
    datasize = sizeof(double);
  } else if (datatype.compare("Real") == 0) {
    datasize = sizeof(Real);
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Unrecognized datatype '" << datatype << "'" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // Write data using standard C functions
  std::fseek(fh_, offset, SEEK_SET);
  std::size_t written = std::fwrite(buf, datasize, cnt, fh_);
  if (written != cnt) {
    std::cerr << "Error writing data. Expected to write " << cnt
              << " elements, but wrote " << written << std::endl;
  }
  return written;
#endif
}

//----------------------------------------------------------------------------------------
//! \fn int IOWrapper::Write_any_type_at_all()
//! \brief wrapper for {MPI_File_write_at_all} versus {std::fseek+std::fwrite} for writing
//! any type of data, specified by the datatype argument.
//! Returns number of data elements of given "type" actually written.

std::size_t IOWrapper::Write_any_type_at_all(const void *buf, IOWrapperSizeT cnt,
                                            IOWrapperSizeT offset, std::string datatype,
                                            bool use_serial_io) {
#if MPI_PARALLEL_ENABLED
  if (use_serial_io) {
    // set appropriate datasize
    std::size_t datasize;
    if (datatype.compare("byte") == 0) {
      datasize = sizeof(char);
    } else if (datatype.compare("int") == 0) {
      datasize = sizeof(int);
    } else if (datatype.compare("float") == 0) {
      datasize = sizeof(float);
    } else if (datatype.compare("double") == 0) {
      datasize = sizeof(double);
    } else if (datatype.compare("Real") == 0) {
      datasize = sizeof(Real);
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Unrecognized datatype '" << datatype << "'" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    // Write data using standard C functions
    std::fseek(reinterpret_cast<FILE*>(fh_), offset, SEEK_SET);
    std::size_t written = std::fwrite(buf, datasize, cnt, reinterpret_cast<FILE*>(fh_));
    if (written != cnt) {
      std::cerr << "Error writing data. Expected to write " << cnt
                << " elements, but wrote " << written << std::endl;
    }
    return written;
  } else {
    if (cnt == 0) {
      char dummy = '\0';
      if (datatype.compare("byte") == 0) {
        return ChunkedMpiByteWriteAtAll(fh_, comm_, &dummy, 0, offset);
      }
      return 0;
    }
    // set appropriate MPI_Datatype
    MPI_Datatype mpitype;
    if (datatype.compare("byte") == 0) {
      mpitype = MPI_BYTE;
    } else if (datatype.compare("int") == 0) {
      mpitype = MPI_INT;
    } else if (datatype.compare("float") == 0) {
      mpitype = MPI_FLOAT;
    } else if (datatype.compare("double") == 0) {
      mpitype = MPI_DOUBLE;
    } else if (datatype.compare("Real") == 0) {
      mpitype = MPI_ATHENA_REAL;
    } else {
      MPI_Abort(MPI_COMM_WORLD, 1);
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Unrecognized datatype '" << datatype << "'" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (datatype.compare("byte") == 0) {
      return ChunkedMpiByteWriteAtAll(fh_, comm_, reinterpret_cast<const char*>(buf), cnt,
                                      offset);
    }
    // Now write data using MPI-IO
    MPI_Status status;
    int errcode = MPI_File_write_at_all(fh_, offset, buf, cnt, mpitype, &status);
    if (errcode != MPI_SUCCESS) {
      char msg[MPI_MAX_ERROR_STRING];
      int resultlen;
      MPI_Error_string(errcode, msg, &resultlen);
      Kokkos::printf("%.*s\n", resultlen, msg);
      return 0;
    }
    int nwrite;
    if (MPI_Get_count(&status,mpitype,&nwrite) == MPI_UNDEFINED) {return 0;}
    return nwrite;
  }
#else
  // set appropriate datasize
  std::size_t datasize;
  if (datatype.compare("byte") == 0) {
    datasize = sizeof(char);
  } else if (datatype.compare("int") == 0) {
    datasize = sizeof(int);
  } else if (datatype.compare("float") == 0) {
    datasize = sizeof(float);
  } else if (datatype.compare("double") == 0) {
    datasize = sizeof(double);
  } else if (datatype.compare("Real") == 0) {
    datasize = sizeof(Real);
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Unrecognized datatype '" << datatype << "'" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // Write data using standard C functions
  std::fseek(fh_, offset, SEEK_SET);
  std::size_t written = std::fwrite(buf, datasize, cnt, fh_);
  if (written != cnt) {
    std::cerr << "Error writing data. Expected to write " << cnt
              << " elements, but wrote " << written << std::endl;
  }
  return written;
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void IOWrapper::Close(bool use_serial_io)
//  \brief wrapper for {MPI_File_close} versus {std::fclose}

int IOWrapper::Close(bool use_serial_io) {
#if MPI_PARALLEL_ENABLED
  if (!use_serial_io) {
    return MPI_File_close(&fh_);
  } else {
    return std::fclose(reinterpret_cast<FILE*>(fh_));
  }
#else
  return std::fclose(fh_);
#endif
}

//----------------------------------------------------------------------------------------
//! \fn int IOWrapper::Seek(IOWrapperSizeT offset, bool use_serial_io)
//  \brief wrapper for {MPI_File_seek} versus {std::fseek}

int IOWrapper::Seek(IOWrapperSizeT offset, bool use_serial_io) {
#if MPI_PARALLEL_ENABLED
  if (!use_serial_io) {
    return MPI_File_seek(fh_, offset, MPI_SEEK_SET);
  } else {
    return std::fseek(reinterpret_cast<FILE*>(fh_), offset, SEEK_SET);
  }
#else
  return std::fseek(fh_, offset, SEEK_SET);
#endif
}

//----------------------------------------------------------------------------------------
//! \fn IOWrapperSizeT IOWrapper::GetPosition(bool use_serial_io)
//  \brief wrapper for {MPI_File_get_position} versus {ftell}

IOWrapperSizeT IOWrapper::GetPosition(bool use_serial_io) {
#if MPI_PARALLEL_ENABLED
  if (!use_serial_io) {
    MPI_Offset position;
    MPI_File_get_position(fh_, &position);
    return position;
  } else {
    int64_t pos = ftell(reinterpret_cast<FILE*>(fh_));
    return pos;
  }
#else
  int64_t pos = ftell(fh_);
  return pos;
#endif
}
