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

int IOWrapper::Open(const char* fname, FileMode rw, bool single_file_per_rank) {
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
  if (!single_file_per_rank) {
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
                                  bool single_file_per_rank) {
#if MPI_PARALLEL_ENABLED
  if (!single_file_per_rank) {
    MPI_Status status;
    int errcode = MPI_File_read(fh_, buf, cnt*size, MPI_BYTE, &status);
    if (errcode != MPI_SUCCESS) {
      char msg[MPI_MAX_ERROR_STRING];
      int resultlen;
      MPI_Error_string(errcode, msg, &resultlen);
      Kokkos::printf("%.*s\n", resultlen, msg);
      return 0;
    }
    int nread;
    if (MPI_Get_count(&status,MPI_BYTE,&nread) == MPI_UNDEFINED) {return 0;}
    return nread/size;
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
                                     bool single_file_per_rank) {
#if MPI_PARALLEL_ENABLED
  if (!single_file_per_rank) {
    MPI_Status status;
    int errcode = MPI_File_read_at(fh_, offset, buf, cnt*size, MPI_BYTE, &status);
    if (errcode != MPI_SUCCESS) {
      char msg[MPI_MAX_ERROR_STRING];
      int resultlen;
      MPI_Error_string(errcode, msg, &resultlen);
      Kokkos::printf("%.*s\n", resultlen, msg);
      return 0;
    }
    int nread;
    if (MPI_Get_count(&status,MPI_BYTE,&nread) == MPI_UNDEFINED) {return 0;}
    return nread/size;
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
                                         bool single_file_per_rank) {
#if MPI_PARALLEL_ENABLED
  if (!single_file_per_rank) {
    MPI_Status status;
    int errcode = MPI_File_read_at_all(fh_, offset, buf, cnt*size, MPI_BYTE, &status);
    if (errcode != MPI_SUCCESS) {
      char msg[MPI_MAX_ERROR_STRING];
      int resultlen;
      MPI_Error_string(errcode, msg, &resultlen);
      Kokkos::printf("%.*s\n", resultlen, msg);
      return 0;
    }
    int nread;
    if (MPI_Get_count(&status,MPI_BYTE,&nread) == MPI_UNDEFINED) {return 0;}
    return nread/size;
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
                                  bool single_file_per_rank) {
#if MPI_PARALLEL_ENABLED
  if (!single_file_per_rank) {
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
                                     IOWrapperSizeT offset, bool single_file_per_rank) {
#if MPI_PARALLEL_ENABLED
  if (!single_file_per_rank) {
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
                                         bool single_file_per_rank) {
#if MPI_PARALLEL_ENABLED
  if (!single_file_per_rank) {
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
                                      std::string datatype, bool single_file_per_rank) {
#if MPI_PARALLEL_ENABLED
  if (single_file_per_rank) {
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
                                         bool single_file_per_rank) {
#if MPI_PARALLEL_ENABLED
  if (single_file_per_rank) {
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
                                            bool single_file_per_rank) {
#if MPI_PARALLEL_ENABLED
  if (single_file_per_rank) {
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
//! \fn void IOWrapper::Close(bool single_file_per_rank)
//  \brief wrapper for {MPI_File_close} versus {std::fclose}

int IOWrapper::Close(bool single_file_per_rank) {
#if MPI_PARALLEL_ENABLED
  if (!single_file_per_rank) {
    return MPI_File_close(&fh_);
  } else {
    return std::fclose(reinterpret_cast<FILE*>(fh_));
  }
#else
  return std::fclose(fh_);
#endif
}

//----------------------------------------------------------------------------------------
//! \fn int IOWrapper::Seek(IOWrapperSizeT offset, bool single_file_per_rank)
//  \brief wrapper for {MPI_File_seek} versus {std::fseek}

int IOWrapper::Seek(IOWrapperSizeT offset, bool single_file_per_rank) {
#if MPI_PARALLEL_ENABLED
  if (!single_file_per_rank) {
    return MPI_File_seek(fh_, offset, MPI_SEEK_SET);
  } else {
    return std::fseek(reinterpret_cast<FILE*>(fh_), offset, SEEK_SET);
  }
#else
  return std::fseek(fh_, offset, SEEK_SET);
#endif
}

//----------------------------------------------------------------------------------------
//! \fn IOWrapperSizeT IOWrapper::GetPosition(bool single_file_per_rank)
//  \brief wrapper for {MPI_File_get_position} versus {ftell}

IOWrapperSizeT IOWrapper::GetPosition(bool single_file_per_rank) {
#if MPI_PARALLEL_ENABLED
  if (!single_file_per_rank) {
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
