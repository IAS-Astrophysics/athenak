cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(athena VERSION 20.0)

# ----- default values for compile time options  -----------------------------------------
option(ATHENA_SINGLE_PRECISION "Compile for single precision" OFF)
option(ATHENA_MPI_ENABLED "Compile with MPI parallelism enabled" OFF)
option(ATHENA_OPENMP_ENABLED "Compile with OpenMP parallelism enabled" OFF)

# ----- set macros exported to config.hpp ------------------------------------------------
# set single precision macro (true/false)
if ( ATHENA_SINGLE_PRECISION )
  set(SINGLE_PRECISION_ENABLED 1)
else()
  set(SINGLE_PRECISION_ENABLED 0)
endif()

# set MPI macro (true/false)
set(ENABLE_MPI OFF)
if ( ATHENA_MPI_ENABLED )
  find_package(MPI COMPONENTS CXX)
  if (NOT MPI_FOUND)
    message(FATAL_ERROR "MPI package required but could not be found.")
  endif()
  set(ENABLE_MPI ON)
endif()
if (ENABLE_MPI)
  set(MPI_PARALLEL_ENABLED 1)
else ()
  set(MPI_PARALLEL_ENABLED 0)
endif()

# set OpenMP macro (true/false)
set(ENABLE_OPENMP OFF)
if ( ATHENA_OPENMP_ENABLED )
  find_package(OpenMP COMPONENTS CXX)
  if (NOT OpenMP_FOUND)
    message(FATAL_ERROR "OpenMP package is required but could not be found.")
  endif()
  set(ENABLE_OPENMP ON)
endif()
if (ENABLE_OPENMP)
  set(OPENMP_PARALLEL_ENABLED 1)
else()
  set(OPENMP_PARALLEL_ENABLED 0)
endif()

#-----------------------------------------------------------------------------------------
# create configuration header file from template in /src directory
configure_file(config.hpp.in config.hpp)

# add subdirectories containing source code
add_subdirectory(src)

# enable include of header files created in build directories
target_include_directories(athena PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)
