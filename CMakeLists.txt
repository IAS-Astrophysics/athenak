# Master CMakeLists.txt file for AthenaNike in root of source directories

cmake_minimum_required(VERSION 3.10)
project(athena VERSION 20.0 LANGUAGES C CXX)

# Disable in-source builds to prevent source tree corruption.
if( "${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}" )
  message( FATAL_ERROR "FATAL: In-source builds disabled. Please create a separate build directory in which to run cmake, e.g. mkdir build; cd build; cmake ../" )
endif()

# AthenaNike (and Kokkos) require C++14 standard as minimum
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Kokkos recommends NOT using default compiler extensions
set(CMAKE_CXX_EXTENSIONS OFF)
#set(CMAKE_CXX_FLAGS -O3)
# required to enable target_link_libraries() below 
cmake_policy(SET CMP0079 NEW)

#if(NOT CMAKE_BUILD_TYPE)
#  set(CMAKE_BUILD_TYPE Release)
#endif()
#
#set(CMAKE_CXX_FLAGS "-Wall -Wextra")
#set(CMAKE_CXX_FLAGS_DEBUG "-g")
#set(CMAKE_CXX_FLAGS_RELEASE "-O3")


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

# Tell Kokkos to vectorize aggressively
# Kokkos prefers this capitalization for debugging reasons
SET (Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION ON CACHE BOOL
     "Kokkos aggressive vectorization")

# Tell Kokkos we need lambdas in Cuda.
if (Kokkos_ENABLE_CUDA)
  SET (Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL
       "Enable lambda expressions in CUDA")
endif()

#-----------------------------------------------------------------------------------------
# create configuration header file from template in /src directory
configure_file(config.hpp.in config.hpp)

# add subdirectories for source code and include files
add_subdirectory(src)
add_subdirectory(kokkos)
include_directories(${Kokkos_INCLUDE_DIRS_RET})
target_link_libraries(athena kokkos)

# enable include of header files created in build directories
target_include_directories(athena PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)
