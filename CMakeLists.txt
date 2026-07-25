# Master CMakeLists.txt file for Athena++ (Kokkos version)
#
#------ start with setting various compiler, environment, and build defaults -------------

cmake_minimum_required(VERSION 3.10)
project(athena VERSION 0.1 LANGUAGES C CXX)

# Disable in-source builds to prevent source tree corruption.
if( "${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}" )
  message(FATAL_ERROR "FATAL: In-source builds disabled. Please create a separate build directory in which to run cmake, e.g. mkdir build; cd build; cmake ../" )
endif()

# Athena++ (and Kokkos) require C++17 standard as minimum
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force flag to denote C++ standard, as deprecated c++1z is not supported by nvcc.
set(CMAKE_CXX17_STANDARD_COMPILE_OPTION "-std=c++17")

# Kokkos recommends NOT using default compiler extensions
set(CMAKE_CXX_EXTENSIONS OFF)

# following policy required to enable target_link_libraries() below 
cmake_policy(SET CMP0079 NEW)

# Use 'Release' as default build type. Use 'Debug' for testing.
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release)
endif()

#------ default values for compile time options  -----------------------------------------

option(Athena_SINGLE_PRECISION "Compile for single precision" OFF)
option(Athena_ENABLE_MPI "Compile with MPI parallelism enabled" OFF)
option(Athena_ENABLE_OPENMP "Compile with OpenMP parallelism enabled" OFF)
option(Athena_ENABLE_NURATES "Compile with bns_nurates enabled" OFF)
option(Athena_ENABLE_RNS "Compile with RNS support enabled" OFF)
option(Athena_ENABLE_TORCH "Compile with LibTorch (Rhea flavor mixing) enabled" OFF)
set(PROBLEM built_in_pgens CACHE STRING "Name of problem generator function")

#------ set macros exported to config.hpp ------------------------------------------------

# set single precision macro (true/false)
if (Athena_SINGLE_PRECISION)
  set(SINGLE_PRECISION_ENABLED 1)
else()
  set(SINGLE_PRECISION_ENABLED 0)
endif()

# set MPI macro (true/false)
set(ENABLE_MPI OFF)
if (Athena_ENABLE_MPI)
  find_package(MPI COMPONENTS CXX)
  if (NOT MPI_FOUND)
    message(FATAL_ERROR "MPI package required but could not be found.")
  endif()
  set(ENABLE_MPI ON)
  set(Kokkos_ENABLE_MPI ON CACHE BOOL "Compile Kokkos with MPI")
endif()
if (ENABLE_MPI)
  set(MPI_PARALLEL_ENABLED 1)
else ()
  set(MPI_PARALLEL_ENABLED 0)
endif()

# set OpenMP macro (true/false)
set(ENABLE_OPENMP OFF)
if (Athena_ENABLE_OPENMP)
  find_package(OpenMP COMPONENTS CXX)
  if (NOT OpenMP_FOUND)
    message(FATAL_ERROR "OpenMP package is required but could not be found.")
  endif()
  set(ENABLE_OPENMP ON)
  set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Compile Kokkos with OpenMP")
endif()
if (ENABLE_OPENMP)
  set(OPENMP_PARALLEL_ENABLED 1)
else()
  set(OPENMP_PARALLEL_ENABLED 0)
endif()

# set nurates macro (true/false)
if(Athena_ENABLE_NURATES)
  set(ENABLE_NURATES 1)
  set(M1_TOTAL_NUM_SPECIES 4)
  message(STATUS "Compiling with bns-nurates")
else()
  set(ENABLE_NURATES 0)
  set(M1_TOTAL_NUM_SPECIES 1)
endif ()

# set torch macro (true/false) and locate LibTorch (Rhea flavor mixing, radiation_m1).
# Pin: LibTorch 2.6.0 (PI decision; matches Rhea's own Python/training pin). Point CMake at
# an install via -DCMAKE_PREFIX_PATH=<path-to-libtorch> (standard LibTorch convention) or
# -DTorch_DIR=<path-to-libtorch>/share/cmake/Torch.
#
# KNOWN, DOCUMENTED FRICTION (see design doc rhea_athenak_port_design.md §7/§11.5): unlike
# bns_nurates above (a plain in-tree header directory with no CMake package of its own),
# LibTorch ships a real CMake package, so find_package(Torch) is used rather than
# reimplemented -- but this means it also injects its own CMAKE_CXX_FLAGS and
# _GLIBCXX_USE_CXX11_ABI setting (via the imported `torch` target's
# INTERFACE_COMPILE_OPTIONS, propagated to `athena` through target_link_libraries below).
# That setting MUST agree with whatever Kokkos and the device compiler
# (nvcc_wrapper/hipcc/icpx) were built with, or the result is either a hard link error
# (undefined std::__cxx11::... symbols) or, worse, a silent ODR violation across the
# LibTorch/Kokkos boundary that only misbehaves at runtime. AthenaK's in-repo CI only
# exercises the CUDA+nvcc_wrapper path (.gitlab-ci.yml); there is NO in-repo precedent for
# combining find_package(Torch) with hipcc or icpx -- budget real time for ABI-flag friction
# on the first HIP/SYCL attempt, and check the actual TORCH_CXX_FLAGS value LibTorch's
# TorchConfig.cmake sets (it is empty for at least one CPU-only distribution actually tested
# while wiring this up, see Package 0's final report -- do not assume it is always empty).
if(Athena_ENABLE_TORCH)
  # Rhea flavor mixing hard-requires nspecies == 4 (design doc §1), but M1_TOTAL_NUM_SPECIES
  # is only set to 4 by Athena_ENABLE_NURATES above -- with TORCH=ON and NURATES=OFF it stays
  # 1, RadiationM1::FlavorMix silently early-exits (nspecies <= 1), and flavor_mix = rhea
  # would load the model and then do nothing, with no error (final-review finding, design doc
  # §11 item 13a). Fail loudly at configure time instead. (Longer-term fix is to decouple
  # M1_TOTAL_NUM_SPECIES from nurates; for now Rhea always builds alongside nurates, which is
  # also the only combination the port was actually tested against.)
  if(NOT Athena_ENABLE_NURATES)
    message(FATAL_ERROR
      "Athena_ENABLE_TORCH=ON requires Athena_ENABLE_NURATES=ON: Rhea flavor mixing needs "
      "M1_TOTAL_NUM_SPECIES=4, which is currently only set by Athena_ENABLE_NURATES. "
      "Re-run cmake with -DAthena_ENABLE_NURATES=ON.")
  endif()
  set(ENABLE_TORCH 1)
  find_package(Torch REQUIRED)
  message(STATUS "Compiling with LibTorch (Rhea flavor mixing): ${TORCH_INSTALL_PREFIX}")
else()
  set(ENABLE_TORCH 0)
endif()

#set user problem generator flag
if (NOT ${PROBLEM} STREQUAL "built_in_pgens")
  message(STATUS "Including user-specified problem generator file: ${PROBLEM}")
  set(USER_PROBLEM_ENABLED 1)
else()
  set(USER_PROBLEM_ENABLED 0)
endif()

#------ set various Kokkos option --------------------------------------------------------

# Tell Kokkos to vectorize aggressively
# Kokkos prefers this capitalization for debugging reasons
SET (Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION ON CACHE BOOL
     "Kokkos aggressive vectorization")

# Tell Kokkos we need lambdas in Cuda.
if (Kokkos_ENABLE_CUDA)
  SET (Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "Enable lambda expressions in CUDA")
endif()

# If this is a debug build, set kokkos debug on
if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  add_definitions(-DDEBUG_BUILD)
  message(STATUS "Enabling Kokkos debug mode")
  set(Kokkos_ENABLE_DEBUG ON CACHE BOOL "Most general debug settings")
  set(Kokkos_ENABLE_DEBUG_BOUNDS_CHECK ON CACHE BOOL
     "Bounds checking on Kokkos views")
  set(Kokkos_ENABLE_DEBUG_DUALVIEW_MODIFY_CHECK ON CACHE BOOL
      "Sanity checks on Kokkos DualView")
endif()

#------ create configuration header file from template in /src directory -----------------

# add subdirectories for source code and include files
add_subdirectory(src)
add_subdirectory(kokkos)
include_directories(${Kokkos_INCLUDE_DIRS_RET})

find_package(Threads REQUIRED)
target_link_libraries(athena PUBLIC Threads::Threads)

target_link_libraries(athena PUBLIC Kokkos::kokkos)
if (ENABLE_MPI)
  target_link_libraries(athena PUBLIC MPI::MPI_CXX)
endif()
if (ENABLE_OPENMP)
  target_link_libraries(athena PUBLIC OpenMP::OpenMP_CXX)
endif()
if (ENABLE_TORCH)
  # PRIVATE: LibTorch is an implementation detail of radiation_m1's Rhea flavor-mixing path,
  # not part of athena's own public interface (there is no installed athena library/headers
  # consumed by anything downstream today, but PRIVATE is the correct default regardless).
  target_link_libraries(athena PRIVATE ${TORCH_LIBRARIES})
endif()
if (${PROBLEM} STREQUAL "z4c_two_puncture")
	target_include_directories(athena PRIVATE ${CMAKE_SOURCE_DIR}/twopuncturesc/include)
	target_link_libraries(athena PUBLIC ${CMAKE_SOURCE_DIR}/twopuncturesc/lib/libTwoPunctures.a)
	#target_include_directories(athena PRIVATE ${CMAKE_SOURCE_DIR}/gsl/include)
	#target_link_libraries(athena PUBLIC ${CMAKE_SOURCE_DIR}/gsl/lib/libgsl.a)
        #target_link_libraries(athena PUBLIC ${CMAKE_SOURCE_DIR}/gsl/lib/libgslcblas.a)
	find_package(GSL REQUIRED)
	target_link_libraries(athena PUBLIC GSL::gsl GSL::gslcblas)
	set(TWO_PUNCTURES 1)
else()
	set(TWO_PUNCTURES 0)
endif()

# Optionally include bns_nurates headers
if (ENABLE_NURATES)
  target_include_directories(athena PRIVATE ${CMAKE_SOURCE_DIR}/bns_nurates/include)
endif()

if (${PROBLEM} STREQUAL "z4c_spectre_bbh")
  # Load binary black hole initial data from the SpECTRE code:
  # https://spectre-code.org
  #
  # Uses the `BundledExporter` library from SpECTRE to read the initial data.
  # You can help CMake find the library by setting
  # `-D SPECTRE_ROOT=path/to/spectre/build` to the path where SpECTRE is
  # installed.
  find_library(
    SPECTRE_EXPORTER_LIB
    NAMES BundledExporter
    PATH_SUFFIXES lib
    PATHS ${SPECTRE_ROOT})
  find_path(
    SPECTRE_EXPORTER_INCLUDE_DIR
    NAMES spectre/Exporter.hpp
    PATH_SUFFIXES include
    PATHS ${SPECTRE_ROOT})
  add_library(spectre::Exporter UNKNOWN IMPORTED)
  set_target_properties(
    spectre::Exporter
    PROPERTIES IMPORTED_LOCATION ${SPECTRE_EXPORTER_LIB}
               INTERFACE_INCLUDE_DIRECTORIES ${SPECTRE_EXPORTER_INCLUDE_DIR})
  target_link_libraries(athena PRIVATE spectre::Exporter)
endif()

if (${PROBLEM} STREQUAL "elliptica_bns")
  target_include_directories(athena PRIVATE
      ${CMAKE_SOURCE_DIR}/Elliptica_ID_Reader/include)
  target_link_libraries(athena PUBLIC
      ${CMAKE_SOURCE_DIR}/Elliptica_ID_Reader/lib/libelliptica_id_reader.a)
  if (Athena_ENABLE_OPENMP)
  else ()
    find_package(OpenMP COMPONENTS CXX)
    target_link_libraries(athena PUBLIC OpenMP::OpenMP_CXX)
  endif()
endif()

if (Athena_ENABLE_RNS)
  # Support for RNS library for rotating neutron stars. See
  # https://github.com/computationalrelativity/RNSC
  target_include_directories(athena PRIVATE
      ${CMAKE_SOURCE_DIR}/rnsc/include)
  target_link_libraries(athena PUBLIC
      ${CMAKE_SOURCE_DIR}/rnsc/lib/libRNS.a)
endif()

if (${PROBLEM} STREQUAL "sgrid_bns")
  target_link_libraries(athena PUBLIC
      ${CMAKE_SOURCE_DIR}/sgrid/lib/libsgrid.a)
endif()

if (${PROBLEM} STREQUAL "kadath_bns")
  # Kadath (FUKa) BNS initial data reader.
  # Expects Kadath to be linked at ${CMAKE_SOURCE_DIR}/kadath
  # with libkadath.a in its lib/ directory.
  find_package(Boost REQUIRED)
  target_include_directories(athena PRIVATE
      ${CMAKE_SOURCE_DIR}/kadath/include
      ${CMAKE_SOURCE_DIR}/kadath/include/Kadath_point_h
      #${CMAKE_SOURCE_DIR}/kadath/include/Solvers
      ${Boost_INCLUDE_DIRS})
  target_link_libraries(athena PUBLIC
      ${CMAKE_SOURCE_DIR}/kadath/lib/libkadath.a)
  # Kadath depends on Boost, FFTW3 (spectral transforms) and LAPACK (linear algebra)
  find_package(PkgConfig QUIET)
  if (PKG_CONFIG_FOUND)
    pkg_check_modules(FFTW3 fftw3)
  endif()
  if (FFTW3_FOUND)
    target_link_libraries(athena PUBLIC ${FFTW3_LIBRARIES})
  else()
    find_library(FFTW3_LIB NAMES fftw3)
    target_link_libraries(athena PUBLIC ${FFTW3_LIB})
  endif()
  find_package(LAPACK REQUIRED)
  target_link_libraries(athena PUBLIC ${LAPACK_LIBRARIES})
  find_package(GSL REQUIRED)
  target_link_libraries(athena PUBLIC GSL::gsl GSL::gslcblas)
  if (Athena_ENABLE_OPENMP)
  else ()
    find_package(OpenMP COMPONENTS CXX)
    target_link_libraries(athena PUBLIC OpenMP::OpenMP_CXX)
  endif()
endif()

if (${PROBLEM} STREQUAL "lorene_bns")
  # FIXME(user): The way the Fortran libraries here are linked is not great. Because these
  # are implicit libraries, we shouldn't have to add them manually, but at least on some
  # machines this seems to be necessary. We should find out if there's a better way to
  # do this.
  enable_language(Fortran)
  include(FortranCInterface)
  FortranCInterface_VERIFY(CXX)
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
  find_package(GSL REQUIRED)
  target_include_directories(athena PRIVATE
      ${CMAKE_SOURCE_DIR}/Lorene/Export/C++/Include
      ${CMAKE_SOURCE_DIR}/Lorene/C++/Include)
  target_link_libraries(athena PUBLIC
      ${CMAKE_SOURCE_DIR}/Lorene/Lib/liblorene_export.a
      ${CMAKE_SOURCE_DIR}/Lorene/Lib/liblorene.a
      ${CMAKE_SOURCE_DIR}/Lorene/Lib/liblorenef77.a
      ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}
      GSL::gsl ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES})
endif()

# enable include of header files created in build directories
target_include_directories(athena PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

configure_file(config.hpp.in config.hpp)

install( TARGETS athena DESTINATION bin )

