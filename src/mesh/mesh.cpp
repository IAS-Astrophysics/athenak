//==================================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//==================================================================================================
//! \file mesh.cpp
//  \brief implementation of functions in Mesh class

#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "outputs/io_wrapper.hpp"
#include "mesh.hpp"
#include "meshblock.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

//----------------------------------------------------------------------------------------
// Mesh constructor, builds mesh at start of calculation using parameters in input file

Mesh::Mesh(std::unique_ptr<ParameterInput> &pin) {

  // Read properties of mesh from input parameters
  RegionSize root_size;
  root_size.x1min = pin->GetReal("mesh", "x1min");
  root_size.x1max = pin->GetReal("mesh", "x1max");
  root_size.x2min = pin->GetReal("mesh", "x2min");
  root_size.x2max = pin->GetReal("mesh", "x2max");
  root_size.x3min = pin->GetReal("mesh", "x3min");
  root_size.x3max = pin->GetReal("mesh", "x3max");
  root_size.x1rat = pin->GetOrAddReal("mesh", "x1rat", 1.0);
  root_size.x2rat = pin->GetOrAddReal("mesh", "x2rat", 1.0);
  root_size.x3rat = pin->GetOrAddReal("mesh", "x3rat", 1.0);
  root_size.nx1   = pin->GetInteger("mesh", "nx1");
  root_size.nx2   = pin->GetInteger("mesh", "nx2");
  root_size.nx3   = pin->GetInteger("mesh", "nx3");
  root_size.nghost = pin->GetOrAddReal("mesh", "nghost", 2);

  // following sets boolean flags to true/false depending on input strings
  adaptive = (pin->GetOrAddString("mesh", "refinement", "none") == "adaptive") ? true : false;
  multilevel = ((adaptive) || (pin->GetOrAddString("mesh", "refinement", "none") == "static")) ?
               true : false;

  // error check physical size of mesh (root level) from input file.
  if (root_size.x1max <= root_size.x1min) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Input x1max must be larger than x1min: x1min=" << root_size.x1min
        << " x1max=" << root_size.x1max << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (root_size.x2max <= root_size.x2min) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Input x2max must be larger than x2min: x2min=" << root_size.x2min
        << " x2max=" << root_size.x2max << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (root_size.x3max <= root_size.x3min) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Input x3max must be larger than x3min: x3min=" << root_size.x3min
        << " x3max=" << root_size.x3max << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // error check requested number of grid cells for entire root domain
  if (root_size.nx1 < 4) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "In mesh block in input file nx1 must be >= 4, but nx1=" << root_size.nx1 << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (root_size.nx2 < 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "In mesh block in input file nx2 must be >= 1, but nx2=" << root_size.nx2 << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (root_size.nx3 < 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "In mesh block in input file nx3 must be >= 1, but nx3=" << root_size.nx3 << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (root_size.nx2 == 1 && root_size.nx3 > 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "In mesh block in input file: nx2=1, nx3=" << root_size.nx3 
        << ", but 2D problems in x1-x3 plane not supported" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // error check number of ghost zones
  if (root_size.nghost < 2) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "More than 2 ghost zones required, but nghost=" << root_size.nghost << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if ((multilevel) && (root_size.nghost % 2 != 0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Number of ghost zones must be divisible by two for SMR/AMR calculations, but nghost="
        << root_size.nghost << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // define some useful variables used to save space in rest of function
  bool nx2gt1 = (root_size.nx2 > 1) ? true : false;
  bool nx3gt1 = (root_size.nx3 > 1) ? true : false;

  // read and set MeshBlock parameters
  RegionSize block_size;
  block_size.x1rat = root_size.x1rat;
  block_size.x2rat = root_size.x2rat;
  block_size.x3rat = root_size.x3rat;
  block_size.nx1 = pin->GetOrAddInteger("meshblock", "nx1", root_size.nx1);
  if (nx2gt1) {
    block_size.nx2 = pin->GetOrAddInteger("meshblock", "nx2", root_size.nx2);
  } else {
    block_size.nx2 = root_size.nx2;
  }
  if (nx3gt1) {
    block_size.nx3 = pin->GetOrAddInteger("meshblock", "nx3", root_size.nx3);
  } else {
    block_size.nx3 = root_size.nx3;
  }

  // error check consistency of the block and mesh
  if (   root_size.nx1 % block_size.nx1 != 0
      || root_size.nx2 % block_size.nx2 != 0
      || root_size.nx3 % block_size.nx3 != 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Mesh must be evenly divisible by MeshBlocks" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (block_size.nx1 < 4 || (block_size.nx2 < 4 && nx2gt1) || (block_size.nx3 < 4 && nx3gt1)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "MeshBlock size must be larger than or equal to 4 cells." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // calculate the number of MeshBlocks in root level in each dir
  nrmbx1 = root_size.nx1/block_size.nx1;
  nrmbx2 = root_size.nx2/block_size.nx2;
  nrmbx3 = root_size.nx3/block_size.nx3;


#ifdef MPI_PARALLEL
  // check if there are sufficient blocks
  if (nbtotal < Globals::nranks) {
    if (mesh_test == 0) {
      msg << "### FATAL ERROR in Mesh constructor" << std::endl
          << "Too few mesh blocks: nbtotal ("<< nbtotal <<") < nranks ("
          << Globals::nranks << ")" << std::endl;
      ATHENA_ERROR(msg);
    } else { // test
      std::cout << "### Warning in Mesh constructor" << std::endl
                << "Too few mesh blocks: nbtotal ("<< nbtotal <<") < nranks ("
                << Globals::nranks << ")" << std::endl;
    }
  }
#endif

}

//--------------------------------------------------------------------------------------------------
// Mesh constructor for restarts. Load the restart file


//--------------------------------------------------------------------------------------------------
// destructor

Mesh::~Mesh() {
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::OutputMeshStructure(int ndim)
//  \brief print the mesh structure information

void Mesh::OutputMeshStructure() {

  // open 'mesh_structure.dat' file
  FILE *fp = nullptr;
  if ((fp = std::fopen("mesh_structure.dat","wb")) == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Cannot open 'mesh_structure.dat' file for output" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Write overall Mesh structure to stdout and file
  std::cout << std::endl;
  std::cout <<"Root grid = "<< nrmbx1 <<" x "<< nrmbx2 <<" x "<< nrmbx3 <<" MeshBlocks"<< std::endl;
  std::cout <<"Total number of MeshBlocks = " << nmbtotal << std::endl;
  std::cout <<"Number of physical refinement levels = "<< (max_level - root_level) << std::endl;
  std::cout <<"Number of logical  refinement levels = "<< max_level << std::endl;


  return;
}
