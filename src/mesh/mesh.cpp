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
#include "meshblock_tree.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

//--------------------------------------------------------------------------------------------------
// Mesh constructor, builds mesh at start of calculation using parameters in input file

Mesh::Mesh(std::unique_ptr<ParameterInput> &pin) {

  //=== Step 1 ===============================================
  // Set properties of Mesh from input parameters, error check
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

  // define some useful variables that indicate 2D/3D calculations
  nx2gt1_ = (root_size.nx2 > 1) ? true : false;
  nx3gt1_ = (root_size.nx3 > 1) ? true : false;

  // set boolean flags indicating type of refinement (if any) depending on input strings
  adaptive = (pin->GetOrAddString("mesh", "refinement", "none") == "adaptive") ? true : false;
  multilevel = ((adaptive) || (pin->GetString("mesh", "refinement") == "static")) ?  true : false;

  //=== Step 2 =======================================================
  // Set properties of MeshBlock(s) from input parameters, error check
  RegionSize block_size;
  block_size.x1rat = root_size.x1rat;
  block_size.x2rat = root_size.x2rat;
  block_size.x3rat = root_size.x3rat;
  block_size.nx1 = pin->GetOrAddInteger("meshblock", "nx1", root_size.nx1);
  if (nx2gt1_) {
    block_size.nx2 = pin->GetOrAddInteger("meshblock", "nx2", root_size.nx2);
  } else {
    block_size.nx2 = root_size.nx2;
  }
  if (nx3gt1_) {
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
  if (block_size.nx1 < 4 || (block_size.nx2 < 4 && nx2gt1_) || (block_size.nx3 < 4 && nx3gt1_)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "MeshBlock must be >= 4 cells in each active dimension" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // calculate the number of MeshBlocks in root level in each dir
  nrmbx1 = root_size.nx1/block_size.nx1;
  nrmbx2 = root_size.nx2/block_size.nx2;
  nrmbx3 = root_size.nx3/block_size.nx3;

  // find maximum number of MeshBlocks in any dir
  int nbmax = (nrmbx1 > nrmbx2) ? nrmbx1 : nrmbx2;
  nbmax = (nbmax > nrmbx3) ? nbmax : nrmbx3;

  // find smallest N such that 2^N > maximum number of MeshBlocks in any dimension (nbmax).
  // Then N is the logical level of root grid. Note 2^N implemented as left-shift (1<<root_level)
  for (root_level=0; ((1<<root_level) < nbmax); root_level++) {}
  int current_level = root_level; 

  ptree = new MeshBlockTree(this);
  ptree->CreateRootGrid();

  //=== Step 3 =======================================================
  // Set properties for SMR/AMR meshes, including regions specified by
  // "refinement" blocks in input file:
  if (adaptive) {
    max_level = pin->GetOrAddInteger("mesh", "numlevel", 1) + root_level - 1;
    if (max_level > 31) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
         << "Number of refinement levels must be smaller than " << 31 - root_level + 1 << std::endl;
      std::exit(EXIT_FAILURE);
    }
  } else {
    max_level = 31;
  }

  if (multilevel) {
    // error check that number of cells in MeshBlock divisible by two
    if (block_size.nx1 % 2 != 0 || 
       (block_size.nx2 % 2 != 0 && nx2gt1_) ||
       (block_size.nx3 % 2 != 0 && nx3gt1_)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
          << "Number of cells in MeshBlock must be divisible by 2 with SMR or AMR." << std::endl;
      std::exit(EXIT_FAILURE);
    }

    // cycle through ParameterInput list and find "refinement" blocks (SMR), extract values
    for (auto it = pin->block.begin(); it != pin->block.end(); ++it) {
      if (it->block_name.compare(0, 10, "refinement") == 0) {
        RegionSize ref_size;
        ref_size.x1min = pin->GetReal(it->block_name, "x1min");
        ref_size.x1max = pin->GetReal(it->block_name, "x1max");
        if (nx2gt1_) {
          ref_size.x2min = pin->GetReal(it->block_name, "x2min");
          ref_size.x2max = pin->GetReal(it->block_name, "x2max");
        } else {
          ref_size.x2min = root_size.x2min;
          ref_size.x2max = root_size.x2max;
        }
        if (nx3gt1_) { 
          ref_size.x3min = pin->GetReal(it->block_name, "x3min");
          ref_size.x3max = pin->GetReal(it->block_name, "x3max");
        } else {
          ref_size.x3min = root_size.x3min;
          ref_size.x3max = root_size.x3max;
        }
        int phy_ref_lev = pin->GetInteger(it->block_name, "level");
        int log_ref_lev = phy_ref_lev + root_level;
        if (log_ref_lev > current_level) current_level = log_ref_lev;

        // error check parameters in "refinement" blocks
        if (phy_ref_lev < 1) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Refinement level must be larger than 0 (root level = 0)" << std::endl;
          std::exit(EXIT_FAILURE);
        }
        if (log_ref_lev > max_level) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Refinement level exceeds maximum allowed (" << max_level << ")" << std::endl
              << "Reduce/specify 'numlevel' in <mesh> input block if using AMR"<< std::endl;
          std::exit(EXIT_FAILURE);
        }
        if (   ref_size.x1min > ref_size.x1max
            || ref_size.x2min > ref_size.x2max
            || ref_size.x3min > ref_size.x3max)  {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Invalid refinement region (xmax < xmin in one direction)."<< std::endl;
          std::exit(EXIT_FAILURE);
        }
        if (   ref_size.x1min < root_size.x1min || ref_size.x1max > root_size.x1max
            || ref_size.x2min < root_size.x2min || ref_size.x2max > root_size.x2max
            || ref_size.x3min < root_size.x3min || ref_size.x3max > root_size.x3max) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Refinement region must be fully contained within root mesh." << std::endl;
          std::exit(EXIT_FAILURE);
        }

        // note: if following is too slow, it could be replaced with bi-section search.
        // Suppose entire root domain is tiled with MeshBlocks at the desired refinement level
        // Find range of x1-integer indices of such MeshBlocks that cover the refinement region
        std::int32_t lx1min = 0, lx1max = 0, lx2min = 0, lx2max = 0, lx3min = 0, lx3max = 0;
        std::int32_t lxmax = nrmbx1*(1<<phy_ref_lev);
        for (lx1min=0; lx1min<lxmax; lx1min++) {
          if (LeftEdgePosition(lx1min+1, lxmax, root_size.x1min, root_size.x1max) > ref_size.x1min)
            break;
        }
        for (lx1max=lx1min; lx1max<lxmax; lx1max++) {
          if (LeftEdgePosition(lx1max+1, lxmax, root_size.x1min, root_size.x1max) >= ref_size.x1max)
            break;
        }
        if (lx1min % 2 == 1) lx1min--;
        if (lx1max % 2 == 0) lx1max++;

        // Find range of x2-integer indices of such MeshBlocks that cover the refinement region
        if (nx2gt1_) { // 2D or 3D
          lxmax = nrmbx2*(1<<phy_ref_lev);
          for (lx2min=0; lx2min<lxmax; lx2min++) {
          if (LeftEdgePosition(lx2min+1, lxmax, root_size.x2min, root_size.x2max) > ref_size.x2min)
            break;
          }
          for (lx2max=lx2min; lx2max<lxmax; lx2max++) {
          if (LeftEdgePosition(lx2max+1, lxmax, root_size.x2min, root_size.x2max) >= ref_size.x2max)
            break;
          }
          if (lx2min % 2 == 1) lx2min--;
          if (lx2max % 2 == 0) lx2max++;
        }

        // Find range of x3-integer indices of such MeshBlocks that cover the refinement region
        if (nx3gt1_) { // 3D
          lxmax = nrmbx3*(1<<phy_ref_lev);
          for (lx3min=0; lx3min<lxmax; lx3min++) {
          if (LeftEdgePosition(lx3min+1, lxmax, root_size.x3min, root_size.x3max) > ref_size.x3min)
            break;
          }
          for (lx3max=lx3min; lx3max<lxmax; lx3max++) {
          if (LeftEdgePosition(lx3max+1, lxmax, root_size.x3min, root_size.x3max) >= ref_size.x3max)
            break;
          }
          if (lx3min % 2 == 1) lx3min--;
          if (lx3max % 2 == 0) lx3max++;
        }

        // Now add these MeshBlocks to the MeshBlockTree (create the finest level)
//        if (ndim == 1) {
//          for (std::int64_t i=lx1min; i<lx1max; i+=2) {
//            LogicalLocation nloc;
//            nloc.level=lrlev, nloc.lx1=i, nloc.lx2=0, nloc.lx3=0;
//            int nnew;
//            tree.AddMeshBlock(nloc, nnew);
//          }
//        }
//        if (ndim == 2) {
//          for (std::int64_t j=lx2min; j<lx2max; j+=2) {
//            for (std::int64_t i=lx1min; i<lx1max; i+=2) {
//              LogicalLocation nloc;
//              nloc.level=lrlev, nloc.lx1=i, nloc.lx2=j, nloc.lx3=0;
//              int nnew;
//              tree.AddMeshBlock(nloc, nnew);
//            }
//          }
//        }
//        if (ndim == 3) {
//          for (std::int64_t k=lx3min; k<lx3max; k+=2) {
//            for (std::int64_t j=lx2min; j<lx2max; j+=2) {
//              for (std::int64_t i=lx1min; i<lx1max; i+=2) {
//                LogicalLocation nloc;
//                nloc.level = lrlev, nloc.lx1 = i, nloc.lx2 = j, nloc.lx3 = k;
//                int nnew;
//                tree.AddMeshBlock(nloc, nnew);
//              }
//            }
//          }
//        }

      }
    }
  } // if(multilevel)

  if (!adaptive) max_level = current_level;

// !!!! finish above block, then test

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
  std::cout <<"Number of physical levels of refinement = "<< (max_level - root_level) 
            <<" (" << (max_level - root_level + 1) << " levels total)" << std::endl;
  std::cout <<"Number of logical  levels of refinement = "<< max_level
            <<" (" << (max_level + 1) << " levels total)" << std::endl;


  return;
}
