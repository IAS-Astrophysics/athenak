//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mesh.cpp
//  \brief implementation of functions in Mesh class

#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "outputs/io_wrapper.hpp"
#include "bvals/bvals.hpp"
#include "mesh.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

//----------------------------------------------------------------------------------------
// Mesh constructor, builds mesh at start of calculation using parameters in input file

Mesh::Mesh(std::unique_ptr<ParameterInput> &pin) : tree(this) {

  //=== Step 1 ===============================================
  // Set properties of Mesh from input parameters, error check
  mesh_size.x1min = pin->GetReal("mesh", "x1min");
  mesh_size.x1max = pin->GetReal("mesh", "x1max");
  mesh_size.x2min = pin->GetReal("mesh", "x2min");
  mesh_size.x2max = pin->GetReal("mesh", "x2max");
  mesh_size.x3min = pin->GetReal("mesh", "x3min");
  mesh_size.x3max = pin->GetReal("mesh", "x3max");
  mesh_size.x1rat = pin->GetOrAddReal("mesh", "x1rat", 1.0);
  mesh_size.x2rat = pin->GetOrAddReal("mesh", "x2rat", 1.0);
  mesh_size.x3rat = pin->GetOrAddReal("mesh", "x3rat", 1.0);
  mesh_size.nx1   = pin->GetInteger("mesh", "nx1");
  mesh_size.nx2   = pin->GetInteger("mesh", "nx2");
  mesh_size.nx3   = pin->GetInteger("mesh", "nx3");
  mesh_size.nghost = pin->GetOrAddReal("mesh", "nghost", 2);

  mesh_bcs[BoundaryFace::inner_x1] =
    GetBoundaryFlag(pin->GetOrAddString("mesh", "ix1_bc", "none"));
  mesh_bcs[BoundaryFace::outer_x1] =
    GetBoundaryFlag(pin->GetOrAddString("mesh", "ox1_bc", "none"));
  mesh_bcs[BoundaryFace::inner_x2] =
    GetBoundaryFlag(pin->GetOrAddString("mesh", "ix2_bc", "none"));
  mesh_bcs[BoundaryFace::outer_x2] =
    GetBoundaryFlag(pin->GetOrAddString("mesh", "ox2_bc", "none"));
  mesh_bcs[BoundaryFace::inner_x3] =
    GetBoundaryFlag(pin->GetOrAddString("mesh", "ix3_bc", "none"));
  mesh_bcs[BoundaryFace::outer_x3] =
    GetBoundaryFlag(pin->GetOrAddString("mesh", "ox3_bc", "none"));

  // define some useful variables that indicate 2D/3D calculations
  nx2gt1 = (mesh_size.nx2 > 1) ? true : false;
  nx3gt1 = (mesh_size.nx3 > 1) ? true : false;

  // set boolean flags indicating type of refinement (if any) depending on input strings
  adaptive = 
    (pin->GetOrAddString("mesh", "refinement", "none") == "adaptive") ? true : false;
  multilevel =
    ((adaptive) || (pin->GetString("mesh", "refinement") == "static")) ?  true : false;

  // error check physical size of mesh (root level) from input file.
  if (mesh_size.x1max <= mesh_size.x1min) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Input x1max must be larger than x1min: x1min=" << mesh_size.x1min
        << " x1max=" << mesh_size.x1max << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (mesh_size.x2max <= mesh_size.x2min) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Input x2max must be larger than x2min: x2min=" << mesh_size.x2min
        << " x2max=" << mesh_size.x2max << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (mesh_size.x3max <= mesh_size.x3min) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Input x3max must be larger than x3min: x3min=" << mesh_size.x3min
        << " x3max=" << mesh_size.x3max << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // error check requested number of grid cells for entire root domain
  if (mesh_size.nx1 < 4) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "In mesh block in input file nx1 must be >= 4, but nx1=" << mesh_size.nx1
        << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (mesh_size.nx2 < 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "In mesh block in input file nx2 must be >= 1, but nx2=" << mesh_size.nx2
        << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (mesh_size.nx3 < 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "In mesh block in input file nx3 must be >= 1, but nx3=" << mesh_size.nx3
        << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (mesh_size.nx2 == 1 && mesh_size.nx3 > 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "In mesh block in input file: nx2=1, nx3=" << mesh_size.nx3 
        << ", but 2D problems in x1-x3 plane not supported" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // error check number of ghost zones
  if (mesh_size.nghost < 2) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
      << "More than 2 ghost zones required, but nghost=" << mesh_size.nghost << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if ((multilevel) && (mesh_size.nghost % 2 != 0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
      << "Number of ghost zones must be divisible by two for SMR/AMR calculations, "
      << "but nghost=" << mesh_size.nghost << std::endl;
    std::exit(EXIT_FAILURE);
  }

  //=== Step 2 =======================================================
  // Set properties of MeshBlock(s) from input parameters, error check
  RegionSize block_size;
  block_size.x1rat = mesh_size.x1rat;
  block_size.x2rat = mesh_size.x2rat;
  block_size.x3rat = mesh_size.x3rat;
  block_size.nx1 = pin->GetOrAddInteger("meshblock", "nx1", mesh_size.nx1);
  if (nx2gt1) {
    block_size.nx2 = pin->GetOrAddInteger("meshblock", "nx2", mesh_size.nx2);
  } else {
    block_size.nx2 = mesh_size.nx2;
  }
  if (nx3gt1) {
    block_size.nx3 = pin->GetOrAddInteger("meshblock", "nx3", mesh_size.nx3);
  } else {
    block_size.nx3 = mesh_size.nx3;
  }

  // error check consistency of the block and mesh
  if (   mesh_size.nx1 % block_size.nx1 != 0
      || mesh_size.nx2 % block_size.nx2 != 0
      || mesh_size.nx3 % block_size.nx3 != 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Mesh must be evenly divisible by MeshBlocks" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if ( block_size.nx1 < 4 ||
      (block_size.nx2 < 4 && nx2gt1) ||
      (block_size.nx3 < 4 && nx3gt1) ) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "MeshBlock must be >= 4 cells in each active dimension" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // calculate the number of MeshBlocks in root level in each dir
  nmbx1_r = mesh_size.nx1/block_size.nx1;
  nmbx2_r = mesh_size.nx2/block_size.nx2;
  nmbx3_r = mesh_size.nx3/block_size.nx3;

  // find maximum number of MeshBlocks in any dir
  int nmbmax = (nmbx1_r > nmbx2_r) ? nmbx1_r : nmbx2_r;
  nmbmax = (nmbmax > nmbx3_r) ? nmbmax : nmbx3_r;

  // Find smallest N such that 2^N > max number of MeshBlocks in any dimension (nmbmax)
  // Then N is logical level of root grid.  2^N implemented as left-shift (1<<root_level)
  for (root_level=0; ((1<<root_level) < nmbmax); root_level++) {}
  int current_level = root_level; 

  //=== Step 3 =======================================================
  //

  tree.CreateRootGrid();

  //=== Step 4 =======================================================
  // Error check properties for SMR/AMR meshes.
  // Expand MeshBlockTree to include "refinement" regions specified in input file:

  if (adaptive) {
    max_level = pin->GetOrAddInteger("mesh", "numlevel", 1) + root_level - 1;
    if (max_level > 31) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Number of refinement levels must be smaller than " 
                << 31 - root_level + 1 << std::endl;
      std::exit(EXIT_FAILURE);
    }
  } else {
    max_level = 31;
  }

  if (multilevel) {
    // error check that number of cells in MeshBlock divisible by two
    if (block_size.nx1 % 2 != 0 || 
       (block_size.nx2 % 2 != 0 && nx2gt1) ||
       (block_size.nx3 % 2 != 0 && nx3gt1)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Number of cells in MeshBlock must be divisible by 2 "
                << "with SMR or AMR." << std::endl;
      std::exit(EXIT_FAILURE);
    }

    // cycle through ParameterInput list and find "refinement" blocks (SMR), extract data
    for (auto it = pin->block.begin(); it != pin->block.end(); ++it) {
      if (it->block_name.compare(0, 10, "refinement") == 0) {
        RegionSize ref_size;
        ref_size.x1min = pin->GetReal(it->block_name, "x1min");
        ref_size.x1max = pin->GetReal(it->block_name, "x1max");
        if (nx2gt1) {
          ref_size.x2min = pin->GetReal(it->block_name, "x2min");
          ref_size.x2max = pin->GetReal(it->block_name, "x2max");
        } else {
          ref_size.x2min = mesh_size.x2min;
          ref_size.x2max = mesh_size.x2max;
        }
        if (nx3gt1) { 
          ref_size.x3min = pin->GetReal(it->block_name, "x3min");
          ref_size.x3max = pin->GetReal(it->block_name, "x3max");
        } else {
          ref_size.x3min = mesh_size.x3min;
          ref_size.x3max = mesh_size.x3max;
        }
        int phy_ref_lev = pin->GetInteger(it->block_name, "level");
        int log_ref_lev = phy_ref_lev + root_level;
        if (log_ref_lev > current_level) current_level = log_ref_lev;

        // error check parameters in "refinement" blocks
        if (phy_ref_lev < 1) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ 
              << std::endl << "Refinement level must be larger than 0 (root level = 0)"
              << std::endl;
          std::exit(EXIT_FAILURE);
        }
        if (log_ref_lev > max_level) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Refinement level exceeds maximum allowed ("
              << max_level << ")" << std::endl << "Reduce/specify 'numlevel' in <mesh> "
              << "input block if using AMR" << std::endl;
          std::exit(EXIT_FAILURE);
        }
        if (   ref_size.x1min > ref_size.x1max
            || ref_size.x2min > ref_size.x2max
            || ref_size.x3min > ref_size.x3max)  {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Invalid refinement region (xmax < xmin in one direction)."
              << std::endl;
          std::exit(EXIT_FAILURE);
        }
        if (   ref_size.x1min < mesh_size.x1min || ref_size.x1max > mesh_size.x1max
            || ref_size.x2min < mesh_size.x2min || ref_size.x2max > mesh_size.x2max
            || ref_size.x3min < mesh_size.x3min || ref_size.x3max > mesh_size.x3max) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Refinement region must be fully contained within root mesh"
              << std::endl;
          std::exit(EXIT_FAILURE);
        }

        // note: if following is too slow, it could be replaced with bi-section search.
        // Suppose entire root domain is tiled with MeshBlocks at the desired refinement
        // level. Find range of x1-integer indices of such MeshBlocks that cover the
        // refinement region
        std::int32_t lx1min = 0, lx1max = 0;
        std::int32_t lx2min = 0, lx2max = 0;
        std::int32_t lx3min = 0, lx3max = 0;
        std::int32_t lxmax = nmbx1_r*(1<<phy_ref_lev);
        for (lx1min=0; lx1min<lxmax; lx1min++) {
          if (LeftEdgePosition(lx1min+1, lxmax, mesh_size.x1min, mesh_size.x1max) >
              ref_size.x1min)
            break;
        }
        for (lx1max=lx1min; lx1max<lxmax; lx1max++) {
          if (LeftEdgePosition(lx1max+1, lxmax, mesh_size.x1min, mesh_size.x1max) >=
              ref_size.x1max)
            break;
        }
        if (lx1min % 2 == 1) lx1min--;
        if (lx1max % 2 == 0) lx1max++;

        // Find range of x2-indices of such MeshBlocks that cover the refinement region
        if (nx2gt1) { // 2D or 3D
          lxmax = nmbx2_r*(1<<phy_ref_lev);
          for (lx2min=0; lx2min<lxmax; lx2min++) {
            if (LeftEdgePosition(lx2min+1, lxmax, mesh_size.x2min, mesh_size.x2max) >
                ref_size.x2min)
            break;
          }
          for (lx2max=lx2min; lx2max<lxmax; lx2max++) {
            if (LeftEdgePosition(lx2max+1, lxmax, mesh_size.x2min, mesh_size.x2max) >=
                ref_size.x2max)
            break;
          }
          if (lx2min % 2 == 1) lx2min--;
          if (lx2max % 2 == 0) lx2max++;
        }

        // Find range of x3-indices of such MeshBlocks that cover the refinement region
        if (nx3gt1) { // 3D
          lxmax = nmbx3_r*(1<<phy_ref_lev);
          for (lx3min=0; lx3min<lxmax; lx3min++) {
            if (LeftEdgePosition(lx3min+1, lxmax, mesh_size.x3min, mesh_size.x3max) >
                ref_size.x3min)
            break;
          }
          for (lx3max=lx3min; lx3max<lxmax; lx3max++) {
            if (LeftEdgePosition(lx3max+1, lxmax, mesh_size.x3min, mesh_size.x3max) >=
                ref_size.x3max)
            break;
          }
          if (lx3min % 2 == 1) lx3min--;
          if (lx3max % 2 == 0) lx3max++;
        }

        // Now add these MeshBlocks to the MeshBlockTree (create the finest level)
        if ( !(nx2gt1) && !(nx3gt1)) {  // 1D
          for (std::int32_t i=lx1min; i<lx1max; i+=2) {
            LogicalLocation nloc;
            nloc.level=log_ref_lev, nloc.lx1=i, nloc.lx2=0, nloc.lx3=0;
            int nnew;
            tree.AddMeshBlock(nloc, nnew);
          }
        }
        if (nx2gt1 && !(nx3gt1)) {  // 2D
          for (std::int32_t j=lx2min; j<lx2max; j+=2) {
            for (std::int32_t i=lx1min; i<lx1max; i+=2) {
              LogicalLocation nloc;
              nloc.level=log_ref_lev, nloc.lx1=i, nloc.lx2=j, nloc.lx3=0;
              int nnew;
              tree.AddMeshBlock(nloc, nnew);
            }
          }
        }
        if (nx2gt1 && nx3gt1) {  // 3D
          for (std::int32_t k=lx3min; k<lx3max; k+=2) {
            for (std::int32_t j=lx2min; j<lx2max; j+=2) {
              for (std::int32_t i=lx1min; i<lx1max; i+=2) {
                LogicalLocation nloc;
                nloc.level = log_ref_lev, nloc.lx1 = i, nloc.lx2 = j, nloc.lx3 = k;
                int nnew;
                tree.AddMeshBlock(nloc, nnew);
              }
            }
          }
        }

      }
    }
  } // if(multilevel)

  if (!adaptive) max_level = current_level;

  // initial mesh hierarchy construction is completed here
  tree.CountMeshBlock(nmbtotal);
  loclist = new LogicalLocation[nmbtotal];
  tree.GetMeshBlockList(loclist, nullptr, nmbtotal);

  //=== Step 5 =======================================================

#if MPI_PARALLEL_ENABLED
  // check there is at least one MeshBlock per MPI rank
  if (nmbtotal < global_variable::nranks) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Fewer MeshBlocks (nmbtotal="<<nmbtotal<") than MPI ranks (nranks="
        << global_variable::nranks << ")" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif

  ranklist = new int[nmbtotal];
  nslist = new int[global_variable::nranks];
  nblist = new int[global_variable::nranks];
  costlist = new double[nmbtotal];
  if (adaptive) { // allocate arrays for AMR
    nref = new int[global_variable::nranks];
    nderef = new int[global_variable::nranks];
    rdisp = new int[global_variable::nranks];
    ddisp = new int[global_variable::nranks];
    bnref = new int[global_variable::nranks];
    bnderef = new int[global_variable::nranks];
    brdisp = new int[global_variable::nranks];
    bddisp = new int[global_variable::nranks];
  }

  // initialize cost array with the simplest estimate; all the blocks are equal
  for (int i=0; i<nmbtotal; i++) {costlist[i] = 1.0;}
  LoadBalance(costlist, ranklist, nslist, nblist, nmbtotal);

  //=== Step 5 =======================================================
  // create MeshBlock list for this process

  gids = nslist[global_variable::my_rank];
  gide = gids + nblist[global_variable::my_rank] - 1;
  nmbthisrank = nblist[global_variable::my_rank];
  
// create MeshBlocks for this node
  for (int i=gids; i<=gide; i++) {
    BoundaryFlag block_bcs[6];
    SetBlockSizeAndBoundaries(loclist[i], block_size, block_bcs);
    MeshBlock new_block(this, pin, block_size, block_bcs);
    my_blocks.push_back(new_block);  // this requires copy operator!
//    my_blocks(i-gids_)->pbval->SearchAndSetNeighbors(tree, ranklist, nslist);
  }

  ResetLoadBalance();
}

//----------------------------------------------------------------------------------------
// Mesh constructor for restarts. Load the restart file


//----------------------------------------------------------------------------------------
// destructor

Mesh::~Mesh() {
  delete [] ranklist;
  delete [] costlist;
  delete [] nslist;
  delete [] nblist;
  if (adaptive) { // deallocate arrays for AMR
    delete [] nref;
    delete [] nderef;
    delete [] rdisp;
    delete [] ddisp;
    delete [] bnref;
    delete [] bnderef;
    delete [] brdisp;
    delete [] bddisp;
  }
  delete [] loclist;
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
  std::cout <<"Root grid = "<<nmbx1_r <<" x "<<nmbx2_r <<" x "<<nmbx3_r <<" MeshBlocks"<< std::endl;
  std::cout <<"Total number of MeshBlocks = " << nmbtotal << std::endl;
  std::cout <<"Number of physical levels of refinement = "<< (max_level - root_level) 
            <<" (" << (max_level - root_level + 1) << " levels total)" << std::endl;
  std::cout <<"Number of logical  levels of refinement = "<< max_level
            <<" (" << (max_level + 1) << " levels total)" << std::endl;


  return;
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::SetBlockSizeAndBoundaries(LogicalLocation loc,
//                 RegionSize &block_size, BundaryFlag *block_bcs)
// \brief Set the physical part of a block_size structure and block boundary conditions

void Mesh::SetBlockSizeAndBoundaries(LogicalLocation loc, RegionSize &block_size,
                                     BoundaryFlag *block_bcs) {
  std::int32_t &lx1 = loc.lx1;
  std::int32_t nmbx1_l = nmbx1_r << (loc.level - root_level);

  // calculate physical size of MeshBlock in x1
  if (lx1 == 0) {
    block_size.x1min = mesh_size.x1min;
    block_bcs[BoundaryFace::inner_x1] = mesh_bcs[BoundaryFace::inner_x1];
  } else {
    block_size.x1min = LeftEdgePosition(lx1, nmbx1_l, mesh_size.x1min, mesh_size.x1max);
    block_bcs[BoundaryFace::inner_x1] = BoundaryFlag::block;
  }

  if (lx1 == nmbx1_l - 1) {
    block_size.x1max = mesh_size.x1max;
    block_bcs[BoundaryFace::outer_x1] = mesh_bcs[BoundaryFace::outer_x1];
  } else {
    block_size.x1max = LeftEdgePosition(lx1+1, nmbx1_l, mesh_size.x1min, mesh_size.x1max);
    block_bcs[BoundaryFace::outer_x1] = BoundaryFlag::block;
  }

  // calculate physical size of MeshBlock in x2
  if (mesh_size.nx2 == 1) {
    block_size.x2min = mesh_size.x2min;
    block_size.x2max = mesh_size.x2max;
    block_bcs[BoundaryFace::inner_x2] = mesh_bcs[BoundaryFace::inner_x2];
    block_bcs[BoundaryFace::outer_x2] = mesh_bcs[BoundaryFace::outer_x2];
  } else {

    std::int32_t &lx2 = loc.lx2;
    std::int32_t nmbx2_l = nmbx2_r << (loc.level - root_level);
    if (lx2 == 0) {
      block_size.x2min = mesh_size.x2min;
      block_bcs[BoundaryFace::inner_x2] = mesh_bcs[BoundaryFace::inner_x2];
    } else {
      block_size.x2min = LeftEdgePosition(lx2, nmbx2_l, mesh_size.x2min, mesh_size.x2max);
      block_bcs[BoundaryFace::inner_x2] = BoundaryFlag::block;
    }

    if (lx2 == (nmbx2_l) - 1) {
      block_size.x2max = mesh_size.x2max;
      block_bcs[BoundaryFace::outer_x2] = mesh_bcs[BoundaryFace::outer_x2];
    } else {
      block_size.x2max = LeftEdgePosition(lx2+1, nmbx2_l,mesh_size.x2min,mesh_size.x2max);
      block_bcs[BoundaryFace::outer_x2] = BoundaryFlag::block;
    }

  }

  // calculate physical size of MeshBlock in x3
  if (mesh_size.nx3 == 1) {
    block_size.x3min = mesh_size.x3min;
    block_size.x3max = mesh_size.x3max;
    block_bcs[BoundaryFace::inner_x3] = mesh_bcs[BoundaryFace::inner_x3];
    block_bcs[BoundaryFace::outer_x3] = mesh_bcs[BoundaryFace::outer_x3];
  } else {
    std::int32_t &lx3 = loc.lx3;
    std::int32_t nmbx3_l = nmbx3_r << (loc.level - root_level);
    if (lx3 == 0) {
      block_size.x3min = mesh_size.x3min;
      block_bcs[BoundaryFace::inner_x3] = mesh_bcs[BoundaryFace::inner_x3];
    } else {
      block_size.x3min = LeftEdgePosition(lx3, nmbx3_l, mesh_size.x3min, mesh_size.x3max);
      block_bcs[BoundaryFace::inner_x3] = BoundaryFlag::block;
    }
    if (lx3 == (nmbx3_l) - 1) {
      block_size.x3max = mesh_size.x3max;
      block_bcs[BoundaryFace::outer_x3] = mesh_bcs[BoundaryFace::outer_x3];
    } else {
      block_size.x3max = LeftEdgePosition(lx3+1, nmbx3_l,mesh_size.x3min,mesh_size.x3max);
      block_bcs[BoundaryFace::outer_x3] = BoundaryFlag::block;
    }
  }
  block_size.x1rat = mesh_size.x1rat;
  block_size.x2rat = mesh_size.x2rat;
  block_size.x3rat = mesh_size.x3rat;

  return;
}
