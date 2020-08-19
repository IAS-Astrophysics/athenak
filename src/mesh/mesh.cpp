//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mesh.cpp
//  \brief implementation of functions in Mesh class

#include <iostream>
#include <cinttypes>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "outputs/io_wrapper.hpp"
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

  mesh_cells.ng  = pin->GetOrAddReal("mesh", "nghost", 2);
  mesh_cells.nx1 = pin->GetInteger("mesh", "nx1");
  mesh_cells.nx2 = pin->GetInteger("mesh", "nx2");
  mesh_cells.nx3 = pin->GetInteger("mesh", "nx3");

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
  nx2gt1 = (mesh_cells.nx2 > 1) ? true : false;
  nx3gt1 = (mesh_cells.nx3 > 1) ? true : false;

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
  if (mesh_cells.nx1 < 4) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "In mesh block in input file nx1 must be >= 4, but nx1=" << mesh_cells.nx1
        << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (mesh_cells.nx2 < 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "In mesh block in input file nx2 must be >= 1, but nx2=" << mesh_cells.nx2
        << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (mesh_cells.nx3 < 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "In mesh block in input file nx3 must be >= 1, but nx3=" << mesh_cells.nx3
        << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (mesh_cells.nx2 == 1 && mesh_cells.nx3 > 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "In mesh block in input file: nx2=1, nx3=" << mesh_cells.nx3 
        << ", but 2D problems in x1-x3 plane not supported" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // error check number of ghost zones
  if (mesh_cells.ng < 2) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
      << "More than 2 ghost zones required, but nghost=" <<mesh_cells.ng << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if ((multilevel) && (mesh_cells.ng % 2 != 0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
      << "Number of ghost zones must be divisible by two for SMR/AMR calculations, "
      << "but nghost=" << mesh_cells.ng << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // passed error checks, compute grid spacing in (virtual) mesh grid
  mesh_cells.dx1 = (mesh_size.x1max-mesh_size.x1min)/static_cast<Real>(mesh_cells.nx1);
  mesh_cells.dx2 = (mesh_size.x2max-mesh_size.x2min)/static_cast<Real>(mesh_cells.nx2);
  mesh_cells.dx3 = (mesh_size.x3max-mesh_size.x3min)/static_cast<Real>(mesh_cells.nx3);

  //=== Step 2 =========================================================
  // Set # of cells in MeshBlock read from input parameters, error check

  RegionSize inblock_size;
  RegionCells inblock_cells;
  inblock_cells.nx1 = pin->GetOrAddInteger("meshblock", "nx1", mesh_cells.nx1);
  if (nx2gt1) {
    inblock_cells.nx2 = pin->GetOrAddInteger("meshblock", "nx2", mesh_cells.nx2);
  } else {
    inblock_cells.nx2 = mesh_cells.nx2;
  }
  if (nx3gt1) {
    inblock_cells.nx3 = pin->GetOrAddInteger("meshblock", "nx3", mesh_cells.nx3);
  } else {
    inblock_cells.nx3 = mesh_cells.nx3;
  }

  // error check consistency of the block and mesh
  if (   mesh_cells.nx1 % inblock_cells.nx1 != 0
      || mesh_cells.nx2 % inblock_cells.nx2 != 0
      || mesh_cells.nx3 % inblock_cells.nx3 != 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Mesh must be evenly divisible by MeshBlocks" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if ( inblock_cells.nx1 < 4 ||
      (inblock_cells.nx2 < 4 && nx2gt1) ||
      (inblock_cells.nx3 < 4 && nx3gt1) ) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "MeshBlock must be >= 4 cells in each active dimension" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // calculate the number of MeshBlocks in root level in each dir
  nmbroot_x1 = mesh_cells.nx1/inblock_cells.nx1;
  nmbroot_x2 = mesh_cells.nx2/inblock_cells.nx2;
  nmbroot_x3 = mesh_cells.nx3/inblock_cells.nx3;

  // find maximum number of MeshBlocks in any dir
  int nmbmax = (nmbroot_x1 > nmbroot_x2) ? nmbroot_x1 : nmbroot_x2;
  nmbmax = (nmbmax > nmbroot_x3) ? nmbmax : nmbroot_x3;

  // Find smallest N such that 2^N > max number of MeshBlocks in any dimension (nmbmax)
  // Then N is logical level of root grid.  2^N implemented as left-shift (1<<root_level)
  for (root_level=0; ((1<<root_level) < nmbmax); root_level++) {}
  int current_level = root_level; 

  //=== Step 3 =======================================================
  //

  tree.CreateRootGrid();

  //=== Step 4 =======================================================
  // Error check properties of input paraemters for SMR/AMR meshes.

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
    if (inblock_cells.nx1 % 2 != 0 || 
       (inblock_cells.nx2 % 2 != 0 && nx2gt1) ||
       (inblock_cells.nx3 % 2 != 0 && nx3gt1)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Number of cells in MeshBlock must be divisible by 2 "
                << "with SMR or AMR." << std::endl;
      std::exit(EXIT_FAILURE);
    }

    // cycle through ParameterInput list and find "refinement" blocks (SMR), extract data
    // Expand MeshBlockTree to include "refinement" regions specified in input file:
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
        std::int32_t lxmax = nmbroot_x1*(1<<phy_ref_lev);
        for (lx1min=0; lx1min<lxmax; lx1min++) {
          if (LeftEdgeX(lx1min+1,lxmax,mesh_size.x1min,mesh_size.x1max) > ref_size.x1min)
            break;
        }
        for (lx1max=lx1min; lx1max<lxmax; lx1max++) {
          if (LeftEdgeX(lx1max+1,lxmax,mesh_size.x1min,mesh_size.x1max) >= ref_size.x1max)
            break;
        }
        if (lx1min % 2 == 1) lx1min--;
        if (lx1max % 2 == 0) lx1max++;

        // Find range of x2-indices of such MeshBlocks that cover the refinement region
        if (nx2gt1) { // 2D or 3D
          lxmax = nmbroot_x2*(1<<phy_ref_lev);
          for (lx2min=0; lx2min<lxmax; lx2min++) {
            if (LeftEdgeX(lx2min+1, lxmax, mesh_size.x2min, mesh_size.x2max) >
                ref_size.x2min)
            break;
          }
          for (lx2max=lx2min; lx2max<lxmax; lx2max++) {
            if (LeftEdgeX(lx2max+1, lxmax, mesh_size.x2min, mesh_size.x2max) >=
                ref_size.x2max)
            break;
          }
          if (lx2min % 2 == 1) lx2min--;
          if (lx2max % 2 == 0) lx2max++;
        }

        // Find range of x3-indices of such MeshBlocks that cover the refinement region
        if (nx3gt1) { // 3D
          lxmax = nmbroot_x3*(1<<phy_ref_lev);
          for (lx3min=0; lx3min<lxmax; lx3min++) {
            if (LeftEdgeX(lx3min+1, lxmax, mesh_size.x3min, mesh_size.x3max) >
                ref_size.x3min)
            break;
          }
          for (lx3max=lx3min; lx3max<lxmax; lx3max++) {
            if (LeftEdgeX(lx3max+1, lxmax, mesh_size.x3min, mesh_size.x3max) >=
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
  // following returns LogicalLocation list sorted by Z-ordering
  tree.GetMeshBlockList(loclist, nullptr, nmbtotal);

  //=== Step 5 =======================================================
  // compute properties of MeshBlocks and initialize 

#if MPI_PARALLEL_ENABLED
  // check there is at least one MeshBlock per MPI rank
  if (nmbtotal < global_variable::nranks) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Fewer MeshBlocks (nmbtotal="<<nmbtotal<") than MPI ranks (nranks="
        << global_variable::nranks << ")" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif

  costlist = new double[nmbtotal];
  ranklist = new int[nmbtotal];
  nslist = new int[global_variable::nranks];
  nblist = new int[global_variable::nranks];
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

  // create MeshBlock list for this process

  gids = nslist[global_variable::my_rank];
  gide = gids + nblist[global_variable::my_rank] - 1;
  nmbthisrank = nblist[global_variable::my_rank];
  
  // create MeshBlocks for this node
  for (int i=gids; i<=gide; i++) {
    BoundaryFlag inblock_bcs[6];
    SetBlockSizeAndBoundaries(loclist[i], inblock_size, inblock_cells, inblock_bcs);
    MeshBlock new_block(this, pin, i, inblock_size, inblock_cells, inblock_bcs);
    new_block.SetNeighbors(tree, ranklist);
    mblocks.push_back(new_block);  // MB vector elements stored in order gids->gide
  }

/*******/
  for (auto it=mblocks.begin(); it<mblocks.end(); ++it) {
    for (int n=0; n<26; ++n) {
      std::cout << "n=" << n << " gid=" << it->neighbor[n].ngid << " level=" << it->neighbor[n].nlevel << " rank=" << it->neighbor[n].nrank << std::endl;
    }
  }
/**********/

  ResetLoadBalance();

  //=== Step 5 =======================================================
  // set initial time/cycle parameters

  time = pin->GetOrAddReal("time", "start_time", 0.0);
  dt   = std::numeric_limits<float>::max();
  cfl_no = pin->GetReal("time", "cfl_number");
  ncycle = 0;
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
//  \brief outputs information about mesh structure, creates file containing MeshBlock
//  positions and sizes that can be used to create plots using 'plot_mesh.py' script 

void Mesh::OutputMeshStructure(int flag) {

  // Write overall Mesh structure to stdout and file
  std::cout << std::endl;
  std::cout <<"Root grid = "<< nmbroot_x1 <<" x "<< nmbroot_x2 <<" x "<< nmbroot_x3
            <<" MeshBlocks"<< std::endl;
  std::cout <<"Total number of MeshBlocks = " << nmbtotal << std::endl;
  std::cout <<"Number of logical  levels of refinement = "<< max_level
            <<" (" << (max_level + 1) << " levels total)" << std::endl;
  std::cout <<"Number of physical levels of refinement = "<< (max_level - root_level) 
            <<" (" << (max_level - root_level + 1) << " levels total)" << std::endl;

  // if more than one physical level: compute/output # of blocks and cost per level
  if ((max_level - root_level) > 1) {
    int nb_per_plevel[max_level];
    float cost_per_plevel[max_level];
    for (int i=0; i<=max_level; ++i) {
      nb_per_plevel[i] = 0;
      cost_per_plevel[i] = 0.0;
    }
    for (int i=0; i<nmbtotal; i++) {
      nb_per_plevel[(loclist[i].level - root_level)]++;
      cost_per_plevel[(loclist[i].level - root_level)] += costlist[i];
    }
    for (int i=root_level; i<=max_level; i++) {
      if (nb_per_plevel[i-root_level] != 0) {
        std::cout << "  Physical level = " << i-root_level << " (logical level = " << i
                  << "): " << nb_per_plevel[i-root_level] << " MeshBlocks, cost = "
                  << cost_per_plevel[i-root_level] <<  std::endl;
      }
    }
  }

  std::cout << "Number of parallel ranks = " << global_variable::nranks << std::endl;
  // if more than one rank: compute/output # of blocks and cost per rank
  if (global_variable::nranks > 1) {
    int nb_per_rank[global_variable::nranks];
    int cost_per_rank[global_variable::nranks];
    for (int i=0; i<global_variable::nranks; ++i) {
      nb_per_rank[i] = 0;
      cost_per_rank[i] = 0;
    }
    for (int i=0; i<nmbtotal; i++) {
      nb_per_rank[ranklist[i]]++;
      cost_per_rank[ranklist[i]] += costlist[i];
    }
    for (int i=0; i<global_variable::nranks; ++i) {
      std::cout << "  Rank = " << i << ": " << nb_per_rank[i] <<" MeshBlocks, cost = "
                << cost_per_rank[i] << std::endl;
    }

    // output total cost and load balancing info
    double mincost = std::numeric_limits<double>::max();
    double maxcost = 0.0, totalcost = 0.0;
    for (int i=root_level; i<=max_level; i++) {
      for (int j=0; j<nmbtotal; j++) {
        if (loclist[j].level == i) {
          mincost = std::min(mincost,costlist[i]);
          maxcost = std::max(maxcost,costlist[i]);
          totalcost += costlist[i];
        }
      }
    }
    std::cout << "Load Balancing:" << std::endl;
    std::cout << "  Minimum cost = " << mincost << ", Maximum cost = " << maxcost
              << ", Average cost = " << totalcost/nmbtotal << std::endl;
  }

  // if -m argument given on command line, and for 2D/3D:
  // output relative size/locations of meshblock to file, for plotting
  if (flag && nx2gt1) {
    FILE *fp = nullptr;
    if ((fp = std::fopen("mesh_structure.dat","wb")) == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ 
          << std::endl << "Cannot open 'mesh_structure.dat' file for output" << std::endl;
      std::exit(EXIT_FAILURE);
    }

    for (int i=root_level; i<=max_level; i++) {
      for (int j=0; j<nmbtotal; j++) {
        if (loclist[j].level == i) {
          RegionSize b_size;
          RegionCells b_cells;
          BoundaryFlag b_bcs[6];
          SetBlockSizeAndBoundaries(loclist[j], b_size, b_cells, b_bcs);
          std::int32_t &lx1 = loclist[j].lx1;
          std::int32_t &lx2 = loclist[j].lx2;
          std::int32_t &lx3 = loclist[j].lx3;
          std::fprintf(fp,"#MeshBlock %d on rank=%d with cost=%g\n", j, ranklist[j],
                       costlist[j]);
          std::fprintf(
              fp,"#  Logical level %d, location = (%" PRId32 " %" PRId32 " %" PRId32")\n",
              loclist[j].level, lx1, lx2, lx3);
          if (nx2gt1 && !(nx3gt1)) { // 2D
            std::fprintf(fp, "%g %g\n", b_size.x1min, b_size.x2min);
            std::fprintf(fp, "%g %g\n", b_size.x1max, b_size.x2min);
            std::fprintf(fp, "%g %g\n", b_size.x1max, b_size.x2max);
            std::fprintf(fp, "%g %g\n", b_size.x1min, b_size.x2max);
            std::fprintf(fp, "%g %g\n", b_size.x1min, b_size.x2min);
            std::fprintf(fp, "\n\n");
          }
          if (nx3gt1) { // 3D
            std::fprintf(fp, "%g %g %g\n", b_size.x1min, b_size.x2min, b_size.x3min);
            std::fprintf(fp, "%g %g %g\n", b_size.x1max, b_size.x2min, b_size.x3min);
            std::fprintf(fp, "%g %g %g\n", b_size.x1max, b_size.x2max, b_size.x3min);
            std::fprintf(fp, "%g %g %g\n", b_size.x1min, b_size.x2max, b_size.x3min);
            std::fprintf(fp, "%g %g %g\n", b_size.x1min, b_size.x2min, b_size.x3min);
            std::fprintf(fp, "%g %g %g\n", b_size.x1min, b_size.x2min, b_size.x3max);
            std::fprintf(fp, "%g %g %g\n", b_size.x1max, b_size.x2min, b_size.x3max);
            std::fprintf(fp, "%g %g %g\n", b_size.x1max, b_size.x2min, b_size.x3min);
            std::fprintf(fp, "%g %g %g\n", b_size.x1max, b_size.x2min, b_size.x3max);
            std::fprintf(fp, "%g %g %g\n", b_size.x1max, b_size.x2max, b_size.x3max);
            std::fprintf(fp, "%g %g %g\n", b_size.x1max, b_size.x2max, b_size.x3min);
            std::fprintf(fp, "%g %g %g\n", b_size.x1max, b_size.x2max, b_size.x3max);
            std::fprintf(fp, "%g %g %g\n", b_size.x1min, b_size.x2max, b_size.x3max);
            std::fprintf(fp, "%g %g %g\n", b_size.x1min, b_size.x2max, b_size.x3min);
            std::fprintf(fp, "%g %g %g\n", b_size.x1min, b_size.x2max, b_size.x3max);
            std::fprintf(fp, "%g %g %g\n", b_size.x1min, b_size.x2min, b_size.x3max);
            std::fprintf(fp, "%g %g %g\n", b_size.x1min, b_size.x2min, b_size.x3min);
            std::fprintf(fp, "\n\n");
          }
        }
      }
    }
    std::fclose(fp);

    std::cout << "See the 'mesh_structure.dat' file for a complete list of MeshBlocks."
              << std::endl;
    std::cout << "Use 'plot_mesh.py' script to visualize data in 'mesh_structure.dat'"
              << " file" << std::endl << std::endl;
  }

  return;
}


//----------------------------------------------------------------------------------------
// \!fn void Mesh::SetBlockSizeAndBoundaries(LogicalLocation loc,
//                 RegionSize &block_size, BundaryFlag *block_bcs)
// \brief Set the physical part of a block_size structure and block boundary conditions

void Mesh::SetBlockSizeAndBoundaries(LogicalLocation loc, RegionSize &block_size,
                                     RegionCells &block_cells, BoundaryFlag *block_bcs) {
  std::int32_t &lx1 = loc.lx1;
  std::int32_t nmbx1_l = nmbroot_x1 << (loc.level - root_level);

  // calculate physical size of MeshBlock in x1
  if (lx1 == 0) {
    block_size.x1min = mesh_size.x1min;
    block_bcs[BoundaryFace::inner_x1] = mesh_bcs[BoundaryFace::inner_x1];
  } else {
    block_size.x1min = LeftEdgeX(lx1, nmbx1_l, mesh_size.x1min, mesh_size.x1max);
    block_bcs[BoundaryFace::inner_x1] = BoundaryFlag::block;
  }

  if (lx1 == nmbx1_l - 1) {
    block_size.x1max = mesh_size.x1max;
    block_bcs[BoundaryFace::outer_x1] = mesh_bcs[BoundaryFace::outer_x1];
  } else {
    block_size.x1max = LeftEdgeX(lx1+1, nmbx1_l, mesh_size.x1min, mesh_size.x1max);
    block_bcs[BoundaryFace::outer_x1] = BoundaryFlag::block;
  }

  // calculate physical size of MeshBlock in x2
  if (mesh_cells.nx2 == 1) {
    block_size.x2min = mesh_size.x2min;
    block_size.x2max = mesh_size.x2max;
    block_bcs[BoundaryFace::inner_x2] = mesh_bcs[BoundaryFace::inner_x2];
    block_bcs[BoundaryFace::outer_x2] = mesh_bcs[BoundaryFace::outer_x2];
  } else {

    std::int32_t &lx2 = loc.lx2;
    std::int32_t nmbx2_l = nmbroot_x2 << (loc.level - root_level);
    if (lx2 == 0) {
      block_size.x2min = mesh_size.x2min;
      block_bcs[BoundaryFace::inner_x2] = mesh_bcs[BoundaryFace::inner_x2];
    } else {
      block_size.x2min = LeftEdgeX(lx2, nmbx2_l, mesh_size.x2min, mesh_size.x2max);
      block_bcs[BoundaryFace::inner_x2] = BoundaryFlag::block;
    }

    if (lx2 == (nmbx2_l) - 1) {
      block_size.x2max = mesh_size.x2max;
      block_bcs[BoundaryFace::outer_x2] = mesh_bcs[BoundaryFace::outer_x2];
    } else {
      block_size.x2max = LeftEdgeX(lx2+1, nmbx2_l,mesh_size.x2min,mesh_size.x2max);
      block_bcs[BoundaryFace::outer_x2] = BoundaryFlag::block;
    }

  }

  // calculate physical size of MeshBlock in x3
  if (mesh_cells.nx3 == 1) {
    block_size.x3min = mesh_size.x3min;
    block_size.x3max = mesh_size.x3max;
    block_bcs[BoundaryFace::inner_x3] = mesh_bcs[BoundaryFace::inner_x3];
    block_bcs[BoundaryFace::outer_x3] = mesh_bcs[BoundaryFace::outer_x3];
  } else {
    std::int32_t &lx3 = loc.lx3;
    std::int32_t nmbx3_l = nmbroot_x3 << (loc.level - root_level);
    if (lx3 == 0) {
      block_size.x3min = mesh_size.x3min;
      block_bcs[BoundaryFace::inner_x3] = mesh_bcs[BoundaryFace::inner_x3];
    } else {
      block_size.x3min = LeftEdgeX(lx3, nmbx3_l, mesh_size.x3min, mesh_size.x3max);
      block_bcs[BoundaryFace::inner_x3] = BoundaryFlag::block;
    }
    if (lx3 == (nmbx3_l) - 1) {
      block_size.x3max = mesh_size.x3max;
      block_bcs[BoundaryFace::outer_x3] = mesh_bcs[BoundaryFace::outer_x3];
    } else {
      block_size.x3max = LeftEdgeX(lx3+1, nmbx3_l,mesh_size.x3min,mesh_size.x3max);
      block_bcs[BoundaryFace::outer_x3] = BoundaryFlag::block;
    }
  }
  // grid spacing at this level.  Ensure all MeshBlocks at same level have same dx
  block_cells.dx1 = mesh_cells.dx1*static_cast<Real>(1<<(loc.level - root_level));
  block_cells.dx2 = mesh_cells.dx2*static_cast<Real>(1<<(loc.level - root_level));
  block_cells.dx3 = mesh_cells.dx3*static_cast<Real>(1<<(loc.level - root_level));
  // everything else
  block_cells.nx1 = mesh_cells.nx1;
  block_cells.nx2 = mesh_cells.nx2;
  block_cells.nx3 = mesh_cells.nx3;
  block_cells.ng  = mesh_cells.ng;

  return;
}
