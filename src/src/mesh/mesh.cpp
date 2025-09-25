//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mesh.cpp
//  \brief implementation of constructor and functions in Mesh class

#include <algorithm>
#include <cinttypes>
#include <iostream>
#include <limits>
#include <cstdio> // fclose
#include <string> // string

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "refinement_criteria.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "z4c/z4c.hpp"
#include "diffusion/viscosity.hpp"
#include "diffusion/resistivity.hpp"
#include "diffusion/conduction.hpp"
#include "radiation/radiation.hpp"
#include "particles/particles.hpp"
#include "srcterms/srcterms.hpp"
#include "outputs/io_wrapper.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

//----------------------------------------------------------------------------------------
//! Mesh constructor:
//! initializes some mesh variables using parameters in input file.
//! The MeshBlockPack, MeshRefinement, and ShearingBox objects are constructed in
//! BuildTreeFromScratch() or BuildTreeFromRestart()
//! The MeshBlockTree and ProblemGenerator objects are constructed in main().
//! This is so that they can store a pointer to the Mesh which can be reliably referenced
//! only after the Mesh constructor has finished.

Mesh::Mesh(ParameterInput *pin) :
  one_d(false),
  two_d(false),
  three_d(false),
  multi_d(false),
  strictly_periodic(true),
  nmb_packs_thisrank(1),
  nprtcl_thisrank(0),
  nprtcl_total(0),
  dtold(0.) {
  // Set physical size and number of cells in mesh (root level)
  mesh_size.x1min = pin->GetReal("mesh", "x1min");
  mesh_size.x1max = pin->GetReal("mesh", "x1max");
  mesh_size.x2min = pin->GetReal("mesh", "x2min");
  mesh_size.x2max = pin->GetReal("mesh", "x2max");
  mesh_size.x3min = pin->GetReal("mesh", "x3min");
  mesh_size.x3max = pin->GetReal("mesh", "x3max");

  mesh_indcs.ng  = pin->GetOrAddReal("mesh", "nghost", 2);
  mesh_indcs.nx1 = pin->GetInteger("mesh", "nx1");
  mesh_indcs.nx2 = pin->GetInteger("mesh", "nx2");
  mesh_indcs.nx3 = pin->GetInteger("mesh", "nx3");

  // define some useful flags that indicate 1D/2D/3D calculations
  if (mesh_indcs.nx3 > 1) {
    three_d = true;
    multi_d = true;
  } else if (mesh_indcs.nx2 > 1) {
    two_d = true;
    multi_d = true;
  } else {
    one_d = true;
  }

  // Set BC flags for ix1/ox1 boundaries and error check
  mesh_bcs[BoundaryFace::inner_x1] = GetBoundaryFlag(pin->GetString("mesh", "ix1_bc"));
  mesh_bcs[BoundaryFace::outer_x1] = GetBoundaryFlag(pin->GetString("mesh", "ox1_bc"));
  if ((mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::periodic ||
       mesh_bcs[BoundaryFace::outer_x1] == BoundaryFlag::periodic) &&
       mesh_bcs[BoundaryFace::inner_x1] != mesh_bcs[BoundaryFace::outer_x1]) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Both inner and outer x1 bcs must be periodic" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (mesh_bcs[BoundaryFace::inner_x1] != BoundaryFlag::periodic) {
    strictly_periodic = false;
  }

  // Error checks if one of x1 boundaries set to shear_periodic.
  if (mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::shear_periodic &&
      mesh_bcs[BoundaryFace::outer_x1] == BoundaryFlag::shear_periodic) {
    if (one_d) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Shear Periodic Boundaries require 2D or 3D" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (!(pin->DoesBlockExist("shearing_box"))) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Shear Periodic Boundaries set but no <shearing_box>"
                << " block in input file" <<std::endl;
      std::exit(EXIT_FAILURE);
    }
  } else if ((mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::shear_periodic &&
              mesh_bcs[BoundaryFace::outer_x1] != BoundaryFlag::shear_periodic) ||
             (mesh_bcs[BoundaryFace::inner_x1] != BoundaryFlag::shear_periodic &&
              mesh_bcs[BoundaryFace::outer_x1] == BoundaryFlag::shear_periodic)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "In shearing box, both x1 bcs must be shear_periodic"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Set BC flags for ix2/ox2 boundaries and error check
  if (multi_d) {
    mesh_bcs[BoundaryFace::inner_x2] = GetBoundaryFlag(pin->GetString("mesh", "ix2_bc"));
    mesh_bcs[BoundaryFace::outer_x2] = GetBoundaryFlag(pin->GetString("mesh", "ox2_bc"));
    if ((mesh_bcs[BoundaryFace::inner_x2] == BoundaryFlag::periodic ||
         mesh_bcs[BoundaryFace::outer_x2] == BoundaryFlag::periodic) &&
         mesh_bcs[BoundaryFace::inner_x2] != mesh_bcs[BoundaryFace::outer_x2]) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "Both inner and outer x2 bcs must be periodic" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (mesh_bcs[BoundaryFace::inner_x2] != BoundaryFlag::periodic) {
      strictly_periodic = false;
    }
    if (mesh_bcs[BoundaryFace::inner_x2] == BoundaryFlag::shear_periodic ||
        mesh_bcs[BoundaryFace::outer_x2] == BoundaryFlag::shear_periodic) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Shear Periodic Boundaries cannot be applied in x2"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  } else {
    // ix2/ox2 BC flags set to undef for 1D problems
    mesh_bcs[BoundaryFace::inner_x2] = BoundaryFlag::undef;
    mesh_bcs[BoundaryFace::outer_x2] = BoundaryFlag::undef;
  }

  // Set BC flags for ix3/ox3 boundaries and error check
  if (three_d) {
    mesh_bcs[BoundaryFace::inner_x3] = GetBoundaryFlag(pin->GetString("mesh", "ix3_bc"));
    mesh_bcs[BoundaryFace::outer_x3] = GetBoundaryFlag(pin->GetString("mesh", "ox3_bc"));
    if ((mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::periodic ||
         mesh_bcs[BoundaryFace::outer_x3] == BoundaryFlag::periodic) &&
         mesh_bcs[BoundaryFace::inner_x3] != mesh_bcs[BoundaryFace::outer_x3]) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "Both inner and outer x3 bcs must be periodic" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (mesh_bcs[BoundaryFace::inner_x3] != BoundaryFlag::periodic) {
      strictly_periodic = false;
    }
    if (mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::shear_periodic ||
        mesh_bcs[BoundaryFace::outer_x3] == BoundaryFlag::shear_periodic) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Shear Periodic Boundaries cannot be applied in x3"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  } else {
    // ix3/ox3 BC flags set to undef for 1D or 2D problems
    mesh_bcs[BoundaryFace::inner_x3] = BoundaryFlag::undef;
    mesh_bcs[BoundaryFace::outer_x3] = BoundaryFlag::undef;
  }

  // set boolean flags indicating type of refinement (if any), and whether mesh is
  // periodic, depending on input strings
  adaptive = (pin->GetOrAddString("mesh_refinement","refinement","none") == "adaptive")
    ?  true : false;
  multilevel = (adaptive || pin->GetString("mesh_refinement","refinement") == "static")
    ?  true : false;

  // FIXME: The shearing box is not currently compatible with SMR/AMR
  if (multilevel && pin->DoesBlockExist("shearing_box")) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Shearing box is not currently compatible with mesh refinement"
        << std::endl;
    std::exit(EXIT_FAILURE);
  }

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
  if ( mesh_indcs.nx1 < 4 ||
      (mesh_indcs.nx2 < 4 && multi_d) ||
      (mesh_indcs.nx3 < 4 && three_d) ) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Mesh must be >= 4 cells in each active dimension" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (mesh_indcs.nx2 < 1 || mesh_indcs.nx3 < 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "In <mesh> block nx2 and nx3 must both be >= 1" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (mesh_indcs.nx2 == 1 && mesh_indcs.nx3 > 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "In <mesh> block in input file: nx2=1, nx3=" << mesh_indcs.nx3
        << ", but 2D problems in x1-x3 plane not supported" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // error check number of ghost zones
  if (mesh_indcs.ng < 2) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
      << "More than 2 ghost zones required, but nghost=" <<mesh_indcs.ng << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if ((multilevel) && (mesh_indcs.ng % 2 != 0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
      << "Number of ghost zones must be divisible by two for SMR/AMR calculations, "
      << "but nghost=" << mesh_indcs.ng << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // passed error checks, compute grid spacing in (virtual) mesh grid
  mesh_size.dx1 = (mesh_size.x1max-mesh_size.x1min)/static_cast<Real>(mesh_indcs.nx1);
  mesh_size.dx2 = (mesh_size.x2max-mesh_size.x2min)/static_cast<Real>(mesh_indcs.nx2);
  mesh_size.dx3 = (mesh_size.x3max-mesh_size.x3min)/static_cast<Real>(mesh_indcs.nx3);

  // Read # of cells in MeshBlock from input parameters, error check
  mb_indcs.nx1 = pin->GetOrAddInteger("meshblock", "nx1", mesh_indcs.nx1);
  if (multi_d) {
    mb_indcs.nx2 = pin->GetOrAddInteger("meshblock", "nx2", mesh_indcs.nx2);
  } else {
    mb_indcs.nx2 = mesh_indcs.nx2;
  }
  if (three_d) {
    mb_indcs.nx3 = pin->GetOrAddInteger("meshblock", "nx3", mesh_indcs.nx3);
  } else {
    mb_indcs.nx3 = mesh_indcs.nx3;
  }

  // error check consistency of the block and mesh
  if (mesh_indcs.nx1 % mb_indcs.nx1 != 0 ||
      mesh_indcs.nx2 % mb_indcs.nx2 != 0 ||
      mesh_indcs.nx3 % mb_indcs.nx3 != 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Mesh must be evenly divisible by MeshBlocks" << std::endl
              << "Check Mesh and MeshBlock dimensions in input file" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if ((mb_indcs.nx1 < 4) ||
      (mb_indcs.nx2 < 4 && multi_d) ||
      (mb_indcs.nx3 < 4 && three_d) ) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "MeshBlock must be >= 4 cells in each active dimension" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if ( (multilevel) &&
      ((mb_indcs.nx1 %2 != 0) ||
       (mb_indcs.nx2 %2 != 0 && multi_d) ||
       (mb_indcs.nx3 %2 != 0 && three_d)) ) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
      << "Number of cells in MeshBlock must be divisible by two in each dimension for "
      << "SMR/AMR calculations." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // initialize indices for Mesh cells, MeshBlock cells, and MeshBlock coarse cells
  mb_indcs.ng  = mesh_indcs.ng;
  mb_indcs.cnx1 = mb_indcs.nx1/2;
  mb_indcs.cnx2 = std::max(1,(mb_indcs.nx2/2));
  mb_indcs.cnx3 = std::max(1,(mb_indcs.nx3/2));

  mesh_indcs.is = mesh_indcs.ng;
  mb_indcs.is   = mb_indcs.ng;
  mb_indcs.cis  = mb_indcs.ng;

  mesh_indcs.ie = mesh_indcs.is + mesh_indcs.nx1 - 1;
  mb_indcs.ie   = mb_indcs.is + mb_indcs.nx1 - 1;
  mb_indcs.cie  = mb_indcs.cis + mb_indcs.cnx1 - 1;

  if (multi_d) {
    mesh_indcs.js = mesh_indcs.ng;
    mb_indcs.js   = mb_indcs.ng;
    mb_indcs.cjs  = mb_indcs.ng;

    mesh_indcs.je = mesh_indcs.js + mesh_indcs.nx2 - 1;
    mb_indcs.je   = mb_indcs.js + mb_indcs.nx2 - 1;
    mb_indcs.cje  = mb_indcs.cjs + mb_indcs.cnx2 - 1;
  } else {
    mesh_indcs.js = 0;
    mb_indcs.js   = 0;
    mb_indcs.cjs  = 0;

    mesh_indcs.je = 0;
    mb_indcs.je   = 0;
    mb_indcs.cje  = 0;
  }

  if (three_d) {
    mesh_indcs.ks = mesh_indcs.ng;
    mb_indcs.ks   = mb_indcs.ng;
    mb_indcs.cks  = mb_indcs.ng;

    mesh_indcs.ke = mesh_indcs.ks + mesh_indcs.nx3 - 1;
    mb_indcs.ke   = mb_indcs.ks + mb_indcs.nx3 - 1;
    mb_indcs.cke  = mb_indcs.cks + mb_indcs.cnx3 - 1;
  } else {
    mesh_indcs.ks = 0;
    mb_indcs.ks   = 0;
    mb_indcs.cks  = 0;

    mesh_indcs.ke = 0;
    mb_indcs.ke   = 0;
    mb_indcs.cke  = 0;
  }
}

//----------------------------------------------------------------------------------------
// destructor

Mesh::~Mesh() {
  delete [] cost_eachmb;
  delete [] rank_eachmb;
  delete [] lloc_eachmb;
  delete [] gids_eachrank;
  delete [] nmb_eachrank;
  delete pmb_pack;
  if (multilevel) {
    delete pmr;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::PrintMeshDiagnostics()
//  \brief prints information about mesh structure, always called at start of every
//  calculation at end of BuildTree

void Mesh::PrintMeshDiagnostics() {
  std::cout << std::endl;
  std::cout <<"Root grid = "<< nmb_rootx1 <<" x "<< nmb_rootx2 <<" x "<< nmb_rootx3
            <<" MeshBlocks"<< std::endl;
  std::cout <<"Total number of MeshBlocks = " << nmb_total << std::endl;
  std::cout <<"Number of logical  levels of refinement = "<< max_level
            <<" (" << (max_level + 1) << " levels total)" << std::endl;
  std::cout <<"Number of physical levels of refinement = "<< (max_level - root_level)
            <<" (" << (max_level - root_level + 1) << " levels total)" << std::endl;

  // if more than one physical level: compute/output # of blocks and cost per level
  if ((max_level - root_level) > 1) {
    int nb_per_plevel[max_level];      // NOLINT(runtime/arrays)
    float cost_per_plevel[max_level];  // NOLINT(runtime/arrays)
    for (int i=0; i<max_level; ++i) {
      nb_per_plevel[i] = 0;
      cost_per_plevel[i] = 0.0;
    }
    for (int i=0; i<nmb_total; i++) {
      nb_per_plevel[(lloc_eachmb[i].level - root_level)]++;
      cost_per_plevel[(lloc_eachmb[i].level - root_level)] += cost_eachmb[i];
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
    int nb_per_rank[global_variable::nranks];    // NOLINT(runtime/arrays)
    int cost_per_rank[global_variable::nranks];  // NOLINT(runtime/arrays)
    for (int i=0; i<global_variable::nranks; ++i) {
      nb_per_rank[i] = 0;
      cost_per_rank[i] = 0;
    }
    for (int i=0; i<nmb_total; i++) {
      nb_per_rank[rank_eachmb[i]]++;
      cost_per_rank[rank_eachmb[i]] += cost_eachmb[i];
    }
    int mincost = std::numeric_limits<int>::max();
    int maxcost = 0, totalcost = 0;
    for (int i=0; i<global_variable::nranks; ++i) {
      std::cout << "  Rank = " << i << ": " << nb_per_rank[i] <<" MeshBlocks, cost = "
                << cost_per_rank[i] << std::endl;
      mincost = std::min(mincost,cost_per_rank[i]);
      maxcost = std::max(maxcost,cost_per_rank[i]);
      totalcost += cost_per_rank[i];
    }

    // output normalized costs per rank
    std::cout << "Load Balancing:" << std::endl;
    std::cout << "  Maximum normalized cost = "
      << static_cast<float>(maxcost)/static_cast<float>(mincost) << ", Average = "
      << static_cast<float>(totalcost)/static_cast<float>(global_variable::nranks*mincost)
      << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::WriteMeshStructure(int ndim)
//  \brief writes file containing MeshBlock positions and sizes that can be used to create
//  plots using 'plot_mesh.py' script.  Only works for 2D/3D data.  Called from main if
//  '-m' option is given on command line.

void Mesh::WriteMeshStructure() {
  if (one_d) {
    std::cout << "WARNING in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Mesh only 1D, so no 'mesh_structure.dat' file produced" << std::endl;
    return;
  }

  FILE *fp = nullptr;
  if ((fp = std::fopen("mesh_structure.dat","wb")) == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "Cannot open 'mesh_structure.dat' file for output" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  for (int i=root_level; i<=max_level; i++) {
    for (int j=0; j<nmb_total; j++) {
      if (lloc_eachmb[j].level == i) {
        MeshBlock block(this->pmb_pack, j, 1);
        std::int32_t &lx1 = lloc_eachmb[j].lx1;
        std::int32_t &lx2 = lloc_eachmb[j].lx2;
        std::int32_t &lx3 = lloc_eachmb[j].lx3;
        std::fprintf(fp,"#MeshBlock %d on rank=%d with cost=%g\n", j, rank_eachmb[j],
                     cost_eachmb[j]);
        std::fprintf(
            fp,"#  Logical level %d, location = (%" PRId32 " %" PRId32 " %" PRId32")\n",
            lloc_eachmb[j].level, lx1, lx2, lx3);
        if (two_d) { // 2D
          Real &x1min = block.mb_size.h_view(0).x1min;
          Real &x1max = block.mb_size.h_view(0).x1max;
          Real &x2min = block.mb_size.h_view(0).x2min;
          Real &x2max = block.mb_size.h_view(0).x2max;
          std::fprintf(fp,"%g %g\n", x1min, x2min);
          std::fprintf(fp,"%g %g\n", x1max, x2min);
          std::fprintf(fp,"%g %g\n", x1max, x2max);
          std::fprintf(fp,"%g %g\n", x1min, x2max);
          std::fprintf(fp,"%g %g\n", x1min, x2min);
          std::fprintf(fp,"\n\n");
        }
        if (three_d) { // 3D
          Real &x1min = block.mb_size.h_view(0).x1min;
          Real &x1max = block.mb_size.h_view(0).x1max;
          Real &x2min = block.mb_size.h_view(0).x2min;
          Real &x2max = block.mb_size.h_view(0).x2max;
          Real &x3min = block.mb_size.h_view(0).x3min;
          Real &x3max = block.mb_size.h_view(0).x3max;
          std::fprintf(fp,"%g %g %g\n", x1min, x2min, x3min);
          std::fprintf(fp,"%g %g %g\n", x1max, x2min, x3min);
          std::fprintf(fp,"%g %g %g\n", x1max, x2max, x3min);
          std::fprintf(fp,"%g %g %g\n", x1min, x2max, x3min);
          std::fprintf(fp,"%g %g %g\n", x1min, x2min, x3min);
          std::fprintf(fp,"%g %g %g\n", x1min, x2min, x3max);
          std::fprintf(fp,"%g %g %g\n", x1max, x2min, x3max);
          std::fprintf(fp,"%g %g %g\n", x1max, x2min, x3min);
          std::fprintf(fp,"%g %g %g\n", x1max, x2min, x3max);
          std::fprintf(fp,"%g %g %g\n", x1max, x2max, x3max);
          std::fprintf(fp,"%g %g %g\n", x1max, x2max, x3min);
          std::fprintf(fp,"%g %g %g\n", x1max, x2max, x3max);
          std::fprintf(fp,"%g %g %g\n", x1min, x2max, x3max);
          std::fprintf(fp,"%g %g %g\n", x1min, x2max, x3min);
          std::fprintf(fp,"%g %g %g\n", x1min, x2max, x3max);
          std::fprintf(fp,"%g %g %g\n", x1min, x2min, x3max);
          std::fprintf(fp,"%g %g %g\n", x1min, x2min, x3min);
          std::fprintf(fp, "\n\n");
        }
      }
    }
  }
  std::fclose(fp);
  std::cout << "See the 'mesh_structure.dat' file for MeshBlock data" << std::endl;
  std::cout << "Use 'plot_mesh.py' script to visualize data" << std::endl << std::endl;

  return;
}

//----------------------------------------------------------------------------------------
//! \fn GetBoundaryFlag(std::string input_string)
//  \brief Parses input string to return scoped enumerator flag specifying boundary
//  condition. Typically called in Mesh() ctor and in pgen/*.cpp files.

BoundaryFlag Mesh::GetBoundaryFlag(const std::string& input_string) {
  if (input_string == "reflect") {
    return BoundaryFlag::reflect;
  } else if (input_string == "outflow") {
    return BoundaryFlag::outflow;
  } else if (input_string == "inflow") {
    return BoundaryFlag::inflow;
  } else if (input_string == "diode") {
    return BoundaryFlag::diode;
  } else if (input_string == "user") {
    return BoundaryFlag::user;
  } else if (input_string == "periodic") {
    return BoundaryFlag::periodic;
  } else if (input_string == "vacuum") {
    return BoundaryFlag::vacuum;
  } else if (input_string == "shear_periodic") {
    return BoundaryFlag::shear_periodic;
  } else if (input_string == "undef") {
    return BoundaryFlag::undef;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Input string = '" << input_string << "' is an invalid boundary type"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------
//! \fn GetBoundaryString(BoundaryFlag input_flag)
//  \brief Parses enumerated type BoundaryFlag internal integer representation to return
//  string describing the boundary condition. Typicall used to format descriptive errors
//  or diagnostics. Inverse of GetBoundaryFlag().

std::string Mesh::GetBoundaryString(BoundaryFlag input_flag) {
  switch (input_flag) {
    case BoundaryFlag::block:  // 0
      return "block";
    case BoundaryFlag::reflect:
      return "reflect";
    case BoundaryFlag::inflow:
      return "inflow";
    case BoundaryFlag::outflow:
      return "outflow";
    case BoundaryFlag::diode:
      return "diode";
    case BoundaryFlag::user:
      return "user";
    case BoundaryFlag::periodic:
      return "periodic";
    case BoundaryFlag::shear_periodic:
      return "shear_periodic";
    case BoundaryFlag::undef:
      return "undef";
    default:
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
         << std::endl << "Input enum class BoundaryFlag=" << static_cast<int>(input_flag)
         << " is an invalid boundary type" << std::endl;
      std::exit(EXIT_FAILURE);
      break;
  }
}

//----------------------------------------------------------------------------------------
// \fn Mesh::NewTimeStep()

void Mesh::NewTimeStep(const Real tlim) {
  // save old timestep
  dtold = dt;
  if (dt == std::numeric_limits<float>::max()) {
    dtold = 0.;
  }

  // cycle over all MeshBlocks on this rank and find minimum dt
  // Requires at least ONE of the physics modules to be defined.
  // limit increase in timestep to 2x old value
  dt = 2.0*dt;

  // Hydro timestep
  if (pmb_pack->phydro != nullptr) {
    dt = std::min(dt, (cfl_no)*(pmb_pack->phydro->dtnew) );
    // viscosity timestep
    if (pmb_pack->phydro->pvisc != nullptr) {
      dt = std::min(dt, (cfl_no)*(pmb_pack->phydro->pvisc->dtnew) );
    }
    // thermal conduction timestep
    if (pmb_pack->phydro->pcond != nullptr) {
      dt = std::min(dt, (cfl_no)*(pmb_pack->phydro->pcond->dtnew) );
    }
    // source terms timestep
    if (pmb_pack->phydro->psrc != nullptr) {
      dt = std::min(dt, (cfl_no)*(pmb_pack->phydro->psrc->dtnew) );
    }
  }
  // MHD timestep
  if (pmb_pack->pmhd != nullptr) {
    dt = std::min(dt, (cfl_no)*(pmb_pack->pmhd->dtnew) );
    // viscosity timestep
    if (pmb_pack->pmhd->pvisc != nullptr) {
      dt = std::min(dt, (cfl_no)*(pmb_pack->pmhd->pvisc->dtnew) );
    }
    // resistivity timestep
    if (pmb_pack->pmhd->presist != nullptr) {
      dt = std::min(dt, (cfl_no)*(pmb_pack->pmhd->presist->dtnew) );
    }
    // thermal conduction timestep
    if (pmb_pack->pmhd->pcond != nullptr) {
      dt = std::min(dt, (cfl_no)*(pmb_pack->pmhd->pcond->dtnew) );
    }
    // source terms timestep
    if (pmb_pack->pmhd->psrc != nullptr) {
      dt = std::min(dt, (cfl_no)*(pmb_pack->pmhd->psrc->dtnew) );
    }
  }
  // z4c timestep
  if (pmb_pack->pz4c != nullptr) {
    dt = std::min(dt, (cfl_no)*(pmb_pack->pz4c->dtnew) );
  }
  // Radiation timestep
  if (pmb_pack->prad != nullptr) {
    dt = std::min(dt, (cfl_no)*(pmb_pack->prad->dtnew) );
  }
  // Particles timestep
  if (pmb_pack->ppart != nullptr) {
    dt = std::min(dt, (pmb_pack->ppart->dtnew) );
  }

#if MPI_PARALLEL_ENABLED
  // get minimum dt over all MPI ranks
  MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
#endif

  // limit last time step to stop at tlim *exactly*
  if ( (time < tlim) && ((time + dt) > tlim) ) {dt = tlim - time;}

  return;
}

//----------------------------------------------------------------------------------------
// \fn Mesh::AddCoordinatesAndPhysics

void Mesh::AddCoordinatesAndPhysics(ParameterInput *pinput) {
  // cycle over MeshBlockPacks on this rank and add Coordinates and Physics
  for (int n=0; n<nmb_packs_thisrank; ++n) {
    pmb_pack->AddCoordinates(pinput);
    pmb_pack->AddPhysics(pinput);
  }

  // Determine total number of particles across all ranks
  particles::Particles *ppart = pmb_pack->ppart;
  if (ppart != nullptr) {
    nprtcl_thisrank = 0;
    for (int n=0; n<nmb_packs_thisrank; ++n) {
      nprtcl_thisrank += pmb_pack->ppart->nprtcl_thispack;
    }
    nprtcl_eachrank = new int[global_variable::nranks];
    nprtcl_eachrank[global_variable::my_rank] = nprtcl_thisrank;
#if MPI_PARALLEL_ENABLED
    // Share number of particles on each rank with all ranks
    MPI_Allgather(&nprtcl_thisrank,1,MPI_INT,nprtcl_eachrank,1,MPI_INT,MPI_COMM_WORLD);
#endif
    for (int n=0; n<global_variable::nranks; ++n) {
      nprtcl_total += nprtcl_eachrank[n];
    }
    // Assign particle IDs
    if (pmb_pack->ppart != nullptr) {
      pmb_pack->ppart->CreateParticleTags(pinput);
    }
  }

  // Call RefinementCriteria constructor to enroll various criteria
  // can only be done after the physics modules have been constructed
  if (adaptive) {
    pmr->pmrc = new RefinementCriteria(this, pinput);
  }
}
