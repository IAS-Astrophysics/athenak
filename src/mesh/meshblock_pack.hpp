#ifndef MESH_MESHBLOCK_PACK_HPP_
#define MESH_MESHBLOCK_PACK_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file meshblock_pack.hpp
//  \brief defines MeshBlockPack class, a container for MeshBlocks

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "parameter_input.hpp"
#include "coordinates/coordinates.hpp"
#include "driver/driver.hpp"
#include "tasklist/task_list.hpp"

// Forward declarations
class MeshBlock;
class ADM;
class Tmunu;
namespace hydro {class Hydro;}
namespace mhd {class MHD;}
namespace ion_neutral {class IonNeutral;}
namespace radiation {class Radiation;}
namespace dyngr {class DynGRMHD;}
namespace numrel {class NumericalRelativity;}
class TurbulenceDriver;
namespace radiation {class Radiation;}
namespace z4c {class Z4c;}
namespace z4c {class CCE;}
namespace adm {class ADM;}
namespace particles {class Particles;}
namespace units {class Units;}

//----------------------------------------------------------------------------------------
//! \class MeshBlockPack
//! \brief data/functions associated with a single block

class MeshBlockPack {
  // mesh classes (Mesh, MeshBlock, MeshBlockPack, MeshBlockTree) like to play together
  friend class Mesh;
  friend class MeshBlock;
  friend class MeshBlockTree;

 public:
  MeshBlockPack(Mesh *pm, int igids, int igide);
  ~MeshBlockPack();

  // data
  Mesh *pmesh;            // ptr to Mesh containing this MeshBlockPack
  int gids, gide;         // start/end of global IDs in this MeshBlockPack
  int nmb_thispack;       // number of MBs in this pack

  // following Grid/Physics objects are all pointers so they can be allocated after
  // MeshBlockPack is constructed with pointer to my_pack.

  MeshBlock* pmb;         // MeshBlocks in this MeshBlockPack
  Coordinates* pcoord;

  // physics (controlled by AddPhysics() function in meshblock_pack.cpp)
  hydro::Hydro *phydro=nullptr;
  mhd::MHD *pmhd=nullptr;
  adm::ADM *padm=nullptr;
  Tmunu *ptmunu=nullptr;
  z4c::Z4c *pz4c=nullptr;
  dyngr::DynGRMHD *pdyngr=nullptr;
  numrel::NumericalRelativity *pnr=nullptr;
  ion_neutral::IonNeutral *pionn=nullptr;
  TurbulenceDriver *pturb=nullptr;
  radiation::Radiation *prad=nullptr;
  std::vector<z4c::CCE *> pz4c_cce;
  particles::Particles *ppart=nullptr;

  // units (needed to convert code units to cgs for, e.g., cooling or radiation)
  units::Units *punit=nullptr;

  // map for task lists which operate over all MeshBlocks in this MeshBlockPack
  std::map<std::string, std::shared_ptr<TaskList>> tl_map;

  // functions
  void AddPhysics(ParameterInput *pin);
  void AddMeshBlocks(ParameterInput *pin);
  void AddCoordinates(ParameterInput *pin);

 private:
  // data

  // functions
  void SetNeighbors(std::unique_ptr<MeshBlockTree> &ptree, int *ranklist);
};

#endif // MESH_MESHBLOCK_PACK_HPP_
