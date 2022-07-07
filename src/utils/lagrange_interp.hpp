//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file lagrange_interp.hpp

#include "athena.hpp"
#include "mesh/mesh.hpp"
class LagrangeInterp1D {
    public:
    LagrangeInterp1D(MeshBlockPack *pmbp, int *meshblock_ind, int *coordinate_ind, Real *coordinate, int *axis);
    DualArray1D<Real> interp_weight; 
    int nghost;
    int mb_ind;
    int coord_ind;
    int ax;
    Real coord;
    void CalculateWeight(MeshBlockPack *pmbp);
    Real Evaluate(DualArray1D<Real> &value);
};

class LagrangeInterp2D {
    public:
    LagrangeInterp2D(MeshBlockPack *pmbp, int *meshblock_ind, int coordinate_ind[2], Real coordinate[2], int axis[2]);
    int nghost;
    int mb_ind;
    int coord_ind[2];
    int ax[2];
    Real coord[2];
    Real Evaluate(MeshBlockPack *pmbp, DualArray2D<Real> &value);
};

class LagrangeInterp3D {
    public:
    LagrangeInterp3D(MeshBlockPack *pmbp, int *meshblock_ind, int coordinate_ind[3], Real coordinate[3], int axis[3]);
    int nghost;
    int mb_ind;
    int coord_ind[3];
    int ax[3];
    Real coord[3];
    Real Evaluate(MeshBlockPack *pmbp, DualArray3D<Real> &value);
};