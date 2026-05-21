#ifndef GRAVITY_GRAVITY_HPP_
#define GRAVITY_GRAVITY_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mg_gravity.hpp
//! \brief defines MGGravity class

// C headers

// C++ headers

// Athena++ headers
#include "../athena.hpp"
#include "../mesh/meshblock_pack.hpp"
#include "../parameter_input.hpp"
#include "../multigrid/multigrid.hpp"
#include "../coordinates/coordinates.hpp"
#include "mg_gravity.hpp"

class MeshBlockPack;
class ParameterInput;
class Coordinates;
class Multigrid;
namespace gravity {
class Gravity {
    public:
        Gravity(MeshBlockPack *pmbp, ParameterInput *pin);
        ~Gravity();

        MeshBlockPack* pmy_pack;  // ptr to MeshBlock containing this Field
        DvceArray5D<Real> phi, coarse_phi;   // gravitational potential
        DvceArray5D<Real> def;   // defect from the Multigrid solver
        Real four_pi_G;
        bool output_defect;
        bool fill_ghost;
        MGGravityDriver *pmgd;
        MGGravity *pmg;
        void SaveFaceBoundaries();
        void RestoreFaceBoundaries();

        friend class MGGravityDriver;

    private:
        DvceArray5D<Real> fbuf_[6];
};
} // namespace gravity
#endif // GRAVITY_GRAVITY_HPP_