#ifndef ATHENA_RADIATION_FEMN_HPP
#define ATHENA_RADIATION_FEMN_HPP

//========================================================================================
// Radiation FEM_N code for Athena
// Copyright (C) 2023 Maitraya Bhattacharyya <mbb6217@psu.edu> and David Radice <dur566@psu.edu>
// AthenaXX copyright(C) James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn.hpp
//  \brief definitions for Radiation FEM_N class

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"

// forward declarations
class EquationOfState;
class Coordinates;
class Driver;

//----------------------------------------------------------------------------------------------
//! \struct RadiationFEMNTaskIDs
//  \brief container to hold TaskIDs of all radiation FEM_N tasks

struct RadiationFEMNTaskIDs {
    TaskID rad_irecv;
    TaskID copycons;
    TaskID rad_flux;
    TaskID rad_sendf;
    TaskID rad_recvf;
    TaskID rad_expl;
    TaskID rad_limfem;
    TaskID rad_limdg;
    TaskID rad_filterfpn;
    TaskID rad_src;
    TaskID rad_resti;
    TaskID rad_sendi;
    TaskID rad_recvi;
    TaskID bcs;
    TaskID rad_csend;
    TaskID rad_crecv;

};

namespace radiationfemn {

    //----------------------------------------------------------------------------------------------
    //! \class RadiationFEMN
    class RadiationFEMN {
    public:
        RadiationFEMN(MeshBlockPack *ppack, ParameterInput *pin);
        ~RadiationFEMN();

        // ---------------------------------------------------------------------------
        // parameters
        // ---------------------------------------------------------------------------
        int refinement_level;           // a parameter which states the level of refinement (does not change)
        int num_energy_bins;            // number of energy bins
        Real energy_max;                // maximum value of energy
        int num_points;                 // number of points on the grid or total number of (l,m) modes
        int num_ref;                    // current refinement level of geodesic grid
        int num_edges;                  // number of unique edges
        int num_triangles;              // number of unique triangular elements
        int basis;                      // choice of basis functions on the geodesic grid (1: tent 2: small tent 3: honeycomb 4: small honeycomb)

        std::string limiter_dg;         // choice of limiter for DG, set to "minmod2" by default
        std::string limiter_fem;        // choice of limiter for FEM, set to "clp" by default
        bool fpn;                       // flag to enable/disable FP_N, disabled by default
        int lmax;                       // maximum value of l when FP_N is used, set to 0 by default
        int filter_sigma_eff;           // effective opacity of the FP_N filter, set to 0 by default

        bool rad_source;                // flag to enable/disable source terms for radiation, disabled by default
        bool beam_source;               // flag to enable/disable beam sources, disabled by default
        // end of parameters
        // ---------------------------------------------------------------------------


        // ---------------------------------------------------------------------------
        // matrices for the angular grid
        // ---------------------------------------------------------------------------
        DvceArray2D<Real> lm_array;               // store the values of (l,m) for FP_N. Alternatively store (phi,theta) for FEM_N
        DvceArray1D<Real> energy_grid;             // array containing the energy grid

        DvceArray2D<Real> mass_matrix;             // mass matrix (in the special relativistic case)
        DvceArray2D<Real> stiffness_matrix_x;      // x component of the stiffness matrix (in the special relativistic case)
        DvceArray2D<Real> stiffness_matrix_y;      // y component of the stiffness matrix (in the special relativistic case)
        DvceArray2D<Real> stiffness_matrix_z;      // z component of the stiffness matrix (in the special relativistic case)

        DvceArray3D <Real> P_matrix;                // P ^muhat ^A _B
        DvceArray5D<Real> G_matrix;                 // G ^nuhat ^muhat _ihat ^A _B
        DvceArray5D<Real> F_matrix;                 // F ^nuhat ^muhat _ihat ^A _B

        // end of matrices for the angular grid
        // ---------------------------------------------------------------------------


        // ---------------------------------------------------------------------------
        // intensity, flux and other arrays
        // ---------------------------------------------------------------------------
        DvceArray5D <Real> i0;         // intensities
        DvceArray5D <Real> i1;         // intensities at intermediate step
        DvceArray5D <Real> coarse_i0;  // intensities on 2x coarser grid (for SMR/AMR)
        DvceFaceFld5D<Real> iflx;     // spatial fluxes on zone faces

        // intermediate arrays needed for limiting
        DvceArray5D<Real> itemp;
        DvceArray4D<Real> etemp0;
        DvceArray4D<Real> etemp1;

        // Arrays for holding tetrad quantities
        DvceArray6D<Real> g_dd;         // placeholder for spatial metric
        DvceArray5D<Real> u_mu;         // placeholder for fluid velocity in lab frame
        DvceArray5D<Real> n_mu;         // placeholder for normal vector to spatial slice
        DvceArray4D<Real> Lambda;       // placeholder for the Lorentz factor
        DvceArray6D<Real> L_mu_muhat;   // tetrad quantities

        // end of arrays for intensities and fluxes
        // ---------------------------------------------------------------------------


        // ---------------------------------------------------------------------------
        // arrays for source terms
        DvceArray4D<Real> eta;          // emissivity
        DvceArray4D<Real> kappa_s;      // scattering coefficient
        DvceArray4D<Real> kappa_a;      // absorption coefficient

        DvceArray5D<bool> beam_mask;  // boolean mask used for beam source term

        DvceArray1D<Real> e_source;     // defined in Eq. (19) of Radice et. al. [arXiv:1209.1634v3]
        DvceArray2D<Real> S_source;     // also defined in Eq. (19)
        DvceArray2D<Real> W_matrix;     // holds the inverse of (delta^A_B - k * S^A_B) where k = dt or dt/2
        // end of source term arrays
        // ---------------------------------------------------------------------------


        // ---------------------------------------------------------------------------
        // other things
        BoundaryValuesCC *pbval_i;      // Boundary communication buffers and functions for i
        Real dtnew;                     // timestep
        RadiationFEMNTaskIDs id;        // container to hold TaskIDs
        // end of other things
        // ---------------------------------------------------------------------------


        // ---------------------------------------------------------------------------
        // Tasklist & associated functions
        // ...in start task list
        TaskStatus InitRecv(Driver *d, int stage);
        // ...in run task list
        TaskStatus CopyCons(Driver *d, int stage);
        TaskStatus ComputeTetrad(Driver *d, int stage);
        TaskStatus CalculateFluxes(Driver *d, int stage);
        TaskStatus SendFlux(Driver *d, int stage);
        TaskStatus RecvFlux(Driver *d, int stage);
        TaskStatus ExpRKUpdate(Driver *d, int stage);
        TaskStatus ApplyLimiterDG(Driver *pdriver, int stage);
        TaskStatus ApplyLimiterFEM(Driver *pdrive, int stage);
        TaskStatus ApplyFilterLanczos(Driver *pdriver, int stage);
        TaskStatus AddRadiationSourceTerm(Driver *d, int stage);
        void AddBeamSource(DvceArray5D <Real> &i0);
        TaskStatus RestrictI(Driver *d, int stage);
        TaskStatus SendI(Driver *d, int stage);
        TaskStatus RecvI(Driver *d, int stage);
        TaskStatus ApplyPhysicalBCs(Driver *pdrive, int stage);
        TaskStatus NewTimeStep(Driver *d, int stage);
        // ...in end task list
        TaskStatus ClearSend(Driver *d, int stage);
        TaskStatus ClearRecv(Driver *d, int stage);
        void AssembleRadiationFEMNTasks(TaskList &start, TaskList &run, TaskList &end);
        // ---------------------------------------------------------------------------

        // ---------------------------------------------------------------------------
        // Functions for GeodesicGrid
        void LoadFEMNMatrices();
        // ---------------------------------------------------------------------------

        // ---------------------------------------------------------------------------
        // Functions for linear algebra
        template<size_t N>
        void CGSolve(double (&A)[N][N], double (&b)[N], double (&xinit)[N], double (&x)[N], double tolerance = 1e-6);
        template<size_t N>
        void CGMatrixInverse(double (&mat)[N][N], double (&guess)[N][N], double (&matinv)[N][N]);
        // ---------------------------------------------------------------------------

    private:
        MeshBlockPack *pmy_pack;  // ptr to MeshBlockPack containing this RadiationFEMN

    };

} // namespace radiationfemn

#endif //ATHENA_RADIATION_FEMN_HPP
