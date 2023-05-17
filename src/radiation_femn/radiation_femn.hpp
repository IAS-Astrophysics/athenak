#ifndef ATHENA_RADIATION_FEMN_HPP
#define ATHENA_RADIATION_FEMN_HPP

//========================================================================================
// GR radiation code for AthenaK with FEM_N & FP_N
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
        int refinement_level;           // a parameter which states the level of angular refinement (does not change)
        int num_energy_bins;            // number of energy bins
        Real energy_max;                // maximum value of energy
        int num_points;                 // number of points on the angular grid (FEM_n) or total number of (l,m) modes (FP_N)
        int num_ref;                    // current refinement level of geodesic grid
        int num_edges;                  // number of unique edges
        int num_triangles;              // number of unique triangular elements
        int basis;                      // choice of basis functions on the geodesic grid (1: tent - FEM_N)

        std::string limiter_dg;         // choice of limiter for DG, set to "minmod2" by default
        std::string limiter_fem;        // choice of limiter for FEM, set to "clp" by default (FEM_N)
        bool fpn;                       // flag to enable/disable FP_N, disabled by default (FP_N)
        int lmax;                       // maximum value of l when FP_N is used, set to 0 by default (FP_N)
        int filter_sigma_eff;           // effective opacity of the FP_N filter, set to 0 by default (FP_N)

        bool rad_source;                // flag to enable/disable source terms for radiation, disabled by default
        bool beam_source;               // flag to enable/disable beam sources, disabled by default

        // ---------------------------------------------------------------------------
        // matrices for the angular grid
        // ---------------------------------------------------------------------------
        DvceArray2D<Real> angular_grid;            // store the values of (l,m) for FP_N. Alternatively store (phi,theta) for FEM_N
        DvceArray1D<Real> energy_grid;             // array containing the energy grid

        DvceArray2D<Real> mass_matrix;             // mass matrix (in the special relativistic case) [Eqn. 12 of arXiv:2212.01409]
        DvceArray2D<Real> stiffness_matrix_x;      // x component of the stiffness matrix (in the special relativistic case) [Eqn. 12 of arXiv:2212.01409]
        DvceArray2D<Real> stiffness_matrix_y;      // y component of the stiffness matrix (in the special relativistic case) [Eqn. 12 of arXiv:2212.01409]
        DvceArray2D<Real> stiffness_matrix_z;      // z component of the stiffness matrix (in the special relativistic case) [Eqn. 12 of arXiv:2212.01409]

        DvceArray3D <Real> P_matrix;                // P ^muhat ^A _B
        DvceArray5D<Real> G_matrix;                 // G ^nuhat ^muhat _ihat ^A _B
        DvceArray5D<Real> F_matrix;                 // F ^nuhat ^muhat _ihat ^A _B

        // ---------------------------------------------------------------------------
        // distribution function, flux and other arrays
        // ---------------------------------------------------------------------------
        DvceArray6D<Real> f0;         // distribution function
        DvceArray6D<Real> f1;         // distribution at intermediate step
        DvceArray6D<Real> coarse_f0;  // distribution function on 2x coarser grid (for SMR/AMR)
        DvceFaceFld6D<Real> iflx;      // spatial fluxes on zone faces

        // intermediate arrays needed for limiting
        DvceArray6D<Real> ftemp;
        DvceArray4D<Real> etemp0;
        DvceArray4D<Real> etemp1;

        // Arrays for holding tetrad quantities
        DvceArray6D<Real> L_mu_muhat0;   // tetrad quantities
        DvceArray6D<Real> L_mu_muhat1;   // tetrad quantities at intermediate step

        // information from other projects (metric and fluid velocity in lab frame)
        DvceArray6D<Real> g_dd;         // placeholder for spatial metric
        DvceArray4D<Real> sqrt_det_g;   // square root of determinant of matrix
        DvceArray5D<Real> u_mu;         // placeholder for fluid velocity in lab frame
        DvceArray6D<Real> Gamma;        // Christoeffel symbol in lab frame
        // ---------------------------------------------------------------------------
        // arrays for source terms
        // ---------------------------------------------------------------------------
        DvceArray5D<Real> eta;          // emissivity [assume isotropic]
        DvceArray5D<Real> kappa_s;      // scattering coefficient [assume isotropic]
        DvceArray5D<Real> kappa_a;      // absorption coefficient [assume isotropic]

        DvceArray6D<bool> beam_mask;    // boolean mask used for beam source term

        DvceArray1D<Real> e_source;     // [Eq. (19) of Radice et. al. arXiv:1209.1634v3]
        DvceArray2D<Real> S_source;     // [Eq. (19) of Radice et. al. arXiv:1209.1634v3]
        DvceArray2D<Real> W_matrix;     // holds the inverse of (delta^A_B - k * S^A_B) where k = dt or dt/2

        // ---------------------------------------------------------------------------
        // other things
        BoundaryValuesCC *pbval_f;      // Boundary communication buffers and functions for f
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
        // Functions for GeodesicGrid & tetrad
        void LoadFEMNMatrices();
        void TetradInitialize();
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
