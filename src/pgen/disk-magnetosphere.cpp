//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file disk.cpp
//! \brief Initializes stratified Keplerian accretion disk in both cylindrical and
//! spherical polar coordinates.  Initial conditions are in vertical hydrostatic eqm.

// C headers

// C++ headers
#include <algorithm>  // min
#include <cmath>      // sqrt
#include <cstdlib>    // srand
#include <cstring>    // strcmp()
#include <cstdio>     // snprintf()
#include <fstream>
#include <iostream>   // endl
#include <limits>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../orbital_advection/orbital_advection.hpp"
#include "../parameter_input.hpp"

//#include "../particles/particles.hpp"


namespace {
void GetCylCoord(Real &rad,Real &phi,Real &z,Real &x1,Real &x2,Real &x3);
void GetSphCoord(Real &rc,Real &theta,Real &phi, Real &x1, Real &x2, Real &x3);
Real DenProfileCyl(MeshBlock *pmb,const Real rad, const Real phi, const Real z,
  int k, int j, int i);
Real PoverR(const Real rad, const Real phi, const Real z);
void VelProfileCyl(MeshBlock *pmb, const Real rad, const Real phi, const Real z,
  int i, int j, int k, Real &v1, Real &v2, Real &v3);
// problem parameters which are useful to make global to this file
Real gm0, r0, rho0, dslope, p0_over_r0, pslope, tcool, gamma_gas;
Real dfloor, rho_floor0, slope_rho_floor;
Real Omega0;
Real rs, smoothin, smoothtr, rfix;
Real Rmin, Ri, Ro, Rmax;
Real thmin, thi, tho, thmax;
Real rrigid, origid, rmagsph, denstar, ratmagfloor, ratmagfslope;
Real mm, b0, beta;
static Real firsttime;
static int bc_ix3, bc_ox3;
} // namespace

Real rho_floor(Real rc);
void AllSourceTerms(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar);
Real GravPot_coe(Real rc);
void Cooling(MeshBlock *pmb, const Real time, const Real dt,const AthenaArray<Real> &prim,
             const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);

void SetBfield(MeshBlock *pmb, Coordinates *pco, FaceField &b,
               int is, int ie, int js, int je, int ks, int ke, int ngh);
static Real A1(const Real x1, const Real x2, const Real x3);
static Real A2(const Real x1, const Real x2, const Real x3);
static Real A3(const Real x1, const Real x2, const Real x3);

void StarSourceTerms(MeshBlock *pmb, const Real time, const Real dt,
  const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);
void PlanetarySourceTerms(MeshBlock *pmb, const Real time, const Real dt,
  const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);



// Planet Potential
Real force_car_btoa_coe(const Real xca, const Real yca, const Real zca,
        const Real xcb, const Real ycb, const Real zcb, const Real gb, const Real rsoft2);
Real force_car_ind_coe(const Real xpp, const Real ypp, const Real zpp, const Real gmp);
// Force on the planet
Real PlanetForce(MeshBlock *pmb, int iout);

// adding planetary system
static Real insert_time;
class PlanetarySystem {
 public:
  int np;
  int ind;
  Real rsoft2;
  double *mass;
  double *massset;
  double *xp, *yp, *zp;         // position in Cartesian coord.
  explicit PlanetarySystem(int np);
  ~PlanetarySystem();
 public:
  void orbit(double dt);      // circular planetary orbit
};

//------------------------------------------
// constructor for planetary system for np planets
PlanetarySystem::PlanetarySystem(int np0) {
  np   = np0;
  ind  = 1;
  rsoft2 = 0.0;
  mass = new double[np];
  massset = new double[np];
  xp   = new double[np];
  yp   = new double[np];
  zp   = new double[np];
}

//---------------------------------------------
// destructor for planetary system

PlanetarySystem::~PlanetarySystem() {
  delete[] mass;
  delete[] massset;
  delete[] xp;
  delete[] yp;
  delete[] zp;
}

// Planet Potential
Real grav_pot_car_btoa(const Real xca, const Real yca, const Real zca,
        const Real xcb, const Real ycb, const Real zcb, const Real gb, const Real rsoft2);
Real grav_pot_car_ind(const Real xca, const Real yca, const Real zca,
        const Real xpp, const Real ypp, const Real zpp, const Real gmp);
// Force on the planet
Real PlanetForce(MeshBlock *pmb, int iout);
// planetary system
static PlanetarySystem *psys;


// User-defined boundary conditions for disk simulations
void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskInnerX1steadyhydro(
                 MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX1steadyhydro(
                 MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskInnerX2steadyhydro(
                 MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX2steadyhydro(
                 MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskInnerX3steadyhydro(
                 MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskInnerX3outflowhydro(
                 MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX3steadyhydro(
                 MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX3outflowhydro(
                 MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskInnerX1steadymag(
                 MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX1steadymag(
                 MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskInnerX2steadymag(
                 MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX2steadymag(
                 MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskInnerX3steadymag(
                 MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskInnerX3outflowmag(
                 MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX3steadymag(
                 MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX3outflowmag(
                 MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief Function to initialize problem-specific data in mesh class.  Can also be used
//! to initialize variables which are global to (and therefore can be passed to) other
//! functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  firsttime=time;

//;  if (PARTICLES) {
//;    omega0_taus = omega0 * DustParticles::GetStoppingTime();
//;    dust2gas = pin->GetOrAddReal("problem", "dust2gas", dust2gas);
//;  }


  // Get parameters for gravitatonal potential of central point mass
  if (std::strcmp(COORDINATE_SYSTEM, "cartesian") != 0
      && std::strcmp(COORDINATE_SYSTEM, "spherical_polar") != 0 ) {
    gm0 = pin->GetOrAddReal("problem","GM",0.0);
  } else {
    gm0 = 1.0;
  }
  r0 = pin->GetOrAddReal("problem","r0",1.0);

  // Get parameters for initial density and velocity
  rho0 = pin->GetReal("problem","rho0");
  dslope = pin->GetOrAddReal("problem","dslope",0.0);

  // Get parameters of initial pressure and cooling parameters
  if (NON_BAROTROPIC_EOS) {
    p0_over_r0 = pin->GetOrAddReal("problem","p0_over_r0",0.0025);
    pslope = pin->GetOrAddReal("problem","pslope",0.0);
    tcool = pin->GetOrAddReal("problem","tcool",0.0);
    gamma_gas = pin->GetReal("hydro","gamma");
  } else {
    p0_over_r0=SQR(pin->GetReal("hydro","iso_sound_speed"));
  }
  Real float_min = std::numeric_limits<float>::min();
  dfloor=pin->GetOrAddReal("hydro","dfloor",(1024*(float_min)));
  rho_floor0 = pin->GetReal("problem","rho_floor0");
  slope_rho_floor = pin->GetOrAddReal("problem","slope_rho_floor",0.0);

  rs = pin->GetOrAddReal("problem", "rsmooth",0.0);
  Omega0 = pin->GetOrAddReal("orbital_advection","Omega0",0.0);
  rfix = pin->GetOrAddReal("problem", "rfix",0.1);

  smoothin = pin->GetOrAddReal("problem","smoothin",0.0);
  smoothtr = pin->GetOrAddReal("problem","smoothtr",0.0);

  rrigid = pin->GetOrAddReal("problem","rrigid",0.0);
  origid = pin->GetOrAddReal("problem","origid",0.0);
  rmagsph = pin->GetOrAddReal("problem","rmagsph",0.0);
  denstar = pin->GetOrAddReal("problem","denstar",0.0);
  ratmagfloor = pin->GetOrAddReal("problem","ratmagfloor",1.0e6);
  ratmagfslope = pin->GetOrAddReal("problem","ratmagfslope", 5.5);

  // Set inner/outer boundaries for damping zones.
  // Define using maximum mesh size so damping zones are consistent across
  // coordinate systems
  Rmin = pin->GetOrAddReal("problem", "Rmin", 0.0);
  Ri =   pin->GetOrAddReal("problem", "Ri", 0.5);
  Ro =   pin->GetOrAddReal("problem", "Ro", 4.0);
  Rmax = pin->GetOrAddReal("problem", "Rmax", 5.0);
  thmin = pin->GetOrAddReal("problem", "thmin", 1.27); // 3 scale heights
  thi =   pin->GetOrAddReal("problem", "thi", 1.37);   // 2 scale heights
  tho =   pin->GetOrAddReal("problem", "tho", 1.77);   // 2 scale heights
  thmax = pin->GetOrAddReal("problem", "thmax", 1.87); // 3 scale heights

  // boundary flag
  bc_ix3 = pin->GetOrAddInteger("problem","bc_ix3",0);
  bc_ox3 = pin->GetOrAddInteger("problem","bc_ox3",0);

  // set particle properties
//;  if (PARTICLES) {
//;    mparticle = pin->GetOrAddReal("problem","dustmass",0.);
//;  }

  // Initialize the magnetic fields
  if (MAGNETIC_FIELDS_ENABLED) {
    beta = pin->GetReal("problem","beta");
    mm = pin->GetOrAddReal("problem","mm",0.0);
    b0=sqrt(2.*p0_over_r0*rho0/beta);
  }

  // set up the planetary system
  insert_time= pin->GetOrAddReal("problem","insert_time",5.0);
  int np = pin->GetOrAddInteger("planets","np",0);
  psys = new PlanetarySystem(np);
  psys->ind = pin->GetOrAddInteger("planets","ind",1);
  psys->rsoft2 = pin->GetOrAddReal("planets","rsoft2",0.0016);

  // set initial planet properties
  for(int ip=0; ip<psys->np; ++ip) {
    char pname[10];
    // All sprintf in original code have been changed to snprintf to pass linter
    // But snprintf needs different arguments so this needs to be fixed
    snprintf(pname,"mass%d",ip);
    psys->massset[ip]=pin->GetOrAddReal("planets",pname,1.0e-2);
    psys->mass[ip]=0.0;
    snprintf(pname,"x%d",ip);
    psys->xp[ip]=pin->GetOrAddReal("planets",pname,1.0);
    snprintf(pname,"y%d",ip);
    psys->yp[ip]=pin->GetOrAddReal("planets",pname,0.0);
    snprintf(pname,"z%d",ip);
    psys->zp[ip]=pin->GetOrAddReal("planets",pname,0.0);
  }

/*
  insert_time = 5.;
  int np=1;
  psys = new PlanetarySystem(np);
  psys->ind = 1;
  psys->rsoft2 = 0.0016;

  psys->massset[0]=1.e-3;
  psys->mass[0]=0.0;
  psys->xp[0]=1.0;
  psys->yp[0]=0.0;
  psys->zp[0]=0.0;
*/
  // enroll planetary potential
  AllocateUserHistoryOutput(14*np);
  for(int ip=0; ip<psys->np; ++ip) {
    char pname[10];
    snprintf(pname,"fr%d",ip);
    EnrollUserHistoryOutput(0+14*ip, PlanetForce, pname);
    snprintf(pname,"ft%d",ip);
    EnrollUserHistoryOutput(1+14*ip, PlanetForce, pname);
    snprintf(pname,"fp%d",ip);
    EnrollUserHistoryOutput(2+14*ip, PlanetForce, pname);
    snprintf(pname,"fxpp%d",ip);
    EnrollUserHistoryOutput(3+14*ip, PlanetForce, pname);
    snprintf(pname,"fypp%d",ip);
    EnrollUserHistoryOutput(4+14*ip, PlanetForce, pname);
    snprintf(pname,"fzpp%d",ip);
    EnrollUserHistoryOutput(5+14*ip, PlanetForce, pname);
    snprintf(pname,"torque%d",ip);
    EnrollUserHistoryOutput(6+14*ip, PlanetForce, pname);
    snprintf(pname,"xpp%d",ip);
    EnrollUserHistoryOutput(7+14*ip, PlanetForce, pname);
    snprintf(pname,"ypp%d",ip);
    EnrollUserHistoryOutput(8+14*ip, PlanetForce, pname);
    snprintf(pname,"zpp%d",ip);
    EnrollUserHistoryOutput(9+14*ip, PlanetForce, pname);
    snprintf(pname,"rpp%d",ip);
    EnrollUserHistoryOutput(10+14*ip, PlanetForce, pname);
    snprintf(pname,"tpp%d",ip);
    EnrollUserHistoryOutput(11+14*ip, PlanetForce, pname);
    snprintf(pname,"ppp%d",ip);
    EnrollUserHistoryOutput(12+14*ip, PlanetForce, pname);
    snprintf(pname,"mp%d",ip);
    EnrollUserHistoryOutput(13+14*ip, PlanetForce, pname);
  }

  // enroll user-defined boundary condition
  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, DiskInnerX1);
  }
  if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, DiskOuterX1);
  }
  if (mesh_bcs[BoundaryFace::inner_x2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x2, DiskInnerX2);
  }
  if (mesh_bcs[BoundaryFace::outer_x2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x2, DiskOuterX2);
  }
  if (mesh_bcs[BoundaryFace::inner_x3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x3, DiskInnerX3);
  }
  if (mesh_bcs[BoundaryFace::outer_x3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x3, DiskOuterX3);
  }

  EnrollUserExplicitSourceFunction(AllSourceTerms);

  return;
}


//========================================================================================
//! \fn void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in MeshBlock class.  Can also be
//  used to initialize variables which are global to other functions in this file.
//  Called in MeshBlock constructor before ProblemGenerator.
//========================================================================================
void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  // store planet properties
  AllocateRealUserMeshBlockDataField(psys->np+1);
  for (int ip=0; ip< psys->np; ++ip) {
    ruser_meshblock_data[ip].NewAthenaArray(14);
  }
  ruser_meshblock_data[psys->np].NewAthenaArray(ncells3,ncells2,ncells1);

  // set total number of particles in the block.
//;  if(PARTICLES){
//;    ppar->npar = 1;
//;  }
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Initializes Keplerian accretion disk.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real x1, x2, x3;
  Real v1, v2, v3;

  if (MAGNETIC_FIELDS_ENABLED) {
    SetBfield(this, pcoord, pfield->b, is, ie, js, je, ks, ke, NGHOST);
    pfield->CalculateCellCenteredField(pfield->b,pfield->bcc,pcoord,is,ie,js,je,ks,ke);
  }
  //  Initialize density and momenta
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        // convert to cylindrical coordinates
        GetCylCoord(rad,phi,z,pcoord->x1v(i),pcoord->x2v(j),pcoord->x3v(k));
        // compute initial conditions in cylindrical coordinates
        phydro->u(IDN,k,j,i) = DenProfileCyl(this,rad,phi,z,k,j,i);
        VelProfileCyl(this,rad,phi,z,i,j,k,v1,v2,v3);

        phydro->u(IM1,k,j,i) = phydro->u(IDN,k,j,i)*v1;
        phydro->u(IM2,k,j,i) = phydro->u(IDN,k,j,i)*v2;
        phydro->u(IM3,k,j,i) = phydro->u(IDN,k,j,i)*v3;

        if (NON_BAROTROPIC_EOS) {
          Real p_over_r = PoverR(rad,phi,z);
          phydro->u(IEN,k,j,i) = p_over_r*phydro->u(IDN,k,j,i)/(gamma_gas - 1.0);
          phydro->u(IEN,k,j,i) += 0.5*(SQR(phydro->u(IM1,k,j,i))+SQR(phydro->u(IM2,k,j,i))
                                       + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);
          if (MAGNETIC_FIELDS_ENABLED)
            phydro->u(IEN,k,j,i) +=
              0.5*(SQR((pfield->bcc(IB1,k,j,i)))
                 + SQR((pfield->bcc(IB2,k,j,i)))
                 + SQR((pfield->bcc(IB3,k,j,i))));

//   phydro->u(IEN,k,j,i) += 0.5*(SQR(0.5*(pfield->b.x1f(k,j,i+1) + pfield->b.x1f(k,j,i)))
//                     + SQR(0.5*(pfield->b.x2f(k,j+1,i) + pfield->b.x2f(k,j,i)))
//                     + SQR(0.5*(pfield->b.x3f(k+1,j,i) + pfield->b.x3f(k,j,i))));
        }
      }
    }
  }

//;  if (PARTICLES) {
//;    DustParticles::SetOneParticleMass(mparticle);
//;    for (int k = 0; k < ppar->npar; ++k) {
//;      // Choose position.
//;    }
//;  }

  return;
}

// Set B field from is to ke
// We normally need to adjut js and je limist when it is polar boundary,
// since the B field at the pole
// is not set by the user. It also avoids deviding by zero.
// When we set the boundary conditions (e.g. i),
// we also don't change the first active zone interface B field
// (e.g. we only change the left face of
// is-ngh to is-1). However, to use a single function to set the field,
// we set the pole B even if it is
// polar boundary. To avoid deviding by zero, we add a small number protector.
// We also set the B fields
// at the the active zone interfaces.

void SetBfield(MeshBlock *pmb, Coordinates *pco, FaceField &b,
               int is, int ie, int js, int je, int ks, int ke, int ngh) {
  if (!MAGNETIC_FIELDS_ENABLED) return;
  BoundaryValues *pbval= pmb->pbval;
  AthenaArray<Real> a1, a2, a3;
  int nx1 = pmb->block_size.nx1 + 2*ngh+1;
  int nx2 = pmb->block_size.nx2 + 2*ngh+1;
  int nx3 = pmb->block_size.nx3 + 2*ngh+1;
  a1.NewAthenaArray(nx3, nx2, nx1);
  a2.NewAthenaArray(nx3, nx2, nx1);
  a3.NewAthenaArray(nx3, nx2, nx1);

  int level = pmb->loc.level;
  for (int k=ks; k<=ke+1; k++) {
    for (int j=js; j<=je+1; j++) {
      for (int i=is; i<=ie+1; i++) {
        if ((pbval->nblevel[1][0][1]>level && j==js)
            || (pbval->nblevel[1][2][1]>level && j==je+1)
            || (pbval->nblevel[0][1][1]>level && k==ks)
            || (pbval->nblevel[2][1][1]>level && k==ke+1)
            || (pbval->nblevel[0][0][1]>level && j==js   && k==ks)
            || (pbval->nblevel[0][2][1]>level && j==je+1 && k==ks)
            || (pbval->nblevel[2][0][1]>level && j==js   && k==ke+1)
            || (pbval->nblevel[2][2][1]>level && j==je+1 && k==ke+1)) {
          Real x1l = pco->x1f(i)+0.25*pco->dx1f(i);
          Real x1r = pco->x1f(i)+0.75*pco->dx1f(i);
          a1(k,j,i) = 0.5*(A1(x1l, pco->x2f(j), pco->x3f(k)) +
                           A1(x1r, pco->x2f(j), pco->x3f(k)));
//          Real rcl, rcr, thetal, thetar, phil, phir;
//          GetSphCoord(rcl,thetal,phil,pcoord->x1f(i)+0.25*pcoord->dx1f(i),
//                                      pcoord->x2f(j),pcoord->x3f(k));
//          GetSphCoord(rcr,thetar,phir,pcoord->x1f(i)+0.75*pcoord->dx1f(i),
//                                      pcoord->x2f(j),pcoord->x3f(k));
//          a1(k,j,i) = 0.5*(A1(rcl, thetal, phil) +
//                           A1(rcr, thetar, phir));
        } else {
          a1(k,j,i) = A1(pco->x1v(i), pco->x2f(j), pco->x3f(k));
//          Real rc, theta, phi;
//          GetSphCoord(rc,theta,phi,pcoord->x1v(i),pcoord->x2f(j),pcoord->x3f(k));
//          a1(k,j,i) = A1(rc, theta, phi);
        }

        if ((pbval->nblevel[1][1][0]>level && i==is)
            || (pbval->nblevel[1][1][2]>level && i==ie+1)
            || (pbval->nblevel[0][1][1]>level && k==ks)
            || (pbval->nblevel[2][1][1]>level && k==ke+1)
            || (pbval->nblevel[0][1][0]>level && i==is   && k==ks)
            || (pbval->nblevel[0][1][2]>level && i==ie+1 && k==ks)
            || (pbval->nblevel[2][1][0]>level && i==is   && k==ke+1)
            || (pbval->nblevel[2][1][2]>level && i==ie+1 && k==ke+1)) {
          Real x2l = pco->x2f(j)+0.25*pco->dx2f(j);
          Real x2r = pco->x2f(j)+0.75*pco->dx2f(j);
          a2(k,j,i) = 0.5*(A2(pco->x1f(i), x2l, pco->x3f(k)) +
                           A2(pco->x1f(i), x2r, pco->x3f(k)));
//          Real rcl, rcr, thetal, thetar, phil, phir;
//          GetSphCoord(rcl,thetal,phil,pcoord->x1f(i),
//                      pcoord->x2f(j)+0.25*pcoord->dx2f(j),pcoord->x3f(k));
//          GetSphCoord(rcr,thetar,phir,pcoord->x1f(i),
//                      pcoord->x2f(j)+0.75*pcoord->dx2f(j),pcoord->x3f(k));
//          a2(k,j,i) = 0.5*(A2(rcl, thetal, phil) +
//                           A2(rcr, thetar, phir));
        } else {
          a2(k,j,i) = A2(pco->x1f(i), pco->x2v(j), pco->x3f(k));
//          Real rc, theta, phi;
//          GetSphCoord(rc,theta,phi,pcoord->x1f(i),pcoord->x2v(j),pcoord->x3f(k));
//          a2(k,j,i) = A2(rc, theta, phi);
        }

        if ((pbval->nblevel[1][1][0]>level && i==is)
            || (pbval->nblevel[1][1][2]>level && i==ie+1)
            || (pbval->nblevel[1][0][1]>level && j==js)
            || (pbval->nblevel[1][2][1]>level && j==je+1)
            || (pbval->nblevel[1][0][0]>level && i==is   && j==js)
            || (pbval->nblevel[1][0][2]>level && i==ie+1 && j==js)
            || (pbval->nblevel[1][2][0]>level && i==is   && j==je+1)
            || (pbval->nblevel[1][2][2]>level && i==ie+1 && j==je+1)) {
          Real x3l = pco->x3f(k)+0.25*pco->dx3f(k);
          Real x3r = pco->x3f(k)+0.75*pco->dx3f(k);
          a3(k,j,i) = 0.5*(A3(pco->x1f(i), pco->x2f(j), x3l) +
                           A3(pco->x1f(i), pco->x2f(j), x3r));
//          Real rcl, rcr, thetal, thetar, phil, phir;
//          GetSphCoord(rcl,thetal,phil,pcoord->x1f(i),pcoord->x2f(j),
//                                      pcoord->x3f(k)+0.25*pcoord->dx3f(k));
//          GetSphCoord(rcr,thetar,phir,pcoord->x1f(i),pcoord->x2f(j),
//                                      pcoord->x3f(k)+0.75*pcoord->dx3f(k));
//          a3(k,j,i) = 0.5*(A3(rcl, thetal, phil) +
//                           A3(rcr, thetar, phir));
        } else {
          a3(k,j,i) = A3(pco->x1f(i), pco->x2f(j), pco->x3v(k));
//          Real rc, theta, phi;
//          GetSphCoord(rc,theta,phi,pcoord->x1f(i),pcoord->x2f(j),pcoord->x3v(k));
//          a3(k,j,i) = A3(rc, theta, phi);
        }
      }
    }
  }

  // Initialize interface fields
  AthenaArray<Real> area, len, len_p1;
  area.NewAthenaArray(nx1);
  len.NewAthenaArray(nx1);
  len_p1.NewAthenaArray(nx1);
  Real real_min = std::numeric_limits<Real>::min();

  // for 1,2,3-D
  for (int k=ks; k<=ke; ++k) {
    // reset loop limits for polar boundary
    int jl=js; int ju=je+1;
//    if (pbval->block_bcs[BoundaryFace::inner_x2] == BoundaryFlag::polar) jl=js+1;
//    if (pbval->block_bcs[BoundaryFace::outer_x2] == BoundaryFlag::polar) ju=je;
    for (int j=jl; j<=ju; ++j) {
      pco->Face2Area(k,j,is,ie,area);
      pco->Edge3Length(k,j,is,ie+1,len);
      for (int i=is; i<=ie; ++i) {
        area(i) = (area(i)<real_min)? real_min : area(i);
        b.x2f(k,j,i) = -1.0*(len(i+1)*a3(k,j,i+1) - len(i)*a3(k,j,i))/area(i);
      }
    }
  }
  for (int k=ks; k<=ke+1; ++k) {
    for (int j=js; j<=je; ++j) {
      pco->Face3Area(k,j,is,ie,area);
      pco->Edge2Length(k,j,is,ie+1,len);
      for (int i=is; i<=ie; ++i) {
        b.x3f(k,j,i) = (len(i+1)*a2(k,j,i+1) - len(i)*a2(k,j,i))/area(i);
      }
    }
  }

  // for 2D and 3D
  if (pmb->block_size.nx2 > 1) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        pco->Face1Area(k,j,is,ie+1,area);
        pco->Edge3Length(k,j  ,is,ie+1,len);
        pco->Edge3Length(k,j+1,is,ie+1,len_p1);
        for (int i=is; i<=ie+1; ++i) {
          b.x1f(k,j,i) = (len_p1(i)*a3(k,j+1,i) - len(i)*a3(k,j,i))/area(i);
        }
      }
    }
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je; ++j) {
        pco->Face3Area(k,j,is,ie,area);
        pco->Edge1Length(k,j  ,is,ie,len);
        pco->Edge1Length(k,j+1,is,ie,len_p1);
        for (int i=is; i<=ie; ++i) {
          b.x3f(k,j,i) -= (len_p1(i)*a1(k,j+1,i) - len(i)*a1(k,j,i))/area(i);
        }
      }
    }
  }
  // for 3D only
  if (pmb->block_size.nx3 > 1) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        pco->Face1Area(k,j,is,ie+1,area);
        pco->Edge2Length(k  ,j,is,ie+1,len);
        pco->Edge2Length(k+1,j,is,ie+1,len_p1);
        for (int i=is; i<=ie+1; ++i) {
          b.x1f(k,j,i) -= (len_p1(i)*a2(k+1,j,i) - len(i)*a2(k,j,i))/area(i);
        }
      }
    }
    for (int k=ks; k<=ke; ++k) {
      // reset loop limits for polar boundary
      int jl=js; int ju=je+1;
//      if (pbval->block_bcs[BoundaryFace::inner_x2] == BoundaryFlag::polar) jl=js+1;
//      if (pbval->block_bcs[BoundaryFace::outer_x2] == BoundaryFlag::polar) ju=je;
      for (int j=jl; j<=ju; ++j) {
        pco->Face2Area(k,j,is,ie,area);
        pco->Edge1Length(k  ,j,is,ie,len);
        pco->Edge1Length(k+1,j,is,ie,len_p1);
        for (int i=is; i<=ie; ++i) {
          area(i) = (area(i)<real_min)? real_min : area(i);
          b.x2f(k,j,i) += (len_p1(i)*a1(k+1,j,i) - len(i)*a1(k,j,i))/area(i);
        }
      }
    }
  }
  return;
}


namespace {
//----------------------------------------------------------------------------------------
//! transform to cylindrical coordinate

void GetCylCoord(Real &rad,Real &phi,Real &z, Real &x1, Real &x2, Real &x3) {
  if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
    rad=sqrt(x1*x1 + x2*x2);
    phi=atan2(x2,x1);
    z=x3;
  } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    rad=x1;
    phi=x2;
    z=x3;
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    rad=std::abs(x1*std::sin(x2));
    phi=x3;
    z=x1*std::cos(x2);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! transform to spherical coordinate

void GetSphCoord(Real &rc,Real &theta,Real &phi, Real &x1, Real &x2, Real &x3) {
  if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
    rc = sqrt(x1*x1 + x2*x2 + x3*x3);
    theta = acos(x3/rc);
    phi = atan2(x2,x1);
  } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    rc = sqrt(x1*x1 + x3*x3);
    theta = acos(x3/rc);
    phi = x2;
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    rc = x1;
    theta = x2;
    phi = x3;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! computes density in cylindrical coordinates

Real DenProfileCyl(MeshBlock *pmb, const Real rad, const Real phi,
                   const Real z, int k, int j, int i) {
  Real den;
  if(pmb->pmy_mesh->time==firsttime) {
    Real r = std::max(rad, rs);
    Real p_over_r = p0_over_r0;
    if (NON_BAROTROPIC_EOS) p_over_r = PoverR(r, phi, z);
    Real denmid = rho0*std::pow(r/r0,dslope);
    Real dentem = denmid*std::exp(gm0/p_over_r*(1./std::sqrt(SQR(r)+SQR(z))-1./r));
    den = dentem;

    Real rc = sqrt(rad*rad+z*z);
    Real sinsq = rad*rad/rc/rc;
    if (rc<rmagsph) {
      den = den*exp(-SQR((rc-rmagsph)/smoothtr));
      if (NON_BAROTROPIC_EOS) p_over_r = PoverR(rs, phi, z);
      Real pre=denstar*p_over_r;
      Real rint = 0;
      Real dr = rs/100.;
      while(rint<rc) {
        if(rint<rs) {
          pre = pre +dr*origid*origid*rint*sinsq*pre/p_over_r;
        } else {
          pre = pre - dr*gm0/rint/rint*(rint-smoothin)*
                    (rint-smoothin)/((rint-smoothin)*(rint-smoothin)+smoothtr*smoothtr)*
                     pre/p_over_r+dr*origid*origid*rint*sinsq*pre/p_over_r;
        }
        rint = rint + dr;
// temperature at the equation is actually changing.
// But we still assume it is the temperature at the inner boundary
      }
      den += pre/p_over_r;
    }
    den = std::max(den,rho_floor(rc));
    pmb->ruser_meshblock_data[psys->np](k,j,i) = den;
  } else {
    den = pmb->ruser_meshblock_data[psys->np](k,j,i);
  }

  return den;
}

//----------------------------------------------------------------------------------------
//! computes pressure/density in cylindrical coordinates

Real PoverR(const Real rad, const Real phi, const Real z) {
  Real poverr;
  Real r = std::max(rad, rfix);
  poverr = p0_over_r0*std::pow(r/r0, pslope);
  return poverr;
}

//----------------------------------------------------------------------------------------
//! computes rotational velocity in cylindrical coordinates

void VelProfileCyl(MeshBlock *pmb, const Real rad, const Real phi, const Real z,
                   int i, int j, int k, Real &v1, Real &v2, Real &v3) {
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  Real r = std::max(rad, rs);
  Real p_over_r = PoverR(r, phi, z);
  Real vel = (dslope+pslope)*p_over_r/(gm0/r) + (1.0+pslope)
             - pslope*r/std::sqrt(r*r+z*z);
  vel = std::sqrt(gm0/r)*std::sqrt(vel) - r*Omega0;
  Real rc=sqrt(rad*rad+z*z);
  if (rc<rmagsph) vel = vel*exp(-SQR((rc-rmagsph)/smoothtr));
  if (rc<=rrigid) vel += origid*rad;

  if (pmb->porb->orbital_advection_defined && COORDINATE_SYSTEM != "cartesian")
    vel -= vK(pmb->porb, pmb->pcoord->x1v(i), pmb->pcoord->x2v(j), pmb->pcoord->x3v(k));

  if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
    v1=-1*vel*sin(phi);
    v2=vel*cos(phi);
    v3=0.0;
  } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    v1=0.0;
    v2=vel;
    v3=0.0;
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    v1=0.0;
    v2=0.0;
    v3=vel;
  }

  return;
}
} // namespace

Real rho_floor(Real rc) {
  Real rhofloor = 0.0;
  if (rc > rs) rhofloor = rho_floor0*pow(rc/r0, slope_rho_floor);
//  if (rc < rmagsph) rhofloor += rho0*mm*mm/beta/ratmagfloor*pow((r0/rc),7);
  if (mm != 0. && rc > rs) rhofloor += 4.*rho0*mm*mm/beta/ratmagfloor*
                                         pow((r0/rc),ratmagfslope);
  return std::max(rhofloor,dfloor);
}

static Real A1(const Real x1, const Real x2, const Real x3) {
  Real a1=0.0;
  if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
    Real rc = std::max(sqrt(x1*x1+x2*x2+x3*x3),rs/2.);
    a1 = mm*b0/rc/rc/rc*(-1.*x2);
  } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    a1 = 0.;
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    a1 = 0.;
  }
  return(a1);
}

static Real A2(const Real x1, const Real x2, const Real x3) {
  Real a2=0.0;
  if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
    Real rc = std::max(sqrt(x1*x1+x2*x2+x3*x3),rs/2.);
    a2 = mm*b0/rc/rc/rc*(1.*x1);
  } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    Real rc = std::max(sqrt(x1*x1+x3*x3),rs/2.);
    a2 = mm*b0/rc/rc/rc*(1.*x1);
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    a2 = 0.;
  }
  return(a2);
}

static Real A3(const Real x1, const Real x2, const Real x3) {
  Real a3=0.0;
  if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
    a3 = 0.0;
  } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    a3 = 0.0;
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    Real r = fabs(x1*sin(x2));
    Real rc = std::max(x1,rs/2.);
    a3 = mm*b0/rc/rc/rc*r;
  }
  return(a3);
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values
//

// inner x1
//

void DiskInnerX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
       Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  if(pmb->pmy_mesh->time==firsttime) {
    DiskInnerX1steadyhydro(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke,ngh);
    if (MAGNETIC_FIELDS_ENABLED) {
      DiskInnerX1steadymag(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke,ngh);
    }
  }
  return;
}

void DiskInnerX1steadyhydro(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim,
       FaceField &b,
       Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  Real rad,phi,z;
  Real v1, v2, v3;
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=ngh; ++i) {
        GetCylCoord(rad,phi,z,pco->x1v(is-i),pco->x2v(j),pco->x3v(k));
        prim(IDN,k,j,is-i) = DenProfileCyl(pmb,rad,phi,z,k,j,i);
        VelProfileCyl(pmb,rad,phi,z,is-i,j,k,v1,v2,v3);
        prim(IM1,k,j,is-i) = v1;
        prim(IM2,k,j,is-i) = v2;
        prim(IM3,k,j,is-i) = v3;
        if (NON_BAROTROPIC_EOS)
          prim(IEN,k,j,is-i) = PoverR(rad, phi, z)*prim(IDN,k,j,is-i);
      }
    }
  }
}

void DiskInnerX1steadymag(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim,
       FaceField &b,
       Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  if (MAGNETIC_FIELDS_ENABLED)
    SetBfield(pmb, pco, b, is-ngh, is-1, js, je, ks, ke, ngh);
  return;
}

// outer x1
//

void DiskOuterX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
       Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  if (pmb->pmy_mesh->time==firsttime) {
    DiskOuterX1steadyhydro(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke,ngh);
    if (MAGNETIC_FIELDS_ENABLED) {
      DiskOuterX1steadymag(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke,ngh);
    }
  }
  return;
}

void DiskOuterX1steadyhydro(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim,
       FaceField &b,
       Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  Real rad,phi,z;
  Real v1, v2, v3;
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=ngh; ++i) {
        GetCylCoord(rad,phi,z,pco->x1v(ie+i),pco->x2v(j),pco->x3v(k));
        prim(IDN,k,j,ie+i) = DenProfileCyl(pmb,rad,phi,z,k,j,i);
        VelProfileCyl(pmb,rad,phi,z,ie+i,j,k,v1,v2,v3);
        prim(IM1,k,j,ie+i) = v1;
        prim(IM2,k,j,ie+i) = v2;
        prim(IM3,k,j,ie+i) = v3;
        if (NON_BAROTROPIC_EOS)
          prim(IEN,k,j,ie+i) = PoverR(rad, phi, z)*prim(IDN,k,j,ie+i);
      }
    }
  }
  return;
}

void DiskOuterX1steadymag(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim,
       FaceField &b,
       Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  if (MAGNETIC_FIELDS_ENABLED)
    SetBfield(pmb, pco, b, ie+1, ie+ngh, js, je, ks, ke, ngh);
  return;
}

// inner x2
//

void DiskInnerX2(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
       Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  if (pmb->pmy_mesh->time==firsttime) {
    DiskInnerX2steadyhydro(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke,ngh);
    if (MAGNETIC_FIELDS_ENABLED) {
      DiskInnerX2steadymag(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke,ngh);
    }
  }
  return;
}

void DiskInnerX2steadyhydro(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim,
       FaceField &b,
       Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  Real rad,phi,z;
  Real v1, v2, v3;
  for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=is; i<=ie; ++i) {
        GetCylCoord(rad,phi,z,pco->x1v(i),pco->x2v(js-j),pco->x3v(k));
        prim(IDN,k,js-j,i) = DenProfileCyl(pmb,rad,phi,z,k,j,i);
        VelProfileCyl(pmb,rad,phi,z,i,js-j,k,v1,v2,v3);
        prim(IM1,k,js-j,i) = v1;
        prim(IM2,k,js-j,i) = v2;
        prim(IM3,k,js-j,i) = v3;
        if (NON_BAROTROPIC_EOS)
          prim(IEN,k,js-j,i) = PoverR(rad, phi, z)*prim(IDN,k,js-j,i);
      }
    }
  }
  return;
}

void DiskInnerX2steadymag(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim,
       FaceField &b,
       Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  if (MAGNETIC_FIELDS_ENABLED)
    SetBfield(pmb, pco, b, is, ie, js-ngh, js-1, ks, ke, ngh);
  return;
}

// outer x2
//

void DiskOuterX2(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
       Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  if(pmb->pmy_mesh->time==firsttime) {
    DiskOuterX2steadyhydro(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke,ngh);
    if (MAGNETIC_FIELDS_ENABLED) {
      DiskOuterX2steadymag(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke,ngh);
    }
  }
  return;
}

void DiskOuterX2steadyhydro(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim,
       FaceField &b,
       Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  Real rad,phi,z;
  Real v1, v2, v3;
  for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=is; i<=ie; ++i) {
        GetCylCoord(rad,phi,z,pco->x1v(i),pco->x2v(je+j),pco->x3v(k));
        prim(IDN,k,je+j,i) = DenProfileCyl(pmb,rad,phi,z,k,j,i);
        VelProfileCyl(pmb,rad,phi,z,i,je+j,k,v1,v2,v3);
        prim(IM1,k,je+j,i) = v1;
        prim(IM2,k,je+j,i) = v2;
        prim(IM3,k,je+j,i) = v3;
        if (NON_BAROTROPIC_EOS)
          prim(IEN,k,je+j,i) = PoverR(rad, phi, z)*prim(IDN,k,je+j,i);
      }
    }
  }
  return;
}

void DiskOuterX2steadymag(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim,
       FaceField &b,
       Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  if (MAGNETIC_FIELDS_ENABLED)
    SetBfield(pmb, pco, b, is, ie, je+1, je+ngh, ks, ke, ngh);
  return;
}

// inner x3
//

void DiskInnerX3(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
       Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  if (bc_ix3 == 0) {
    if(pmb->pmy_mesh->time==firsttime) {
      DiskInnerX3steadyhydro(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke,ngh);
      if (MAGNETIC_FIELDS_ENABLED) {
        DiskInnerX3steadymag(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke,ngh);
      }
    }
  } else if (bc_ix3==1) {
    DiskInnerX3outflowhydro(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke,ngh);
    if (MAGNETIC_FIELDS_ENABLED) {
      DiskInnerX3outflowmag(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke,ngh);
    }
  }
  return;
}

void DiskInnerX3steadyhydro(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim,
       FaceField &b,
       Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  Real rad,phi,z;
  Real v1, v2, v3;
  for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        GetCylCoord(rad,phi,z,pco->x1v(i),pco->x2v(j),pco->x3v(ks-k));
        prim(IDN,ks-k,j,i) = DenProfileCyl(pmb,rad,phi,z,k,j,i);
        VelProfileCyl(pmb,rad,phi,z,i,j,ks-k,v1,v2,v3);
        prim(IM1,ks-k,j,i) = v1;
        prim(IM2,ks-k,j,i) = v2;
        prim(IM3,ks-k,j,i) = v3;
        if (NON_BAROTROPIC_EOS)
          prim(IEN,ks-k,j,i) = PoverR(rad, phi, z)*prim(IDN,ks-k,j,i);
      }
    }
  }
  return;
}

void DiskInnerX3outflowhydro(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim,
       FaceField &b,
       Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          prim(n,ks-k,j,i) = prim(n,ks,j,i);
          if(n==IVZ && prim(n,ks-k,j,i)>0.0) prim(n,ks-k,j,i)=0.0;
        }
      }
    }
  }
  return;
}


void DiskInnerX3steadymag(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim,
       FaceField &b,
       Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  if (MAGNETIC_FIELDS_ENABLED)
    SetBfield(pmb, pco, b, is, ie, js, je, ks-ngh, ks-1, ngh);
  return;
}

void DiskInnerX3outflowmag(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim,
       FaceField &b,
       Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie+1; ++i) {
          b.x1f((ks-k),j,i) = b.x1f(ks,j,i);
        }
      }
    }
    for (int k=1; k<=ngh; ++k) {
      for (int j=js; j<=je+1; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          b.x2f((ks-k),j,i) = b.x2f(ks,j,i);
        }
      }
    }
    for (int k=1; k<=ngh; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          b.x3f((ks-k),j,i) = b.x3f(ks,j,i);
        }
      }
    }
  }
  return;
}


// outer x3
//

void DiskOuterX3(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
       Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  if (bc_ox3==0) {
    if (pmb->pmy_mesh->time==firsttime) {
      DiskOuterX3steadyhydro(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke,ngh);
      if (MAGNETIC_FIELDS_ENABLED) {
        DiskOuterX3steadymag(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke,ngh);
      }
    }
  } else if (bc_ox3==1) {
    DiskOuterX3outflowhydro(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke,ngh);
    if (MAGNETIC_FIELDS_ENABLED)
      DiskOuterX3outflowmag(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke,ngh);
  }
  return;
}

void DiskOuterX3steadyhydro(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim,
       FaceField &b,
       Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  Real rad,phi,z;
  Real v1, v2, v3;
  for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        GetCylCoord(rad,phi,z,pco->x1v(i),pco->x2v(j),pco->x3v(ke+k));
        prim(IDN,ke+k,j,i) = DenProfileCyl(pmb,rad,phi,z,k,j,i);
        VelProfileCyl(pmb,rad,phi,z,i,j,ke+k,v1,v2,v3);
        prim(IM1,ke+k,j,i) = v1;
        prim(IM2,ke+k,j,i) = v2;
        prim(IM3,ke+k,j,i) = v3;
        if (NON_BAROTROPIC_EOS)
          prim(IEN,ke+k,j,i) = PoverR(rad, phi, z)*prim(IDN,ke+k,j,i);
      }
    }
  }
  return;
}

void DiskOuterX3outflowhydro(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim,
       FaceField &b,
       Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          prim(n,ke+k,j,i) = prim(n,ke,j,i);
          if (n==IVZ && prim(n,ke+k,j,i)<0.0) prim(n,ke+k,j,i)=0.0;
        }
      }
    }
  }
  return;
}


void DiskOuterX3steadymag(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim,
       FaceField &b,
       Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  if (MAGNETIC_FIELDS_ENABLED)
    SetBfield(pmb, pco, b, is, ie, js, je, ke+1, ke+ngh, ngh);
  return;
}

void DiskOuterX3outflowmag(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim,
       FaceField &b,
       Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie+1; ++i) {
          b.x1f((ke+k  ),j,i) = b.x1f((ke  ),j,i);
        }
      }
    }
    for (int k=1; k<=ngh; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          b.x2f((ke+k  ),j,i) = b.x2f((ke  ),j,i);
        }
      }
    }
    for (int k=1; k<=ngh; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          b.x3f((ke+k+1),j,i) = b.x3f((ke+1),j,i);
        }
      }
    }
  }
  return;
}


void AllSourceTerms(MeshBlock *pmb, const Real time, const Real dt,
  const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
  const AthenaArray<Real> &bcc, AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar) {
  StarSourceTerms(pmb,time,dt,prim,bcc,cons);
  if(psys->np>0) PlanetarySourceTerms(pmb,time,dt,prim,bcc,cons);
  if(NON_BAROTROPIC_EOS&&tcool>0.0) Cooling(pmb,time,dt,prim,bcc,cons);
  return;
}


void StarSourceTerms(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons) {
  Real src[NHYDRO];
  Coordinates *pco = pmb->pcoord;
  Real f_x1=0.0, f_x2=0.0, f_x3=0.0, rc, rm;
  Real rad,phi,z,v1,v2,v3;
  Real fcoe,fsmooth;
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
          rc = sqrt(pco->x1v(i)*pco->x1v(i)+pco->x2v(j)*pco->x2v(j)+
               pco->x3v(k)*pco->x3v(k));
          fcoe=GravPot_coe(rc);
          f_x1 = fcoe*pco->x1v(i);
          f_x2 = fcoe*pco->x2v(j);
          f_x3 = fcoe*pco->x3v(k);
        } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
          f_x1 = fcoe*pco->x1v(i);
        }
        Real rcv2 = (rc-smoothin)*(rc-smoothin);
        fsmooth = rcv2/(rcv2+smoothtr*smoothtr);
        if (rc<smoothin) fsmooth=0.;

        src[IM1] = dt*prim(IDN,k,j,i)*f_x1*fsmooth;
        src[IM2] = dt*prim(IDN,k,j,i)*f_x2*fsmooth;
        src[IM3] = dt*prim(IDN,k,j,i)*f_x3*fsmooth;

        cons(IM1,k,j,i) += src[IM1];
        cons(IM2,k,j,i) += src[IM2];
        cons(IM3,k,j,i) += src[IM3];
        if(NON_BAROTROPIC_EOS) {
          cons(IEN,k,j,i) += src[IM1]*prim(IM1,k,j,i) +
                             src[IM2]*prim(IM2,k,j,i) + src[IM3]*prim(IM3,k,j,i);
        }

        if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
          GetCylCoord(rad,phi,z,pco->x1v(i),pco->x2v(j),pco->x3v(k));
          Real rsph=sqrt(pco->x1v(i)*pco->x1v(i)+pco->x2v(j)*pco->x2v(j)+
                        pco->x3v(k)*pco->x3v(k));
          if (rsph<rfix) {
            cons(IDN,k,j,i) = DenProfileCyl(pmb,rad,phi,z,k,j,i);
            VelProfileCyl(pmb,rad,phi,z,i,j,k,v1,v2,v3);
            cons(IM1,k,j,i) = v1*cons(IDN,k,j,i);
            cons(IM2,k,j,i) = v2*cons(IDN,k,j,i);
            cons(IM3,k,j,i) = v3*cons(IDN,k,j,i);
            if (NON_BAROTROPIC_EOS) {
              cons(IEN,k,j,i) = PoverR(rad, phi, z)*cons(IDN,k,j,i)/(gamma_gas - 1.0)+
                                0.5*(SQR(v1)+SQR(v2)+SQR(v3))*cons(IDN,k,j,i);
              if (MAGNETIC_FIELDS_ENABLED) {
                cons(IEN,k,j,i) = cons(IEN,k,j,i)+0.5*(SQR(bcc(IB1,k,j,i))+
                                  SQR(bcc(IB2,k,j,i))+SQR(bcc(IB3,k,j,i)));
              }
            }
          }
        }
        cons(IDN,k,j,i) = std::max(cons(IDN,k,j,i),rho_floor(rc));
      }
    }
  }
}

// stellar force coefficient Fstar=coe*vec(r)

Real GravPot_coe(Real rc) {
  return(-gm0/rc/rc/rc);
}

// Set planetary source term
void PlanetarySourceTerms(MeshBlock *pmb, const Real time, const Real dt,
  const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons) {
  Real cosx3, sinx3, x3, cosx2, sinx2, x2;
  Real xcar, ycar, zcar, coeind, coeba;
  Coordinates *pco = pmb->pcoord;
  AthenaArray<Real> vol;
  vol.NewAthenaArray((pmb->ie-pmb->is)+1+2*(NGHOST));

  psys->orbit(time);

  Real src[NHYDRO];
  for (int ip=0; ip< psys->np; ++ip) {
    Real xpp=psys->xp[ip];
    Real ypp=psys->yp[ip];
    Real zpp=psys->zp[ip];
    Real mp=psys->mass[ip];
    Real rpp=sqrt(xpp*xpp+ypp*ypp+zpp*zpp);
    Real thepp=acos(zpp/rpp);
    Real phipp=atan2(ypp,xpp);
    Real rsoft2=psys->rsoft2;
    Real f_xpp=0., f_ypp=0., f_zpp=0.;
    Real dens = 0.;
    coeind = force_car_ind_coe(xpp,ypp,zpp,mp);
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      x3=pco->x3v(k);
      if(std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
        cosx3=cos(x3);
        sinx3=sin(x3);
      }
      for (int j=pmb->js; j<=pmb->je; ++j) {
        pco->CellVolume(k,j,pmb->is,pmb->ie,vol);
        x2=pco->x2v(j);
        if(std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0 ||
           std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
          cosx2=cos(x2);
          sinx2=sin(x2);
        }
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          if(std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
            xcar = pco->x1v(i)*sinx2*cosx3;
            ycar = pco->x1v(i)*sinx2*sinx3;
            zcar = pco->x1v(i)*cosx2;
          }
          if(std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
            xcar = pco->x1v(i)*cosx2;
            ycar = pco->x1v(i)*sinx2;
            zcar = x3;
          }
          if(std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
            xcar = pco->x1v(i);
            ycar = x2;
            zcar = x3;
          }
          coeba = force_car_btoa_coe(xcar, ycar, zcar,xpp,ypp,zpp,mp,rsoft2);
          Real f_x1 = 0.0;
          Real f_x2 = 0.0;
          Real f_x3 = 0.0;
          Real f_xca = coeba*(xcar-xpp);
          Real f_yca = coeba*(ycar-ypp);
          Real f_zca = coeba*(zcar-zpp);

          // Masking the Hill sphere around the planet
          Real rpfram = sqrt((xcar-xpp)*(xcar-xpp) + (ycar-ypp)*(ycar-ypp) +
                         (zcar-zpp)*(zcar-zpp));
          Real distst = sqrt(xcar*xcar+ycar*ycar+zcar*zcar);
          Real rH = rpp*pow(mp/3./gm0,1./3.);

          dens = pmb->phydro->u(IDN,k,j,i);

          if (rpfram < rH || distst< 2.5*rs) {
            dens = 0.;
          }
          // end masking

          f_xpp += f_xca*vol(i)*dens;
          f_ypp += f_yca*vol(i)*dens;
          f_zpp += f_zca*vol(i)*dens;
          if(psys->ind!=0) {
            f_xca += coeind*xpp;
            f_yca += coeind*ypp;
            f_zca += coeind*zpp;
          }
          if(std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
            f_x1 += f_xca*sinx2*cosx3+f_yca*sinx2*sinx3+f_zca*cosx2;
            f_x2 += f_xca*cosx2*cosx3+f_yca*cosx2*sinx3-f_zca*sinx2;
            f_x3 += f_xca*(-sinx3) + f_yca*cosx3;
          }
          if(std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
            f_x1 += f_xca*cosx2 + f_yca*sinx2;
            f_x2 += -f_xca*sinx2 + f_yca*cosx2;
            f_x3 += f_zca;
          }
          if(std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
            f_x1 += f_xca;
            f_x2 += f_yca;
            f_x3 += f_zca;
          }

          src[IM1] = dt*prim(IDN,k,j,i)*f_x1;
          src[IM2] = dt*prim(IDN,k,j,i)*f_x2;
          src[IM3] = dt*prim(IDN,k,j,i)*f_x3;

          cons(IM1,k,j,i) += src[IM1];
          cons(IM2,k,j,i) += src[IM2];
          cons(IM3,k,j,i) += src[IM3];
          if(NON_BAROTROPIC_EOS) {
            src[IEN] = src[IM1]*prim(IM1,k,j,i)+ src[IM2]*prim(IM2,k,j,i)
                        + src[IM3]*prim(IM3,k,j,i);
            cons(IEN,k,j,i) += src[IEN];
          }
        } // i
      } // j
    } // k
    Real f_r=f_xpp*sin(thepp)*cos(phipp) + f_ypp*sin(thepp)*sin(phipp) + f_zpp*cos(thepp);
    Real f_t=f_xpp*cos(thepp)*cos(phipp) + f_ypp*cos(thepp)*sin(phipp) - f_zpp*sin(thepp);
    Real f_p=f_xpp*(-sin(phipp)) + f_ypp*cos(phipp);

    pmb->ruser_meshblock_data[ip](0)= f_r;
    pmb->ruser_meshblock_data[ip](1)= f_t;
    pmb->ruser_meshblock_data[ip](2)= f_p;
    pmb->ruser_meshblock_data[ip](3)= f_xpp;
    pmb->ruser_meshblock_data[ip](4)= f_ypp;
    pmb->ruser_meshblock_data[ip](5)= f_zpp;
    pmb->ruser_meshblock_data[ip](6)= f_p*rpp;
    pmb->ruser_meshblock_data[ip](7)= xpp;
    pmb->ruser_meshblock_data[ip](8)= ypp;
    pmb->ruser_meshblock_data[ip](9)= zpp;
    pmb->ruser_meshblock_data[ip](10)= rpp;
    pmb->ruser_meshblock_data[ip](11)= thepp;
    pmb->ruser_meshblock_data[ip](12)= phipp;
    pmb->ruser_meshblock_data[ip](13)= mp;
  } // ip
}

Real PlanetForce(MeshBlock *pmb, int iout) {
  Real x1, x2, x3, cosx2, sinx2, cosx3, sinx3, xcar, ycar, zcar;
  if (psys->np > 0) {
    int ip= iout/14; // integer for the planet
    int re=iout%14;
    return pmb->ruser_meshblock_data[ip](re);
  }
  return 0;
}

//----------------------------------------------------------------------------------------
//!\f: calculate forces coefficient from b to a (Fbtoa=coe*vec(xca-xcb))
//

Real force_car_btoa_coe(const Real xca, const Real yca, const Real zca,
     const Real xcb, const Real ycb, const Real zcb, const Real gb, const Real rsoft2) {
  Real dist=sqrt((xca-xcb)*(xca-xcb) + (yca-ycb)*(yca-ycb) + (zca-zcb)*(zca-zcb));
  Real rsoft=sqrt(rsoft2);
  Real coe;
  if (dist>=rsoft) {
     coe=-gb/dist/dist/dist;
  } else {
     coe=gb/rsoft/rsoft/rsoft*(3*dist/rsoft-4);
  }
  return(coe);
}

//----------------------------------------------------------------------------------------
//!\f: calculate indirect forces coefficient due to gmp (Find=coe*vec(xpp))
//

Real force_car_ind_coe(const Real xpp, const Real ypp, const Real zpp, const Real gmp) {
  Real pdist=sqrt(xpp*xpp+ypp*ypp+zpp*zpp);
  Real coe = -gmp/pdist/pdist/pdist;
  return(coe);
}

//----------------------------------------------------------------------------------------
//!\f: Fix planetary orbit
//
void PlanetarySystem::orbit(double time) {
  int i;
  for (i=0; i<np; ++i) {
    if (time<insert_time*2.*PI) {
      mass[i]=massset[i]*sin(time/insert_time/4.)*sin(time/insert_time/4.);
    } else {
      mass[i]=massset[i];
    }
    double dis=sqrt(xp[i]*xp[i]+yp[i]*yp[i]);
    double ome=(sqrt((gm0+mass[i])/dis/dis/dis)-Omega0);
    double ang=acos(xp[i]/dis);
    ang = ome*time;
    xp[i]=dis*cos(ang);
    yp[i]=dis*sin(ang);
  }
  return;
}


// Set cooling

void Cooling(MeshBlock *pmb, const Real time, const Real dt,const AthenaArray<Real> &prim,
             const AthenaArray<Real> &bcc, AthenaArray<Real> &cons) {
  Real rad, phi, z;
  if (tcool>0.0) {
    Coordinates *pco = pmb->pcoord;
    for(int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          Real eint = cons(IEN,k,j,i)-0.5*(SQR(cons(IM1,k,j,i))+SQR(cons(IM2,k,j,i))
                                          +SQR(cons(IM3,k,j,i)))/cons(IDN,k,j,i);
          if (MAGNETIC_FIELDS_ENABLED) {
            eint = eint-0.5*(SQR(bcc(IB1,k,j,i))+SQR(bcc(IB2,k,j,i))+SQR(bcc(IB3,k,j,i)));
          }
          GetCylCoord(rad,phi,z,pco->x1v(i),pco->x2v(j),pco->x3v(k));
          Real p_over_r = PoverR(rad,phi,z);
          Real dtr = std::max(tcool*2.*PI/sqrt(gm0/rad/rad/rad),dt);
          Real dfrac=dt/dtr;
          Real dE=eint-p_over_r/(gamma_gas-1.0)*cons(IDN,k,j,i);
          cons(IEN,k,j,i) -= dE*dfrac;
        }
      }
    }
  }
}

//======================================================================================
//! \fn void DustParticles::UserSourceTerms(Real t, Real dt,
//                              const AthenaArray<Real>& meshsrc)
//  \brief Adds source terms to the particles.
//======================================================================================

// void DustParticles::UserSourceTerms(Real t,Real dt,const AthenaArray<Real>& meshsrc) {
// Nothing to do for tracer particles.
//;  if (taus == 0.0) return;

  // Add central gravity from the star.
//;  const Real a0 = dt ;
//;  for (int k = 0; k < npar; ++k) {
//;    Real a = a0 * std::pow(std::pow(xp0(k), 2) +
//;                           std::pow(yp0(k), 2) +
//;                           std::pow(zp0(k), 2), -1.5);
//;    vpx(k) -= a * xp0(k);
//;    vpy(k) -= a * yp0(k);
//;    vpz(k) -= a * zp0(k);
//;  }
//;}


