//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================

#include <assert.h>
#include <unistd.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include "compact_object_tracker.hpp"

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "utils/lagrange_interpolator.hpp"
#include "coordinates/adm.hpp"
#include "mhd/mhd.hpp"
#include "z4c/z4c.hpp"

//----------------------------------------------------------------------------------------
CompactObjectTracker::CompactObjectTracker(Mesh *pmesh, ParameterInput *pin, int n):
              owns_compact_object{false}, pos{NAN, NAN, NAN}, vel{NAN, NAN, NAN},
              pmesh{pmesh}, out_every{1} {
  std::string nstr = std::to_string(n);
  std::string ofname = pin->GetString("job", "basename") + ".";
  ofname += pin->GetOrAddString("z4c", "filename", "co_");
  ofname += nstr + ".txt";

  std::string cotype = pin->GetString("z4c", "co_" + nstr + "_type");
  if (cotype == "BH" || cotype == "BlackHole") {
    type = BlackHole;
  } else if (cotype == "NS" || cotype == "NeutronStar") {
    type = NeutronStar;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line "
              << __LINE__ << std::endl;
    std::cout << "Unknown compact object type: " << cotype << std::endl;
    std::exit(EXIT_FAILURE);
  }

  pos[0] = pin->GetOrAddReal("z4c", "co_" + nstr + "_x", 0.0);
  pos[1] = pin->GetOrAddReal("z4c", "co_" + nstr + "_y", 0.0);
  pos[2] = pin->GetOrAddReal("z4c", "co_" + nstr + "_z", 0.0);

  reflevel = pin->GetOrAddInteger("z4c", "co_" + nstr + "_reflevel", -1);
  radius = pin->GetOrAddReal("z4c", "co_" + nstr + "_radius", 0.0);

  out_every = pin->GetOrAddInteger("z4c", "co_" + nstr + "_out_every", 1);

  if (0 == global_variable::my_rank) {
    ofile.open(ofname.c_str());

    if (type == BlackHole) {
      ofile << "# Black Hole";
    } else {
      ofile << "# Neutron Star";
    }
    ofile << std::endl;

    ofile << "# 1:iter 2:time 3:x 4:y 5:z 6:vx 7:vy 8:vz\n";
    ofile << std::flush;
    ofile << std::setprecision(19);
  }
}

//----------------------------------------------------------------------------------------
CompactObjectTracker::~CompactObjectTracker() { }

//----------------------------------------------------------------------------------------
void CompactObjectTracker::InterpolateVelocity(MeshBlockPack *pmbp) {
  auto &padm = pmbp->padm;
  auto &pmhd = pmbp->pmhd;
  auto &pz4c = pmbp->pz4c;
  auto *S    = new LagrangeInterpolator(pmbp, pos);

  if (S->point_exist) {
    owns_compact_object = true;

    Real betax = S->Interpolate(pz4c->u0, pz4c->I_Z4C_BETAX);
    Real betay = S->Interpolate(pz4c->u0, pz4c->I_Z4C_BETAY);
    Real betaz = S->Interpolate(pz4c->u0, pz4c->I_Z4C_BETAZ);
    vel[0] = - betax;
    vel[1] = - betay;
    vel[2] = - betaz;
    if (type == NeutronStar) {
      Real alp = S->Interpolate(pz4c->u0, pz4c->I_Z4C_ALPHA);

      Real zx = S->Interpolate(pmhd->w0, IVX);
      Real zy = S->Interpolate(pmhd->w0, IVY);
      Real zz = S->Interpolate(pmhd->w0, IVZ);

      Real gxx = S->Interpolate(padm->u_adm, padm->I_ADM_GXX);
      Real gxy = S->Interpolate(padm->u_adm, padm->I_ADM_GXY);
      Real gxz = S->Interpolate(padm->u_adm, padm->I_ADM_GXZ);
      Real gyy = S->Interpolate(padm->u_adm, padm->I_ADM_GYY);
      Real gyz = S->Interpolate(padm->u_adm, padm->I_ADM_GYZ);
      Real gzz = S->Interpolate(padm->u_adm, padm->I_ADM_GZZ);

      Real z_x = gxx*zx + gxy*zy + gxz*zz;
      Real z_y = gxy*zx + gyy*zy + gyz*zz;
      Real z_z = gxz*zx + gyz*zy + gzz*zz;
      Real W = std::sqrt(z_x*zx + z_y*zy + z_z*zz + 1);

      vel[0] += alp*zx/W;
      vel[1] += alp*zy/W;
      vel[2] += alp*zz/W;
    }
  } else {
    owns_compact_object = false;
  }

  delete S;
}

//----------------------------------------------------------------------------------------
void CompactObjectTracker::EvolveTracker() {
  if (owns_compact_object) {
    for (int a = 0; a < NDIM; ++a) {
      pos[a] += pmesh->dt * vel[a];
    }
#if !(MPI_PARALLEL_ENABLED)
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl;
    std::cout << "couldn't find the compact object!" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#else
  }
  Real buf[2 * NDIM + 1] = {0., 0., 0., 0., 0., 0., 0.};
  if (owns_compact_object) {
    buf[0] = pos[0];
    buf[1] = pos[1];
    buf[2] = pos[2];
    buf[3] = vel[0];
    buf[4] = vel[1];
    buf[5] = vel[2];
    buf[6] = 1.0;
  }
  MPI_Allreduce(
    MPI_IN_PLACE, buf, 2 * NDIM + 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  if (buf[6] < 0.5) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl;
    std::cout << "The compact object has left the grid" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  pos[0] = buf[0] / buf[6];
  pos[1] = buf[1] / buf[6];
  pos[2] = buf[2] / buf[6];
  vel[0] = buf[3] / buf[6];
  vel[1] = buf[4] / buf[6];
  vel[2] = buf[5] / buf[6];
#endif // MPI_PARALLEL_ENABLED

  // After the compact object has moved it might have changed ownership
  owns_compact_object = false;
}

//----------------------------------------------------------------------------------------
void CompactObjectTracker::WriteTracker() {
  if (0 == global_variable::my_rank && 0 == pmesh->ncycle % out_every) {
    ofile << pmesh->ncycle << " "
          << pmesh->time << " "
          << pos[0] << " "
          << pos[1] << " "
          << pos[2] << " "
          << vel[0] << " "
          << vel[1] << " "
          << vel[2] << std::endl << std::flush;
  }
}
