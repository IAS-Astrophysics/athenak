#include <algorithm> // for fill
#include <cmath>     // for NAN
#include <stdexcept>
#include <sstream>
#include <unistd.h> // for F_OK
#include <fstream> 

#define H5_USE_16_API (1)
#include <hdf5.h>


#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#include "cce.hpp"
#include "matrix.hpp"
#include "sYlm.hpp"
#include "myassert.hpp"
#include "decomp.hpp"
#include "../../parameter_input.hpp"
#include "../../mesh/mesh.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../../utils/lagrange_interp.hpp"
#include "../../globals.hpp"
#include "../z4c.hpp"

#define BUFFSIZE  (1024)
#define MAX_RADII (100)
#define ABS(x_) ((x_)>0 ? (x_) : (-(x_)))

#define HDF5_CALL(fn_call)                                           \
{                                                                     \
  /* ex: error_code = group_id = H5Gcreate(file_id, metaname, 0) */   \
  hid_t _error_code = fn_call;                                        \
  if (_error_code < 0)                                                \
  {                                                                   \
    cerr << "File: " << __FILE__ << "\n"                              \
         << "line: " << __LINE__ << "\n"                              \
         << "HDF5 call " << #fn_call << ", "                          \
         << "returned error code: " << (int)_error_code << ".\n";     \
         exit((int)_error_code);                                      \
  }                                                                   \
}

using namespace decomp_matrix_class;
using namespace decomp_sYlm;
using namespace decomp_decompose;


static int output_3Dmodes(const int iter,
       const char *dir,
       const char* name,
       const int obs, Real time,
       int s, int nl,
       int nn, Real rin, Real rout,
       const Real *re, const Real *im);


CCE::CCE(Mesh *const pm, ParameterInput *const pin, std::string name, int rn):
    pm(pm),
    pin(pin),
    fieldname(name),
    spin(0),
    dinfo_pp(nullptr),
    rn(rn),
    count_interp_pnts(0)
{
  bitant     = pin->GetOrAddBoolean("z4c", "bitant", false);
  output_dir = pin->GetString("cce","output_dir");
  rin  = pin->GetReal("cce", "rin_"  + std::to_string(rn));
  rout = pin->GetReal("cce", "rout_" + std::to_string(rn));
  num_mu_points  = pin->GetOrAddInteger("cce","num_theta",41);
  num_phi_points = pin->GetOrAddInteger("cce","num_phi",82);
  num_x_points   = pin->GetOrAddInteger("cce","num_r_inshell",28);
  num_l_modes    = pin->GetOrAddInteger("cce","num_l_modes",7);
  num_n_modes    = pin->GetOrAddInteger("cce","num_radial_modes",7);
  nangle = num_mu_points*num_phi_points;
  npoint = nangle*num_x_points;

  Real *radius    = nullptr;
  Real *mucolloc  = nullptr;
  Real *phicolloc = nullptr;
  myassert (ABS(spin) <= MAX_SPIN);

  nlmmodes = num_l_modes*(num_l_modes+2*ABS(MAX_SPIN));
  dinfo_pp = new const decomp_info* [2*MAX_SPIN+1];
  myassert(dinfo_pp);
  for (int s=-MAX_SPIN; s<=MAX_SPIN; s++)
  {
    dinfo_pp[s+MAX_SPIN] = nullptr;
  }

  // alloc
  radius = new Real [num_x_points];
  xb = new Real [nangle*num_x_points];
  yb = new Real [nangle*num_x_points];
  zb = new Real [nangle*num_x_points];
  mucolloc = new Real [nangle];
  phicolloc = new Real [nangle];
  ifield = new Real [nangle*num_x_points]();// init to 0.
  re_f   = new Real [nangle*num_x_points]();// init to 0.
 
  myassert(radius);
  myassert(xb);
  myassert(yb);
  myassert(zb);
  myassert(mucolloc);
  myassert(phicolloc);
  myassert(ifield);
  myassert(re_f);

  std::fill(radius, radius + (num_x_points),NAN); // init to nan
  std::fill(xb, xb + (nangle*num_x_points),NAN); // init to nan
  std::fill(yb, yb + (nangle*num_x_points),NAN); // init to nan
  std::fill(zb, zb + (nangle*num_x_points),NAN); // init to nan
  std::fill(mucolloc,  mucolloc  + (nangle),NAN); // init to nan
  std::fill(phicolloc, phicolloc + (nangle),NAN); // init to nan

  if (! dinfo_pp[MAX_SPIN + spin])
  {
    dinfo_pp[MAX_SPIN + spin] =  initialize(spin, num_l_modes, num_n_modes,
                                   num_mu_points, num_phi_points, num_x_points);
    myassert(dinfo_pp[MAX_SPIN + spin]);
  }

  dinfo_pp[MAX_SPIN + spin]->get_ncolloc(num_x_points, radius);
  dinfo_pp[MAX_SPIN + spin]->get_mucolloc(nangle, mucolloc);
  dinfo_pp[MAX_SPIN + spin]->get_phicolloc(nangle, phicolloc);

  for (int k=0; k < num_x_points; k++)
  {
    Real xk = radius[k];
    radius[k] = 0.5 * ( (rout - rin) * xk + (rout + rin) );
  }

  for (int k=0; k < num_x_points; k++)
  {
    for (int i=0; i < nangle; i++)
    {
      const Real phi = phicolloc[i];
      const Real mu = mucolloc[i];

      const Real sph = sin(phi);
      const Real cph = cos(phi);
      const Real cth = mu;
      const Real sth = sqrt(1.0 - mu*mu);

      const int indx = i + k*nangle;
      
      xb[indx] = radius[k] * sth*cph;
      yb[indx] = radius[k] * sth*sph;
      zb[indx] = radius[k] * cth;
    
    }
  }
    
  // free mem.
  delete [] mucolloc;
  delete [] phicolloc;
  delete [] radius;
}

CCE::~CCE()
{
  delete [] ifield;
  delete [] re_f;
  delete [] xb;
  delete [] yb;
  delete [] zb;

  for (int i = 0; i < 2*MAX_SPIN+1; ++i)
  {
    if (dinfo_pp[i]) 
      delete dinfo_pp[i];
  }
  delete [] dinfo_pp;
}

// given a Cartesian point, interpolate the pertinent field for that point
void CCE::Interpolate(MeshBlockPack *pmbp)
{
  const int Npoints = nangle*num_x_points;
  auto &pz4c = pmbp->pz4c;
  auto &u0   = pz4c->u0;
  Real bitant_sign  = 0.; // set according to the field and symmetry
  int isrc_field = -1; // index of field to interpolate

  // find the src field
  if (fieldname == "gxx")
  {
    isrc_field = pz4c->I_Z4C_GXX;
    bitant_sign = 1.;
  }
  else if (fieldname == "gxy")
  {
    isrc_field = pz4c->I_Z4C_GXY;
    bitant_sign = 1.;
  }
  else if (fieldname == "gxz")
  {
    isrc_field = pz4c->I_Z4C_GXZ;
    bitant_sign = -1.;
  }
  else if (fieldname == "gyy")
  {
    isrc_field = pz4c->I_Z4C_GYY;
    bitant_sign = 1.;
  }
  else if (fieldname == "gyz")
  {
    isrc_field =  pz4c->I_Z4C_GYZ;
    bitant_sign = -1.;
  }
  else if (fieldname == "gzz")
  {
    isrc_field = pz4c->I_Z4C_GZZ;
    bitant_sign = 1.;
  }
  else if (fieldname == "betax")
  {
    isrc_field = pz4c->I_Z4C_BETAX;
    bitant_sign = 1.;
  }
  else if (fieldname == "betay")
  {
    isrc_field = pz4c->I_Z4C_BETAY;
    bitant_sign = 1.;
  }
  else if (fieldname == "betaz")
  {
    isrc_field = pz4c->I_Z4C_BETAZ;
    bitant_sign = -1.;

  }
  else if (fieldname == "alp")
  {
    isrc_field = pz4c->I_Z4C_ALPHA;
    bitant_sign = 1.;
  }
  else
  {
    std::stringstream msg;
    msg << "### FATAL ERROR in CCE interpolation" << std::endl;
    msg << "Could not find '" << fieldname << "' for interpolation!";
    throw std::runtime_error(msg.str().c_str());
  }
  
  for (int p = 0; p < Npoints; ++p)
  {
    bool IsBitant = (bitant && zb[p] < 0.);
    Real isign = IsBitant ? bitant_sign: 1.;
    Real zsign = IsBitant ? -1.        : 1.;
    Real coord[3] = {xb[p], yb[p], zsign*zb[p]};
    
    auto *intrp = new LagrangeInterpolator(pmbp, coord);
    
    if (intrp->point_exist)
    {
      ifield[p] = isign*intrp(u0,isrc_field);

#pragma omp atomic update
      count_interp_pnts++;
      // printf("f(%g,%g,%g) = %g\n",coord[0],coord[1],coord[2],ifield[p]);
    }
  }
}

// reduce different parts of the interpolation array into one array
void CCE::ReduceInterpolation()
{
  const int Npoints = nangle*num_x_points;
  int counter = 0;
  std::fill(re_f, re_f + Npoints,0.0); // init to 0.
  
# ifdef MPI_PARALLEL
  // we assumed each point interpolated once and only once so we use MPI_SUM
  MPI_Reduce(ifield, re_f, Npoints, MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Allreduce(&count_interp_pnts, &counter, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  
  // ensure all points have been found
  assert(counter == Npoints);
  
# else // no MPI

  // ensure all points have been found
  assert(count_interp_pnts == Npoints);
  
  for (int i = 0; i < Npoints; ++i)
    re_f[i] = ifield[i];

# endif  
  
  std::fill(ifield, ifield + Npoints,0.0); // init to 0.
  count_interp_pnts = 0;// reset for the next time check
}

// decompose the field and write into an h5 file
void CCE::DecomposeAndWrite(int iter/* number of times writes into an h5 file */, Real curr_time)
{
  if (0 != Globals::my_rank) return;
  
  // create workspace
  Real *re_m = new Real [nlmmodes*num_x_points];
  Real *im_m = new Real [nlmmodes*num_x_points];
  Real *im_f = new Real [nangle*num_x_points](); // init to zero
  myassert(re_m);
  myassert(im_m);
  myassert(im_f);
  std::fill(re_m, re_m + (nlmmodes*num_x_points),NAN); // init to nan
  std::fill(im_m, im_m + (nlmmodes*num_x_points),NAN); // init to nan
  
  // decompose the re_f, note im_f is zero 
  decompose3D(dinfo_pp[MAX_SPIN + spin], re_f, im_f, re_m, im_m);

  // dump the modes into an h5 file
  output_3Dmodes(iter, output_dir.c_str(), fieldname.c_str(), rn, curr_time,
     spin, num_l_modes, num_n_modes, rin, rout, re_m, im_m);

  // free workspace
  delete [] re_m;
  delete [] im_m;
  delete [] im_f;
}

// write the decomposed field in an h5 file.
static int output_3Dmodes(const int iter/* output iteration */, const char *dir,
  const char* name, const int obs, Real time,
  int s, int nl,
  int nn, Real rin, Real rout,
  const Real *re, const Real *im)
{
  char filename[BUFFSIZE];
  hid_t   file_id;
  hsize_t dims[2];
  herr_t  status;

  snprintf(filename, sizeof filename, "%s/cce_decomp_shell_%d.h5", dir, obs);

  const int nlmmodes = nl*(nl+2*ABS(s));
  dims[0] = nn;
  dims[1] = nlmmodes;

  static int FirstCall = 1;
  static int last_dump[MAX_RADII];

  const int dump_it = iter;
  hid_t dataset_id, attribute_id, dataspace_id, group_id;

  if (FirstCall)
  {
    FirstCall = 0;
    for (unsigned int i=0; i < sizeof(last_dump) / sizeof(*last_dump); i++)
    {
      last_dump[i] = -1000;
    }
  }

  bool file_exists = false;
  H5E_BEGIN_TRY {
     file_exists = H5Fis_hdf5(filename) > 0;
  } H5E_END_TRY;

  if(file_exists)
  {
    file_id = H5Fopen (filename, H5F_ACC_RDWR, H5P_DEFAULT);
    if (file_id < 0)
    {
      cerr << "Failed to open hdf5 file";
      exit((int)file_id);
    }
  }
  else
  {
    file_id = H5Fcreate (filename, H5F_ACC_TRUNC,
           H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0)
    {
      cerr << "Failed to create hdf5 file";
      exit((int)file_id);
    }

    char metaname[]="/metadata";
    HDF5_CALL(group_id = H5Gcreate(file_id, metaname, 0));

    int ds[2] = {nn, nlmmodes};
    hsize_t oD2 = 2;
    hsize_t oD1 = 1;

    HDF5_CALL(dataspace_id =  H5Screate_simple(1, &oD2, NULL));
    HDF5_CALL(attribute_id = H5Acreate(group_id, "dim", H5T_NATIVE_INT,
                   dataspace_id, H5P_DEFAULT));
    HDF5_CALL(status = H5Awrite(attribute_id, H5T_NATIVE_INT, ds));
    HDF5_CALL(status = H5Aclose(attribute_id));
    HDF5_CALL(status = H5Sclose(dataspace_id));

    HDF5_CALL(dataspace_id =  H5Screate_simple(1, &oD1, NULL));
    HDF5_CALL(attribute_id = H5Acreate(group_id, "spin", H5T_NATIVE_INT,
                    dataspace_id, H5P_DEFAULT));
    HDF5_CALL(status = H5Awrite(attribute_id, H5T_NATIVE_INT, &s));
    HDF5_CALL(status = H5Aclose(attribute_id));
    HDF5_CALL(status = H5Sclose(dataspace_id));

    HDF5_CALL(dataspace_id =  H5Screate_simple(1, &oD1, NULL));
    HDF5_CALL(attribute_id = H5Acreate(group_id, "Rin", H5T_NATIVE_DOUBLE,
                    dataspace_id, H5P_DEFAULT));
    HDF5_CALL(status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &rin));
    HDF5_CALL(status = H5Aclose(attribute_id));
    HDF5_CALL(status = H5Sclose(dataspace_id));

    HDF5_CALL(dataspace_id =  H5Screate_simple(1, &oD1, NULL));
    HDF5_CALL(attribute_id = H5Acreate(group_id, "Rout", H5T_NATIVE_DOUBLE,
                    dataspace_id, H5P_DEFAULT));
    HDF5_CALL(status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &rout));
    HDF5_CALL(status = H5Aclose(attribute_id));
    HDF5_CALL(status = H5Sclose(dataspace_id));

    HDF5_CALL(H5Gclose(group_id));

  }
  
  char buff[BUFFSIZE];
  // NOTE: the dump_it should be the same for all vars that's why we need this
  if (dump_it > last_dump[obs]) {
    hsize_t oneD = 1;
    snprintf(buff, sizeof buff, "/%d", dump_it);
    if (!(H5Lexists(file_id, buff, H5P_DEFAULT ) > 0)) {
      HDF5_CALL(group_id = H5Gcreate(file_id, buff, 0));
      HDF5_CALL(dataspace_id =  H5Screate_simple(1, &oneD, NULL));
      HDF5_CALL(attribute_id = H5Acreate(group_id, "Time", H5T_NATIVE_DOUBLE,
        dataspace_id, H5P_DEFAULT));
      HDF5_CALL(status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &time));
      HDF5_CALL(status = H5Aclose(attribute_id));
      HDF5_CALL(status = H5Sclose(dataspace_id));
      HDF5_CALL(H5Gclose(group_id));
    }
  }
  last_dump[obs] = dump_it;

  snprintf(buff, sizeof buff, "/%d/%s", dump_it, name);
  if (!(H5Lexists(file_id, buff, H5P_DEFAULT ) > 0)) {
    HDF5_CALL(group_id = H5Gcreate(file_id, buff, 0));
    HDF5_CALL(H5Gclose(group_id));
  }

  snprintf(buff, sizeof buff, "/%d/%s/re", dump_it, name);
  if (!(H5Lexists(file_id, buff, H5P_DEFAULT ) > 0)) {
    HDF5_CALL(dataspace_id =  H5Screate_simple(2, dims, NULL));
    HDF5_CALL(dataset_id =  H5Dcreate(file_id, buff, H5T_NATIVE_DOUBLE,
           dataspace_id, H5P_DEFAULT));
    HDF5_CALL(status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL,
                     H5S_ALL, H5P_DEFAULT, re));
    HDF5_CALL(status = H5Dclose(dataset_id));
    HDF5_CALL(status = H5Sclose(dataspace_id));
  }

  
  snprintf(buff, sizeof buff, "/%d/%s/im", dump_it, name);
  if (!(H5Lexists(file_id, buff, H5P_DEFAULT ) > 0)) {
    HDF5_CALL(dataspace_id =  H5Screate_simple(2, dims, NULL));
    HDF5_CALL(dataset_id =  H5Dcreate(file_id, buff, H5T_NATIVE_DOUBLE,
           dataspace_id, H5P_DEFAULT));
    HDF5_CALL(status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL,
                     H5S_ALL, H5P_DEFAULT, im));
    HDF5_CALL(status = H5Dclose(dataset_id));
    HDF5_CALL(status = H5Sclose(dataspace_id));
  }

  HDF5_CALL(H5Fclose(file_id));

  return 0;
}

