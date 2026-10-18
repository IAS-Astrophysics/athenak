// Direct native-Jacobian test against independent analytic rest-frame MHD modes.
#include <sstream>
#include <iostream>
#include <algorithm>
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "dyn_grmhd/characteristic_reservoir.hpp"
#include "dyn_grmhd/characteristic_ghost_ct.hpp"
using NativeEOS=PrimitiveSolverHydro<Primitive::IdealGas,Primitive::ResetFloor>;
template<int ivx>
int Check(const NativeEOS &eos,Real &worst) {
  constexpr int d=ivx-IVX,t=(d+1)%3;
  const Real p=.002414486921529175, rho=1, gamma=5.0/3.0;
  const Real B=sqrt(2*p), H=rho+gamma*p/(gamma-1)+B*B;
  const Real cs=sqrt(gamma*p/(rho+gamma*p/(gamma-1))),ca=B/sqrt(H),eps=1e-7;
  Real g[NSPMETRIC]={1,0,0,1,0,1};int count=0;
  for(int metric=0;metric<3;++metric)for(Real alpha:{.7,1.3})for(Real shift:{-2.,-.2,0.,.2,2.})for(int sign:{-1,1})
    for(int family=0;family<3;++family)for(int prop:{-1,1}) {
      if(family==2&&prop==-1)continue;
      g[S11]=metric?1.4:1;g[S22]=metric?.8:1;g[S33]=metric?1.1:1;
      if(metric==2){g[S12]=.15;g[S13]=.05;g[S23]=-.04;}else{g[S12]=g[S13]=g[S23]=0;}
      int diag[3]={S11,S22,S33};Real tangent_scale=sqrt(g[diag[t]]),gi[6];
      Real det=adm::SpatialDet(g[0],g[1],g[2],g[3],g[4],g[5]);
      adm::SpatialInv(1/det,g[0],g[1],g[2],g[3],g[4],g[5],&gi[0],&gi[1],&gi[2],&gi[3],&gi[4],&gi[5]);
      Real inv[3][3]={{gi[0],gi[1],gi[2]},{gi[1],gi[3],gi[4]},{gi[2],gi[4],gi[5]}};
      Real normal[3];for(int j=0;j<3;++j)normal[j]=inv[j][d]/sqrt(inv[d][d]);
      Real beta[3]={};beta[d]=shift;
      Real ref[7]={rho,p,0,0,0,B*normal[t],B*normal[(d+2)%3]},inside[7],out[7],delta[7]={};
      Real speed=0;
      if(family==0){delta[0]=eps;delta[1]=gamma*p*eps;for(int j=0;j<3;++j)delta[2+j]=prop*cs*eps*normal[j];speed=prop*cs*sqrt(inv[d][d]);}
      if(family==1){delta[2+t]=eps/tangent_scale;delta[5]=-prop*sqrt(H)*eps/tangent_scale;speed=prop*ca*sqrt(inv[d][d]);}
      if(family==2)delta[0]=eps;
      for(int j=0;j<7;++j)inside[j]=ref[j]+delta[j];
      bool limited=false;Real projector[7][7];
      if(!dyngr::char_reservoir::Project<ivx>(eos,ref,inside,B*normal[d],g,beta,alpha,sign,out,limited,projector)||limited)
        throw std::runtime_error("Native characteristic projection failed");
      bool outgoing=sign*(alpha*speed-shift)>=0;
      Real scale[7]={rho,p,sqrt(p),sqrt(p),sqrt(p),sqrt(p),sqrt(p)};
      for(int j=0;j<7;++j) {
        Real projected=ref[j];for(int k=0;k<7;++k)projected+=scale[j]*projector[j][k]*(inside[k]-ref[k])/scale[k];
        if(fabs(projected-out[j])>1e-12*scale[j])throw std::runtime_error("Returned projector disagrees with native state projection");
        Real expected=ref[j]+(outgoing?delta[j]:0.0);
        Real error=fabs(out[j]-expected)/(eps*scale[j]);worst=std::max(worst,error);
        if(error>1e-3) {std::cerr<<"axis="<<d<<" family="<<family<<" alpha="<<alpha<<" shift="<<shift<<" outward="<<sign<<" propagation="<<prop<<" component="<<j<<" normalized_error="<<error<<std::endl;throw std::runtime_error("Wrong native incoming/outgoing mode");}
      }
      ++count;
    }
  return count;
}
int main(int argc,char **argv) {
  Kokkos::initialize(argc,argv);
  int status=0;
  try {
    ParameterInput pin;std::stringstream input;
    input<<"<mhd>\ngamma=1.6666666666666667\ngamma_max=10\n";pin.LoadFromStream(input);
    NativeEOS eos("mhd",nullptr,&pin);Real worst=0;
    // Independent cell-average polynomials fix the extrapolation coefficients.
    auto increment=dyngr::char_reservoir::TraceIncrement;
    if(fabs(increment(9.5,8.5,7.5,3)-.5)>1e-13 ||
       fabs(9.533333333333333+increment(9.533333333333333,8.733333333333333,8.133333333333333,3)-10)>1e-13 ||
       increment(1,1,1,3)!=0 || increment(2,1,2,3)!=0)
      throw std::runtime_error("Boundary trace fails constant/linear/quadratic/extremum check");
    auto extend=dyngr::char_ghost::Extend;
    if(fabs(extend(9.5,8.5,7.5,2,3)-11.5)>1e-13 ||
       fabs(extend(9.533333333333333,8.733333333333333,8.133333333333333,2,3)-11.733333333333333)>1e-13 ||
       extend(1,1,1,3,3)!=1 || extend(2,1,2,3,3)!=2)
      throw std::runtime_error("Ghost extension polynomial/extremum check failed");
    int n=Check<IVX>(eos,worst)+Check<IVY>(eos,worst)+Check<IVZ>(eos,worst);
    std::cout<<"{\"status\":\"passed\",\"trace_polynomial_cases\":4,\"ghost_polynomial_cases\":4,\"native_mode_cases\":"<<n<<",\"max_normalized_error\":"<<worst<<"}"<<std::endl;
  } catch(const std::exception &e){std::cerr<<e.what()<<std::endl;status=1;}
  Kokkos::finalize();return status;
}
