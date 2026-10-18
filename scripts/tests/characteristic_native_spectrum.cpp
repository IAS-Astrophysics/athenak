// Export the native seven-variable primitive symbol, without changing the solver.
// The Python companion independently evaluates the covariant RMHD dispersion law.
#include <sstream>
#include <iostream>
#include <iomanip>
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "dyn_grmhd/characteristic_reservoir.hpp"
using NativeEOS=PrimitiveSolverHydro<Primitive::IdealGas,Primitive::ResetFloor>;
void vector(const Real *v,int n) {
  std::cout<<'[';for(int i=0;i<n;++i){if(i)std::cout<<',';std::cout<<v[i];}std::cout<<']';
}
void matrix(const Real a[7][7]) {
  std::cout<<'[';for(int i=0;i<7;++i){if(i)std::cout<<',';vector(a[i],7);}std::cout<<']';
}
template<int ivx> void Sample(const NativeEOS &eos,int metric,int field,int flow,
                              Real alpha,Real shift) {
  constexpr int d=ivx-IVX,t=(d+1)%3,s=(d+2)%3;
  Real g[6]={1,0,0,1,0,1};
  if(metric){g[S11]=1.4;g[S22]=.8;g[S33]=1.1;}
  if(metric==2){g[S12]=.15;g[S13]=.05;g[S23]=-.04;}
  Real p=.002414486921529175,B=sqrt(2*p),bn=0;
  Real q[7]={1,p,0,0,0,0,0},beta[3]={};beta[d]=shift;
  if(field==0)bn=B;                      // parallel/Type II
  if(field==1)q[5]=B;                    // perpendicular/Type I
  if(field==2){bn=.7*B;q[5]=.4*B;q[6]=.6*B;}
  if(field==3){bn=1e-8*B;q[5]=B;}        // near Type I
  if(field==4){}                        // B=0, hydrodynamic degeneracy
  if(field==5){bn=10*B;q[5]=20*B;q[6]=7*B;}
  if(field==6)q[6]=B;
  if(flow==1){q[2]=.3;q[3]=-.2;q[4]=.1;}
  if(flow==2){q[2]=2;q[3]=.5;q[4]=-.7;}
  Real scale[7]={1,p,sqrt(p),sqrt(p),sqrt(p),sqrt(p),sqrt(p)};
  Real du[7][7],df[7][7],inv[7][7],a[7][7],sgn[7][7];
  for(int j=0;j<7;++j){
    Real plus[7],minus[7];for(int k=0;k<7;++k)plus[k]=minus[k]=q[k];
    plus[j]+=1e-5*scale[j];minus[j]-=1e-5*scale[j];
    Real up[7],um[7],fp[7],fm[7];
    dyngr::char_reservoir::Evaluate<ivx>(eos,plus,bn,g,beta,alpha,up,fp);
    dyngr::char_reservoir::Evaluate<ivx>(eos,minus,bn,g,beta,alpha,um,fm);
    for(int k=0;k<7;++k){du[k][j]=(up[k]-um[k])/2e-5;df[k][j]=(fp[k]-fm[k])/2e-5;}
  }
  for(int k=0;k<7;++k){Real n=0;for(int j=0;j<7;++j)n=fmax(n,fabs(du[k][j]));
    for(int j=0;j<7;++j){du[k][j]/=n;df[k][j]/=n;}}
  if(!dyngr::char_reservoir::Inverse(du,inv))throw std::runtime_error("Singular primitive/conservative map");
  for(int i=0;i<7;++i)for(int j=0;j<7;++j){a[i][j]=0;
    for(int k=0;k<7;++k)a[i][j]+=inv[i][k]*df[k][j];sgn[i][j]=a[i][j];}
  bool ok=dyngr::char_reservoir::Sign(sgn);
  std::cout<<"{\"axis\":"<<d<<",\"metric\":"<<metric<<",\"field\":"<<field<<",\"flow\":"<<flow
    <<",\"alpha\":"<<alpha<<",\"beta\":";vector(beta,3);
  std::cout<<",\"g\":";vector(g,6);std::cout<<",\"q\":";vector(q,7);
  std::cout<<",\"bn\":"<<bn<<",\"scale\":";vector(scale,7);
  std::cout<<",\"A\":";matrix(a);std::cout<<",\"sign_ok\":"<<(ok?"true":"false")<<",\"sign\":";matrix(sgn);
  std::cout<<"}\n";
}
template<int ivx> void Axis(const NativeEOS &eos) {
  for(int metric=0;metric<3;++metric)for(int field=0;field<6;++field)
    for(int flow=0;flow<3;++flow)for(Real alpha:{.7,1.3})for(Real shift:{-16.,-.2,0.,.2,16.})
      Sample<ivx>(eos,metric,field,flow,alpha,shift);
}
int main(int argc,char **argv) {
  Kokkos::initialize(argc,argv);int status=0;
  try {
    ParameterInput pin;std::stringstream input;
    input<<"<mhd>\ngamma=1.6666666666666667\ngamma_max=10\n";pin.LoadFromStream(input);
    NativeEOS eos("mhd",nullptr,&pin);std::cout<<std::setprecision(17);
    if(argc>1 && std::string(argv[1])=="--reservoir") {
      Sample<IVX>(eos,0,6,0,1,0);Sample<IVY>(eos,0,1,0,1,0);
      Sample<IVZ>(eos,0,0,0,1,0);
    } else {Axis<IVX>(eos);Axis<IVY>(eos);Axis<IVZ>(eos);}
  }catch(const std::exception &e){std::cerr<<e.what()<<std::endl;status=1;}
  Kokkos::finalize();return status;
}
