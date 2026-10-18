#ifndef DYN_GRMHD_CHARACTERISTIC_RESERVOIR_HPP_
#define DYN_GRMHD_CHARACTERISTIC_RESERVOIR_HPP_
// Experimental local characteristic projector. Numerical Jacobians use the SAME
// nonlinear Valencia flux/EOS as the volume solver, in the actual face metric.
#include "coordinates/coordinates.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/rsolvers/hlle_dyn_grmhd.hpp"
namespace dyngr {
namespace char_reservoir {
KOKKOS_INLINE_FUNCTION
bool Inverse(const Real a[7][7], Real inv[7][7]) {
  Real b[7][14];
  for(int i=0;i<7;++i) for(int j=0;j<14;++j)
    b[i][j]=j<7?a[i][j]:(j-7==i?1.0:0.0);
  for(int k=0;k<7;++k) {
    int pivot=k;for(int i=k+1;i<7;++i) if(fabs(b[i][k])>fabs(b[pivot][k])) pivot=i;
    if(!(fabs(b[pivot][k])>1e-25)) return false;
    for(int j=0;j<14;++j){Real t=b[k][j];b[k][j]=b[pivot][j];b[pivot][j]=t;}
    Real t=b[k][k];for(int j=0;j<14;++j)b[k][j]/=t;
    for(int i=0;i<7;++i) if(i!=k) {t=b[i][k];for(int j=0;j<14;++j)b[i][j]-=t*b[k][j];}
  }
  for(int i=0;i<7;++i)for(int j=0;j<7;++j) inv[i][j]=b[i][j+7];
  return true;
}
KOKKOS_INLINE_FUNCTION
bool Sign(Real a[7][7]) {
  // Shift stationary modes to the outgoing side. This is an explicitly small
  // glancing-mode tolerance, not a normal-velocity diode.
  Real norm=0;for(int i=0;i<7;++i)for(int j=0;j<7;++j)norm=fmax(norm,fabs(a[i][j]));
  norm=fmax(norm,1e-8);
  for(int i=0;i<7;++i)for(int j=0;j<7;++j)a[i][j]=a[i][j]/norm+(i==j?1e-8:0.0);
  for(int iter=0;iter<64;++iter) {
    Real inv[7][7];if(!Inverse(a,inv)) return false;
    Real na=0,ni=0;for(int i=0;i<7;++i)for(int j=0;j<7;++j){na+=a[i][j]*a[i][j];ni+=inv[i][j]*inv[i][j];}
    Real scale=sqrt(sqrt(ni/na));Real change=0;
    for(int i=0;i<7;++i)for(int j=0;j<7;++j){Real v=.5*(scale*a[i][j]+inv[i][j]/scale);change=fmax(change,fabs(v-a[i][j]));a[i][j]=v;}
    if(change<1e-9) {
      Real err=0;for(int i=0;i<7;++i)for(int j=0;j<7;++j){Real v=0;for(int k=0;k<7;++k)v+=a[i][k]*a[k][j];err=fmax(err,fabs(v-(i==j?1.0:0.0)));}
      return err<1e-6;
    }
  }
  return false;
}
template<int ivx,class E,class Err>
KOKKOS_INLINE_FUNCTION
void Evaluate(const PrimitiveSolverHydro<E,Err>& eos,Real q[7],Real bn,
              Real g[NSPMETRIC],Real beta[3],Real alpha,Real U[7],Real F[7]) {
  constexpr int d=ivx-IVX, t=(d+1)%3, s=(d+2)%3;
  Real p[NPRIM]={},b[NPRIM]={},c[NCONS],cr[NCONS],f[NCONS],fr[NCONS],bf[NMAG],bfr[NMAG],bs,bsr;
  p[PRH]=q[0];p[PPR]=q[1];for(int j=0;j<3;++j)p[PVX+j]=q[2+j];
  p[PTM]=eos.ps.GetEOS().GetTemperatureFromP(p[PRH],p[PPR],&p[PYF]);
  b[d]=bn;b[t]=q[5];b[s]=q[6];
  SingleStateFlux<ivx>(eos,p,p,b,b,5,0,g,beta,alpha,c,cr,f,fr,bf,bfr,bs,bsr);
  for(int j=0;j<5;++j){U[j]=c[j];F[j]=alpha*f[j];}
  U[5]=b[t];U[6]=b[s];F[5]=alpha*bf[t];F[6]=alpha*bf[s];
}
template<int ivx,class E,class Err>
KOKKOS_INLINE_FUNCTION
bool Project(const PrimitiveSolverHydro<E,Err>& eos,Real ref[7],Real inside[7],
             Real bn,Real g[NSPMETRIC],Real beta[3],Real alpha,int outward,
             Real result[7],bool &limited) {
  constexpr int d=ivx-IVX;
  constexpr int diag[3]={S11,S22,S33};
  constexpr int off[3]={S23,S13,S12};
  Real ginv=(g[diag[(d+1)%3]]*g[diag[(d+2)%3]]-g[off[d]]*g[off[d]])/Primitive::GetDeterminant(g);
  Real center=-outward*beta[d],radius=alpha*sqrt(ginv);
  limited=false;
  if(center>radius) {for(int i=0;i<7;++i)result[i]=inside[i];return true;}
  if(center<-radius) {for(int i=0;i<7;++i)result[i]=ref[i];return true;}
  // Normalize primitive AND conservative rows to avoid pressure/density scale
  // disparities in the finite differences and the small matrix inversion.
  Real vscale=sqrt(ref[1]/ref[0]);
  Real scale[7]={ref[0],ref[1],vscale,vscale,vscale,
                sqrt(ref[1]),sqrt(ref[1])};
  // A positive physical mean tracks actual inflow/outflow, including the
  // entropy/contact mode. Linearizing permanently at ambient rest would leave
  // that mode glancing even after a nonzero boundary mass flow develops.
  Real linearization[7];for(int k=0;k<7;++k)linearization[k]=.5*(ref[k]+inside[k]);
  Real du[7][7],df[7][7];
  for(int j=0;j<7;++j) {
    Real plus[7],minus[7];for(int k=0;k<7;++k)plus[k]=minus[k]=linearization[k];
    Real h=1e-5*scale[j];plus[j]+=h;minus[j]-=h;
    Real up[7],um[7],fp[7],fm[7];
    Evaluate<ivx>(eos,plus,bn,g,beta,alpha,up,fp);
    Evaluate<ivx>(eos,minus,bn,g,beta,alpha,um,fm);
    for(int k=0;k<7;++k){du[k][j]=(up[k]-um[k])/2e-5;df[k][j]=(fp[k]-fm[k])/2e-5;}
  }
  for(int k=0;k<7;++k){Real n=0;for(int j=0;j<7;++j)n=fmax(n,fabs(du[k][j]));if(!(n>0))return false;for(int j=0;j<7;++j){du[k][j]/=n;df[k][j]/=n;}}
  Real inv[7][7],a[7][7];if(!Inverse(du,inv))return false;
  for(int i=0;i<7;++i)for(int j=0;j<7;++j){a[i][j]=0;for(int k=0;k<7;++k)a[i][j]+=outward*inv[i][k]*df[k][j];}
  if(!Sign(a))return false;
  Real delta[7];for(int i=0;i<7;++i){delta[i]=0;for(int j=0;j<7;++j)delta[i]+=.5*((i==j?1.0:0.0)+a[i][j])*(inside[j]-ref[j])/scale[j];delta[i]*=scale[i];}
  Real theta=1;limited=false;
  for(int trial=0;trial<24;++trial) {
    for(int i=0;i<7;++i)result[i]=ref[i]+theta*delta[i];
    Real u[3]={result[2],result[3],result[4]};
    Real usq=Primitive::SquareVector(u,g);
    bool finite=true;for(int i=0;i<7;++i)finite=finite&&Kokkos::isfinite(result[i]);
    Real vmax=eos.ps.GetEOS().GetMaxVelocity();
    if(finite&&result[0]>0&&result[1]>0&&usq>=0&&usq/(1+usq)<vmax*vmax)return true;
    limited=true;theta*=.5;
  }
  return false;
}
} // namespace char_reservoir

// First-order face-interior trace, characteristic exterior, shared HLLE fluid
// flux and induction flux -> standard edge EMF -> standard CT. Runs before the
// FOFC predictor and after fallback replacement.
// Physical face-normal B is read, never overwritten. Tangential ghost strips are
// included to supply the existing unsplit CT corner construction.
template<int ivx,class E,class Err>
void CharacteristicFaces(MeshBlockPack *pack,const PrimitiveSolverHydro<E,Err>& eos) {
  auto cfg=pack->pcoord->coord_data;auto count=pack->pcoord->characteristic_counts;
  auto ind=pack->pmesh->mb_indcs;auto size=pack->pmb->mb_size;auto adm=pack->padm->adm;
  auto w=pack->pmhd->w0;auto bc=pack->pmhd->bcc0;auto b=pack->pmhd->b0;
  constexpr int d=ivx-IVX,t=(d+1)%3,s=(d+2)%3;
  auto flux=d==0?pack->pmhd->uflx.x1f:(d==1?pack->pmhd->uflx.x2f:pack->pmhd->uflx.x3f);
  auto ey=d==0?pack->pmhd->e3x1:(d==1?pack->pmhd->e1x2:pack->pmhd->e2x3);
  auto ez=d==0?pack->pmhd->e2x1:(d==1?pack->pmhd->e3x2:pack->pmhd->e1x3);
  int lo[3]={ind.is,ind.js,ind.ks},hi[3]={ind.ie,ind.je,ind.ke};
  par_for("characteristic_faces",DevExeSpace(),0,pack->nmb_thispack-1,
          0,1,lo[t]-1,hi[t]+1,lo[s]-1,hi[s]+1,
    KOKKOS_LAMBDA(int m,int side,int tang_t,int tang_s) {
      int cell[3];cell[d]=side==0?lo[d]:hi[d]+1;cell[t]=tang_t;cell[s]=tang_s;
      int i=cell[0],j=cell[1],k=cell[2];
      auto box=size.d_view(m);
      Real mn[3]={box.x1min,box.x2min,box.x3min},mx[3]={box.x1max,box.x2max,box.x3max};
      int sign=0;
      if(cell[d]==lo[d]&&fabs(mn[d]-cfg.reservoir_min[d])<1e-9)sign=-1;
      if(cell[d]==hi[d]+1&&fabs(mx[d]-cfg.reservoir_max[d])<1e-9)sign=1;
      if(!sign||!cfg.reservoir_user_faces[2*d+(sign>0)])return;
      Real g[NSPMETRIC],beta[3],alpha;
      if constexpr(d==0)adm::Face1Metric(m,k,j,i,adm.g_dd,adm.beta_u,adm.alpha,g,beta,alpha);
      if constexpr(d==1)adm::Face2Metric(m,k,j,i,adm.g_dd,adm.beta_u,adm.alpha,g,beta,alpha);
      if constexpr(d==2)adm::Face3Metric(m,k,j,i,adm.g_dd,adm.beta_u,adm.alpha,g,beta,alpha);
      Real det=Primitive::GetDeterminant(g),root=sqrt(det);
      Real x=CellCenterX(i-ind.is,ind.nx1,box.x1min,box.x1max);
      Real y=CellCenterX(j-ind.js,ind.nx2,box.x2min,box.x2max);
      if constexpr(d==0)x=LeftEdgeX(i-ind.is,ind.nx1,box.x1min,box.x1max);
      if constexpr(d==1)y=LeftEdgeX(j-ind.js,ind.nx2,box.x2min,box.x2max);
      Real v[3]={(beta[0]+cfg.reservoir_omega*y)/alpha,
                 (beta[1]-cfg.reservoir_omega*x)/alpha,beta[2]/alpha};
      Real vsq=Primitive::SquareVector(v,g);if(!(vsq<1&&vsq>=0))Kokkos::abort("Characteristic reservoir not timelike");
      Real W=1/sqrt(1-vsq);
      Real bn=d==0?b.x1f(m,k,j,i):(d==1?b.x2f(m,k,j,i):b.x3f(m,k,j,i));bn/=root;
      // Use the evolved CT normal field for BOTH states. Ambient tangential
      // field remains the prescribed densitized uniform vertical field.
      Real mb=eos.ps.GetEOS().GetBaryonMass();
      Real ref[7]={cfg.reservoir_rho/mb,cfg.reservoir_p,W*v[0],W*v[1],W*v[2],
                   t==2?cfg.reservoir_bz/root:0.0,s==2?cfg.reservoir_bz/root:0.0};
      int ci=i-(d==0&&sign==1),cj=j-(d==1&&sign==1),ck=k-(d==2&&sign==1);
      Real inside[7]={w(m,IDN,ck,cj,ci)/mb,w(m,IPR,ck,cj,ci),w(m,IVX,ck,cj,ci),
          w(m,IVY,ck,cj,ci),w(m,IVZ,ck,cj,ci),bc(m,t,ck,cj,ci)/root,bc(m,s,ck,cj,ci)/root};
      // Trace the PERTURBATION from the adjacent cell, not its background
      // Eulerian velocity. The inertial-rest reference varies with ADM geometry.
      Real xc=CellCenterX(ci-ind.is,ind.nx1,box.x1min,box.x1max);
      Real yc=CellCenterX(cj-ind.js,ind.nx2,box.x2min,box.x2max);
      Real vc[3]={(adm.beta_u(m,0,ck,cj,ci)+cfg.reservoir_omega*yc)/adm.alpha(m,ck,cj,ci),
                  (adm.beta_u(m,1,ck,cj,ci)-cfg.reservoir_omega*xc)/adm.alpha(m,ck,cj,ci),
                   adm.beta_u(m,2,ck,cj,ci)/adm.alpha(m,ck,cj,ci)};
      Real normc=0;for(int a=0;a<3;++a)for(int z=0;z<3;++z)
        normc+=adm.g_dd(m,a,z,ck,cj,ci)*vc[a]*vc[z];
      if(!(normc>=0&&normc<1))Kokkos::abort("Interior reservoir reference not timelike");
      Real wc=1/sqrt(1-normc);
      for(int a=0;a<3;++a)inside[2+a]+=ref[2+a]-wc*vc[a];
      Real out[7];bool limited;
      if(!char_reservoir::Project<ivx>(eos,ref,inside,bn,g,beta,alpha,sign,out,limited))
        Kokkos::abort("Characteristic projector failed; no silent boundary fallback");
      Kokkos::atomic_increment(&count(0));if(limited)Kokkos::atomic_increment(&count(1));
      Real pl[NPRIM]={},pr[NPRIM]={},bl[NPRIM]={},br[NPRIM]={};
      Real *left=sign==1?inside:out,*right=sign==1?out:inside;
      pl[PRH]=left[0];pl[PPR]=left[1];pr[PRH]=right[0];pr[PPR]=right[1];
      for(int a=0;a<3;++a){pl[PVX+a]=left[2+a];pr[PVX+a]=right[2+a];}
      pl[PTM]=eos.ps.GetEOS().GetTemperatureFromP(pl[PRH],pl[PPR],&pl[PYF]);
      pr[PTM]=eos.ps.GetEOS().GetTemperatureFromP(pr[PRH],pr[PPR],&pr[PYF]);
      bl[d]=br[d]=bn*root;bl[t]=left[5]*root;bl[s]=left[6]*root;br[t]=right[5]*root;br[s]=right[6]*root;
      Real f[NCONS],bf[NMAG];SingleStateHLLE_DYNGR<ivx>(eos,pl,pr,bl,br,5,0,g,beta,alpha,f,bf);
      flux(m,IDN,k,j,i)=f[CDN];flux(m,IEN,k,j,i)=f[CTA];
      for(int a=0;a<3;++a)flux(m,IVX+a,k,j,i)=f[CSX+a];
      ey(m,k,j,i)=bf[IBY];ez(m,k,j,i)=bf[IBZ];
    });
}
} // namespace dyngr
#endif
