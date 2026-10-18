#ifndef DYN_GRMHD_CHARACTERISTIC_GHOST_CT_HPP_
#define DYN_GRMHD_CHARACTERISTIC_GHOST_CT_HPP_
// Opt-in characteristic ghost closure. After primitive recovery, extend outgoing
// perturbations, construct exterior staggered fields with discrete divB=0, then
// use native reconstruction/HLLE/SG07/CT without any boundary flux/EMF overwrite.
#include "dyn_grmhd/characteristic_reservoir.hpp"
namespace dyngr {
namespace char_ghost {
KOKKOS_INLINE_FUNCTION
bool Outside(const CoordData &cfg,const RegionSize &box,const RegionIndcs &ind,
             int d,int index,bool face=false) {
  Real mn[3]={box.x1min,box.x2min,box.x3min},mx[3]={box.x1max,box.x2max,box.x3max};
  int lo[3]={ind.is,ind.js,ind.ks},hi[3]={ind.ie,ind.je,ind.ke};
  return (cfg.reservoir_user_faces[2*d]&&fabs(mn[d]-cfg.reservoir_min[d])<1e-9&&index<lo[d]) ||
    (cfg.reservoir_user_faces[2*d+1]&&fabs(mx[d]-cfg.reservoir_max[d])<1e-9&&index>hi[d]+face);
}
template<class A>
KOKKOS_INLINE_FUNCTION
void Rest(const A &adm,const RegionSize &box,const RegionIndcs &ind,Real omega,
          int m,const int c[3],Real u[3],Real &root) {
  int i=c[0],j=c[1],k=c[2];Real g[6];
  g[0]=adm.g_dd(m,0,0,k,j,i);g[1]=adm.g_dd(m,0,1,k,j,i);g[2]=adm.g_dd(m,0,2,k,j,i);
  g[3]=adm.g_dd(m,1,1,k,j,i);g[4]=adm.g_dd(m,1,2,k,j,i);g[5]=adm.g_dd(m,2,2,k,j,i);
  Real x=CellCenterX(i-ind.is,ind.nx1,box.x1min,box.x1max);
  Real y=CellCenterX(j-ind.js,ind.nx2,box.x2min,box.x2max),alpha=adm.alpha(m,k,j,i);
  Real v[3]={(adm.beta_u(m,0,k,j,i)+omega*y)/alpha,
             (adm.beta_u(m,1,k,j,i)-omega*x)/alpha,adm.beta_u(m,2,k,j,i)/alpha};
  Real vsq=Primitive::SquareVector(v,g);if(!(vsq>=0&&vsq<1))Kokkos::abort("Ghost reservoir not timelike");
  Real W=1/sqrt(1-vsq);for(int a=0;a<3;++a)u[a]=W*v[a];root=sqrt(Primitive::GetDeterminant(g));
}
KOKKOS_INLINE_FUNCTION
Real Extend(Real q0,Real q1,Real q2,int layer,int order) {
  if(order==1)return q0;
  Real a=q0-q1,b=q1-q2;if(a*b<=0)return q0;
  Real direction=a>0?1:-1;
  if(order==2)return q0+layer*direction*fmin(fabs(a),fabs(b));
  // Exact continuation of cell-average quadratic data when the limiter is inactive.
  Real increment=layer*a+.5*layer*(layer+1)*(a-b);
  Real bound=2*layer*fmax(fabs(a),fabs(b));
  return q0+direction*fmin(bound,fmax(0.0,direction*increment));
}
} // namespace char_ghost

template<int ivx,class E,class Err>
void CharacteristicGhostColumns(MeshBlockPack *pack,const PrimitiveSolverHydro<E,Err>& eos,
                                const DvceArray5D<Real> &temperature) {
  constexpr int d=ivx-IVX,t=(d+1)%3,s=(d+2)%3;
  auto cfg=pack->pcoord->coord_data;auto ind=pack->pmesh->mb_indcs;auto sizes=pack->pmb->mb_size;
  auto adm=pack->padm->adm;auto w=pack->pmhd->w0;auto bc=pack->pmhd->bcc0;auto b=pack->pmhd->b0;
  auto counts=pack->pcoord->characteristic_counts;
  int lo[3]={ind.is,ind.js,ind.ks},hi[3]={ind.ie,ind.je,ind.ke};
  int n[3]={ind.nx1+2*ind.ng,ind.nx2+2*ind.ng,ind.nx3+2*ind.ng};
  // Ordered tensor-product corners: x slabs, then y including x ghosts, then z
  // including x/y ghosts. Unique state per ghost cell; no concurrent face writes.
  par_for("characteristic_ghost_columns",DevExeSpace(),0,pack->nmb_thispack-1,
    0,1,0,n[t]-1,0,n[s]-1,KOKKOS_LAMBDA(int m,int side,int ct,int cs) {
      auto box=sizes.d_view(m);Real mn[3]={box.x1min,box.x2min,box.x3min},mx[3]={box.x1max,box.x2max,box.x3max};
      int sign=side?1:-1;
      if(!cfg.reservoir_user_faces[2*d+side] || fabs((side?mx[d]:mn[d])-(side?cfg.reservoir_max[d]:cfg.reservoir_min[d]))>=1e-9)return;
      int c[3];c[d]=side?hi[d]:lo[d];c[t]=ct;c[s]=cs;
      for(int a=d+1;a<3;++a)if(char_ghost::Outside(cfg,box,ind,a,c[a]))return;
      int face[3]={c[0],c[1],c[2]};face[d]+=side;
      Real g[6],beta[3],alpha;
      if constexpr(d==0)adm::Face1Metric(m,face[2],face[1],face[0],adm.g_dd,adm.beta_u,adm.alpha,g,beta,alpha);
      if constexpr(d==1)adm::Face2Metric(m,face[2],face[1],face[0],adm.g_dd,adm.beta_u,adm.alpha,g,beta,alpha);
      if constexpr(d==2)adm::Face3Metric(m,face[2],face[1],face[0],adm.g_dd,adm.beta_u,adm.alpha,g,beta,alpha);
      Real root=sqrt(Primitive::GetDeterminant(g)),ur[3],rc;
      char_ghost::Rest(adm,box,ind,cfg.reservoir_omega,m,c,ur,rc);
      // Projection reference is at the anchor cell, with the face metric used for
      // both states; use its inertial-rest velocity at the actual face location.
      Real x=CellCenterX(face[0]-ind.is,ind.nx1,box.x1min,box.x1max);
      Real y=CellCenterX(face[1]-ind.js,ind.nx2,box.x2min,box.x2max);
      if constexpr(d==0)x=LeftEdgeX(face[0]-ind.is,ind.nx1,box.x1min,box.x1max);
      if constexpr(d==1)y=LeftEdgeX(face[1]-ind.js,ind.nx2,box.x2min,box.x2max);
      Real vf[3]={(beta[0]+cfg.reservoir_omega*y)/alpha,(beta[1]-cfg.reservoir_omega*x)/alpha,beta[2]/alpha};
      Real vsq=Primitive::SquareVector(vf,g);if(!(vsq>=0&&vsq<1))Kokkos::abort("Face reservoir not timelike");
      Real W=1/sqrt(1-vsq),mb=eos.ps.GetEOS().GetBaryonMass();
      Real ref[7]={cfg.reservoir_rho/mb,cfg.reservoir_p,W*vf[0],W*vf[1],W*vf[2],t==2?cfg.reservoir_bz/root:0.0,s==2?cfg.reservoir_bz/root:0.0};
      Real bn=(d==0?b.x1f(m,face[2],face[1],face[0]):(d==1?b.x2f(m,face[2],face[1],face[0]):b.x3f(m,face[2],face[1],face[0])))/root;
      Real scale[7]={ref[0],ref[1],sqrt(ref[1]/ref[0]),sqrt(ref[1]/ref[0]),sqrt(ref[1]/ref[0]),sqrt(ref[1]),sqrt(ref[1])};
      Real q[3][7],P[7][7],out[7],delta[3][7];bool limited;
      for(int a=0;a<3;++a) {
        int ci[3]={c[0],c[1],c[2]};ci[d]-=sign*a;Real uref[3],rt;
        char_ghost::Rest(adm,box,ind,cfg.reservoir_omega,m,ci,uref,rt);
        q[a][0]=w(m,IDN,ci[2],ci[1],ci[0])/mb;q[a][1]=w(m,IPR,ci[2],ci[1],ci[0]);
        for(int z=0;z<3;++z)q[a][2+z]=ref[2+z]+w(m,IVX+z,ci[2],ci[1],ci[0])-uref[z];
        q[a][5]=bc(m,t,ci[2],ci[1],ci[0])/root;q[a][6]=bc(m,s,ci[2],ci[1],ci[0])/root;
      }
      if(!char_reservoir::Project<ivx>(eos,ref,q[0],bn,g,beta,alpha,sign,out,limited,P))Kokkos::abort("Ghost characteristic projector failed");
      Kokkos::atomic_inc(&counts(0));
      for(int a=0;a<3;++a)for(int z=0;z<7;++z){delta[a][z]=0;for(int j=0;j<7;++j)delta[a][z]+=P[z][j]*(q[a][j]-ref[j])/scale[j];delta[a][z]*=scale[z];}
      Real normal=bc(m,d,c[2],c[1],c[0]);
      for(int layer=1;layer<=ind.ng;++layer) {
        int ci[3]={c[0],c[1],c[2]};ci[d]+=sign*layer;Real ug[3],rg;
        char_ghost::Rest(adm,box,ind,cfg.reservoir_omega,m,ci,ug,rg);
        Real dq[7];for(int z=0;z<7;++z)dq[z]=char_ghost::Extend(delta[0][z],delta[1][z],delta[2][z],layer,cfg.reservoir_trace_order);
        // Component limiting need not preserve an eigenspace. Project the
        // limited continuation again with the same column projector, then use
        // one positivity contraction for all components.
        Real projected[7];for(int z=0;z<7;++z){projected[z]=0;for(int j=0;j<7;++j)projected[z]+=P[z][j]*dq[j]/scale[j];projected[z]*=scale[z];}
        for(int z=0;z<7;++z)dq[z]=projected[z];
        Real theta=1;bool good=false;Real val[7];
        for(int trial=0;trial<24;++trial){
          for(int z=0;z<7;++z)val[z]=ref[z]+theta*dq[z];
          Real u[3]={ug[0]+theta*dq[2],ug[1]+theta*dq[3],ug[2]+theta*dq[4]};Real usq=0;
          for(int a=0;a<3;++a)for(int z=0;z<3;++z)usq+=adm.g_dd(m,a,z,ci[2],ci[1],ci[0])*u[a]*u[z];
          good=val[0]>=eos.ps.GetEOS().GetMinimumDensity()&&val[1]>0&&usq>=0&&usq/(1+usq)<SQR(eos.ps.GetEOS().GetMaxVelocity());
          for(int z=0;z<7;++z)good=good&&Kokkos::isfinite(val[z]);
          if(good)good=eos.ps.GetEOS().GetTemperatureFromP(val[0],val[1],nullptr)>=eos.ps.GetEOS().GetMinimumTemperature();
          if(good)break;theta*=.5;
        }
        if(!good)Kokkos::abort("Characteristic ghost positivity failed");
        if(theta<1)Kokkos::atomic_inc(&counts(1));
        w(m,IDN,ci[2],ci[1],ci[0])=mb*val[0];w(m,IPR,ci[2],ci[1],ci[0])=val[1];
        for(int a=0;a<3;++a)w(m,IVX+a,ci[2],ci[1],ci[0])=ug[a]+theta*dq[2+a];
        bc(m,d,ci[2],ci[1],ci[0])=normal;
        bc(m,t,ci[2],ci[1],ci[0])=(t==2?cfg.reservoir_bz:0)+root*theta*dq[5];
        bc(m,s,ci[2],ci[1],ci[0])=(s==2?cfg.reservoir_bz:0)+root*theta*dq[6];
        temperature(m,0,ci[2],ci[1],ci[0])=eos.ps.GetEOS().GetTemperatureFromP(val[0],val[1],nullptr);
      }
    });
}

template<class E,class Err>
void CharacteristicGhostCT(MeshBlockPack *pack,const PrimitiveSolverHydro<E,Err>& eos,
                           const DvceArray5D<Real> &temperature) {
  auto mesh=pack->pmesh;
  if(!mesh->three_d || mesh->mb_indcs.nx1<3 || mesh->mb_indcs.nx2<3 || mesh->mb_indcs.nx3<3)
    Kokkos::abort("Characteristic ghost/CT closure requires 3D blocks with at least three cells per axis");
  // Interior static AMR is supported. Refined physical faces require an additional
  // coarse physical-ghost/reflux closure and are deliberately rejected here.
  for(int m=0;m<pack->nmb_thispack;++m) {
    if(pack->pmb->mb_lev.h_view(m)==mesh->root_level)continue;
    auto box=pack->pmb->mb_size.h_view(m);auto cfg=pack->pcoord->coord_data;
    Real mn[3]={box.x1min,box.x2min,box.x3min},mx[3]={box.x1max,box.x2max,box.x3max};
    for(int d=0;d<3;++d)if((cfg.reservoir_user_faces[2*d]&&fabs(mn[d]-cfg.reservoir_min[d])<1e-9)||
       (cfg.reservoir_user_faces[2*d+1]&&fabs(mx[d]-cfg.reservoir_max[d])<1e-9))
      Kokkos::abort("Characteristic ghost/CT closure requires unrefined physical faces; interior AMR is allowed");
  }
  CharacteristicGhostColumns<IVX>(pack,eos,temperature);
  CharacteristicGhostColumns<IVY>(pack,eos,temperature);
  CharacteristicGhostColumns<IVZ>(pack,eos,temperature);
  auto cfg=pack->pcoord->coord_data;auto ind=pack->pmesh->mb_indcs;auto sizes=pack->pmb->mb_size;
  auto bc=pack->pmhd->bcc0;auto b=pack->pmhd->b0;
  int n1=ind.nx1+2*ind.ng,n2=ind.nx2+2*ind.ng,n3=ind.nx3+2*ind.ng;
  // Only exterior faces are proposed. Evolved physical normal faces and all
  // active faces remain bitwise untouched. CT alone evolves their normal flux.
  par_for("characteristic_ghost_faces",DevExeSpace(),0,pack->nmb_thispack-1,0,n3,0,n2,0,n1,
    KOKKOS_LAMBDA(int m,int k,int j,int i){
      auto box=sizes.d_view(m);int c[3]={i,j,k},n[3]={n1,n2,n3};
      for(int d=0;d<3;++d){
        if(c[(d+1)%3]>=n[(d+1)%3]||c[(d+2)%3]>=n[(d+2)%3])continue;
        bool exterior=false;for(int a=0;a<3;++a)exterior=exterior||char_ghost::Outside(cfg,box,ind,a,c[a],a==d);
        if(!exterior)continue;
        int l[3]={i,j,k},r[3]={i,j,k};l[d]=c[d]>0?c[d]-1:0;r[d]=c[d]<n[d]?c[d]:n[d]-1;
        Real value=.5*(bc(m,d,l[2],l[1],l[0])+bc(m,d,r[2],r[1],r[0]));
        if(d==0)b.x1f(m,k,j,i)=value;if(d==1)b.x2f(m,k,j,i)=value;if(d==2)b.x3f(m,k,j,i)=value;
      }
    });
  // Divergence-free exterior extension, ordered by the last exterior direction.
  // Each cell's outward face is solved once; later slabs never change its faces.
  // Normal faces on the physical surface are the integration constants.
  for(int d=0;d<3;++d)for(int layer=1;layer<=ind.ng;++layer) {
    int t=(d+1)%3,s=(d+2)%3,n[3]={n1,n2,n3};
    par_for("characteristic_ghost_divergence",DevExeSpace(),0,pack->nmb_thispack-1,0,1,0,n[t]-1,0,n[s]-1,
      KOKKOS_LAMBDA(int m,int side,int ct,int cs){
        auto box=sizes.d_view(m);int c[3],lo[3]={ind.is,ind.js,ind.ks},hi[3]={ind.ie,ind.je,ind.ke};
        c[d]=side?hi[d]+layer:lo[d]-layer;c[t]=ct;c[s]=cs;
        int i=c[0],j=c[1],k=c[2];
        if(!char_ghost::Outside(cfg,box,ind,d,c[d]))return;
        for(int a=d+1;a<3;++a)if(char_ghost::Outside(cfg,box,ind,a,c[a]))return;
        Real dx[3]={box.dx1,box.dx2,box.dx3};
        Real deriv[3]={(b.x1f(m,k,j,i+1)-b.x1f(m,k,j,i))/dx[0],(b.x2f(m,k,j+1,i)-b.x2f(m,k,j,i))/dx[1],(b.x3f(m,k+1,j,i)-b.x3f(m,k,j,i))/dx[2]};
        Real tangent=deriv[(d+1)%3]+deriv[(d+2)%3];
        if(d==0){if(side)b.x1f(m,k,j,i+1)=b.x1f(m,k,j,i)-dx[d]*tangent;else b.x1f(m,k,j,i)=b.x1f(m,k,j,i+1)+dx[d]*tangent;}
        if(d==1){if(side)b.x2f(m,k,j+1,i)=b.x2f(m,k,j,i)-dx[d]*tangent;else b.x2f(m,k,j,i)=b.x2f(m,k,j+1,i)+dx[d]*tangent;}
        if(d==2){if(side)b.x3f(m,k+1,j,i)=b.x3f(m,k,j,i)-dx[d]*tangent;else b.x3f(m,k,j,i)=b.x3f(m,k+1,j,i)+dx[d]*tangent;}
      });
  }
  // Ghost bcc must come from the SAME staggered field used by the Riemann solver
  // and cell-centered CT electric field. Do not leave the proposed cell B behind.
  par_for("characteristic_ghost_bcc",DevExeSpace(),0,pack->nmb_thispack-1,0,n3-1,0,n2-1,0,n1-1,
    KOKKOS_LAMBDA(int m,int k,int j,int i){
      auto box=sizes.d_view(m);if(!(char_ghost::Outside(cfg,box,ind,0,i)||char_ghost::Outside(cfg,box,ind,1,j)||char_ghost::Outside(cfg,box,ind,2,k)))return;
      bc(m,0,k,j,i)=.5*(b.x1f(m,k,j,i)+b.x1f(m,k,j,i+1));
      bc(m,1,k,j,i)=.5*(b.x2f(m,k,j,i)+b.x2f(m,k,j+1,i));
      bc(m,2,k,j,i)=.5*(b.x3f(m,k,j,i)+b.x3f(m,k+1,j,i));
    });
}
} // namespace dyngr
#endif
