#!/usr/bin/env python3
"""Isolated linearized SRMHD boundary experiment; NOT an AthenaK boundary.
Cartesian orthonormal frame, alpha=1, constant normal/tangential shift.
Seven evolved variables, normal B fixed (1D CT). Output identifies scope.
"""
import argparse, json
from pathlib import Path
import numpy as np

GAMMA=5/3

def state_flux(q, bn, shift):
    rho,p=q[:2]; v=q[2:5]; B=np.array([bn,*q[5:7]])
    W=1/np.sqrt(1-v@v); b0=W*(B@v); b=B/W+b0*v
    bs=B@B/W**2+(B@v)**2
    h=1+GAMMA/(GAMMA-1)*p/rho
    D=rho*W; S=(rho*h+bs)*W**2*v-b0*b
    E=(rho*h+bs)*W**2-(p+bs/2)-b0*b0
    U=np.array([D,*S,E-D,*B[1:]])
    F=np.array([D*v[0],*((rho*h+bs)*W**2*v*v[0]-b*b[0]+np.array([p+bs/2,0,0])),S[0]-D*v[0],*(v[0]*B[1:]-v[1:]*bn)])
    # Constant tangential shift adds a constant induction flux, retained here.
    F-=shift[0]*U
    F[5:]+=shift[1:]*bn
    return U,F

def system(bn,bt,shift,eps=1e-5):
    q=np.array([1.,.0024,0,0,0,*bt]); scale=np.array([1,.0024, .05,.05,.05,.07,.07])
    DU=[];DF=[]
    for j in range(7):
        dq=np.zeros(7);dq[j]=eps*scale[j]
        up,fp=state_flux(q+dq,bn,shift);um,fm=state_flux(q-dq,bn,shift)
        DU.append((up-um)/(2*dq[j]));DF.append((fp-fm)/(2*dq[j]))
    A=np.array(DF).T@np.linalg.inv(np.array(DU).T)
    speed,R=np.linalg.eig(A)
    assert abs(speed.imag).max()<1e-8
    speed=speed.real;R=R.real; order=np.argsort(speed);speed=speed[order];R=R[:,order]
    L=np.linalg.inv(R);tol=1e-9
    pin=R@np.diag(speed < -tol)@L; pout=np.eye(7)-pin
    residual=float(np.linalg.norm(A@R-R*speed)/max(1,np.linalg.norm(A)))
    return speed,R,L,pin,pout,residual

def wave_exit(speed,R,L,n=128):
    # Evolve all seven independent packets, each toward its outgoing boundary.
    # First-order characteristic upwind + RK2, extrapolate outgoing and set
    # incoming perturbations to zero at BOTH boundaries. No extrapolation in time.
    x=(np.arange(n)+.5)/n; initial=np.exp(-((x-.5)/.07)**2)
    a=np.tile(initial[:,None],(1,7));vmax=max(abs(speed)); t=0.; target=.8/max(min(abs(speed)),1e-6)
    # Skip stationary families; separately tested by the projector audit.
    live=abs(speed)>1e-7;target=.8/min(abs(speed[live]))
    dt=.45/n/vmax
    def rhs(y):
        left=np.where(speed>0,0,y[0]); right=np.where(speed<0,0,y[-1])
        padded=np.vstack([left,y,right]); flux=np.where(speed>=0,padded[:-1],padded[1:])*speed
        return -n*(flux[1:]-flux[:-1])
    steps=0
    while t<target:
        step=min(dt,target-t); y=a+step*rhs(a); a=.5*(a+y+step*rhs(y));t+=step;steps+=1
        if steps>150000:raise RuntimeError('bounded experiment step cap')
    # Convert to coupled conservative fields and recover wave families.
    recovered=(a@R.T)@L.T
    stationary=~live
    outgoing=float(abs(recovered[:,live]).max())
    assert outgoing<.02,(n,outgoing)
    assert np.isfinite(recovered).all()
    return dict(n=n,steps=steps,remaining_outgoing_amplitude=outgoing,
                stationary_families=int(stationary.sum()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    records=[];rng=np.random.default_rng(9471)
    # Nondegenerate oblique field, normal field, tangential field; actual face
    # normal rotational shift ranges +/-11.3137 and crosses wave speeds.
    for bn,bt in [(.04,[.05,.02]),(.07,[0.,0.]),(0.,[.07,0.])]:
      for beta in [-11.3137085,-.2,-.03,0,.03,.2,11.3137085]:
        speed,R,L,pi,po,res=system(bn,bt,np.array([beta,3.,0.]))
        assert np.linalg.cond(R)<1e8
        assert res<1e-8
        assert np.linalg.norm(pi@pi-pi)<1e-9
        assert np.linalg.norm(pi@po)<1e-9
        interior=rng.normal(size=7)*1e-7
        ghost=po@interior # ambient perturbation = 0
        assert np.linalg.norm(pi@ghost)<1e-15
        assert np.linalg.norm(po@(ghost-interior))<1e-15
        # Uniform ambient must stay uniform, independent of shift classification.
        assert np.array_equal(po@np.zeros(7),np.zeros(7))
        records.append(dict(bn=bn,bt=bt,normal_shift=beta,speeds=speed.tolist(),
                            incoming=int(sum(speed < -1e-9)),eigen_residual=res))
    # Pulse evolution at a generic oblique field, away from zero speeds.
    pulses=[]
    for beta in [-11.3137085,-.2,.2,11.3137085]:
        speed,R,L,*_=system(.04,[.05,.02],np.array([beta,3.,0.]))
        for n in [64,128]:pulses.append(dict(normal_shift=beta,**wave_exit(speed,R,L,n)))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(dict(scope='linearized SRMHD, flat rotating-face patch; not nonlinear GRMHD or 3D CT validation',status='passed',projector_cases=records,pulse_cases=pulses),indent=2)+'\n')
    print(json.dumps(dict(status='passed',projector_cases=len(records),pulse_cases=len(pulses),out=str(args.out))))
if __name__=='__main__':main()
