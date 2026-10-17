"""Stationary spatial metric for this equal-mass, nonspinning circular pilot.

Specializes dynbbh.cpp's BoostedSpatialCoordinates/BuildBoostJacobian and
SuperposedBBHTemplateMixed at corotating t=0. Spatial gamma is unchanged by
the rotating time-space transformation at that instant. This is pointwise
metric evaluation, not interpolation/restriction of a finer saved metric.
"""
import numpy as np


def config(header):
    result={};block=''
    for line in header:
        line=line.split('#',1)[0].strip()
        if line.startswith('<') and line.endswith('>'):block=line[1:-1]
        elif '=' in line:
            key,value=line.split('=',1);result[block+'/'+key.strip()]=value.strip()
    return result


def check_pilot(header):
    c=config(header)
    for key in ('problem/a1','problem/a2'):assert float(c[key])==0
    assert float(c['problem/q'])==1 and float(c['problem/sep'])==32
    assert c['problem/corotating'].lower() in ('1','true')
    assert c.get('problem/single_black_hole','0').lower() in ('0','false')
    assert c.get('problem/use_traj_table','0').lower() in ('0','false')
    assert float(c.get('problem/cutoff_floor','0.0001'))==1e-4
    assert c.get('problem/spin_ramp','0').lower() in ('0','false')
    assert abs(float(c['problem/frame_omega'])-32**-1.5)<1e-15
    return c


def spatial_metric(x,y,z):
    x,y,z=np.broadcast_arrays(x,y,z)
    g=np.zeros((3,3)+x.shape,dtype=np.float64)
    for i in range(3):g[i,i]=1.
    for hx in (16.,-16.):
        v=hx*32**-1.5;lor=1/np.sqrt(1-v*v)
        dx=x-hx;dy=lor*y;dz=z.copy()
        r=np.sqrt(dx*dx+dy*dy+dz*dz)
        # Same cutoff as the zero-spin simulation; all sampled active centers
        # in this pilot are outside it, but retain its behavior explicitly.
        dz=np.where(r<1e-4,np.where(dz>0,1e-4,-1e-4),dz)
        r=np.sqrt(dx*dx+dy*dy+dz*dz)
        ell=np.array([dx/r,lor*(dy/r-v),dz/r])
        g+=np.einsum('i...,j...->ij...',ell,ell)/r  # 2*m=1 per hole
    return g


def beta_from_primitives(g,u,Btilde,pressure):
    det=(g[0,0]*(g[1,1]*g[2,2]-g[1,2]**2)
         -g[0,1]*(g[0,1]*g[2,2]-g[0,2]*g[1,2])
         +g[0,2]*(g[0,1]*g[1,2]-g[0,2]*g[1,1]))
    assert np.isfinite(det).all() and (det>0).all()
    B=Btilde/np.sqrt(det)
    norm=lambda a,b:np.einsum('ij...,i...,j...->...',g,a,b)
    b2=(norm(B,B)+norm(B,u)**2)/(1+norm(u,u))
    assert np.isfinite(b2).all() and (b2>0).all()
    return 2*pressure/b2


def validate_saved_metric(d,fields):
    check_pilot(d['header']);b=d['mb_geometry'];n=d['nx1_out_mb']
    assert n==d['nx1_mb']==d['nx2_mb']==d['nx3_mb']
    xyz=[]
    for axis in range(3):
        q=b[:,2*axis,None]+(np.arange(n)+.5)*(b[:,2*axis+1]-b[:,2*axis])[:,None]/n
        xyz.append(q[:,None,None,:] if axis==0 else q[:,None,:,None] if axis==1 else q[:,:,None,None])
    g=spatial_metric(*xyz);errors={}
    for i,ci in enumerate('xyz'):
      for j,cj in enumerate('xyz'[i:],i):
        saved=fields['adm_g'+ci+cj]
        err=float((abs(g[i,j]-saved)/np.maximum(1.,abs(saved))).max())
        assert err<2e-7,(ci+cj,err)
        errors[ci+cj]=err
    return errors
