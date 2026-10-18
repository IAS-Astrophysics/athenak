#!/usr/bin/env python3
"""Regress positive-cell extrapolation failure and preserve interior interpolation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from analyze_smooth_magnetic_inflow import BoundedAMRSampler,NearestAMRSampler
from plot_bondi_validation import AMRSampler
from bondi_hole_accretion import geometry,flux

def main():
 d={'mb_geometry':np.array([[0.,8.,0.,8.,0.,8.]]),'nx1_out_mb':8,'nx2_out_mb':8,'nx3_out_mb':8,'x1min':0.,'x2min':0.,'x3min':0.,'x1max':8.}
 point=np.array([[.1,2.5,2.5]])
 field=np.ones((1,8,8,8));field[:,:,:,1:]=10
 assert AMRSampler(d,point).sample(field)[0]<0 # positive input, negative extrapolation
 for cls in (BoundedAMRSampler,NearestAMRSampler):
  assert cls(d,point).sample(field)[0]==1
  assert np.allclose(cls(d,point).sample(np.full_like(field,3.)),3.)
 grid=np.arange(8)+.5;z,y,x=np.meshgrid(grid,grid,grid,indexing='ij')
 affine=(1+2*x+3*y+z)[None];points=np.array([[1.2,2.4,3.3],[5.6,4.4,1.7]])
 assert np.allclose(BoundedAMRSampler(d,points).sample(affine),1+2*points[:,0]+3*points[:,1]+points[:,2])
 # Analytic nonzero-lapse/radial-shift flux in a flat spatial metric.
 # utilde=-k*x and beta=b*x are affine; W is constant on each sphere.
 d2={'mb_geometry':np.array([[-4.,4.,-4.,4.,-4.,4.]]),'nx1_out_mb':16,'nx2_out_mb':16,'nx3_out_mb':16,'x1min':-4.,'x2min':-4.,'x3min':-4.,'x1max':4.}
 centers=np.arange(16)*.5-3.75;zz,yy,xx=np.meshgrid(centers,centers,centers,indexing='ij');shape=(1,16,16,16)
 af={}
 for name in ('xx','xy','xz','yy','yz','zz'):af['adm_g'+name]=np.full(shape,1. if name in ('xx','yy','zz') else 0.)
 af['adm_alpha']=np.full(shape,.7)
 fields={'dens':np.full(shape,1.3)}
 for axis,position in zip('xyz',(xx,yy,zz)):
  af['adm_beta'+axis]=(.02*position)[None]
  fields['vel'+axis]=(-.1*position)[None]
 r=2.;expected=4*np.pi*r**3*1.3*(.7*.1+np.sqrt(1+(.1*r)**2)*.02)
 for cls in (AMRSampler,BoundedAMRSampler):
  geo=geometry(d2,af,r,0.,24,sampler_class=cls)
  measured,_=flux(d2,fields,geo,r,sampler_class=cls)
  assert np.isclose(measured,expected,rtol=1e-12),(measured,expected)
 print('Positive edge, affine interior, and analytic lapse/shift flux regressions passed.')

if __name__=='__main__':main()
