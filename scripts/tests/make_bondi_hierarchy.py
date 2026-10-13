#!/usr/bin/env python3
"""Generate matched gas reservoirs with hole-centered hydro cyclic-zoom ladders.

Eq.17 residence times are physical by default. --residence-scale is only for
bounded infrastructure tests, never a relaxation claim.
"""
import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]

def make(single=False, residence_scale=1.0):
    text=(ROOT/'inputs/dyngr/bondi_matched_zoom.athinput').read_text()
    mass=1.0 if single else 0.5
    levels=8 if single else 9
    text=re.sub(r'<refined_region1>.*?(?=<amr_criterion1>)','',text,flags=re.S)
    regions=''
    for i,x in enumerate([0.0] if single else [16.,-16.],1):
        regions+=f'<refined_region{i}>\nlevel = {levels-1}\n'
        for d,c in enumerate([x,0.,0.],1):
            regions+=f'x{d}min = {c-.5*mass}\nx{d}max = {c+.5*mass}\n'
    text=text.replace('<amr_criterion1>',regions+'<amr_criterion1>')
    def val(key,value):
        nonlocal text
        text=re.sub(r'^'+key+r'\s*=.*$',f'{key} = {value}',text,flags=re.M)
    for key,value in {'num_levels':levels,'nlevels':levels,'max_nmb_per_rank':2048,
                      'max_nzmb_per_dvce':2048,'max_nzmb_per_host':2048,
                      'r_0':mass,'trun_pow':1.5,'verbose':'true','nlim':-1,
                      'tlim':100000,'single_black_hole':str(single).lower(),
                      'bondi_cavity_radius':32,'bondi_cavity_core_fraction':0}.items():val(key,value)
    text=text.replace('method = cyclic_zoom',
        f'method = cyclic_zoom\nbondi_hydro_schedule = true\nresidence_scale = {residence_scale}')
    text=text.replace('# Keep the mesh identical for single and binary: refine origin and both holes.',
                      '# Resolve each physical horizon at m_hole/16; no binary COM fine patch.')
    text=text.replace('# Change only problem/single_black_hole to switch black-hole configurations.',
                      '# Generated companion files use the same gas; hierarchy follows the horizons.')
    rows=[]
    for z in range(levels):
        r=mass*2**z; near=not single and r<32
        t=100*mass if z==0 else 0 if z in (1,2) else 10*(r**3/(mass if near else 1))**.5
        rows.append(dict(zone=z,radius=r,centers=2 if near else 1,mass=mass if near else 1,
                         physical_residence=t,actual_residence=t*residence_scale))
    return text,rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--residence-scale',type=float,default=1.)
    a=p.parse_args()
    if not 0<a.residence_scale<=1: p.error('residence scale must be in (0,1]')
    a.out.mkdir(parents=True,exist_ok=True)
    schedules={}
    for name,single in [('single',True),('binary',False)]:
        text,rows=make(single,a.residence_scale)
        (a.out/f'{name}.athinput').write_text(text);schedules[name]=rows
    (a.out/'schedule.json').write_text(json.dumps(schedules,indent=2)+'\n')
    print(a.out)
if __name__=='__main__':main()
