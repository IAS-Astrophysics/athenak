#!/usr/bin/env python3
"""Prepare one serial Aurora debug campaign; does not submit a job."""
import argparse
from pathlib import Path
import re


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--tlim',type=float,default=4.0)
    parser.add_argument('--residence',type=float,default=0.5)
    parser.add_argument('--zone0-residence',type=float,default=1.0)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    text=(root/'inputs/dyngr/bondi_matched_zoom.athinput').read_text()
    levels=int(re.search(r'^nlevels\s*=\s*(\d+)',text,re.M).group(1))
    return_time=args.zone0_residence+(2*levels-3)*args.residence
    if not (args.residence > 0 and args.zone0_residence > 0 and
            return_time < args.tlim < return_time+args.zone0_residence):
        parser.error('target must follow one complete sweep and precede the next')
    text=re.sub(r'^tlim\s*=.*$',f'tlim = {args.tlim}',text,flags=re.M)
    text=re.sub(r'^trun_fac\s*=.*$',
                f'trun_fac = {args.residence}\ntrun_fac_0 = {args.zone0_residence}',
                text,flags=re.M)
    text=re.sub(r'^dt\s*=\s*10\s*$',f'dt = {args.residence}',text,flags=re.M)
    text=re.sub(r'^dt\s*=\s*120\s*$',f'dt = {args.tlim}',text,flags=re.M)
    (out/'matched.athinput').write_text(text)
    # Original hydro pgen, with its central mask and reduced dynamic range.
    legacy=text.replace('variable = mhd_w_bcc','variable = hydro_w')
    for block in ('refined_region2','refined_region3','adm','output2','output3'):
        legacy=re.sub(r'<'+block+r'>\n.*?(?=\n<|\Z)','',legacy,flags=re.S)
    legacy=re.sub(r'<mhd>\n.*?(?=\n<)',
                  '<hydro>\neos = ideal\nreconstruct = ppm4\nrsolver = hlle\n'
                  'gamma = 1.6666666666666667\n',legacy,flags=re.S)
    legacy=legacy.replace('<problem>','<problem>\npgen_name = gr_bondi')
    legacy=legacy.replace('r_0 = 32','r_0 = 1')
    (out/'upstream.athinput').write_text(legacy)
    (out/'campaign.txt').write_text(
        'upstream upstream.athinput\n'
        'single_inertial matched.athinput problem/single_black_hole=true '
        'problem/corotating=false problem/frame_omega=0\n'
        'single_rotating matched.athinput problem/single_black_hole=true\n'
        'binary_rotating matched.athinput problem/single_black_hole=false\n')
    print(out)


if __name__ == '__main__':
    main()
