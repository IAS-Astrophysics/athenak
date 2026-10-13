#!/usr/bin/env python3
"""Prepare one serial Aurora debug campaign; does not submit a job."""
import argparse
from pathlib import Path
import re


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    text=(root/'inputs/dyngr/bondi_matched_zoom.athinput').read_text()
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
