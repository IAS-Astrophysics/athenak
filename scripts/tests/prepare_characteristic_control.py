#!/usr/bin/env python3
"""Prepare isolated controls; AthenaK CLI overrides require existing keys."""
import argparse,re
from pathlib import Path

def prepare(text):
    for block, key, value in [('coord','smooth_excision','true'),
                              ('output1','ghost_zones','true'),
                              ('output2','ghost_zones','true')]:
        pattern=rf'(<{block}>\s*\n)(.*?)(?=\n<|\Z)'
        def update(m):
            body=m[2]
            if re.search(rf'^\s*{key}\s*=',body,re.M):
                body=re.sub(rf'^(\s*{key}\s*=).*$',rf'\g<1> {value}',body,flags=re.M)
            else:body=f'{key} = {value}\n'+body
            return m[1]+body
        text,n=re.subn(pattern,update,text,flags=re.S)
        if n!=1:raise ValueError(f'Expected one {block} block, found {n}')
    return text

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('target',type=Path)
    a=p.parse_args();a.target.write_text(prepare(a.source.read_text()))
