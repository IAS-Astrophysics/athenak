#!/usr/bin/env python3
"""Same rotating packets as v2; enable post-C2P ghost/native-CT closure."""
from characteristic_rotating_cases import flags
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('family');p.add_argument('frame');p.add_argument('choice');p.add_argument('n',type=int);p.add_argument('--block-z',type=int,default=8);a=p.parse_args()
 print('\n'.join(flags(a.family,a.frame,a.choice,a.n,a.block_z)+[f'problem/reservoir_ghost_ct={str(a.choice in ("characteristic","firstorder")).lower()}']))
