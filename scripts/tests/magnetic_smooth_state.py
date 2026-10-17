"""Native restart metadata for the prescribed-ADM, no-forcing MHD configuration.

Layout follows outputs/restart.cpp (double Real, 19-int RegionIndcs). Rejects
unknown configurations rather than guessing an offset into checkpoint arrays.
"""
import math
import struct


def checkpoint_state(path):
    with path.open('rb') as f:
        head=f.read(65536);end=head.index(b'<par_end>\n');base=end+10
        params={};block=''
        for line in head[:end].decode().splitlines():
            line=line.split('#',1)[0].strip()
            if line.startswith('<') and line.endswith('>'):block=line[1:-1]
            elif '=' in line:
                k,v=line.split('=',1);params[block+'/'+k.strip()]=v.strip()
        assert params['adm/dynamic']=='false'
        assert not any(k.startswith(('z4c/','turbulence/','hydro/','radiation/','dyn_radiation/')) for k in params)
        blocks,root=struct.unpack_from('<ii',head,base)
        t,dt,cycle=struct.unpack_from('<ddi',head,base+232)
        assert blocks>0 and root>=0 and cycle>=0 and math.isfinite(t) and dt>0
        sizepos=base+252+20*blocks
        f.seek(sizepos);perblock=struct.unpack('<Q',f.read(8))[0]
        assert 0<perblock<100_000_000
        zoompos=sizepos+8+blocks*perblock
        f.seek(zoompos);magic,last,event,zone,previous,direction,nexttime=struct.unpack('<Q5i4xd',f.read(40))
        assert magic==0x435A424248000002 and 0<=last<=cycle and event>=0
        assert 0<=zone<int(params['amr_criterion1/nlevels']) and direction in (-1,1)
        assert math.isfinite(nexttime)
    return dict(time=t,dt=dt,cycle=cycle,blocks=blocks,id=event,zone=zone,
                last_cycle=last,last_zone=previous,direction=direction,next_time=nexttime,
                parameters=params)
