#!/usr/bin/env python

from circumbinary_parfile import Parfile


initial_data = {
    "par_b": 3.257,
    "ncells": 128,
    "mbsize": 32
}

name = f"circumbinary_N{initial_data['ncells']}.athinput"
par = Parfile(**initial_data)
with open(name, "w") as f:
    f.write(str(par))
