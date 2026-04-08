#!/usr/bin/env python

"""Read AthenaK spherical slice (sphslice) binary output files.

A sphslice file has an ASCII preheader, followed by an input parameter dump,
followed by a binary payload.  Two payload layouts:

  * shared mode (single_file_per_rank=0):
        <nvars * ntheta * nphi  float32>, ordering (var, itheta, iphi)

  * per-rank mode (single_file_per_rank=1):
        <npoints  int32 angle indices a>
        <nvars * npoints  float32 values>, var-major
    where the global angle index a = itheta*nphi + iphi.

Sample points are placed on a uniform-in-cos(theta), uniform-in-phi grid (the
equal-area cells produced by the C++ writer).

If pointed at a single_file_per_rank rank_00000000 file, this reader discovers
all sibling rank files automatically and reassembles the full surface.  This
matches the convention used by read_pdf.py in this directory.
"""

import glob as _glob
import os
import re

import numpy as np


_HEADER_FIRST_RE = re.compile(r'^Athena spherical slice version=(.+)$')

_INT_KEYS = frozenset({
    'cycle', 'ntheta', 'nphi', 'number_of_variables', 'size_of_variable',
    'npoints', 'rank', 'single_file_per_rank', 'header_offset',
})
_FLOAT_KEYS = frozenset({'time', 'radius'})


def _is_single_file_per_rank(path):
    """True iff `path` lives under a rank_00000000 directory."""
    return 'rank_00000000' in path


def _glob_rank_files(rank0_path):
    """Given a rank_00000000 path, list every matching rank_* sibling."""
    pattern = rank0_path.replace('rank_00000000', 'rank_*')
    files = sorted(_glob.glob(pattern))
    if not files:
        raise RuntimeError('No rank files found matching {}'.format(pattern))
    return files


def _read_header(f):
    """Parse the ASCII preheader from an open binary file.

    On return, the file position is at the start of the input parameter dump
    (i.e. just past the trailing newline of the `header offset=` line).
    """
    first = f.readline()
    if not first:
        raise RuntimeError('Empty spherical slice file')
    match = _HEADER_FIRST_RE.match(first.decode('ascii').rstrip())
    if match is None:
        raise RuntimeError('Not a spherical slice file: {!r}'.format(first))

    header = {'version': match.group(1)}
    while True:
        raw = f.readline()
        if not raw:
            raise RuntimeError('Unexpected EOF in spherical slice header')
        line = raw.decode('ascii').strip()
        if line.startswith('variables:'):
            header['variables'] = line[len('variables:'):].split()
            continue
        if '=' not in line:
            continue
        key, _, value = line.partition('=')
        key = key.strip().replace(' ', '_')
        value = value.strip()
        if key in _INT_KEYS:
            header[key] = int(value)
        elif key in _FLOAT_KEYS:
            header[key] = float(value)
        else:
            header[key] = value
        if key == 'header_offset':
            break

    if 'header_offset' not in header:
        raise RuntimeError('Missing "header offset" in spherical slice header')
    if 'variables' not in header:
        raise RuntimeError('Missing "variables" line in spherical slice header')

    header['nvars'] = header['number_of_variables']
    return header


def _read_payload(f, header):
    """Read the binary payload starting at the input dump position.

    Returns (idxs, vals):
      * shared mode: idxs is None and vals has length nvars*ntheta*nphi
      * per-rank mode: idxs is an int32 array of length npoints,
        vals has length nvars*npoints (var-major)
    """
    f.seek(header['header_offset'], os.SEEK_CUR)
    nvars = header['nvars']
    if header.get('single_file_per_rank', 0):
        npoints = header['npoints']
        idxs = np.fromfile(f, dtype=np.int32, count=npoints)
        if idxs.size != npoints:
            raise RuntimeError('Truncated index block in {}'.format(f.name))
        vals = np.fromfile(f, dtype=np.float32, count=nvars*npoints)
        if vals.size != nvars*npoints:
            raise RuntimeError('Truncated value block in {}'.format(f.name))
        return idxs, vals
    ntheta = header['ntheta']
    nphi = header['nphi']
    n = nvars*ntheta*nphi
    vals = np.fromfile(f, dtype=np.float32, count=n)
    if vals.size != n:
        raise RuntimeError('Truncated payload in {}'.format(f.name))
    return None, vals


def read_sphslice_header(path):
    """Parse only the ASCII preheader of a sphslice file."""
    with open(path, 'rb') as f:
        return _read_header(f)


def read_sphslice(path):
    """Read a sphslice binary file and reassemble its (theta, phi) surface.

    If `path` lives under a rank_00000000 directory, all sibling rank files are
    read and stitched into a single global surface.

    Returns a dict with keys:
        'header'    -- metadata dict (ntheta, nphi, nvars, time, cycle, radius,
                       variables, ...)
        'data'      -- float32 ndarray with shape (ntheta, nphi, nvars)
        'theta'     -- float64 ndarray, length ntheta -- cell-center theta values
                       in radians (uniform in cos(theta))
        'phi'       -- float64 ndarray, length nphi  -- cell-center phi values
                       in radians (uniform in phi)
        'time'      -- simulation time
        'radius'    -- sphere radius
        'variables' -- list of variable name strings (length nvars)
    """
    if _is_single_file_per_rank(path):
        files = _glob_rank_files(path)
    else:
        files = [path]

    header = None
    full = None
    covered = None
    ntheta = nphi = nvars = 0

    for fp in files:
        with open(fp, 'rb') as f:
            h = _read_header(f)
            idxs, vals = _read_payload(f, h)

        if header is None:
            header = h
            ntheta = h['ntheta']
            nphi = h['nphi']
            nvars = h['nvars']
            full = np.zeros((nvars, ntheta*nphi), dtype=np.float32)
            covered = np.zeros(ntheta*nphi, dtype=bool)
        else:
            for key in ('ntheta', 'nphi', 'nvars', 'radius'):
                if h[key] != header[key]:
                    raise RuntimeError(
                        '{} mismatch across rank files: {} vs {}'.format(
                            key, h[key], header[key]))
            if h['variables'] != header['variables']:
                raise RuntimeError('variable list mismatch across rank files')

        if idxs is None:
            full[...] = vals.reshape(nvars, ntheta*nphi)
            covered[...] = True
        elif idxs.size > 0:
            full[:, idxs] = vals.reshape(nvars, idxs.size)
            covered[idxs] = True

    missing = int((~covered).sum())
    if missing > 0:
        raise RuntimeError(
            'sphslice reassembly missing {} of {} points (path={})'.format(
                missing, covered.size, path))

    # (var, itheta*nphi) -> (ntheta, nphi, nvars)
    data = full.reshape(nvars, ntheta, nphi).transpose(1, 2, 0)

    theta = np.arccos(-1.0 + 2.0*(np.arange(ntheta) + 0.5)/ntheta)
    phi = 2.0*np.pi*(np.arange(nphi) + 0.5)/nphi

    return {
        'header': header,
        'data': data,
        'theta': theta,
        'phi': phi,
        'time': header['time'],
        'radius': header['radius'],
        'variables': header['variables'],
    }


__all__ = ['read_sphslice', 'read_sphslice_header']
