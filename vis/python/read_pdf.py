#!/usr/bin/env python

"""Read AthenaK N-D PDF output files.

Binary layout:
  - First 8 bytes: time (float64)
  - Next: flattened histogram values (float64), length = total_bins

Header layout (ASCII):
  - <basename>.header.pdf produced once per output stream.
"""

import os
import re

import numpy as np


_HEADER_KEY_RE = re.compile(r'^([A-Za-z0-9_]+)\s*=\s*(.*)$')
_DIM_RE = re.compile(r'(\d+)$')
_BIN_EDGE_RE = re.compile(r'^bin_edges_(\d+)$')
_PDF_DATA_RE = re.compile(r'^(.*)\.(\d{5})\.pdf$')


def _parse_bool(value):
    value = value.strip().lower()
    if value in ('true', 't', '1', 'yes', 'y'):
        return True
    if value in ('false', 'f', '0', 'no', 'n'):
        return False
    raise ValueError('Unrecognized boolean value: {}'.format(value))


def _get_dim_index(key):
    match = _DIM_RE.search(key)
    if match is None:
        return None
    return int(match.group(1))


def _infer_header_path(data_path):
    match = _PDF_DATA_RE.match(os.path.basename(data_path))
    if match is None:
        raise ValueError('Unable to infer header path from {}'.format(data_path))
    base = match.group(1)
    return os.path.join(os.path.dirname(data_path), base + '.header.pdf')


def read_pdf_header(header_path):
    """Parse a PDF header file and return a metadata dict."""
    header = {}
    dims = {}

    with open(header_path, 'r') as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            match = _HEADER_KEY_RE.match(line)
            if match is None:
                continue
            key, value = match.group(1), match.group(2).strip()

            if key == 'ndim':
                header['ndim'] = int(value)
            elif key == 'weight':
                header['weight'] = value
            elif key == 'weight_variable':
                header['weight_variable'] = value
            elif key == 'total_bins':
                header['total_bins'] = int(value)
            else:
                edge_match = _BIN_EDGE_RE.match(key)
                if edge_match:
                    dim = int(edge_match.group(1))
                    dims.setdefault(dim, {})['bin_edges'] = np.array(
                        [float(v) for v in value.split()],
                        dtype=np.float64
                    )
                    continue

                dim = _get_dim_index(key)
                if dim is None:
                    continue
                dims.setdefault(dim, {})

                if key.startswith('variable_'):
                    dims[dim]['variable'] = value
                elif key.startswith('nbin'):
                    dims[dim]['nbin'] = int(value)
                elif key.startswith('bin') and key.endswith('_min'):
                    dims[dim]['bin_min'] = float(value)
                elif key.startswith('bin') and key.endswith('_max'):
                    dims[dim]['bin_max'] = float(value)
                elif key.startswith('logscale'):
                    dims[dim]['logscale'] = _parse_bool(value)
                elif key.startswith('stride'):
                    dims[dim]['stride'] = int(value)

    if 'ndim' not in header:
        raise RuntimeError('Missing ndim in header {}'.format(header_path))

    dimensions = []
    for dim in range(1, header['ndim'] + 1):
        if dim not in dims:
            raise RuntimeError('Missing metadata for dimension {} in {}'.format(dim,
                                                                                header_path))
        info = dims[dim]
        if 'nbin' not in info:
            raise RuntimeError('Missing nbin{} in {}'.format(dim, header_path))
        info['nbin_with_overflow'] = info['nbin'] + 2
        if 'bin_edges' in info:
            edges = info['bin_edges']
            if edges.size == info['nbin'] + 1:
                if info.get('logscale', False):
                    info['bin_centers'] = np.sqrt(edges[:-1] * edges[1:])
                else:
                    info['bin_centers'] = 0.5 * (edges[:-1] + edges[1:])
        dimensions.append(info)

    header['dimensions'] = dimensions
    header['shape'] = tuple([dim['nbin_with_overflow'] for dim in dimensions])

    if 'total_bins' in header:
        expected = 1
        for dim in dimensions:
            expected *= dim['nbin_with_overflow']
        if expected != header['total_bins']:
            raise RuntimeError('Header total_bins mismatch: expected {}, got {}'.format(
                expected, header['total_bins']))

    return header


def read_pdf(data_path, header_path=None, reshape=True):
    """Read a PDF data file and return a dict with time, data, and header."""
    if header_path is None:
        header_path = _infer_header_path(data_path)

    header = read_pdf_header(header_path)
    raw = np.fromfile(data_path, dtype=np.float64)

    if raw.size == 0:
        raise RuntimeError('No data found in {}'.format(data_path))

    expected = header.get('total_bins')
    if expected is not None and raw.size != expected + 1:
        raise RuntimeError('Unexpected data size in {} (expected {}, got {})'.format(
            data_path, expected + 1, raw.size))

    time_val = float(raw[0])
    data = raw[1:]
    if reshape:
        data = data.reshape(header['shape'], order='C')

    return {
        'time': time_val,
        'pdf': data,
        'header': header,
    }


__all__ = ['read_pdf', 'read_pdf_header']
