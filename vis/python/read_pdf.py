#!/usr/bin/env python

"""Read AthenaK N-D PDF output files.

Two on-disk formats (selected by the header 'format' field):

  dense (used when ranks are globally reduced to rank 0):
    - float64 time
    - float64 values[total_bins]

  sparse_coo (used for partitioned rank- or node-sharded output):
    - float64 time
    - uint32  nnz
    - uint32  idx[nnz]   (flat bin indices)
    - float64 val[nnz]

Header layout (ASCII): <basename>.header.pdf, written once per stream.

For sparse_coo, point read_pdf() at any `rank_*` or `node_*` file and all
matching shard files are discovered automatically and summed into a dense
histogram.
"""

import glob as _glob
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


def _symlog_forward(values, linthresh):
    values = np.asarray(values, dtype=np.float64)
    sign = np.sign(values)
    abs_values = np.abs(values)
    transformed = np.empty_like(abs_values)
    linear_mask = abs_values <= linthresh
    transformed[linear_mask] = abs_values[linear_mask] / linthresh
    transformed[~linear_mask] = 1.0 + np.log10(abs_values[~linear_mask] / linthresh)
    return sign * transformed


def _symlog_inverse(values, linthresh):
    values = np.asarray(values, dtype=np.float64)
    sign = np.sign(values)
    abs_values = np.abs(values)
    physical = np.empty_like(abs_values)
    linear_mask = abs_values <= 1.0
    physical[linear_mask] = abs_values[linear_mask] * linthresh
    physical[~linear_mask] = linthresh * np.power(10.0, abs_values[~linear_mask] - 1.0)
    return sign * physical


def _bin_centers_from_edges(edges, scale, linthresh):
    if scale == 'log':
        return np.sqrt(edges[:-1] * edges[1:])
    if scale == 'symlog':
        transformed = _symlog_forward(edges, linthresh)
        return _symlog_inverse(0.5 * (transformed[:-1] + transformed[1:]), linthresh)
    return 0.5 * (edges[:-1] + edges[1:])


def _glob_partition_files(path):
    """Given a node/rank shard path, find all matching sibling shard files."""
    shard_dir = os.path.dirname(path)
    shard_base = os.path.basename(shard_dir)
    parent_dir = os.path.dirname(shard_dir)
    file_base = os.path.basename(path)

    if shard_base.startswith('rank_'):
        pattern = os.path.join(parent_dir, 'rank_*', file_base)
    elif shard_base.startswith('node_'):
        pattern = os.path.join(parent_dir, 'node_*', file_base)
    else:
      return [path]

    files = sorted(_glob.glob(pattern))
    if not files:
        raise RuntimeError(
            'No partition files found matching {}'.format(pattern))
    return files


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

            if key == 'format':
                header['format'] = value
            elif key == 'distribution':
                header['distribution'] = value
            elif key == 'ndim':
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
                elif key.startswith('scale'):
                    dims[dim]['scale'] = value
                elif key.startswith('linthresh'):
                    dims[dim]['linthresh'] = float(value)
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
        if 'scale' not in info:
            info['scale'] = 'log' if info.get('logscale', False) else 'linear'
        if info['scale'] == 'symlog':
            if 'linthresh' not in info:
                raise RuntimeError('Missing linthresh{} in {}'.format(dim, header_path))
        else:
            info.setdefault('linthresh', 1.0)
        if 'bin_edges' in info:
            edges = info['bin_edges']
            if edges.size == info['nbin'] + 1:
                info['bin_centers'] = _bin_centers_from_edges(
                    edges, info['scale'], info['linthresh']
                )
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


def _read_sparse_rank_file(path):
    """Read one sparse_coo rank file. Returns (time, idx, val)."""
    with open(path, 'rb') as handle:
        tv = np.fromfile(handle, dtype=np.float64, count=1)
        if tv.size == 0:
            return None, None, None
        nnz_arr = np.fromfile(handle, dtype=np.uint32, count=1)
        if nnz_arr.size == 0:
            raise RuntimeError(
                'Truncated sparse PDF file {}: missing nnz header'.format(path))
        nnz = int(nnz_arr[0])
        if nnz == 0:
            return float(tv[0]), None, None
        idx = np.fromfile(handle, dtype=np.uint32, count=nnz)
        val = np.fromfile(handle, dtype=np.float64, count=nnz)
        if idx.size != nnz or val.size != nnz:
            raise RuntimeError(
                'Truncated sparse PDF file {}: expected nnz={}'.format(path, nnz))
    return float(tv[0]), idx, val


def read_pdf(data_path, header_path=None, reshape=True):
    """Read a PDF data file and return a dict with time, data, and header.

    For sparse_coo output, pass any node/rank shard file and all matching
    partition files are discovered, decoded, and summed into a dense histogram.
    """
    if header_path is None:
        header_path = _infer_header_path(data_path)

    header = read_pdf_header(header_path)
    fmt = header.get('format', 'dense')
    total_bins = header.get('total_bins')

    if fmt == 'sparse_coo':
        if total_bins is None:
            raise RuntimeError(
                'Sparse PDF header missing total_bins: {}'.format(header_path))
        rank_files = _glob_partition_files(data_path)
        data = np.zeros(total_bins, dtype=np.float64)
        time_val = None
        for rf in rank_files:
            tv, idx, val = _read_sparse_rank_file(rf)
            if tv is None:
                continue
            if time_val is None:
                time_val = tv
            if idx is None:
                continue
            data[idx] += val
    elif fmt == 'dense':
        raw = np.fromfile(data_path, dtype=np.float64)
        if raw.size == 0:
            raise RuntimeError('No data found in {}'.format(data_path))
        if total_bins is not None and raw.size != total_bins + 1:
            raise RuntimeError(
                'Unexpected data size in {} (expected {}, got {})'.format(
                    data_path, total_bins + 1, raw.size))
        time_val = float(raw[0])
        data = raw[1:]
    else:
        raise RuntimeError('Unknown PDF format {!r} in {}'.format(fmt, header_path))

    if time_val is None:
        raise RuntimeError('No data found for {}'.format(data_path))

    if reshape:
        data = data.reshape(header['shape'], order='C')

    return {
        'time': time_val,
        'pdf': data,
        'header': header,
    }


__all__ = ['read_pdf', 'read_pdf_header']
