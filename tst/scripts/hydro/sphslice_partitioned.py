# Regression test for shared, per-rank, and per-node spherical-slice output.

import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import scripts.utils.athena as athena

_REPO_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str((_REPO_DIR / 'vis' / 'python').resolve()))
import read_sphslice  # noqa

logger = logging.getLogger('athena' + __name__[7:])

_NPROC = 2
_INPUT = 'tests/sphslice_partitioned.athinput'
_BUILD_DIR = _REPO_DIR / 'tst' / 'build'
_EXE_DIR = _BUILD_DIR / 'src'
_BIN_DIR = _EXE_DIR / 'bin'
_SKIP_REASON = None
_STATS_ENV = {'ATHENAK_OUTPUT_IO_STATS': '1'}


def _mpi_enabled():
    config_path = _BUILD_DIR / 'config.hpp'
    if not config_path.is_file():
        return False
    return '#define MPI_PARALLEL_ENABLED 1' in config_path.read_text()


def _run_cmd(cmd, cwd, env=None, capture_output=False):
    logger.debug('Executing: %s (cwd=%s)', ' '.join(cmd), cwd)
    run_env = os.environ.copy()
    if env is not None:
        run_env.update(env)
    result = subprocess.run(cmd, cwd=str(cwd), env=run_env, text=True,
                            capture_output=capture_output)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd,
                                            output=result.stdout,
                                            stderr=result.stderr)
    if capture_output:
        return result.stdout + result.stderr
    return ''


def _run_athena(extra_args, env=None, capture_output=False):
    exe_path = str((_EXE_DIR / 'athena').resolve())
    input_path = str((_REPO_DIR / 'inputs' / _INPUT).resolve())
    cmd = ['mpiexec', '-n', str(_NPROC), exe_path, '-i', input_path] + extra_args
    return _run_cmd(cmd, _EXE_DIR, env=env, capture_output=capture_output)


def _clean_bin_dir():
    if _BIN_DIR.is_dir():
        shutil.rmtree(_BIN_DIR)


def _latest_sphfile(root):
    files = sorted(root.rglob('*.sph.bin'))
    assert files, f'no sphslice outputs found under {root}'
    return files[-1]


def _parse_sphslice_stats(output):
    stats = []
    for line in output.splitlines():
        if not line.startswith('[output-io] type=sphslice'):
            continue
        fields = {}
        for token in line.split():
            if '=' not in token:
                continue
            key, value = token.split('=', 1)
            fields[key] = value
        stats.append(fields)
    return stats


def _assert_no_sphslice_stats(output):
    assert not _parse_sphslice_stats(output), (
        'shared sphslice path should not emit sparse output I/O stats')


def _assert_partitioned_stats(output, mode):
    stats = _parse_sphslice_stats(output)
    assert stats, f'expected sphslice output I/O stats for mode={mode}'
    for fields in stats:
        assert fields.get('mode') == mode, f'unexpected stats mode: {fields}'
        sparse_bytes = int(fields['sparse_payload_bytes'])
        dense_bytes = int(fields['dense_baseline_bytes'])
        assert sparse_bytes < dense_bytes, (
            f'expected sparse payload to be smaller than dense baseline for {mode}')
        if mode == 'rank':
            assert fields['local_points'] == fields['node_points'], (
                'per-rank stats should report node_points == local_points')
        if mode == 'node':
            assert int(fields['node_points']) >= int(fields['local_points']), (
                'per-node stats should aggregate at least the local point count')


def _assert_matching_data(reference, candidate):
    ref_header = reference['header']
    cand_header = candidate['header']
    for key in ('ntheta', 'nphi', 'radius'):
        assert cand_header[key] == ref_header[key], f'header mismatch for {key}'
    assert cand_header['variables'] == ref_header['variables'], 'variable mismatch'
    assert np.array_equal(candidate['data'], reference['data']), 'sphslice data mismatch'


def run(**kwargs):
    global _SKIP_REASON
    _SKIP_REASON = None
    logger.debug('Running test %s', __name__)

    if not _mpi_enabled():
        _SKIP_REASON = 'MPI is not enabled in build/config.hpp'
        logger.warning(_SKIP_REASON)
        return
    if shutil.which('mpiexec') is None:
        _SKIP_REASON = 'mpiexec is not available'
        logger.warning(_SKIP_REASON)
        return

    common_args = ['time/nlim=0', 'time/tlim=0.0']

    _clean_bin_dir()
    shared_output = _run_athena(['job/basename=sphslice_shared'] + common_args,
                                env=_STATS_ENV, capture_output=True)
    _assert_no_sphslice_stats(shared_output)
    shared_file = _latest_sphfile(_BIN_DIR)
    shared = read_sphslice.read_sphslice(str(shared_file))

    _clean_bin_dir()
    node_output = _run_athena(['job/basename=sphslice_node'] + common_args
                              + ['output1/single_file_per_node=true'],
                              env=_STATS_ENV, capture_output=True)
    _assert_partitioned_stats(node_output, 'node')
    node_file = _latest_sphfile(_BIN_DIR / 'node_00000000')
    node = read_sphslice.read_sphslice(str(node_file))
    _assert_matching_data(shared, node)

    _clean_bin_dir()
    rank_output = _run_athena(['job/basename=sphslice_rank'] + common_args
                              + ['output1/single_file_per_rank=true'],
                              env=_STATS_ENV, capture_output=True)
    _assert_partitioned_stats(rank_output, 'rank')
    rank_file = _latest_sphfile(_BIN_DIR / 'rank_00000000')
    rank = read_sphslice.read_sphslice(str(rank_file))
    _assert_matching_data(shared, rank)


def analyze():
    if _SKIP_REASON is not None:
        logger.warning('Skipping %s: %s', __name__, _SKIP_REASON)
        return True
    return True
