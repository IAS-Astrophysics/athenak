import logging
import os
from pathlib import Path
import shutil
import subprocess
import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])

_REPO_DIR = Path(__file__).resolve().parents[3]
_NPROC = 2
_INPUT = 'tests/restart_per_node.athinput'
_BUILD_DIR = _REPO_DIR / 'tst' / 'build'
_EXE_DIR = _BUILD_DIR / 'src'
_RST_DIR = _EXE_DIR / 'rst'
_SKIP_REASON = None
_STATS_ENV = {'ATHENAK_RESTART_IO_STATS': '1'}
_CHUNKED_ENV = {
    'ATHENAK_RESTART_IO_STATS': '1',
    'ATHENAK_TEST_MAX_MPI_BYTES': '128',
}


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


def _run_athena(input_filename, extra_args, cwd, env=None, capture_output=False):
    exe_path = str((_EXE_DIR / 'athena').resolve())
    input_path = str((_REPO_DIR / 'inputs' / input_filename).resolve())
    cmd = ['mpiexec', '-n', str(_NPROC), exe_path, '-i', input_path] + extra_args
    return _run_cmd(cmd, cwd, env=env, capture_output=capture_output)


def _run_restart(restart_arg, extra_args, cwd, env=None, capture_output=False):
    exe_path = str((_EXE_DIR / 'athena').resolve())
    cmd = ['mpiexec', '-n', str(_NPROC), exe_path, '-r', restart_arg] + extra_args
    return _run_cmd(cmd, cwd, env=env, capture_output=capture_output)


def _clean_path(path):
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _parse_restart_stat_lines(output):
    stats = []
    for line in output.splitlines():
        if not line.startswith('[restart-io]'):
            continue
        fields = {}
        for token in line.split():
            if '=' not in token:
                continue
            key, value = token.split('=', 1)
            fields[key] = value
        stats.append(fields)
    return stats


def _assert_restart_stats(output):
    stats = _parse_restart_stat_lines(output)
    assert stats, 'expected restart I/O stats in output'
    found_merged_span_reduction = False
    for fields in stats:
        assert fields.get('shard_opens') == '1', 'expected one shard open per node'
        assert fields.get('shard_closes') == '1', 'expected one shard close per node'
        raw_requests = int(fields['raw_requests'])
        merged_spans = int(fields['merged_spans'])
        collective_reads = int(fields['collective_reads'])
        assert collective_reads >= 1, 'expected at least one collective read'
        if raw_requests > merged_spans:
            found_merged_span_reduction = True
    assert found_merged_span_reduction, (
        'expected merged per-node restart spans to be fewer than raw requests')


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

    _clean_path(_RST_DIR)
    for rel_dir in (
            'restart_from_manifest',
            'restart_from_node',
            'restart_from_manifest_rel',
            'restart_from_node_rel',
            'restart_from_shared',
            'restart_from_rank',
            'restart_from_manifest_chunked',
            'restart_from_shared_chunked'):
        _clean_path(_EXE_DIR / rel_dir)

    base_args = ['job/basename=restart_per_node', 'time/nlim=0', 'time/tlim=0.0']
    _run_athena(_INPUT, base_args, _EXE_DIR)

    manifest_path = _RST_DIR / 'restart_per_node.00000.rst'
    payload_path = _RST_DIR / 'node_00000000' / 'restart_per_node.00000.rst'
    assert manifest_path.is_file(), 'missing per-node restart manifest'
    assert payload_path.is_file(), 'missing per-node restart payload shard'
    assert manifest_path.stat().st_size < payload_path.stat().st_size, (
        'per-node restart manifest should be smaller than payload shard')

    shared_args = ['job/basename=restart_shared',
                   'time/nlim=0',
                   'time/tlim=0.0',
                   'output1/single_file_per_node=false']
    _run_athena(_INPUT, shared_args, _EXE_DIR)
    shared_path = _RST_DIR / 'restart_shared.00000.rst'
    shared_payload_path = _RST_DIR / 'node_00000000' / 'restart_shared.00000.rst'
    assert shared_path.is_file(), 'missing shared restart file'
    assert not shared_payload_path.exists(), 'shared restart should not create node payload'

    rank_args = ['job/basename=restart_ranked',
                 'time/nlim=0',
                 'time/tlim=0.0',
                 'output1/single_file_per_node=false',
                 'output1/single_file_per_rank=true']
    _run_athena(_INPUT, rank_args, _EXE_DIR)
    rank0_path = _RST_DIR / 'rank_00000000' / 'restart_ranked.00000.rst'
    rank1_path = _RST_DIR / 'rank_00000001' / 'restart_ranked.00000.rst'
    assert rank0_path.is_file(), 'missing rank-0 restart shard'
    assert rank1_path.is_file(), 'missing rank-1 restart shard'
    assert not (_RST_DIR / 'restart_ranked.00000.rst').exists(), (
        'per-rank restart should not create a shared root file')

    restart_args = ['time/nlim=0', 'time/tlim=0.0']
    manifest_output = _run_restart('rst/restart_per_node.00000.rst',
                                   restart_args + ['-d', 'restart_from_manifest'],
                                   _EXE_DIR, env=_STATS_ENV, capture_output=True)
    _assert_restart_stats(manifest_output)
    _run_restart('./rst/node_00000000/restart_per_node.00000.rst',
                 restart_args + ['-d', 'restart_from_node'],
                 _EXE_DIR)

    _run_restart('restart_per_node.00000.rst',
                 restart_args + ['-d', '../restart_from_manifest_rel'],
                 _RST_DIR)
    _run_restart('node_00000000/restart_per_node.00000.rst',
                 restart_args + ['-d', '../restart_from_node_rel'],
                 _RST_DIR)

    _run_restart('rst/restart_shared.00000.rst',
                 restart_args + ['-d', 'restart_from_shared'],
                 _EXE_DIR)
    _run_restart('rst/rank_00000000/restart_ranked.00000.rst',
                 restart_args + ['-d', 'restart_from_rank'],
                 _EXE_DIR)

    chunked_node_args = ['job/basename=restart_node_chunked',
                         'time/nlim=0',
                         'time/tlim=0.0']
    _run_athena(_INPUT, chunked_node_args, _EXE_DIR, env=_CHUNKED_ENV)
    chunked_manifest = _RST_DIR / 'restart_node_chunked.00000.rst'
    chunked_payload = _RST_DIR / 'node_00000000' / 'restart_node_chunked.00000.rst'
    assert chunked_manifest.is_file(), 'missing chunked per-node restart manifest'
    assert chunked_payload.is_file(), 'missing chunked per-node restart payload shard'
    assert chunked_manifest.stat().st_size < chunked_payload.stat().st_size, (
        'chunked per-node restart manifest should be smaller than payload shard')

    chunked_shared_args = ['job/basename=restart_shared_chunked',
                           'time/nlim=0',
                           'time/tlim=0.0',
                           'output1/single_file_per_node=false']
    _run_athena(_INPUT, chunked_shared_args, _EXE_DIR, env=_CHUNKED_ENV)
    chunked_shared = _RST_DIR / 'restart_shared_chunked.00000.rst'
    assert chunked_shared.is_file(), 'missing chunked shared restart file'

    chunked_output = _run_restart('rst/restart_node_chunked.00000.rst',
                                  restart_args + ['-d', 'restart_from_manifest_chunked'],
                                  _EXE_DIR, env=_CHUNKED_ENV, capture_output=True)
    _assert_restart_stats(chunked_output)
    _run_restart('rst/restart_shared_chunked.00000.rst',
                 restart_args + ['-d', 'restart_from_shared_chunked'],
                 _EXE_DIR, env=_CHUNKED_ENV)


def analyze():
    if _SKIP_REASON is not None:
        logger.warning('Skipping %s: %s', __name__, _SKIP_REASON)
        return True
    return True
