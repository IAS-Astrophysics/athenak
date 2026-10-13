# Functions for interfacing with AthenaK during testing

# Modules
import logging
import os
import subprocess
from timeit import default_timer as timer
from .log_pipe import LogPipe

# Global variables
athena_rel_path = '../'


# Function for compiling AthenaK
def make(arguments):
    logger = logging.getLogger('athena.make')
    out_log = LogPipe('athena.make', logging.INFO)
    current_dir = os.getcwd()
    try:
        subprocess.check_call(['mkdir', 'build'], stdout=out_log)
        build_dir = current_dir + '/build/'
        os.chdir(build_dir)
        cmake_command = ['cmake3', '../' + athena_rel_path] + arguments
        make_command = ['make', '-j8']
        try:
            t0 = timer()
            logger.debug('Executing: '+' '.join(cmake_command))
            subprocess.check_call(cmake_command, stdout=out_log)
            logger.debug('Executing: '+' '.join(make_command))
            subprocess.check_call(make_command, stdout=out_log)
            logger.debug('Build took {0:.3g} seconds.'.format(timer() - t0))
        except subprocess.CalledProcessError as err:
            logger.error("Something bad happened", exc_info=True)
            raise AthenaError('Return code {0} from command \'{1}\''
                              .format(err.returncode, ' '.join(err.cmd)))
    finally:
        out_log.close()
        os.chdir(current_dir)


# Function for running AthenaK
def run(input_filename, arguments):
    out_log = LogPipe('athena.run', logging.INFO)
    current_dir = os.getcwd()
    exe_dir = current_dir + '/build/src/'
    os.chdir(exe_dir)
    try:
        input_filename_full = '../../' + athena_rel_path + \
                              'inputs/' + input_filename
        run_command = ['./athena', '-i', input_filename_full]
        try:
            cmd = run_command + arguments
            logging.getLogger('athena.run').debug('Executing: '+' '.join(cmd))
            subprocess.check_call(cmd, stdout=out_log)
        except subprocess.CalledProcessError as err:
            raise AthenaError('Return code {0} from command \'{1}\''
                              .format(err.returncode, ' '.join(err.cmd)))
    finally:
        out_log.close()
        os.chdir(current_dir)


# Function for running AthenaK with MPI
def mpirun(nproc, input_filename, arguments):
    out_log = LogPipe('athena.run', logging.INFO)
    current_dir = os.getcwd()
    exe_dir = current_dir + '/build/src/'
    os.chdir(exe_dir)
    try:
        input_filename_full = '../../' + athena_rel_path + \
                              'inputs/' + input_filename
        run_command = ['mpiexec -n', str(nproc), './athena', '-i',
                       input_filename_full]
        try:
            cmd = run_command + arguments
            logging.getLogger('athena.run').debug('Executing: '+' '.join(cmd))
            subprocess.check_call(cmd, stdout=out_log)
        except subprocess.CalledProcessError as err:
            raise AthenaError('Return code {0} from command \'{1}\''
                              .format(err.returncode, ' '.join(err.cmd)))

    finally:
        out_log.close()
        os.chdir(current_dir)


# General exception class for these functions
class AthenaError(RuntimeError):
    pass
