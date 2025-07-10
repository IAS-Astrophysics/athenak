
# Automatic test based on linear wave convergence in 1D
# In hydro, both L-/R-going sound waves and the entropy wave are tested.
# Note errors are very sensitive to the exact parameters (e.g. cfl_number, time limit)
# used. For the hard-coded error limits to apply, run parameters must not be changed.

# Modules
import sys
sys.path.append('../tst/tests_suite')
import tests_suite.testutils as testutils

from subprocess import Popen, PIPE, CalledProcessError
def run_command(cmd):
    """
    Runs a command and yields its output line by line.
    
    Args:
        cmd (list): Command to run.
    Yields:
        str: Output lines from the command.
    """
    with Popen(cmd, stdout=PIPE, bufsize=1, universal_newlines=True) as p:
        for line in p.stdout:
            print(line, end='')

#run_command(['pytest','tests_suite/style'])
testutils.cmake()
testutils.make()
run_command(['pytest','tests_suite/hydro','-k','cpu'])
run_command(['pytest','tests_suite/mhd','-k','cpu'])