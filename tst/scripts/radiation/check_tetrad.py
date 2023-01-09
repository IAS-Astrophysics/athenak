# Unit test for Cartesian CKS tetrad
#
# Checks orthonormality of radiation tetrad

# Modules
import logging
import scripts.utils.athena as athena
import sys
sys.path.insert(0, '../vis/python')
import athena_read  # noqa
athena_read.check_nan_flag = True
logger = logging.getLogger('athena' + __name__[7:])  # set logger name


# Run AthenaK
def run(**kwargs):
    logger.debug('Runnning test ' + __name__)
    arguments = ['time/nlim=0',
                 'radiation/nlevel=2']
    athena.run('tests/tetrad.athinput', arguments)


# Analyze outputs
def analyze():
    return True
