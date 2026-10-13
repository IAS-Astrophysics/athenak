#!/usr/bin/env python

# Regression test script.

# Usage: From this directory, call this script with python:
#        python run_tests.py

# Notes:
#   - Requires Python 3+.
#   - This file should not be modified when adding new scripts.

# Modules
import argparse
import os
from collections import OrderedDict
from importlib import reload
import logging
import logging.config
from pkgutil import iter_modules
from timeit import default_timer as timer

# Prevent generation of .pyc files
# This should be set before importing any user modules
import sys
sys.dont_write_bytecode = True

# AthenaK modules
import scripts.utils.athena as athena  # noqa

# AthenaK logger
logger = logging.getLogger('athena')


# Main function
def main(**kwargs):
    # Make list of tests to run
    tests = kwargs.pop('tests')
    test_names = []
    if len(tests) == 0:  # run all tests
        for _, directory, ispkg in iter_modules(path=['scripts']):
            if ispkg and (directory != 'utils' and directory != 'style'):
                dir_test_names = [name for _, name, _ in
                                  iter_modules(path=['scripts/'
                                                     + directory],
                                               prefix=directory + '.')]
                test_names.extend(dir_test_names)
    else:  # run selected tests
        for test in tests:
            if test[-1] == '/':
                test = test[:-1]  # remove trailing slash
            if '/' in test:  # specific test specified
                test_names.append(test.replace('/', '.'))
            else:  # test suite specified
                dir_test_names = [name for _, name, _ in
                                  iter_modules(path=['scripts/'
                                                     + test],
                                               prefix=test + '.')]
                test_names.extend(dir_test_names)

    # Remove duplicate test entries while preserving the original order
    test_names = list(OrderedDict.fromkeys(test_names))

    # Run tests
    current_dir = os.getcwd()
    test_times = []
    test_results = []
    test_errors = []
    try:
        # Check that required modules are installed for all test dependencies
        deps_installed = True
        for name in test_names:
            try:
                name_full = 'scripts.' + name
                module = __import__(name_full, globals(), locals(),
                                    fromlist=['run', 'analyze'])
            except ImportError as e:
                if sys.version_info >= (3, 6, 0):  # ModuleNotFoundError
                    missing_module = e.name
                else:
                    missing_module = e.message.split(' ')[-1]
                logger.warning('Unable to '
                               'import "{:}".'.format(missing_module))
                deps_installed = False
        if not deps_installed:
            logger.warning('WARNING! Not all required Python modules '
                           'are available')
        # Build AthenaK
        try:
            os.system('rm -rf {0}/build'.format(current_dir))
            # insert arguments for athena.make()
            athena_cmake_args = kwargs.pop('cmake')
            module.athena.make(athena_cmake_args)
        except Exception:
            logger.error("Exception occurred", exc_info=True)
            test_errors.append('make()')
            raise TestError('Unable to build AthenaK')
        # Run each test
        for name in test_names:
            t0 = timer()
            try:
                name_full = 'scripts.' + name
                module = __import__(name_full, globals(), locals(),
                                    fromlist=['run', 'analyze'])
                reload(module)
                try:
                    run_ret = module.run()
                except Exception:
                    logger.error("Exception occurred", exc_info=True)
                    test_errors.append('run()')
                    raise TestError(name_full.replace('.', '/') + '.py')
                try:
                    result = module.analyze()
                except Exception:
                    logger.error("Exception occurred", exc_info=True)
                    test_errors.append('analyze()')
                    raise TestError(name_full.replace('.', '/') + '.py')
            except TestError as err:
                test_results.append(False)
                logger.error('---> Error in ' + str(err))
                # do not measure runtime for failed/incomplete tests
                test_times.append(None)
            else:
                test_times.append(timer() - t0)
                msg = 'Test {0} took {1:.3g} seconds to complete.'
                msg = msg.format(name, test_times[-1])
                logging.getLogger('athena.tests.' + name).debug(msg)
                test_results.append(result)
                test_errors.append(None)
            # For CI, print after every individual test has finished
            logger.info('{} test: run(), analyze() finished'.format(name))
    finally:
        os.system('rm -rf {0}/build'.format(current_dir))

    # Report test results
    logger.info('\nResults:')
    for name, result, error, time in zip(test_names, test_results, test_errors,
                                         test_times):
        result_string = 'passed' if result else 'failed'
        error_string = ' -- unexpected failure in {0} stage'.format(error) \
                       if error is not None \
                       else '; time elapsed: {0:.3g} s'.format(time)
        logger.info('    {0}: {1}{2}'.format(name, result_string,
                                             error_string))
    logger.info('')
    num_tests = len(test_results)
    num_passed = test_results.count(True)
    test_string = 'test' if num_tests == 1 else 'tests'
    logger.info('Summary: {0} out of {1} {2} '
                'passed\n'.format(num_passed, num_tests, test_string))
    # For CI calling scripts, explicitly raise error if not all tests passed
    if num_passed == num_tests:
        return 0
    else:
        raise TestError()


# Exception for unexpected behavior by individual tests
class TestError(RuntimeError):
    pass


# Filter out critical exceptions
class CriticalExceptionFilter(logging.Filter):
    def filter(self, record):
        return not record.exc_info or record.levelno != logging.CRITICAL


# Initialize log
def log_init(args):
    kwargs = vars(args)
    logging.basicConfig(level=0)  # setting to 0 gives output cntrl to handler
    logger.propagate = False  # don't use default handler
    c_handler = logging.StreamHandler()  # console/terminal handler
    c_handler.setLevel(logging.INFO)
    c_handler.addFilter(CriticalExceptionFilter())  # stderr errors to screen
    c_handler.setFormatter(logging.Formatter('%(message)s'))  # only show msg
    logger.addHandler(c_handler)
    # setup log_file
    log_fn = kwargs.pop('log_file')
    if log_fn:
        f_handler = logging.FileHandler(log_fn)
        f_handler.setLevel(0)  # log everything
        f_format = logging.Formatter('%(asctime)s|%(levelname)s'
                                     ':%(name)s: %(message)s')
        f_handler.setFormatter(f_format)
        logger.addHandler(f_handler)
    logger.debug('Starting AthenaK regression tests')


# Execute main function
if __name__ == '__main__':
    help_msg = ('names of tests to run, relative to scripts/')
    parser = argparse.ArgumentParser()
    parser.add_argument('tests',
                        type=str,
                        default=None,
                        nargs='*',
                        help=help_msg)

    parser.add_argument("--cmake",
                        default=[],
                        action='append',
                        help='architecture specific args to pass to cmake')

    parser.add_argument('--log_file',
                        type=str,
                        default=None,
                        help='set filename of logfile')

    args = parser.parse_args()
    log_init(args)

    try:
        logger.debug('args: ' + str(vars(args)))
        main(**vars(args))
    except Exception:
        logger.critical('', exc_info=True)
        raise
