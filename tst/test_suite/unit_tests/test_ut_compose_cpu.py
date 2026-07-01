"""
Unit tests for EOSCompOSE
"""

# Modules
import test_suite.testutils as testutils


def test_logs():
    input_file = "inputs/ut_compose_log.athinput"
    testutils.run(input_file)


def test_NQTs():
    input_file = "inputs/ut_compose_NQT.athinput"
    testutils.run(input_file)
