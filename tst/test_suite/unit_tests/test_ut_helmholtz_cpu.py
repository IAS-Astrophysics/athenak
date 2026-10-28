"""
Unit tests for EOSHelmholtz
"""

# Modules
import os

import pytest

import test_suite.testutils as testutils

TABLE = "inputs/tables/electron_table.athtab"


def test_helmholtz():
    # The electron table is not part of the repository
    if not os.path.exists(TABLE) and not os.path.exists(os.path.join("..", "..", TABLE)):
        pytest.skip(f"{TABLE} not found")
    input_file = "inputs/ut_helmholtz.athinput"
    testutils.run(input_file)
