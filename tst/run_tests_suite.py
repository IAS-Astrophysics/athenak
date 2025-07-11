
# Modules
import sys
sys.path.append('../tst/tests_suite')
import tests_suite.testutils as testutils
import pytest

pytest.main(["tests_suite/style"])
testutils.clean_make()

pytest.main(["tests_suite/hydro", "-k", "cpu"])
pytest.main(["tests_suite/mhd", "-k", "cpu"])

testutils.clean()