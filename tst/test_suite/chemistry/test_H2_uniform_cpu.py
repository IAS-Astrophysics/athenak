"""
Test for chemistry using the H2 network with uniform initial conditions. Runs tests for
different ODE solvers
"""

# Modules
import pytest
import test_suite.chemistry.test_H2_uniform_gpu as h2_uniform


@pytest.mark.parametrize("ode_solver", h2_uniform.ode_solvers)
def test_h2_uniform_cpu(ode_solver):
    """CPU Test for H2 uniform test problem."""
    h2_uniform.run_h2_uniform(ode_solver)
