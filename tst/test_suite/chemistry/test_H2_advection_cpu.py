"""
Test for chemistry using the H2 network with advecting gaussian initial conditions. Runs
tests for different ODE solvers
"""

# Modules
import pytest
import test_suite.chemistry.test_H2_advection_gpu as h2_advection


@pytest.mark.parametrize("ode_solver", h2_advection.ode_solvers)
def test_h2_uniform_cpu(ode_solver):
    """CPU Test for H2 uniform test problem."""
    h2_advection.run_h2_advection(ode_solver)
