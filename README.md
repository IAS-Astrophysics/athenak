# AthenaK

Block-based AMR framework with fluid, particle and numerical relativity solvers in Kokkos.

## Overview

AthenaK is a complete rewrite of the AMR framework and fluid solvers in the [Athena++](https://github.com/PrincetonUniversity/athena) astrophysical MHD code using the [Kokkos](https://kokkos.org/) programming model.  Note that Athena++ is itself an extension of the original C-version of
[Athena](https://github.com/PrincetonUniversity/Athena-Cversion).

Using Kokkos enables *performance-portability*.  AthenaK will run on any hardware supported by Kokkos, including CPU, GPUs from various vendors, and ARM processors.

AthenaK is targeting challenging problems that require exascale resources, and as such it does not implement all of the features of Athena++.  Current code features are:
- Block-based AMR with dynamical execution via a task list
- Non-relativistic (Newtonian) hydrodynamics and MHD
- Special relativistic (SR) hydrodynamics and MHD
- General relativistic (GR) hydrodynamics and MHD in stationary spacetimes
- Relativistic radiation transport
- Lagrangian tracer particles, and charged test particles
- Numerical relativity solver using the Z4c formalism
- GR hydrodynamics and MHD in dynamical spacetimes

The numerical algorithms implemented in AthenaK are all based on higher-order finite volume methods with a variety of reconstruction algorithms, Riemann solvers, and time integration methods.

AthenaK was developed in conjunction with the 
[Parthenon AMR framework](https://github.com/parthenon-hpc-lab/parthenon) and borrows many features
from that effort. It is also closely related to the 
[AthenaPK MHD code](https://github.com/parthenon-hpc-lab/athenapk), which is another implementation
of Athena++ based on the Parthenon framework.

## Getting Started

See the [Documentation](https://ias-astrophysics.github.io/athenak-docs) to get started.

In particular, see the complete list of [requirements](https://ias-astrophysics.github.io/athenak-docs/requirements.html), or
instructions on how to [download](https://ias-astrophysics.github.io/athenak-docs/download.html) and [build](https://ias-astrophysics.github.io/athenak-docs/build.html) the code for various devices.
Other pages give instructions for running the code.

Since AthenaK is very similar to Athena++, the [Athena++ documention](https://github.com/PrincetonUniversity/athena/wiki) may also be helpful.

## Tutorials ##

A seperate GitHUb repo contains a variety of [Tutorials](https://github.com/IAS-Astrophysics/athenak-gallery) with detailed instructions on how to use AthenaK to solve specific problems.

## Code papers

For more details on the features and algorithms implemented in AthenaK, see the code papers:
- [Stone et al (2024)](https://ui.adsabs.harvard.edu/abs/2024arXiv240916053S/abstract): basic framework
- [Zhu et al. (2024)](https://ui.adsabs.harvard.edu/abs/2024arXiv240910383Z/abstract): numerical relativity solver
- [Fields at al. (2024)](https://ui.adsabs.harvard.edu/abs/2024arXiv240910384F/abstract): GR hydro and MHD solver in dynamical spacetimes

Please reference these papers as appropriate for any publications that use AthenaK.
