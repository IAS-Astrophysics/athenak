# Explicit seven-wave decomposition

The matrix-sign boundary already contains all seven ideal-MHD characteristic
families. It selects the full outgoing invariant subspace, rather than just an
acoustic or fast-wave approximation. An explicit full decomposition is useful
for identifying modal contamination, but does not create additional outgoing
information or an exact exterior solution.

The new offline implementation, `scripts/characteristic_decomposition.py`, returns
right/left bases R,L, seven speeds and degenerate-family groups. It holds the CT
normal magnetic field fixed and uses the same native primitive symbol as the
solver:

    A = (dU/dq)^(-1) dF/dq,   delta q = R (L delta q),
    P_out = R diag[H(outward * lambda)] L.

Primitives are scaled before differentiating, with q=(n,p,utilde^i,Bt,Bs).
`characteristic_native_spectrum.cpp` calls the native ideal-EOS Valencia flux to
export this symbol. This is a diagnostic binary; production source/executable
and the accepted Aurora test binary are unchanged.

Speeds are independently computed from the covariant ideal-RMHD dispersion law.
For a coordinate normal n, x=(lambda+beta^n)/alpha, a=W(v^n-x),
G=gamma^{nn}-x^2, and b.phi=B^n/W+(B.v)a, the four magnetosonic roots solve

    rho*h*(1-cs^2)*a^4 - (rho*h*cs^2+b^2)*a^2*G
        + cs^2*(b.phi)^2*G = 0.

The remaining roots are the entropy speed alpha*v^n-beta^n and two Alfvén
speeds from sqrt(rho*h+b^2)*a = +/- b.phi. Families are fast+/-, Alfvén+/-,
slow+/-, entropy. All metric contractions use the actual spatial metric.
The native diagnostic uses baryon mass one and an ideal gas with gamma=5/3;
it does not establish arbitrary tabulated-EOS or composition support.

SVD nullspaces construct the right eigenspaces; inversion produces the left
basis. Coincident or nearly coincident eigenvalues are grouped (2e-6 of the
coordinate light speed). Their individual polarizations are nonunique. The sum
of reconstructed components in a group is meaningful; the individual basis
amplitudes are not invariant under rotation within that group. A near-degenerate
group has an explicitly reported speed spread and bounded eigen residual. This
is a numerical full decomposition, not the closed-form renormalized analytic
GPU eigensystem of [Antón et al. (2010)](https://arxiv.org/abs/0912.4692).

`test_characteristic_decomposition.py` checks 1,620 native states, including
870 degenerate/near-degenerate states: all three normals; diagonal and off-diagonal
metrics; lapse 0.7/1.3; normal shifts up to 16; rest and relativistic flows; parallel,
perpendicular, oblique, near-zero-normal, zero and strong fields. It checks the
independent dispersion relation, complete basis reconstruction, inverse,
idempotent projectors and causal speeds. For unambiguous nonstationary spectra,
the explicit projection agrees with the native matrix sign to 1.29e-8 maximum
matrix-component error. Near-zero-speed comparisons are excluded because the
native sign convention deliberately assigns glancing modes outward. Eigen and
dispersion residuals reach 1.23e-6 and 2.08e-6 of coordinate light speed, dominated
by the finite-difference native symbol in relativistic/cancellation-sensitive
states. Reconstruction error is below 1.3e-12 in the sampled scaled states.
See `full-decomposition-tests.json` for exact metrics.

For the vertical beta=1 reservoir, parallel acoustic waves are the nondegenerate
slow branches (cs=0.06324555), while fast and Alfvén pairs coalesce
(cA=0.06911635). `plot_characteristic_modes.py` decomposes all seven saved
primitive differences from the matched reference and reconstructs each group's
pressure contribution. This separates the incoming pressure wave from entropy.
The diagnostic remains linear and reference-subtracted; it is not a nonlinear
reflection coefficient, nor a conserved-energy reflection measurement.

Replacing a validated matrix-sign projector with this CPU eigensystem alone
cannot address inconsistent high-order ghost/CT stencils or multidimensional
exterior memory. No performance benefit is established: the native numerical
Jacobian remains, and an explicit eigensystem has its own cost. No production
boundary selection changes are made on the basis of these checks.
