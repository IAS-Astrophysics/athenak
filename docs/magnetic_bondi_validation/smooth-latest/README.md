# Latest saved finest-stage xy view

Selected from the smooth-excision MHD chain after renewed SSH access. Current
physical job 8940293 is running; quiet completion monitoring is active again.
The selected saved fluid output is from completed, validated job 8940212:
`magnetic_bondi_binary.mhd_w_bcc.00395.bin`, cycle 236032, 1408 active blocks,
finest zone 0/event 28. This is the first full return following the smooth
restart at event 14; the requested second roundtrip is still pending.

The binary output header rounds time to 19700.9 M. The matching native final
checkpoint gives 19700.932791173836 M, and this finest visit began at
19586.65420246086 M: finest-visit phase 114.278588712976 M. The metric is reused
from the stationary initial ADM snapshot only after exact logical-location and
block-geometry equality checks. All saved fluid fields are finite; density and
pressure are positive. Fluid-frame magnetic pressure uses the spatial metric,
Lorentz factor and Btilde/sqrt(gamma), not the coordinate Euclidean field norm.

`finest_xy.png` shows actual active AMR cell slices. The spatial resolution
varies away from the holes. Black masks follow the boosted smooth-drain region;
cyan crosses mark the hole centers. Temperature is the dimensionless proxy
p/(rho c^2), not a temperature in kelvin. Plasma beta is 2p/b^2. The diagnostic
is not a convergence or inflow-equilibrium claim.

At the preceding clean wall stop all twelve ranks passed validation;
conservative summed RSS and tracked allocation peaks were 23.474 GB,
maximum scaled magnetic divergence 9.51e-12. The controller continued from the
accepted native checkpoint without resetting its remaining dwell. No extra
job was submitted during this visualization request.
