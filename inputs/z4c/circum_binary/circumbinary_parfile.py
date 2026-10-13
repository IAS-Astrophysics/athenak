#!/usr/bin/env python3
# This script generates an AthenaK input file for a BBH run
import numpy as np

TEMPLATE = """
<comment>
problem   = two punctures run with tracker

<job>
basename  = bbh        # problem ID: basename of output filenames

<mesh>
nghost    = 4          # Number of ghost cells
nx1       = {nx}       # Number of zones in X1-direction
x1min     = -512      # minimum value of X1
x1max     = 512       # maximum value of X1
ix1_bc    = outflow    # inner-X1 boundary flag
ox1_bc    = outflow    # outer-X1 boundary flag

nx2       = {ny}       # Number of zones in X2-direction
x2min     = -512      # minimum value of X2
x2max     = 512       # maximum value of X2
ix2_bc    = outflow    # inner-X2 boundary flag
ox2_bc    = outflow    # outer-X2 boundary flag

nx3       = {nz}       # Number of zones in X3-direction
x3min     = -512      # minimum value of X3
x3max     = 512       # maximum value of X3
ix3_bc    = outflow    # inner-X3 boundary flag
ox3_bc    = outflow    # outer-X3 boundary flag

<meshblock>
nx1       = {mbsize}   # Number of cells in each MeshBlock, X1-dir
nx2       = {mbsize}   # Number of cells in each MeshBlock, X2-dir
nx3       = {mbsize}   # Number of cells in each MeshBlock, X3-dir

<time>
evolution  = dynamic    # dynamic/kinematic/static
integrator = rk3        # time integration algorithm
cfl_number = {cfl}      # The Courant, Friedrichs, & Lewy (CFL) Number
nlim       = -1         # cycle limit
tlim       = 1000       # time limit
ndiag      = 1          # cycles between diagostic output

<mesh_refinement>
refinement       = adaptive    # type of refinement
max_nmb_per_rank = 150
num_levels       = 8
refinement_interval = {refinement_interval}

<z4c_amr>
method = chi
chi_min = 0.2

<refinement1>
level = 7
x1min = {refinement1_x1min}
x1max = {refinement1_x1max}
x2min = -{bh_0_rad}
x2max = {bh_0_rad}
x3min = -{bh_0_rad}
x3max = {bh_0_rad}

<refinement2>
level = 7
x1min = {refinement2_x1min}
x1max = {refinement2_x1max}
x2min = -{bh_1_rad}
x2max = {bh_1_rad}
x3min = -{bh_1_rad}
x3max = {bh_1_rad}

<z4c>
lapse_harmonic  = 0.0       # Harmonic lapse parameter mu_L
lapse_oplog     = 2.0       # 1+log lapse parameter
first_order_shift = true
shift_eta       = 1.0       # Shift damping term
shift_advect    = 0.0
shift_ggamma    = 1.0
diss            = 0.5       # Kreiss-Oliger dissipation parameter
chi_div_floor   = 0.00001
damp_kappa1     = 0.00      # Constraint damping factor 1
damp_kappa2     = 0.01

## Test fluid
back_reaction   = false

## Wave extraction
nrad_wave_extraction = 5
extraction_radius_0  = 50
extraction_radius_1  = 100
extraction_radius_2  = 150
extraction_radius_3  = 200
extraction_radius_4  = 250
extraction_nlev      = 30
waveform_dt          = 1

## Puncture tracker
co_0_type       = BH
co_0_x          = {bh_0_x}
co_0_radius     = {bh_0_rad}
co_1_type       = BH
co_1_x          = {bh_1_x}
co_1_radius     = {bh_1_rad}

## Horizon Finder
dump_horizon_0  = true
co_0_dump_radius = 12
horizon_0_Nx     = 100
dump_horizon_1  = true
co_1_dump_radius = 2
horizon_1_Nx     = 100
horizon_dt = 5

## Slow Start Lapse
slow_start_lapse = true
ssl_damping_time = 20
ssl_damping_amp = 0.6

<coord>
general_rel = true	# general relativity
m           = 1.0
a           = 0.0
excise      = true
excision_scheme = lapse
excise_lapse = 0.5
dexcise = 1.0e-10
pexcise = 1.0e-18

<mhd>
eos         = ideal	 # EOS type
dyn_eos     = ideal	 # EOS type
dyn_error   = reset_floor # EOS type
reconstruct = wenoz	  # spatial reconstruction method
rsolver     = hlle	 # Riemann-solver to be used
dfloor      = 1.0e-10     # floor on density rho
pfloor      = 1.0e-18  # floor on gas pressure p_gas
tfloor      = 1.0e-8  # floor on gas pressure p_gas
gamma       = 1.44444444444 # ratio of specific heats Gamma
fofc        = true	 # Enable first order flux correction
gamma_max   = 20.0	 # Enable ceiling on Lorentz factor
dyn_scratch = 0
enforce_maximum = false
dmp_M       = 1.2

<problem>
################ SPACETIME ############
pgen_name = z4c_two_puncture
# the following are defaults and descriptions
verbose = true
par_b = {par_b}                        # 1/2 separation
par_m_plus = 0.483                  # mass of the m+ puncture
par_m_minus = 0.483                 # mass of the m- puncture
target_M_plus = 1            # target ADM mass for m+
target_M_minus = 1           # target ADM mass for m-
par_P_plus1 = {bh_0_px}                # momentum of the m+ puncture
par_P_plus2 = {bh_0_py}
par_P_plus3 = 0
par_P_minus1 = {bh_1_px}               # momentum of the m- puncture
par_P_minus2 = {bh_1_py}
par_P_minus3 = 0
par_S_plus1 = 0                # spin of the m+ puncture
par_S_plus2 = 0
par_S_plus3 = 0
par_S_minus1 = 0               # spin of the m- puncture
par_S_minus2 = 0
par_S_minus3 = 0
center_offset1 = {center_offset}       # offset b=0 to position (x,y,z)
center_offset2 = 0.0
center_offset3 = 0.0
give_bare_mass = True                  # User provides bare masses not target M_ADM
npoints_A = 48                         # No. coeff in the compactified radial dir.
npoints_B = 48                         # No. coeff in the angular dir.
npoints_phi = 32                       # no. coeff in the phi dir.
Newton_tol = 1e-10                     # Tolerance for Newton solver
Newton_maxit = 5                       # Maximum number of Newton iterations
# A small number to smooth out singularities at the puncture locations
TP_epsilon = 0.0
# Tiny number to avoid nans near or at the pucture locations
TP_Tiny = 0.0
# Radius of an extended spacetime instead of the puncture
TP_Extend_Radius = 0
# Tolerance of ADM masses when give_bare_mass=no
adm_tol = 1e-2
# Output debug information about the residuum
do_residuum_debug_output = false
# Output debug information about initial guess
do_initial_debug_output = false
# Solve for momentum constraint?
solve_momentum_constraint = false
# Interpolation
# Exponent n for psi^-n initial lapse profile (<0)
initial_lapse_psi_exponent = -2.0
# Swap x and z coordinates when interpolating,
# so that the black holes are separated in the z direction
swap_xz = false

########### TORUS ############
chakrabarti_torus = false  # Chakrabarti
fm_torus   = true  # Fishbone & Moncrief
sane       = true  # vector potential for magnetic field (SANE or MAD config.)
r_edge            = 20.0  # radius of inner edge of disk
r_peak            = 30.2  # radius of pressure maximum; use l instead if negative
tilt_angle        = 0.0   # angle (deg) to incl disk spin axis rel. to BH spin in dir of x
potential_beta_min = 100.0  # ratio of gas to magnetic pressure at maxima (diff locations)
potential_cutoff   = 0.2    # amount to subtract from density when calculating potential
potential_rho_pow = 1.0     # dependence of the vector potential on density rho
rho_min   = 1.0e-5    # background on rho given by rho_min ...
rho_pow   = -1.5      # ... * r^rho_pow
pgas_min  = 0.333e-7  # background on p_gas given by pgas_min ...
pgas_pow  = -2.5      # ... * r^pgas_pow
rho_max   = 1.0       # if > 0, rescale rho to have this peak; rescale pres by same factor
l         = 0.0       # const. ang. mom. per unit mass u^t u_phi; only used if r_peak < 0
pert_amp  = 2.0e-2    # perturbation amplitude
user_hist = false      # enroll user-defined history function

<output1>
file_type  = bin     # Binary data dump
variable   = z4c     # variables to be output
slice_x3   = 0
dt         = 5       # time increment between outputs

<output2>
file_type  = bin     # Binary data dump
variable   = con     # variables to be output
slice_x3   = 0
dt         = 5       # time increment between outputs

<output3>
file_type  = bin     # Binary data dump
variable   = mhd_w_bcc     # variables to be output
slice_x3   = 0
dt         = 5       # time increment between outputs

<output4>
file_type  = bin     # Binary data dump
variable   = mhd_w_bcc     # variables to be output
slice_x2   = 0
dt         = 5       # time increment between outputs

<output5>
file_type  = rst    # Binary data dump
dt         = 50.    # time increment between outputs

<output6>
file_type = hst
dt        = 2
"""


class Parfile(object):
    """
    Creates a Z4c two punctures AthenaK parfile
    """

    def __init__(
        self,
        # Grid options
        ncells=128,
        mbsize=32,
        refinement_interval=1,
        bh_0_rad=1.0,
        bh_1_rad=1.0,
        # Other numerical parameters
        cfl=0.25,
        # Initial data (defaults to calibration run)
        par_b = 3.257, # orbital separation in x
        par_P_plus1 = 0,
        par_P_plus2 = -0.133,
        par_P_minus1 = 0,
        par_P_minus2 = 0.133,
    ):
        self.subs = {}

        self.subs["nx"] = ncells
        self.subs["ny"] = ncells
        self.subs["nz"] = ncells
        self.subs["mbsize"] = mbsize
        self.subs["refinement_interval"] = refinement_interval
        self.subs["bh_0_rad"] = bh_0_rad
        self.subs["bh_1_rad"] = bh_1_rad

        self.subs["cfl"] = cfl

        # calculate separation in x
        bh_0_x = par_b
        bh_1_x = -par_b

        self.subs["refinement1_x1min"] = bh_0_x - bh_0_rad
        self.subs["refinement1_x1max"] = bh_0_x + bh_0_rad
        self.subs["refinement2_x1min"] = bh_1_x - bh_1_rad
        self.subs["refinement2_x1max"] = bh_1_x + bh_1_rad

        self.subs["par_b"] = abs(bh_0_x - bh_1_x)/2
        self.subs["center_offset"] = bh_0_x - self.subs["par_b"]

        self.subs["bh_0_x"] = bh_0_x
        self.subs["bh_0_px"] = par_P_plus1
        self.subs["bh_0_py"] = par_P_plus2

        self.subs["bh_1_x"] = bh_1_x
        self.subs["bh_1_px"] = par_P_minus1
        self.subs["bh_1_py"] = par_P_minus2
    def __str__(self):
        return TEMPLATE.format(**self.subs)

if __name__ == "__main__":
    print(Parfile())
