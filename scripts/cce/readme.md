# Background

To compute the CCE strains, we require the spherical harmonics coefficients 
of the ADM variables on a nested set of spherical shells at various time steps.

For the spherical harmonics decomposition, we use an angular grid with Gauss-Legendre 
points in $cos(\theta)$ and uniformly spaced points in $\phi$.

The radii of the nested shells are chosen to lie at Chebyshev collocation
points (the roots of Chebyshev polynomials of the second kind) 
distributed between the innermost and outermost shells.
The ADM variables are interpolated onto these radial collocation points.
This allows the Python script `athk_to_spectre.py` to compute radial derivatives 
of the variables at any specified extraction radius.

Finally, the spherical harmonic coefficients and the radially interpolated 
values of ADM variables are written to a binary format file (`.bin`) 
to be used in world tube boundary data of the `Spectre CCE` code.
---

# Requirements

1. [`Spectre CCE` executable](https://spectre-code.org/tutorial_cce.html)  
2. [`scri` (BMS transformation package)](https://scri.readthedocs.io/en/latest/README.html)  
3. Python packages:  
   - `numpy`  
   - `h5py`  
   - `scipy`  
   - `matplotlib`  

---

# Compute CCE Strains using `Spectre CCE`

### Step 1: Convert Boundary Data to Spectre Format

First, convert the binary boundary condition data into a format readable by 
`Spectre CCE` using the script `athk_to_spectre.py`.

Before launching the script, identify the `dump_extraction_radius`, 
which can be found in the `AthenaK` parameter file. 
This radius is typically set midway between the inner and outer radii 
of the shell where the ADM data are interpolated.

```bash
# Convert binary dump data to Spectre-readable format:
./athk_to_spectre.py -ftype bin -radius [dump_extraction_radius] -d_out [output_directory]
```

### Step 2: Run `Spectre CCE`

After conversion, the output directory will contain an `.h5` file, 
named like `CceRXXXX.h5`, where `XXXX` denotes the extraction radius.

Set the `CharacteristicExtract.yaml` config file as follows:

```yaml
BoundaryDataFilename: path/to/CceRXXXX.h5
H5IsBondiData: false  # Since the data is not in Bondi format
```

Run `Spectre CCE` with:

```bash
./CharacteristicExtract --input-file CharacteristicExtract.yaml
```

For more on the Bondi format, refer to the 
[Spectre CCE tutorial](https://spectre-code.org/tutorial_cce.html#autotoc_md58).

---

# BMS Transformation

The output from `Spectre CCE` is not in the superrest frame. To transform the waveform, apply a
_BMS transformation_ using `scri`. For more detail see:

- [Frame fixing](https://spectre-code.org/tutorial_cce.html#autotoc_md65)
- [Scri tutorial](https://scri.readthedocs.io/en/latest/tutorial_abd.html#loading-cce-data-and-adjusting-the-bms-frame)

Example:

```python
import scri
import matplotlib.pyplot as plt
import numpy as np

# Load and transform waveform to the superrest frame
w = scri.WaveformModes()
abd = scri.create_abd_from_h5(
    file_name="/path/to/CharacteristicExtractReduction.h5",
    file_format="spectrecce_v1",
    ch_mass=1.0,
    t_0_superrest=t0,  # time after junk radiation
    padding_time=w0    # ~2 orbital periods
)
h = abd.h

# Plot h_22 mode
plt.plot(h.t, h.data[:, h.index(2, 2)])

# Access real and imaginary parts
re = h.data[:, h.index(2, 2)].real
im = h.data[:, h.index(2, 2)].imag
```

---

# Debugging Tools

Use the `debug_athk_to_spectre.py` script to analyze and debug 
the output of `athk_to_spectre.py`. You can inspect specific 
spherical harmonic modes over time, or examine convergence and derivatives.

Example: Plot the real part of the $(2,2)$ mode of $g_{xx}(t)$

```bash
./debug_athk_to_spectre.py -debug plot_simple \
  -dout [output_directory] \
  -fpath [/path/to/CceRXXXX.h5] \
  -field_name "gxx" -field_mode "Re(2,2)"
```
