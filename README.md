# slab-multiscattering-recursion

This README was prepared with assistance from GPT-5.5.

This repository contains the minimal reproducibility package for a recursive
formalism for multi-scattering escape probabilities, angular distributions, and
Comptonized spectra in a plane-parallel slab.

The repository is intentionally smaller than the full working directory used to
produce exploratory plots.  The large plotting notebook, generated figures, and
Monte Carlo output tables are not part of the core package.

## Contents

- `theory_recursive_demo.ipynb`  
  A self-contained Python notebook implementing the recursive slab formalism.
  It computes order-resolved escape probabilities, boundary-resolved
  probabilities, angular distributions, spectral radii, approximate photon
  indices, mean scattering counts, and approximate Comptonized spectra.

- `mcrt_ang/`  
  A C++ Monte Carlo code for slab inverse-Compton calculations with a 5 eV
  blackbody seed photon spectrum.  It supports lower-boundary Lambert injection
  and uniform internal isotropic injection, and writes both escaping energy
  histograms and angle-resolved escaping spectra.

- `plot_style.py`  
  Small Matplotlib style helper used by the notebook.

## Python environment

The theory notebook only needs standard scientific Python packages:

```bash
python -m pip install -r requirements.txt
```

Then open:

```bash
jupyter notebook theory_recursive_demo.ipynb
```

## Build the Monte Carlo code

The Monte Carlo code uses HDF5 for thermal Klein-Nishina transport tables.
Set `HDF5_PREFIX` if HDF5 is not installed in the default location:

```bash
make -C mcrt_ang HDF5_PREFIX=/path/to/hdf5
```

On a conda installation, this is often:

```bash
make -C mcrt_ang HDF5_PREFIX="$CONDA_PREFIX"
```

## Thermal-KN transport table

Thermal-KN runs require a precomputed transport table.  A template command is
provided in:

```text
mcrt_ang/examples/generate_thermal_kn_table.sh
```

The same executable also supports Thomson transport through
`--transport-cross-section thomson`.  The current parser still carries a
thermal-KN table path in the configuration, so example commands pass a table
path even when the selected transport mode is Thomson.

## Example Monte Carlo runs

Example shell scripts are provided in:

```text
mcrt_ang/examples/
```

They are meant as transparent command templates rather than a full production
pipeline:

- `run_thomson_smoke.sh` runs a small Thomson slab check.
- `run_blackbody_lambert_spectrum.sh` runs a thermal-KN blackbody slab case.
- `run_angle_resolved_spectra.sh` runs the angle-resolved setup used for the
  angular spectral diagnostics.

Large production runs used in the paper can require many photon histories
(`1e7` in the final high-statistics tests), so the generated CSV files are not
tracked by git.

## What is not included

The following were part of the local exploration workspace but are deliberately
not tracked in this minimal package:

- generated figures,
- large Monte Carlo CSV outputs,
- compiled binaries,
- the exploratory `plot.ipynb`,
- full comparison-table output directories.

If exact archival reproduction of every final figure is required, publish the
large generated CSV tables separately, for example through a release asset,
Zenodo archive, or Git LFS.
