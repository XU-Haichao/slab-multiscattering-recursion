# mcrt_ang

This directory contains the Monte Carlo code used for the blackbody seed-photon
and angle-resolved slab spectrum checks.

It writes one angle-resolved escaping spectrum table per run:

```text
<prefix>_angspec.csv
```

The angle windows are measured from the local outward normal of the escaping
surface and are fixed to 0--5 deg, 25--35 deg, and 55--65 deg.  Both upper and
lower escaping surfaces are included.  The spectra include all escaping photon
histories, including zero-scatter photons.

## Seed photon spectrum

The default seed photon model in this copy is

```text
--seed-photon-model blackbody
--seed-temperature-eV 5.0
```

Photon histories are sampled from the blackbody photon-number spectrum,

```text
p(x) dx ∝ x^2 / (exp(x) - 1) dx,   x = E / kT_seed.
```

For `T_seed = 5 eV`, this gives

```text
kT_seed / (m_e c^2) = 9.7848e-6
<E_seed> / (m_e c^2) = 2.6430e-5
```

`--seed-photon-model monoenergetic` is still available for debugging.

## Build

The code uses HDF5 for the thermal-KN transport table.  If HDF5 is installed in
the active conda environment, this is usually enough:

```bash
make -C mcrt_ang
```

Override `HDF5_PREFIX` if needed:

```bash
make -C mcrt_ang HDF5_PREFIX=/path/to/hdf5
```

## Thermal-KN transport table

One suitable output path is:

```text
mcrt_ang/output/thermal_kn_transport_table_blackbody_5eV.h5
```

The command template is also available as
`mcrt_ang/examples/generate_thermal_kn_table.sh`.  Explicitly, it runs:

```bash
./mcrt_ang/mcrt_ang \
  --mode generate-thermal-kn-transport-table \
  --thermal-kn-table mcrt_ang/output/thermal_kn_transport_table_blackbody_5eV.h5 \
  --output-dir mcrt_ang/output/thermal_kn_table \
  --seed-photon-model blackbody \
  --seed-temperature-eV 5.0 \
  --thermal-kn-energy-min 1e-8 \
  --thermal-kn-energy-max 10 \
  --thermal-kn-theta-min 1e-3 \
  --thermal-kn-theta-max 10 \
  --thermal-kn-energy-points 120 \
  --thermal-kn-theta-points 80 \
  --thermal-kn-z-points 64 \
  --thermal-kn-mu-points 64 \
  --thermal-kn-z-max 60
```

The table can be used in thermal slab runs with
`--transport-cross-section thermal_kn --electron-model thermal`.

## Example runs

The `examples/` directory contains command templates for:

- a small Thomson slab smoke test,
- a thermal-KN blackbody Lambert run,
- angle-resolved blackbody spectrum runs for Lambert and internal isotropic
  seed injection.
