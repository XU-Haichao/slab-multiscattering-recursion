#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXE="${ROOT_DIR}/mcrt_ang/mcrt_ang"
TABLE="${ROOT_DIR}/mcrt_ang/output/thermal_kn_transport_table_blackbody_5eV.h5"

for injection in lambert internal_iso; do
  OUT_DIR="${ROOT_DIR}/mcrt_ang/output/angle_theta0p1_tau1_${injection}"
  mkdir -p "${OUT_DIR}"
  "${EXE}" \
    --mode run \
    --geometry slab \
    --transport-cross-section thermal_kn \
    --thermal-kn-table "${TABLE}" \
    --electron-model thermal \
    --electron-kTe 0.1 \
    --seed-photon-model blackbody \
    --seed-temperature-eV 5.0 \
    --slab-tau 1.0 \
    --slab-injection "${injection}" \
    --events 200000 \
    --max-scatters 120 \
    --energy-bins 400 \
    --energy-bin-spacing log \
    --energy-min 1e-8 \
    --energy-max 10 \
    --mu-bins 60 \
    --label "ang_${injection}_t0p1_tau1" \
    --output-dir "${OUT_DIR}"
  echo "Wrote outputs under ${OUT_DIR}"
done
