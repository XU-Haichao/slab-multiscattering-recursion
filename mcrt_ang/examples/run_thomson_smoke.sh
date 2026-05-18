#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXE="${ROOT_DIR}/mcrt_ang/mcrt_ang"
OUT_DIR="${ROOT_DIR}/mcrt_ang/output/thomson_smoke"
TABLE="${ROOT_DIR}/mcrt_ang/output/thermal_kn_transport_table_blackbody_5eV.h5"

mkdir -p "${OUT_DIR}"

"${EXE}" \
  --mode run \
  --geometry slab \
  --transport-cross-section thomson \
  --thermal-kn-table "${TABLE}" \
  --electron-model thermal \
  --electron-kTe 0.1 \
  --seed-photon-model blackbody \
  --seed-temperature-eV 5.0 \
  --slab-tau 1.0 \
  --slab-injection lambert \
  --events 10000 \
  --max-scatters 80 \
  --energy-bins 200 \
  --energy-bin-spacing log \
  --energy-min 1e-8 \
  --energy-max 10 \
  --mu-bins 50 \
  --label thomson_smoke \
  --output-dir "${OUT_DIR}"

echo "Wrote outputs under ${OUT_DIR}"
