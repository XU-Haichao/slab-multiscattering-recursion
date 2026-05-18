#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXE="${ROOT_DIR}/mcrt_ang/mcrt_ang"
OUT_DIR="${ROOT_DIR}/mcrt_ang/output/thermal_kn_table"
TABLE="${ROOT_DIR}/mcrt_ang/output/thermal_kn_transport_table_blackbody_5eV.h5"

mkdir -p "${OUT_DIR}"

"${EXE}" \
  --mode generate-thermal-kn-transport-table \
  --thermal-kn-table "${TABLE}" \
  --output-dir "${OUT_DIR}" \
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

echo "Wrote ${TABLE}"
