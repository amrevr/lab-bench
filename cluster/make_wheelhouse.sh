#!/usr/bin/env bash
# Run on a machine WITH internet (not the cluster). Downloads wheels for the cluster's Python so the
# cluster can install offline. Several GB. Match PYVER/PLAT to the cluster.
#   PYVER=3.9 VLLM_SPEC="vllm==0.8.5" ./make_wheelhouse.sh && tar cf wheelhouse.tar wheelhouse && scp wheelhouse.tar cluster:
set -euo pipefail
PYVER="${PYVER:-3.9}"; PLAT="${PLAT:-manylinux_2_28_x86_64}"
mkdir -p wheelhouse
python3 -m pip download "${VLLM_SPEC:-vllm}" --dest wheelhouse --python-version "$PYVER" \
  --only-binary=:all: --platform "$PLAT" --platform manylinux2014_x86_64 --platform manylinux1_x86_64 --platform linux_x86_64
