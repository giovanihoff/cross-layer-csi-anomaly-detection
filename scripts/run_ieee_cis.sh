#!/usr/bin/env bash
set -euo pipefail
python -m csi_payment_attestation.pipelines.run_ieee_cis --config configs/ieee_cis_raspberrypi.yaml
