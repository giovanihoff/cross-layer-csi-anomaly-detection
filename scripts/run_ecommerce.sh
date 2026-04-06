#!/usr/bin/env bash
set -euo pipefail
python -m csi_payment_attestation.pipelines.run_ecommerce --config configs/ecommerce_raspberrypi.yaml
