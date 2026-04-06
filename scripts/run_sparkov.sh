#!/usr/bin/env bash
set -euo pipefail
python -m csi_payment_attestation.pipelines.run_sparkov --config configs/sparkov_raspberrypi.yaml
