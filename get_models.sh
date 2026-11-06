#!/usr/bin/env bash
# Download the two trained checkpoints from the GitHub release into outputs/runs/.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
base=https://github.com/vinay3254/lumpy-skin-disease-xai/releases/download/v1.0
for run in hybrid_attention_c3 hybrid_concat_c2_v2; do
    mkdir -p "outputs/runs/$run"
    curl -fL --retry 3 -o "outputs/runs/$run/best.pt" "$base/$run.pt"
done
echo "Models ready. Run ./start.sh"
