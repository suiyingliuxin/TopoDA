#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export MAX_JOBS="${MAX_JOBS:-4}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$(python -c 'import torch; a,b=torch.cuda.get_device_capability(); print(f"{a}.{b}")')}"
python -m pip install --no-build-isolation --no-deps ./DCNv4_op
python -m pip install --no-build-isolation --no-deps ./models/topoda/ops
