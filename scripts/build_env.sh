#!/usr/bin/env bash
# Build the single cspflow environment: mattergen + mattersim + cspflow.
#
# Why one env and why these pins:
#   mattergen declares `mattersim>=1.1` as a dependency, so one environment is
#   the intended arrangement, not a compromise.  Its pins are hard, though:
#   torch==2.2.1+cu118, numpy<2.0, pytorch-lightning==2.0.6.  The existing
#   `mattersim` env has torch 2.10.0+cu128 and numpy 2.2.6, so it cannot be
#   extended in place -- hence a new env rather than a modification.
#   Python 3.10 because mattersim >=1.2.4 requires 3.12 while mattergen needs
#   numpy<2; 1.2.3 is the newest that supports 3.10.
#
# torch's cu118 wheels bundle their own CUDA runtime, so they run fine against
# the cluster's newer driver (module load cuda/12.8).
# NOTE: no `set -u`.  Conda's activate.d hooks are not written to be safe under
# it -- this machine's julia_activate.sh references an unbound JULIA_DEPOT_PATH
# and aborts the whole script at `conda activate`.
set -eo pipefail

ENV_NAME="${1:-cspflow}"
PY=3.10
TORCH=2.2.1
CUDA=cu118
LOG_PREFIX="[build_env]"

log() { echo "$LOG_PREFIX $*"; }

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    log "env '$ENV_NAME' already exists -- reusing it"
else
    log "creating env '$ENV_NAME' (python $PY)"
    conda create -y -n "$ENV_NAME" "python=$PY" pip
fi

conda activate "$ENV_NAME"
log "python: $(python -V)  at $(which python)"

log "step 1/5: torch $TORCH+$CUDA"
pip install --no-cache-dir \
    "torch==${TORCH}+${CUDA}" "torchvision==0.17.1+${CUDA}" "torchaudio==${TORCH}+${CUDA}" \
    --index-url "https://download.pytorch.org/whl/${CUDA}"

log "step 2/5: torch geometric companions (prebuilt against torch ${TORCH}+${CUDA})"
pip install --no-cache-dir torch_scatter torch_sparse torch_cluster \
    -f "https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html"

log "step 3/5: mattergen (pulls mattersim, numpy<2, pytorch-lightning)"
pip install --no-cache-dir mattergen

log "step 4/5: cspflow itself"
pip install --no-cache-dir -e /projects/mmi/Ridwan/cspflow

log "step 5/5: verification"
python - <<'PY'
import importlib, sys
ok = True
for name in ("torch", "numpy", "ase", "pymatgen", "mattersim", "mattergen", "cspflow"):
    try:
        m = importlib.import_module(name)
        print(f"  {name:<12} {getattr(m, '__version__', 'present')}")
    except Exception as exc:
        ok = False
        print(f"  {name:<12} FAILED: {type(exc).__name__}: {exc}")
import torch
print(f"  torch.cuda    compiled={torch.version.cuda} available={torch.cuda.is_available()}")
print("  (cuda unavailable on a login node is expected)")
sys.exit(0 if ok else 1)
PY

log "done. activate with: conda activate $ENV_NAME"
