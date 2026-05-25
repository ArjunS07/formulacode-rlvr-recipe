#!/usr/bin/env bash
# Set up formulacode-rlvr on a new machine.
# Run from the formulacode-rlvr/ directory after git clone.
set -euo pipefail

SKYRL_COMMIT="ccc181e2"  # pin to the commit used during training

echo "=== formulacode-rlvr setup ==="

# 1. Clone SkyRL if not already present
if [ ! -d "SkyRL/.git" ]; then
    echo "[1/3] Cloning SkyRL..."
    git clone https://github.com/NovaSky-AI/SkyRL.git SkyRL
    git -C SkyRL checkout "$SKYRL_COMMIT"
else
    echo "[1/3] SkyRL already present ($(git -C SkyRL rev-parse --short HEAD))"
fi

# 2. Overlay our custom formulacode additions
echo "[2/3] Overlaying formulacode additions..."
cp -r skyrl-overlay/examples/ SkyRL/examples/

# 3. Create results directory skeleton
echo "[3/3] Creating results directories..."
mkdir -p results/trials

echo ""
echo "Done. Before running, update TASKS_BASE in the run script:"
echo "  SkyRL/examples/train_integrations/harbor/formulacode/run_1gpu_a100.sh"
echo ""
echo "Tasks directory (harbor-tasks-may18) must be rsynced separately:"
echo "  rsync -av user@source:/mnt/sdd3/asharma/harbor-tasks-may18/ ./harbor-tasks-may18/"
echo "  Then set TASKS_BASE=\"\$PWD/harbor-tasks-may18\" in the run script."
echo ""
echo "To start training:"
echo "  cd SkyRL"
echo "  CUDA_VISIBLE_DEVICES=0 bash examples/train_integrations/harbor/formulacode/run_1gpu_a100.sh"
