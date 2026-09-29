#!/usr/bin/env bash
# =============================================================================
# Robot Simulation Setup — RTX 4050 Laptop (6 GB VRAM)
# Stack: PyBullet (physics + 3D GUI) + PyTorch + Stable-Baselines3
# Fully isolated conda environment — touches nothing else on your system
# =============================================================================
# Usage:
#   chmod +x setup_robot_sim.sh
#   ./setup_robot_sim.sh
# =============================================================================

set -e
ENV="robot_sim_env"
PY="3.11"

echo "=============================================="
echo "  Robot Sim Setup for RTX 4050 (6 GB VRAM)"
echo "=============================================="
echo ""

# ── 0. Pre-flight ──────────────────────────────────────────────────────────
echo "[0/5] Checking system..."

command -v conda >/dev/null || { echo "ERROR: Install Miniconda first"; exit 1; }
command -v nvidia-smi >/dev/null || { echo "ERROR: NVIDIA driver missing"; exit 1; }

GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
echo "  GPU: $GPU"
echo ""

# ── 1. Isolated conda env ─────────────────────────────────────────────────
echo "[1/5] Creating isolated conda env: $ENV ..."
if conda info --envs | grep -q "^$ENV "; then
    echo "  Removing old env..."
    conda env remove -n "$ENV" -y
fi
conda create -n "$ENV" python="$PY" -y
echo "  ✓ Created"

# ── 2. PyBullet (physics + GUI — uses OpenGL not CUDA, ~0 VRAM) ───────────
echo ""
echo "[2/5] Installing PyBullet (OpenGL GUI, zero VRAM)..."
conda run -n "$ENV" pip install pybullet --quiet
conda run -n "$ENV" pip install pybullet-data --quiet 2>/dev/null || true
echo "  ✓ PyBullet installed"

# ── 3. PyTorch (CUDA 12.1 — conservative for 4050) ───────────────────────
echo ""
echo "[3/5] Installing PyTorch (CUDA 12.1)..."
conda run -n "$ENV" pip install \
    torch==2.2.2 \
    torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cu121 \
    --quiet
echo "  ✓ PyTorch installed"

# ── 4. RL + visualisation libs ───────────────────────────────────────────
echo ""
echo "[4/5] Installing RL + visualisation packages..."
conda run -n "$ENV" pip install \
    stable-baselines3==2.3.0 \
    gymnasium==0.29.1 \
    numpy \
    matplotlib \
    opencv-python \
    open3d \
    rich \
    --quiet
echo "  ✓ All packages installed"

# ── 5. Create demo folder ─────────────────────────────────────────────────
echo ""
echo "[5/5] Creating demo scripts..."
mkdir -p robot_sim_demo
echo "  ✓ robot_sim_demo/ created"

echo ""
echo "=============================================="
echo "  Setup complete!"
echo ""
echo "  Run the simulation:"
echo "    conda activate $ENV"
echo "    python robot_sim_demo/franka_sim.py"
echo ""
echo "  Run RL training:"
echo "    python robot_sim_demo/rl_reach.py"
echo "=============================================="
