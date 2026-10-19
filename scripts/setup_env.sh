#!/usr/bin/env bash
# Linux Personal Laptop Environment Setup (CUDA GPU / CPU Auto-Detect)
set -euo pipefail

echo "=== LLRE Setup: Linux ==="

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Created .env from .env.example"
fi

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment (.venv)..."
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip

if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected via nvidia-smi! Installing JAX with CUDA 12 support..."
    pip install -e ".[gpu,smt,dev]"
else
    echo "No NVIDIA GPU detected; installing CPU dependencies..."
    pip install -e ".[smt,dev]"
fi

echo "Verifying LLRE runtime configuration..."
python -m llre.config
