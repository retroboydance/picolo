"""
Cross-OS Checkpoint & Training Metrics Manager (`llre/checkpoint.py`).

Saves and restores pure-JAX PyTree model parameters (`params`) using portable compressed
NumPy `.npz` archives (100% interoperable between Windows CPU and Linux CUDA GPU),
alongside JSON metadata and `.jsonl` validation curves.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple

from llre.config import PROJECT_ROOT
from llre.latent_core import PyTree
import jax.numpy as jnp
import numpy as np

DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


def save_checkpoint(
    params: PyTree,
    checkpoint_path: Path | str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save JAX PyTree parameters (`Dict[str, Dict[str, jnp.ndarray]]`) to a portable `.npz`
    file and write companion metadata JSON.
    """
    path = Path(checkpoint_path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)

    flat_arrays: Dict[str, np.ndarray] = {}
    for module_name, layer_dict in params.items():
        for param_name, tensor in layer_dict.items():
            key = f"{module_name}__{param_name}"
            flat_arrays[key] = np.asarray(tensor, dtype=np.float32)

    np.savez_compressed(path, **flat_arrays)

    if metadata is not None:
        meta_payload = dict(metadata)
        meta_payload["saved_at_unix"] = time.time()
        meta_payload["checkpoint_file"] = path.name
        meta_path = path.with_suffix(".json")
        meta_path.write_text(json.dumps(meta_payload, indent=2) + "\n", encoding="utf-8", newline="\n")

    return path


def load_checkpoint(
    checkpoint_path: Path | str,
) -> Tuple[PyTree, Dict[str, Any]]:
    """
    Load JAX PyTree parameters from a `.npz` archive and return `(params, metadata)`.
    """
    path = Path(checkpoint_path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    params: PyTree = {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            if "__" not in key:
                continue
            module_name, param_name = key.split("__", 1)
            params.setdefault(module_name, {})[param_name] = jnp.asarray(data[key], dtype=jnp.float32)

    meta_path = path.with_suffix(".json")
    metadata: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}

    return params, metadata


def append_metrics_log(
    log_path: Path | str,
    record: Dict[str, Any],
) -> None:
    """Append a single epoch/validation metrics dictionary to a `.jsonl` log file."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(line)
