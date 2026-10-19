"""
Cross-OS & Cross-Hardware Configuration Loader for LLRE.

Handles seamless interoperability between:
- Windows Work Laptop (Multi-core CPU execution)
- Linux Personal Laptop (Dedicated NVIDIA CUDA GPU execution)

IMPORTANT: This module configures XLA/JAX environment variables and 64-bit precision
BEFORE JAX initializes its device backend.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import platform
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv_file(dotenv_path: Optional[Path] = None, override: bool = False) -> Dict[str, str]:
    """
    Zero-dependency `.env` file parser that works identically on Windows (CRLF/LF)
    and Linux (LF), without requiring external packages to be pre-installed.
    """
    path = dotenv_path or (PROJECT_ROOT / ".env")
    loaded: Dict[str, str] = {}
    if not path.exists():
        fallback = PROJECT_ROOT / ".env.example"
        if fallback.exists():
            path = fallback
        else:
            return loaded

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        # Strip inline comments if unquoted
        if " #" in val and not (val.startswith('"') or val.startswith("'")):
            val = val.split(" #", 1)[0].strip()
        # Strip surrounding quotes
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
            val = val[1:-1]
        loaded[key] = val
        if override or key not in os.environ:
            os.environ[key] = val
    return loaded


def _bootstrap_jax_env() -> None:
    """Configure XLA/JAX environment variables before importing `jax`."""
    load_dotenv_file()

    # 1. Enable 64-bit precision (critical for 64-way bit-parallel uint64 registers)
    enable_x64 = os.environ.get("LLRE_ENABLE_X64", "1").strip().lower() in ("1", "true", "yes")
    if enable_x64:
        os.environ["JAX_ENABLE_X64"] = "True"

    # 2. Configure XLA persistent compilation cache directory (OS-agnostic path)
    cache_dir_str = os.environ.get("LLRE_XLA_CACHE_DIR", ".jax_cache").strip()
    if cache_dir_str:
        cache_dir = Path(cache_dir_str)
        if not cache_dir.is_absolute():
            cache_dir = PROJECT_ROOT / cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(cache_dir))

    # 3. Configure explicit backend if not "auto"
    requested_backend = os.environ.get("LLRE_BACKEND", "auto").strip().lower()
    if requested_backend in ("cpu", "gpu", "tpu"):
        os.environ["JAX_PLATFORMS"] = requested_backend


# Bootstrap environment before importing JAX
_bootstrap_jax_env()

import jax  # noqa: E402

if os.environ.get("JAX_ENABLE_X64", "True") == "True":
    jax.config.update("jax_enable_x64", True)


def detect_active_backend() -> tuple[str, Any]:
    """
    Detect the best available JAX device across Linux (CUDA GPU) and Windows (CPU).
    Falls back gracefully to CPU if a requested GPU backend is unavailable.
    """
    requested = os.environ.get("LLRE_BACKEND", "auto").strip().lower()
    if requested in ("gpu", "auto"):
        try:
            gpu_devices = jax.devices("gpu")
            if gpu_devices:
                return "gpu", gpu_devices[0]
        except RuntimeError:
            pass
    cpu_devices = jax.devices("cpu")
    return "cpu", cpu_devices[0]


@dataclass(frozen=True)
class LLREConfig:
    """Resolved runtime configuration for LLRE across Windows & Linux."""

    os_platform: str
    os_release: str
    active_backend: str
    device_description: str
    enable_x64: bool
    batch_size: int
    max_steps: int
    num_alu_tests: int
    latent_dim: int
    sigreg_projections: int
    sigreg_weight: float
    trajectory_steps: int
    trajectory_lr: float
    random_seed: int
    smt_backend: str
    z3_timeout_ms: int
    primitive_lib_path: Path
    xla_cache_dir: Path

    @classmethod
    def from_env(cls, dotenv_path: Optional[Path] = None) -> "LLREConfig":
        if dotenv_path is not None:
            load_dotenv_file(dotenv_path, override=True)

        backend_name, device = detect_active_backend()
        enable_x64 = os.environ.get("LLRE_ENABLE_X64", "1").strip().lower() in ("1", "true", "yes")

        # Auto-scale batch size based on whether we are on Linux GPU vs Windows CPU
        raw_batch = os.environ.get("LLRE_BATCH_SIZE", "auto").strip().lower()
        if raw_batch == "auto":
            batch_size = 65536 if backend_name == "gpu" else 4096
        else:
            batch_size = int(raw_batch)

        # Auto-detect SMT backend (z3 vs built-in symbolic bit-vector verifier)
        raw_smt = os.environ.get("LLRE_SMT_BACKEND", "auto").strip().lower()
        if raw_smt == "auto":
            try:
                import z3  # noqa: F401
                smt_backend = "z3"
            except ImportError:
                smt_backend = "symbolic"
        else:
            smt_backend = raw_smt

        prim_rel = os.environ.get("LLRE_PRIMITIVE_LIB_PATH", "data/primitive_library.json").strip()
        prim_path = Path(prim_rel) if Path(prim_rel).is_absolute() else (PROJECT_ROOT / prim_rel)

        cache_rel = os.environ.get("LLRE_XLA_CACHE_DIR", ".jax_cache").strip()
        cache_path = Path(cache_rel) if Path(cache_rel).is_absolute() else (PROJECT_ROOT / cache_rel)

        return cls(
            os_platform=platform.system(),
            os_release=platform.release(),
            active_backend=backend_name,
            device_description=f"{device.platform}:{device.device_kind} (id={device.id})",
            enable_x64=enable_x64,
            batch_size=batch_size,
            max_steps=int(os.environ.get("LLRE_MAX_STEPS", "16")),
            num_alu_tests=int(os.environ.get("LLRE_NUM_ALU_TESTS", "64")),
            latent_dim=int(os.environ.get("LLRE_LATENT_DIM", "64")),
            sigreg_projections=int(os.environ.get("LLRE_SIGREG_PROJECTIONS", "64")),
            sigreg_weight=float(os.environ.get("LLRE_SIGREG_WEIGHT", "0.1")),
            trajectory_steps=int(os.environ.get("LLRE_TRAJECTORY_STEPS", "12")),
            trajectory_lr=float(os.environ.get("LLRE_TRAJECTORY_LR", "0.08")),
            random_seed=int(os.environ.get("LLRE_RANDOM_SEED", "42")),
            smt_backend=smt_backend,
            z3_timeout_ms=int(os.environ.get("LLRE_Z3_TIMEOUT_MS", "2000")),
            primitive_lib_path=prim_path,
            xla_cache_dir=cache_path,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["primitive_lib_path"] = str(self.primitive_lib_path)
        d["xla_cache_dir"] = str(self.xla_cache_dir)
        return d

    def summary(self) -> str:
        lines = [
            "============================================================",
            " LLRE Cross-Platform Runtime Configuration",
            "============================================================",
            f" OS Platform       : {self.os_platform} ({self.os_release})",
            f" Active Backend    : {self.active_backend.upper()} [{self.device_description}]",
            f" 64-Bit Precision  : {self.enable_x64} (JAX x64={jax.config.jax_enable_x64})",
            f" Sandbox Batch (B) : {self.batch_size:,} programs/batch",
            f" Max Circuit Depth : {self.max_steps} instructions [B, L, 4]",
            f" Latent Dim (D)    : {self.latent_dim} (SIGReg projections={self.sigreg_projections})",
            f" Formal Verifier   : {self.smt_backend.upper()}",
            f" Primitive Library : {self.primitive_lib_path}",
            "============================================================",
        ]
        return "\n".join(lines)


def get_config() -> LLREConfig:
    """Return the active LLRE configuration."""
    return LLREConfig.from_env()


if __name__ == "__main__":
    cfg = get_config()
    print(cfg.summary())
