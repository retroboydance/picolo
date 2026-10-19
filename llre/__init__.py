"""
Latent-Space Logic Reasoning Engine (LLRE).

A Token-Free, Grounded Paradigm for Native Silicon & Logic Intelligence:
- Tier 1: Peripheral I/O Adapter (`llre.io_adapter`)
- Tier 2: Continuous Latent Reasoning Core with JEPA & SIGReg (`llre.latent_core`, `llre.sigreg`)
- Tier 3: Symbolic Intermediate Logic Language (`llre.lisp_compiler`)
- Tier 4: Bare-Metal JAX Vectorized Logic Sandbox (`llre.sandbox`, `llre.isa`)
- Tier 5: Formal Equivalence Solver (`llre.verifier`)
"""

from llre.checkpoint import load_checkpoint, save_checkpoint
from llre.config import LLREConfig, get_config
from llre.io_adapter import PeripheralIOAdapter
from llre.isa import NUM_OPCODES, NUM_REGS, OUT_REG, Opcode
from llre.lisp_compiler import LispCompiler, LispModule, parse_lisp_module
from llre.metrics import ObservabilityReport, build_observability_report
from llre.sandbox import LogicSandbox, SandboxResult
from llre.self_play import (
    LogicChallenge,
    PrimitiveLibrary,
    SelfPlayEngine,
    VerifiedPrimitive,
    build_core_curriculum,
)
from llre.sigreg import compute_effective_dimensionality, sigreg_loss
from llre.verifier import FormalVerifier, VerificationResult

__version__ = "0.1.0"

__all__ = [
    "LLREConfig",
    "get_config",
    "save_checkpoint",
    "load_checkpoint",
    "Opcode",
    "NUM_OPCODES",
    "NUM_REGS",
    "OUT_REG",
    "LogicSandbox",
    "SandboxResult",
    "LispCompiler",
    "LispModule",
    "parse_lisp_module",
    "FormalVerifier",
    "VerificationResult",
    "sigreg_loss",
    "compute_effective_dimensionality",
    "SelfPlayEngine",
    "PrimitiveLibrary",
    "LogicChallenge",
    "VerifiedPrimitive",
    "build_core_curriculum",
    "ObservabilityReport",
    "build_observability_report",
    "PeripheralIOAdapter",
]
