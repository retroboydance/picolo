# GEMINI.md — Agent Context & Engineering Guide for `picolo` (LLRE)

This file provides architectural context, invariants, and cross-platform development rules for working on the **Latent-Space Logic Reasoning Engine (LLRE)** across both the **Windows Work Laptop (CPU)** and **Linux Personal Laptop (Dedicated NVIDIA CUDA GPU)**.

---

## 1. Project Overview & Core Paradigm

**LLRE** (specified in [`latent_logic_reasoning_architecture.md`](latent_logic_reasoning_architecture.md)) decouples **internal reasoning** from **serial token generation**:
- **Substrate of Thought (Tier 2):** Continuous latent representations ($\mathbf{z} \in \mathbb{R}^D$) governed by a Joint-Embedding Predictive Architecture (JEPA), **Sketched Isotropic Gaussian Regularization (SIGReg)**, and **Dialectical Energy Clamping** ($E_{\text{thesis}} + E_{\text{antithesis}}$).
- **Symbolic Pivot (Tier 3):** Homoiconic LISP S-expressions `(module ...)` compiled deterministically in $O(L)$ to flat fixed-width instruction tensors $\mathbf{P} \in \mathbb{N}^{B \times L \times 4}$ (`uint16`) and `QF_BV` SMT-LIB 2.0 scripts.
- **Bare-Metal Execution (Tier 4):** Branchless JAX/XLA kernels (`jax.jit` + `jax.vmap` + `jax.lax.scan` + `jax.lax.select_n`) supporting both **64-way bit-packed Boolean evaluation** and **vectorized 64-bit word ALU evaluation**, returning exact bitwise Hamming distance (`jax.lax.population_count`) + gate-economy rewards.
- **Formal Verification & CEGIS (Tier 5):** Proves universal 64-bit equivalence ($\forall x, \mathcal{C}(x) = \mathcal{S}(x)$) via `UNSAT` of $\mathcal{C}(x) \neq \mathcal{S}(x)$ using `z3-solver` (when installed) or the built-in symbolic bit-vector solver (`symbolic-bv64`), feeding any `SAT` counterexample directly back into the Tier 4 Sandbox test suite.
- **Edge I/O Peripheral (Tier 1):** Natural language is treated strictly as an I/O peripheral (`llre/io_adapter.py`), mapping queries to formal specifications and rendering verified ASTs into deterministic step-by-step English explanations.

---

## 2. Repository Structure & Module Map

```text
picolo/
├── .env.example                        # Cross-OS environment template (copy to .env)
├── .gitattributes                      # Enforces LF on .py/.sh/.json/.toml, CRLF on .ps1
├── .gitignore                          # Ignores .env, .venv/, .jax_cache/, checkpoints/
├── GEMINI.md                           # This agent guide & architectural reference
├── README.md                           # Quickstart documentation
├── latent_logic_reasoning_architecture.md # Original 5-tier LLRE architecture specification
├── pyproject.toml                      # Package metadata & optional [gpu,smt,dev] extras
├── run_llre_demo.py                    # End-to-end showcase, benchmark & interactive REPL CLI
├── train_self_play.py                  # Self-supervised training loop, curriculum expansion & validation
├── data/
│   └── primitive_library.json          # Git-tracked persistent library of UNSAT-proved circuits
├── scripts/
│   ├── setup_env.ps1                   # Windows PowerShell bootstrap script
│   └── setup_env.sh                    # Linux Bash bootstrap script (auto-detects nvidia-smi -> jax[cuda12])
├── llre/
│   ├── __init__.py                     # Public package exports
│   ├── config.py                       # Zero-dependency .env loader & JAX x64 / CPU / GPU bootstrapper
│   ├── isa.py                          # 16-opcode Orthogonal Logic ISA & 16-register bank layout
│   ├── sandbox.py                      # Tier 4 JAX vectorized bit-packed & 64-bit ALU execution kernels
│   ├── lisp_compiler.py                # Tier 3 LISP parser, bidirectional [L, 4] compiler, DCE & SMT-LIB2
│   ├── verifier.py                     # Tier 5 Z3 + symbolic-bv64 equivalence prover & CEGIS extractor
│   ├── sigreg.py                       # Tier 2 O(N*D) SIGReg regularizer & Effective Dimensionality (RankMe)
│   ├── latent_core.py                  # Tier 2 JEPA predictor, Dialectical Energy, Trajectory GD, AdamW
│   ├── self_play.py                    # Section 6 Asymmetric Self-Play (Levels 1-3, 16 tasks) & validation
│   ├── checkpoint.py                   # Portable .npz PyTree + AdamW opt_state checkpoint manager
│   ├── metrics.py                      # Section 7 Observability dashboard & shallow linear diagnostic probes
│   └── io_adapter.py                   # Tier 1 Natural Language <-> Verified LISP AST & English renderer
└── tests/
    └── test_llre_suite.py              # Automated unit & integration tests covering Phases 0-4
```

---

## 3. Critical Architectural Invariants (MUST PRESERVE)

1. **Import `llre.config` Before `jax` in Entrypoints:**
   - `llre/config.py` loads `.env`, sets `JAX_ENABLE_X64=True`, configures persistent XLA compilation caching (`.jax_cache`), and selects the hardware platform (`cpu` vs `gpu`) **before** JAX initializes XLA.
   - `jax_enable_x64=True` is mandatory because Tier 4 registers are 64-bit unsigned integers (`jnp.uint64`).

2. **16-Register Bank Partitioning (`llre/isa.py`):**
   - `R0..R3` (`0..3`): Read-only problem input registers (`x`, `y`, `z`, `w`).
   - `R4..R10` (`4..10`): Mutable scratch registers (`t0..t6`).
   - `R11` (`11`): Primary output register (`OUT_REG = 11`).
   - `R12..R15` (`12..15`): Read-only hardware constant registers:
     - `R12 = 0` (`REG_ZERO`)
     - `R13 = 1` (`REG_ONE` in 64-bit ALU mode; `0xFFFFFFFFFFFFFFFF` in 64-way bit-packed Boolean mode)
     - `R14 = 2` (`REG_TWO`)
     - `R15 = 0xFFFFFFFFFFFFFFFF` (`REG_ALL_ONES`, i.e., `~0`)
   - Writes to `dest >= 12` or with `opcode == Opcode.NOP (0x00)` are masked out in hardware so constant registers can never be corrupted.

3. **Ternary `SELECT` (`0x32`) Encoding in 4-Tuple `[Opcode, Reg_Dest, Reg_Src1, Reg_Src2]`:**
   - Use `encode_select_operands(cond_reg, true_reg, false_reg)` and `decode_select_operands(s1, s2)` in `llre/isa.py`.
   - `Reg_Src1` stores `true_reg`; `Reg_Src2` packs `((cond_reg + 1) << 4) | false_reg` (if the upper nibble is `0`, it falls back to legacy spec mode where `s1=cond`, `s2=true`, `R0=false`).

4. **Full 64-Bit Output Verification (`llre/sandbox.py`, `llre/verifier.py`):**
   - `evaluate_alu64` and `to_smtlib2` compare all 64 bits of `regs[OUT_REG]`. Circuits with 1-bit boolean outputs (like `parity_checker`) must explicitly clear upper bits `1..63` in the circuit (e.g., `(bit-and x 1)` followed by `(bit-xor t0 1)`).

5. **Active-Step Cross-Entropy Masking & Dialectical Attractor (`llre/latent_core.py`):**
   - In `train_latent_core_step`, operand logits (`dest`, `s1`, `s2`, `cond`) are masked by `active_mask = (target_ops != 0)` so `NOP` padding steps never evaluate cross-entropy against grammar-masked `-1e9` entries.
   - `compute_dialectical_energy` anchors the zero-gradient minimum at `z_hat = predict_jepa_target(params, c)` so continuous latent trajectory optimization (`optimize_latent_trajectory`) monotonically descends into the decoded attractor basin.

6. **Cross-OS File & Git Hygiene:**
   - Always write JSON/JSONL/text files with `encoding="utf-8", newline="\n"` and resolve paths via `pathlib.Path` relative to `PROJECT_ROOT`.

---

## 4. Common Commands (Linux GPU & Windows CPU)

### Setup
- **Linux (CUDA GPU):**
  ```bash
  ./scripts/setup_env.sh
  source .venv/bin/activate
  ```
- **Windows (PowerShell):**
  ```powershell
  python -m llre.config
  ```

### Self-Supervised Training, Dynamic Curriculum Expansion & Checkpointing
```bash
# Train from scratch across Curriculum Levels 1 -> 2 -> 3 (16 tasks)
python train_self_play.py --epochs 60 --val-every 10

# Resume from checkpoints/llre_latest.npz (restores weights, AdamW opt_state, and curriculum level)
python train_self_play.py --resume --epochs 60 --val-every 10
```

### Demo, Benchmarks & Interactive REPL
```bash
# Full 5-tier showcase + Section 7 Observability Dashboard
python run_llre_demo.py

# Interactive mode (accepts natural language, task names, or raw LISP `(module ...)` specs)
python run_llre_demo.py --interactive --smt2

# Single query with SMT-LIB 2.0 proof output
python run_llre_demo.py --query "Clear the two lowest set bits of x" --smt2

# Tier 4 Sandbox throughput benchmark only
python run_llre_demo.py --benchmark-only
```

### Automated Test Suite
```bash
python -m unittest discover -s tests -v
```
