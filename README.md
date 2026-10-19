# picolo — Latent-Space Logic Reasoning Engine (LLRE)

A token-free, formally grounded neuro-symbolic architecture in **JAX** that decouples **continuous latent reasoning** (JEPA + Sketched Isotropic Gaussian Regularization + Dialectical Energy Clamping) from **bare-metal bit/word-parallel execution** and **formal SMT verification (`UNSAT`)**.

Full architecture specification: [`latent_logic_reasoning_architecture.md`](latent_logic_reasoning_architecture.md).

---

## Five-Tier Architecture

1. **Tier 1: Peripheral I/O Adapter (`llre/io_adapter.py`)** — Translates natural-language problem descriptions or LISP specs into formal synthesis challenges and renders verified circuits into deterministic step-by-step English and homoiconic LISP ASTs.
2. **Tier 2: Continuous Latent Reasoning Core (`llre/latent_core.py`, `llre/sigreg.py`)** — Encodes problem specifications into a continuous latent manifold $\mathbf{Z} \in \mathbb{R}^D$ regularized by $O(N \cdot D)$ **SIGReg** characteristic-function sketching, optimizing trajectories $\mathbf{z}_0 \to \dots \to \mathbf{z}^*$ over Dialectical Energy ($E_{\text{thesis}} + E_{\text{antithesis}}$).
3. **Tier 3: Symbolic Intermediate Logic Language (`llre/lisp_compiler.py`)** — Bidirectional compiler between homoiconic LISP S-expressions `(module ...)`, flat instruction tensors `[B, L, 4]`, and SMT-LIB 2.0 (`QF_BV`) bit-vector scripts, with causal SSA grammar masks.
4. **Tier 4: Bare-Metal JAX Vectorized Logic Sandbox (`llre/isa.py`, `llre/sandbox.py`)** — 16-instruction branchless Orthogonal Logic ISA supporting both 64-way bit-packed Boolean evaluation and vectorized 64-bit ALU execution via `jax.jit` + `jax.vmap` + `jax.lax.scan` + `jax.lax.select_n`.
5. **Tier 5: Formal Solver & CEGIS (`llre/verifier.py`, `llre/self_play.py`)** — Proves universal 64-bit equivalence ($\forall x, \mathcal{C}(x) = \mathcal{S}(x)$) via Z3 (`z3-solver`) or the built-in symbolic bit-vector solver, feeding any counterexample directly back into the Tier 4 Sandbox (CEGIS) and persisting verified circuits in `data/primitive_library.json`.

---

## Cross-OS Quickstart (Windows CPU $\leftrightarrow$ Linux CUDA GPU)

 Runtime configuration is managed via `.env` (copied from `.env.example`) and `llre/config.py`, which automatically detects CUDA GPUs on Linux and falls back to multi-core CPU execution on Windows.

### Linux Personal Laptop (Dedicated NVIDIA GPU)
```bash
git clone https://github.com/retroboydance/picolo.git
cd picolo
./scripts/setup_env.sh
source .venv/bin/activate
python run_llre_demo.py
```

### Windows Work Laptop (CPU)
```powershell
git clone https://github.com/retroboydance/picolo.git
cd picolo
.\scripts\setup_env.ps1
python run_llre_demo.py
```

### Running the Test Suite
```bash
python -m unittest discover -s tests -v
```
