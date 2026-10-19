"""
Comprehensive Unit & Integration Test Suite for LLRE (Phases 0 through 4).

Tests:
1. Phase 0: Cross-OS `.env` loading, `jax_enable_x64=True`, and hardware auto-scaling.
2. Phase 1: 16-Instruction Orthogonal Logic ISA, 64-way bit-packed Boolean kernel,
   and vectorized 64-bit ALU kernel.
3. Phase 2: Homoiconic LISP S-expression parser, bidirectional `LISP <-> [L, 4]` compiler,
   Dead-Code Elimination, SMT-LIB 2.0 generation, and Tier 5 Formal Verifier (`UNSAT` & `SAT` CEGIS).
4. Phase 3: SIGReg O(N*D) characteristic-function regularization, Effective Dimensionality,
   Dialectical Energy Trajectory descent, Grammar-Constrained decoding, and Self-Play loop.
5. Phase 4: Peripheral I/O Adapter and Section 7 Observability & Diagnostic Linear Probes.
"""

from __future__ import annotations

import unittest

from llre.config import get_config, load_dotenv_file
from llre.io_adapter import PeripheralIOAdapter
from llre.isa import (
    OUT_REG,
    Opcode,
    REG_ALL_ONES,
    REG_ONE,
    REG_ZERO,
    encode_select_operands,
    make_instruction,
    pad_program,
)
from llre.latent_core import (
    encode_condition,
    extract_spec_features,
    init_latent_core_params,
    optimize_latent_trajectory,
    sample_programs_from_latent,
)
from llre.lisp_compiler import LispCompiler, parse_lisp_module
from llre.metrics import build_observability_report
from llre.sandbox import LogicSandbox
from llre.self_play import SelfPlayEngine, build_core_curriculum
from llre.sigreg import compute_effective_dimensionality, sigreg_loss
from llre.verifier import FormalVerifier
import jax
import jax.numpy as jnp
import numpy as np


class TestLLREArchitecture(unittest.TestCase):
    def test_phase0_cross_os_config(self) -> None:
        cfg = get_config()
        self.assertTrue(cfg.enable_x64)
        self.assertIn(cfg.active_backend, ("cpu", "gpu", "tpu"))
        self.assertGreaterEqual(cfg.batch_size, 512)
        env_map = load_dotenv_file()
        self.assertIn("LLRE_BACKEND", env_map)

    def test_phase1_sandbox_bitpacked_and_alu64(self) -> None:
        sb = LogicSandbox()
        # 1. Bit-packed Boolean XOR test across 64 simultaneous test cases
        prog_xor = pad_program([make_instruction(Opcode.XOR, OUT_REG, 0, 1)], 8)
        in_a = np.uint64(0xAAAAAAAAAAAAAAAA)
        in_b = np.uint64(0xCCCCCCCCCCCCCCCC)
        expected_xor = in_a ^ in_b
        res_bp = sb.evaluate_bitpacked(prog_xor, np.array([in_a, in_b], dtype=np.uint64), expected_xor)
        self.assertEqual(int(res_bp.bit_errors[0]), 0)
        self.assertEqual(int(res_bp.active_steps[0]), 1)
        self.assertAlmostEqual(float(res_bp.rewards[0]), 0.99, places=4)

        # 2. 64-bit ALU Kernighan bit-clear: x & (x - 1)
        prog_kern = pad_program(
            [
                make_instruction(Opcode.SUB, 4, 0, REG_ONE),
                make_instruction(Opcode.AND, OUT_REG, 0, 4),
            ],
            8,
        )
        xs = np.array([0, 1, 2, 3, 7, 8, 12, 0x8000000000000001, 0xFFFFFFFFFFFFFFFF], dtype=np.uint64)
        targets = xs & (xs - np.uint64(1))
        res_alu = sb.evaluate_alu64(prog_kern, xs, targets)
        self.assertEqual(int(res_alu.bit_errors[0]), 0)
        self.assertEqual(int(res_alu.active_steps[0]), 2)
        self.assertAlmostEqual(float(res_alu.rewards[0]), 0.98, places=4)

    def test_phase2_lisp_compiler_and_formal_verifier(self) -> None:
        parity_lisp = """
        (module parity_checker
          (inputs (x :bits 64))
          (outputs (result :bits 1))
          (circuit
            (let ((t0 (bit-and x 1)))
              (let ((result (bit-xor t0 1)))
                result)))
          (specification
            (assert (= result (= (mod x 2) 0)))))
        """
        verifier = FormalVerifier(max_steps=8)
        res = verifier.verify_lisp_module(parity_lisp)
        self.assertTrue(res.proved)
        self.assertEqual(res.status, "UNSAT")
        self.assertEqual(res.active_gates, 2)
        self.assertIn("(check-sat)", res.smt2_script)

        # Verify buggy circuit triggers SAT with concrete counterexample
        buggy_lisp = """
        (module buggy_parity
          (inputs (x :bits 64))
          (outputs (result :bits 1))
          (circuit
            (let ((result (bit-and x 1)))
              result))
          (specification
            (assert (= result (= (mod x 2) 0)))))
        """
        bug_res = verifier.verify_lisp_module(buggy_lisp)
        self.assertFalse(bug_res.proved)
        self.assertEqual(bug_res.status, "SAT")
        self.assertIsNotNone(bug_res.counterexample)

    def test_phase3_sigreg_and_latent_trajectory(self) -> None:
        key = jax.random.PRNGKey(7)
        k1, k2, k3 = jax.random.split(key, 3)

        # Isotropic Gaussian Z should have significantly lower SIGReg loss and higher
        # Effective Dimensionality than a collapsed rank-1 tensor
        z_iso = jax.random.normal(k1, (128, 32), dtype=jnp.float32)
        z_collapsed = jnp.ones((128, 32), dtype=jnp.float32) * jax.random.normal(k2, (128, 1))

        loss_iso = float(sigreg_loss(z_iso, k3))
        loss_col = float(sigreg_loss(z_collapsed, k3))
        self.assertLess(loss_iso, loss_col)

        d_eff_iso = float(compute_effective_dimensionality(z_iso))
        d_eff_col = float(compute_effective_dimensionality(z_collapsed))
        self.assertGreater(d_eff_iso, 15.0)
        self.assertLess(d_eff_col, 1.5)

        # Verify Latent Trajectory Energy Descent E(z*) < E(z_0)
        params = init_latent_core_params(k1, feature_dim=68, latent_dim=32, hidden_dim=64, max_steps=8)
        xs = np.arange(64, dtype=np.uint64)
        ys = ((xs % 2) == 0).astype(np.uint64)
        feats = extract_spec_features(xs, ys, num_inputs=1, output_bits=1)
        cond_c = encode_condition(params, feats)
        traj = optimize_latent_trajectory(params, cond_c, k2, num_steps=10, step_lr=0.1)
        self.assertLess(float(traj.energy_curve[-1]), float(traj.energy_curve[0]))

    def test_phase4_end_to_end_self_play_and_io_adapter(self) -> None:
        import tempfile
        from pathlib import Path
        from llre.checkpoint import load_checkpoint, save_checkpoint

        engine = SelfPlayEngine(max_steps=8, latent_dim=32, explore_batch_size=256)
        challenges = build_core_curriculum()
        sp = engine.run_self_play_curriculum(epochs=15, challenges=challenges)
        self.assertEqual(sp["formal_proof_rate"], 1.0)

        # Verify pure-neural validation (no seed hints) reaches high 1-shot greedy UNSAT accuracy
        val = engine.validate_neural_policy(challenges)
        self.assertGreaterEqual(val["greedy_1shot_unsat_rate"], 0.75)

        # Verify cross-OS checkpoint save & load round-trip
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_ckpt.npz"
            save_checkpoint(engine.params, ckpt_path, metadata={"epoch": 15})
            loaded_params, meta = load_checkpoint(ckpt_path)
            self.assertEqual(meta["epoch"], 15)
            self.assertIn("decoder", loaded_params)

        report = build_observability_report(
            challenges=challenges,
            primitives=sp["primitives"],
            trajectories=sp["trajectories"],
        )
        self.assertEqual(report.formal_proof_rate, 1.0)
        self.assertEqual(report.mean_bitwise_defect_density, 0.0)
        self.assertGreater(report.mean_energy_drop, 0.0)

        adapter = PeripheralIOAdapter(engine=engine)
        q_res = adapter.query("Check whether an integer x is even.")
        prim = q_res["primitive"]
        self.assertEqual(prim.status, "UNSAT")
        self.assertIn("parity_checker", prim.lisp_module)


if __name__ == "__main__":
    unittest.main()
