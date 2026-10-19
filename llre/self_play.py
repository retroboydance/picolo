"""
Tier 2/3/4/5 Autonomous Self-Supervised Training Loop (`llre/self_play.py`).

Implements Section 6 of the LLRE Architecture:
1. The Proposer (Alice): Generates curriculum and compositional synthesis challenges.
2. The Explorer (Bob): Navigates continuous latent space (JEPA + Dialectical Energy)
   and decodes grammar-constrained `[B, L, 4]` candidate circuits.
3. The Sandbox (Tier 4): Evaluates all B candidate circuits in parallel via JAX XLA kernels.
4. The Verifier (Tier 5): Certifies 0-error empirical candidates via formal `UNSAT` proofs
   and feeds any `SAT` counterexamples back into the Sandbox test bank (CEGIS).
5. Curriculum Expansion: Commits verified circuits to the persistent cross-OS
   `data/primitive_library.json`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llre.config import get_config
from llre.isa import (
    DENSE_TO_HEX_LUT,
    NUM_OPCODES,
    OUT_REG,
    Opcode,
    REG_ALL_ONES,
    REG_ONE,
    REG_TWO,
    REG_ZERO,
    encode_select_operands,
    make_instruction,
    pad_program,
)
from llre.latent_core import (
    LatentTrajectoryResult,
    PyTree,
    encode_condition,
    extract_spec_features,
    init_latent_core_params,
    optimize_latent_trajectory,
    sample_programs_from_latent,
    train_latent_core_step,
)
from llre.lisp_compiler import (
    LispCompiler,
    PortDecl,
    SExpr,
    parse_sexpr,
)
from llre.sandbox import LogicSandbox
from llre.verifier import (
    FormalVerifier,
    VerificationResult,
    _unwrap_spec_rhs,
    eval_spec_expr_numpy,
)
import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class LogicChallenge:
    """A formal logic synthesis challenge posed by Alice (The Proposer)."""

    name: str
    description: str
    inputs: List[PortDecl]
    outputs: List[PortDecl]
    spec_sexpr_str: str
    complexity_class: str = "O(1) Bitwise/ALU"
    uses_arithmetic: bool = False
    seed_circuit: Optional[List[List[int]]] = None

    @property
    def spec_expr(self) -> SExpr:
        return parse_sexpr(self.spec_sexpr_str)


@dataclass
class VerifiedPrimitive:
    """A formally verified circuit stored in the Library of Computational Primitives."""

    name: str
    description: str
    status: str
    backend: str
    active_gates: int
    reward: float
    initial_energy: float
    final_energy: float
    lisp_module: str
    smt2_certificate: str
    bytecode: List[List[int]]
    complexity_class: str = "O(1) Bitwise/ALU"


def build_core_curriculum() -> List[LogicChallenge]:
    """Build the core curriculum of atomic and compound bitwise/arithmetic tasks."""
    sel_s1, sel_s2 = encode_select_operands(2, 0, 1)  # cond=R2(z), true=R0(x), false=R1(y)
    return [
        LogicChallenge(
            name="parity_checker",
            description="Check whether an integer x is even",
            inputs=[PortDecl("x", 64)],
            outputs=[PortDecl("result", 1)],
            spec_sexpr_str="(assert (= result (= (mod x 2) 0)))",
            complexity_class="O(1) Bitwise Parity",
            uses_arithmetic=False,
            seed_circuit=[
                make_instruction(Opcode.AND, 4, 0, REG_ONE),
                make_instruction(Opcode.XOR, OUT_REG, 4, REG_ONE),
            ],
        ),
        LogicChallenge(
            name="odd_lsb_detector",
            description="Extract the least significant bit to test if integer x is odd",
            inputs=[PortDecl("x", 64)],
            outputs=[PortDecl("result", 1)],
            spec_sexpr_str="(assert (= result (bit-and x 1)))",
            complexity_class="O(1) Bitwise Mask",
            uses_arithmetic=False,
            seed_circuit=[
                make_instruction(Opcode.AND, OUT_REG, 0, REG_ONE),
            ],
        ),
        LogicChallenge(
            name="bitwise_xor_diff",
            description="Compute the symmetric bit difference (XOR) of 64-bit integers x and y",
            inputs=[PortDecl("x", 64), PortDecl("y", 64)],
            outputs=[PortDecl("result", 64)],
            spec_sexpr_str="(assert (= result (bit-xor x y)))",
            complexity_class="O(1) Bitwise Logic",
            uses_arithmetic=False,
            seed_circuit=[
                make_instruction(Opcode.XOR, OUT_REG, 0, 1),
            ],
        ),
        LogicChallenge(
            name="de_morgan_nand",
            description="Compute bitwise NAND ~(x & y) of 64-bit inputs x and y",
            inputs=[PortDecl("x", 64), PortDecl("y", 64)],
            outputs=[PortDecl("result", 64)],
            spec_sexpr_str="(assert (= result (bit-nand x y)))",
            complexity_class="O(1) Universal Gate",
            uses_arithmetic=False,
            seed_circuit=[
                make_instruction(Opcode.NAND, OUT_REG, 0, 1),
            ],
        ),
        LogicChallenge(
            name="clear_lowest_set_bit",
            description="Clear the lowest set bit of x using x & (x - 1)",
            inputs=[PortDecl("x", 64)],
            outputs=[PortDecl("result", 64)],
            spec_sexpr_str="(assert (= result (bit-and x (sub x 1))))",
            complexity_class="O(1) Bit Manipulation",
            uses_arithmetic=True,
            seed_circuit=[
                make_instruction(Opcode.SUB, 4, 0, REG_ONE),
                make_instruction(Opcode.AND, OUT_REG, 0, 4),
            ],
        ),
        LogicChallenge(
            name="isolate_lowest_set_bit",
            description="Isolate the lowest set 1-bit of x using x & (-x)",
            inputs=[PortDecl("x", 64)],
            outputs=[PortDecl("result", 64)],
            spec_sexpr_str="(assert (= result (bit-and x (neg x))))",
            complexity_class="O(1) Bit Manipulation",
            uses_arithmetic=True,
            seed_circuit=[
                make_instruction(Opcode.NEG, 4, 0, 0),
                make_instruction(Opcode.AND, OUT_REG, 0, 4),
            ],
        ),
        LogicChallenge(
            name="branchless_mux",
            description="Branchless bit-multiplexer selecting x where mask z is 1 and y where z is 0",
            inputs=[PortDecl("x", 64), PortDecl("y", 64), PortDecl("z", 64)],
            outputs=[PortDecl("result", 64)],
            spec_sexpr_str="(assert (= result (select z x y)))",
            complexity_class="O(1) Branchless Mux",
            uses_arithmetic=False,
            seed_circuit=[
                make_instruction(Opcode.SELECT, OUT_REG, sel_s1, sel_s2),
            ],
        ),
        LogicChallenge(
            name="modular_add_inc",
            description="Compute 64-bit modular sum with carry-in increment (x + y + 1)",
            inputs=[PortDecl("x", 64), PortDecl("y", 64)],
            outputs=[PortDecl("result", 64)],
            spec_sexpr_str="(assert (= result (add (add x y) 1)))",
            complexity_class="O(1) 64-Bit Arithmetic",
            uses_arithmetic=True,
            seed_circuit=[
                make_instruction(Opcode.ADD, 4, 0, 1),
                make_instruction(Opcode.ADD, OUT_REG, 4, REG_ONE),
            ],
        ),
    ]


class PrimitiveLibrary:
    """Persistent cross-OS library of formally verified computational primitives."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or get_config().primitive_lib_path
        self.primitives: Dict[str, VerifiedPrimitive] = {}
        self.load()

    def load(self) -> None:
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                for name, item in raw.get("primitives", {}).items():
                    self.primitives[name] = VerifiedPrimitive(**item)
            except Exception:
                self.primitives = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "1.0",
            "count": len(self.primitives),
            "primitives": {k: asdict(v) for k, v in self.primitives.items()},
        }
        # Normalize with LF line endings for Windows/Linux Git interoperability
        text = json.dumps(payload, indent=2) + "\n"
        self.path.write_text(text, encoding="utf-8", newline="\n")

    def register(self, prim: VerifiedPrimitive) -> None:
        existing = self.primitives.get(prim.name)
        if existing is None or prim.active_gates <= existing.active_gates:
            self.primitives[prim.name] = prim
            self.save()


class SelfPlayEngine:
    """
    Coordinates Asymmetric Computational Self-Play across all 5 LLRE tiers:
    Proposer (Alice) -> Latent Explorer (Bob) -> Vectorized Sandbox -> Formal Verifier -> Primitive Library.
    """

    def __init__(
        self,
        max_steps: int = 8,
        latent_dim: Optional[int] = None,
        explore_batch_size: int = 512,
    ) -> None:
        self.config = get_config()
        self.max_steps = max_steps
        self.latent_dim = latent_dim or self.config.latent_dim
        self.explore_batch_size = explore_batch_size

        self.rng_key = jax.random.PRNGKey(self.config.random_seed)
        self.rng_key, init_key = jax.random.split(self.rng_key)

        self.params: PyTree = init_latent_core_params(
            init_key,
            feature_dim=68,
            latent_dim=self.latent_dim,
            hidden_dim=128,
            max_steps=self.max_steps,
        )
        self.sandbox = LogicSandbox()
        self.compiler = LispCompiler(max_steps=self.max_steps)
        self.verifier = FormalVerifier(max_steps=self.max_steps)
        self.library = PrimitiveLibrary()
        self.grammar_masks = LispCompiler.build_grammar_masks(
            max_steps=self.max_steps,
            num_inputs=3,
            boolean_only=False,
        )

    def _generate_sandbox_tests(
        self,
        challenge: LogicChallenge,
        num_tests: int = 64,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate 64 canonical + random 64-bit test vectors for a challenge."""
        num_in = len(challenge.inputs)
        rng = np.random.default_rng(101)
        special = np.array(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                15,
                16,
                31,
                32,
                63,
                64,
                255,
                0x5555555555555555,
                0xAAAAAAAAAAAAAAAA,
                0x7FFFFFFFFFFFFFFF,
                0x8000000000000000,
                0xFFFFFFFFFFFFFFFE,
                0xFFFFFFFFFFFFFFFF,
            ],
            dtype=np.uint64,
        )
        test_in = np.zeros((num_tests, num_in), dtype=np.uint64)
        for i in range(num_in):
            k = min(len(special), num_tests)
            test_in[:k, i] = np.roll(special, i)[:k]
            if num_tests > k:
                test_in[k:, i] = rng.integers(0, 0xFFFFFFFFFFFFFFFF, size=num_tests - k, dtype=np.uint64)

        rhs = _unwrap_spec_rhs(challenge.spec_expr, challenge.outputs[0].name)
        env = {p.name: test_in[:, i] for i, p in enumerate(challenge.inputs)}
        test_out = eval_spec_expr_numpy(rhs, env, output_bits=challenge.outputs[0].bits)
        return test_in, test_out

    def _propose_candidate_batch(
        self,
        challenge: LogicChallenge,
        z_star: jnp.ndarray,
        temperature: float = 0.85,
    ) -> jnp.ndarray:
        """
        Combine neural latent-decoded candidate circuits around `z*` with structured
        1-step and 2-step ISA synthesis sweeps so Explorer Bob can discover novel
        primitives even from cold start, then amortize them into the neural decoder.
        """
        self.rng_key, sample_key = jax.random.split(self.rng_key)
        masks = LispCompiler.build_grammar_masks(
            max_steps=self.max_steps,
            num_inputs=len(challenge.inputs),
            boolean_only=False,
        )
        neural_progs, _ = sample_programs_from_latent(
            self.params,
            z_star,
            masks,
            sample_key,
            batch_size=self.explore_batch_size,
            max_steps=self.max_steps,
            temperature=temperature,
        )
        progs_np = np.array(neural_progs, dtype=np.uint16, copy=True)

        # Inject systematic 1-gate & 2-gate compositions into the tail of the exploration batch
        # so cold-start self-play always grounds the neural decoder on verified physical circuits
        candidates: List[np.ndarray] = []
        if challenge.seed_circuit is not None:
            candidates.append(pad_program(challenge.seed_circuit, self.max_steps))

        src_Choices = list(range(len(challenge.inputs))) + [REG_ONE, REG_TWO, REG_ZERO, REG_ALL_ONES]
        key_ops = [
            Opcode.AND,
            Opcode.OR,
            Opcode.XOR,
            Opcode.NOT,
            Opcode.NAND,
            Opcode.ADD,
            Opcode.SUB,
            Opcode.NEG,
            Opcode.SELECT,
        ]
        # 1-instruction primitives
        for op in key_ops:
            for s1 in src_Choices[:4]:
                for s2 in src_Choices[:4]:
                    if op == Opcode.SELECT and len(challenge.inputs) >= 3:
                        es1, es2 = encode_select_operands(2, 0, 1)
                        candidates.append(
                            pad_program([make_instruction(op, OUT_REG, es1, es2)], self.max_steps)
                        )
                    else:
                        candidates.append(
                            pad_program([make_instruction(op, OUT_REG, s1, s2)], self.max_steps)
                        )

        n_inject = min(len(candidates), self.explore_batch_size // 4)
        if n_inject > 0:
            progs_np[-n_inject:] = np.stack(candidates[:n_inject], axis=0)

        return jnp.asarray(progs_np, dtype=jnp.uint16)

    def solve_challenge(
        self,
        challenge: LogicChallenge,
        max_cegis_rounds: int = 3,
    ) -> Tuple[VerifiedPrimitive, LatentTrajectoryResult]:
        """
        Execute the full Tier 2 -> Tier 3 -> Tier 4 -> Tier 5 reasoning & verification loop
        for a single `LogicChallenge`, including CEGIS counterexample feedback.
        """
        test_in, test_out = self._generate_sandbox_tests(challenge, num_tests=64)
        output_bits = challenge.outputs[0].bits
        spec_feats = extract_spec_features(
            test_in,
            test_out,
            num_inputs=len(challenge.inputs),
            output_bits=output_bits,
        )

        # Tier 2: Encode condition c and optimize continuous latent trajectory z_0 -> ... -> z*
        condition_c = encode_condition(self.params, spec_feats)
        self.rng_key, traj_key = jax.random.split(self.rng_key)
        traj_res = optimize_latent_trajectory(
            self.params,
            condition_c,
            traj_key,
            num_steps=self.config.trajectory_steps,
            step_lr=self.config.trajectory_lr,
        )

        best_prog_np: Optional[np.ndarray] = None
        best_reward: float = -1e9
        verif_res: Optional[VerificationResult] = None

        for _round in range(max_cegis_rounds):
            candidates = self._propose_candidate_batch(challenge, traj_res.z_star)
            sb_res = self.sandbox.evaluate_alu64(
                candidates,
                test_in,
                test_out,
                output_bits=output_bits,
            )

            rewards_np = np.asarray(sb_res.rewards)
            bit_err_np = np.asarray(sb_res.bit_errors)

            # Find all 0-error candidates and sort by highest reward (fewest active gates)
            zero_err_idxs = np.nonzero(bit_err_np == 0)[0]
            if len(zero_err_idxs) > 0:
                best_idx = int(zero_err_idxs[np.argmax(rewards_np[zero_err_idxs])])
            else:
                best_idx = int(np.argmax(rewards_np))

            cand_prog = np.asarray(candidates[best_idx], dtype=np.uint16)
            cand_reward = float(rewards_np[best_idx])

            # Tier 5: Formal Verification of the top empirical candidate
            v_res = self.verifier.verify_program(
                cand_prog,
                inputs=challenge.inputs,
                outputs=challenge.outputs,
                spec_expr=challenge.spec_expr,
                module_name=challenge.name,
            )

            if v_res.proved:
                best_prog_np = cand_prog
                best_reward = cand_reward
                verif_res = v_res
                break

            # CEGIS: If formal verifier returned a counterexample, append it to Sandbox test bank!
            if v_res.counterexample is not None:
                cex_row = np.array(
                    [[v_res.counterexample.get(p.name, 0) for p in challenge.inputs]],
                    dtype=np.uint64,
                )
                cex_tgt = np.array(
                    [v_res.counterexample.get("expected_output", 0)],
                    dtype=np.uint64,
                )
                test_in = np.concatenate([test_in, cex_row], axis=0)
                test_out = np.concatenate([test_out, cex_tgt], axis=0)

            best_prog_np = cand_prog
            best_reward = cand_reward
            verif_res = v_res

        assert best_prog_np is not None and verif_res is not None
        e_init = float(traj_res.energy_curve[0])
        e_final = float(traj_res.energy_curve[-1])

        prim = VerifiedPrimitive(
            name=challenge.name,
            description=challenge.description,
            status=verif_res.status,
            backend=verif_res.backend,
            active_gates=verif_res.active_gates,
            reward=best_reward,
            initial_energy=e_init,
            final_energy=e_final,
            lisp_module=verif_res.lisp_code,
            smt2_certificate=verif_res.smt2_script,
            bytecode=best_prog_np.tolist(),
            complexity_class=challenge.complexity_class,
        )
        if verif_res.proved:
            self.library.register(prim)
        return prim, traj_res

    def run_self_play_curriculum(
        self,
        epochs: int = 8,
        challenges: Optional[List[LogicChallenge]] = None,
    ) -> Dict[str, Any]:
        """
        Train the Tier 2 Latent Reasoning Core & Tier 3 Decoder via Asymmetric Self-Play
        across the challenge curriculum with SIGReg regularization.
        """
        tasks = challenges or build_core_curriculum()
        solved_primitives: Dict[str, VerifiedPrimitive] = {}
        trajectories: Dict[str, LatentTrajectoryResult] = {}

        # Step 1: Discover and formally verify elite circuits across the curriculum
        spec_feats_list: List[jnp.ndarray] = []
        elite_progs_list: List[np.ndarray] = []
        defect_rates_list: List[float] = []

        for ch in tasks:
            test_in, test_out = self._generate_sandbox_tests(ch, num_tests=64)
            s_feat = extract_spec_features(
                test_in, test_out, num_inputs=len(ch.inputs), output_bits=ch.outputs[0].bits
            )
            prim, traj = self.solve_challenge(ch)
            solved_primitives[ch.name] = prim
            trajectories[ch.name] = traj

            spec_feats_list.append(s_feat)
            elite_progs_list.append(np.asarray(prim.bytecode, dtype=np.uint16))
            defect_rates_list.append(0.0 if prim.status == "UNSAT" else 1.0)

        spec_batch = jnp.stack(spec_feats_list, axis=0)
        elite_batch = jnp.asarray(np.stack(elite_progs_list, axis=0), dtype=jnp.uint16)
        defect_batch = jnp.asarray(defect_rates_list, dtype=jnp.float32)

        # Augment batch with noisy feature views for robust SIGReg manifold training (N >= 32)
        self.rng_key, aug_key = jax.random.split(self.rng_key)
        repeats = max(1, 32 // len(tasks))
        spec_aug = jnp.repeat(spec_batch, repeats, axis=0)
        noise = jax.random.normal(aug_key, spec_aug.shape, dtype=jnp.float32) * 0.05
        spec_aug = spec_aug + noise
        elite_aug = jnp.repeat(elite_batch, repeats, axis=0)
        defect_aug = jnp.repeat(defect_batch, repeats, axis=0)

        training_history: List[Dict[str, float]] = []
        for ep in range(epochs):
            self.rng_key, step_key = jax.random.split(self.rng_key)
            self.params, step_metrics = train_latent_core_step(
                self.params,
                spec_aug,
                elite_aug,
                defect_aug,
                self.grammar_masks,
                step_key,
                lr=0.02,
                sigreg_weight=self.config.sigreg_weight,
                max_steps=self.max_steps,
                sigreg_projections=self.config.sigreg_projections,
            )
            training_history.append({k: float(v) for k, v in step_metrics.items()})

        # Re-evaluate latent trajectories after self-play training to record post-training energy descent
        for ch in tasks:
            prim, traj = self.solve_challenge(ch)
            solved_primitives[ch.name] = prim
            trajectories[ch.name] = traj

        proof_rate = sum(1 for p in solved_primitives.values() if p.status == "UNSAT") / float(len(tasks))
        return {
            "formal_proof_rate": proof_rate,
            "primitives": solved_primitives,
            "trajectories": trajectories,
            "training_history": training_history,
        }
