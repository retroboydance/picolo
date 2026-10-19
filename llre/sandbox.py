"""
Tier 4: Bare-Metal Vectorized Logic Sandbox (JAX / XLA).

Implements both execution modes from the architectural review:
1. 64-Way Bit-Packed Boolean Kernel:
   Packs 64 independent Boolean test cases into each `uint64` register (`1 bit = 1 test case`).
   A single bitwise instruction evaluates 64 test cases simultaneously.
2. Vectorized 64-Bit ALU Kernel:
   Evaluates T full 64-bit integer test cases across all 16 bitwise and modular arithmetic
   opcodes (`ADD`, `SUB`, `SHL`, `SHR`, `CMP_EQ`, `SELECT`, etc.), measuring exact bitwise
   Hamming distance via `jax.lax.population_count`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import time
from typing import Dict, Optional, Tuple

from llre.config import get_config  # Ensures JAX x64 is initialized
from llre.isa import (
    HEX_TO_DENSE_LUT,
    NUM_REGS,
    OUT_REG,
    REG_ALL_ONES,
    REG_ONE,
    REG_TWO,
    REG_ZERO,
)
import jax
import jax.numpy as jnp
import numpy as np

MASK64 = jnp.uint64(0xFFFFFFFFFFFFFFFF)
ZERO64 = jnp.uint64(0)
ONE64 = jnp.uint64(1)
TWO64 = jnp.uint64(2)
SHIFT_MASK64 = jnp.uint64(63)


@dataclass(frozen=True)
class SandboxResult:
    """Output of a batched Tier 4 Logic Sandbox evaluation."""

    outputs: jnp.ndarray      # [B] or [B, T] uint64 output register values
    bit_errors: jnp.ndarray   # [B] int32 total bitwise Hamming distance
    case_errors: jnp.ndarray  # [B] int32 number of failed test cases
    active_steps: jnp.ndarray # [B] int32 number of non-NOP instructions
    rewards: jnp.ndarray      # [B] float32 continuous reward signal


def _execute_opcode_bitpacked(
    dense_op: jnp.ndarray,
    v1: jnp.ndarray,
    v2: jnp.ndarray,
    v_cond: jnp.ndarray,
) -> jnp.ndarray:
    """
    Branchless evaluation of all 16 opcodes in 64-way bit-packed Boolean mode.
    Every bit i in [0..63] is an independent 1-bit test case.
    Uses `jax.lax.select_n` for zero-branch SIMD/SIMT instruction fusion.
    """
    sh = v2 & SHIFT_MASK64
    return jax.lax.select_n(
        dense_op,
        v1,                             # 0: NOP
        v1 & v2,                        # 1: AND
        v1 | v2,                        # 2: OR
        v1 ^ v2,                        # 3: XOR
        ~v1,                            # 4: NOT
        v1 >> sh,                       # 5: SHR
        v1 << sh,                       # 6: SHL
        ~(v1 & v2),                     # 7: NAND
        ~(v1 | v2),                     # 8: NOR
        ~(v1 ^ v2),                     # 9: XNOR
        v1 ^ v2,                        # 10: ADD (1-bit GF(2) sum in bit-packed mode)
        v1 ^ v2,                        # 11: SUB (1-bit GF(2) diff in bit-packed mode)
        ~v1,                            # 12: NEG
        ~(v1 ^ v2),                     # 13: CMP_EQ (bitwise equality = XNOR)
        (~v1) & v2,                     # 14: CMP_ULT (1-bit a < b)
        (v_cond & v1) | ((~v_cond) & v2),  # 15: SELECT (c ? a : b)
    )


def _execute_opcode_alu64(
    dense_op: jnp.ndarray,
    v1: jnp.ndarray,
    v2: jnp.ndarray,
    v_cond: jnp.ndarray,
) -> jnp.ndarray:
    """
    Branchless evaluation of all 16 opcodes in 64-bit word ALU mode.
    Each element of v1, v2, v_cond is a full 64-bit unsigned integer word (`[T]`).
    Uses `jax.lax.select_n` for zero-branch SIMD/SIMT instruction fusion.
    """
    sh = v2 & SHIFT_MASK64
    return jax.lax.select_n(
        dense_op,
        v1,                                       # 0: NOP
        v1 & v2,                                  # 1: AND
        v1 | v2,                                  # 2: OR
        v1 ^ v2,                                  # 3: XOR
        ~v1,                                      # 4: NOT
        v1 >> sh,                                 # 5: SHR
        v1 << sh,                                 # 6: SHL
        ~(v1 & v2),                               # 7: NAND
        ~(v1 | v2),                               # 8: NOR
        ~(v1 ^ v2),                               # 9: XNOR
        v1 + v2,                                  # 10: ADD (64-bit modular)
        v1 - v2,                                  # 11: SUB (64-bit modular)
        ZERO64 - v1,                              # 12: NEG (two's complement)
        jnp.where(v1 == v2, MASK64, ZERO64),      # 13: CMP_EQ (~0 : 0)
        jnp.where(v1 < v2, MASK64, ZERO64),       # 14: CMP_ULT (~0 : 0)
        (v_cond & v1) | ((~v_cond) & v2),         # 15: SELECT
    )


def _eval_single_program_bitpacked(
    program: jnp.ndarray,       # [L, 4] uint16
    packed_inputs: jnp.ndarray, # [NumInputs] uint64
    packed_target: jnp.ndarray, # [] uint64
    out_reg: int,
    target_mask: jnp.ndarray,   # [] uint64
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Evaluate one program `[L, 4]` on 64 bit-packed Boolean test cases."""
    regs = jnp.zeros((NUM_REGS,), dtype=jnp.uint64)
    num_in = min(packed_inputs.shape[0], 4)
    for i in range(num_in):
        regs = regs.at[i].set(packed_inputs[i])

    # In 64-way bit-packed Boolean mode, each bit is a test lane:
    # const_0 = 0x0000000000000000, const_1 = 0xFFFFFFFFFFFFFFFF (logical 1 in all 64 lanes)
    regs = regs.at[REG_ZERO].set(ZERO64)
    regs = regs.at[REG_ONE].set(MASK64)
    regs = regs.at[REG_TWO].set(TWO64)
    regs = regs.at[REG_ALL_ONES].set(MASK64)

    def step_fn(reg_state: jnp.ndarray, instr: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        op_raw = instr[0].astype(jnp.int32)
        dest = (instr[1] & jnp.uint16(0x0F)).astype(jnp.int32)
        s1 = (instr[2] & jnp.uint16(0x0F)).astype(jnp.int32)
        s2_raw = instr[3].astype(jnp.int32)
        s2_lo = s2_raw & 0x0F
        s2_hi = (s2_raw >> 4) & 0x0F

        dense_op = HEX_TO_DENSE_LUT[op_raw & 0x3F]

        # Decode SELECT operands: if s2_hi == 0, legacy spec mode (s1=cond, s2_lo=true, R0=false)
        # Otherwise packed mode ((s2_hi - 1)=cond, s1=true, s2_lo=false)
        is_legacy_sel = s2_hi == 0
        cond_idx = jnp.where(is_legacy_sel, s1, (s2_hi - 1) & 0x0F)
        true_idx = jnp.where(is_legacy_sel, s2_lo, s1)
        false_idx = jnp.where(is_legacy_sel, 0, s2_lo)

        v1 = jnp.where(dense_op == 15, reg_state[true_idx], reg_state[s1])
        v2 = jnp.where(dense_op == 15, reg_state[false_idx], reg_state[s2_lo])
        v_cond = reg_state[cond_idx]

        res = _execute_opcode_bitpacked(dense_op, v1, v2, v_cond)

        # Protect constant registers (R12..R15) and ignore NOP (dense_op == 0)
        is_valid_write = (dense_op != 0) & (dest < 12)
        new_val = jnp.where(is_valid_write, res, reg_state[dest])
        reg_state = reg_state.at[dest].set(new_val)
        is_active = (dense_op != 0).astype(jnp.int32)
        return reg_state, is_active

    final_regs, active_flags = jax.lax.scan(step_fn, regs, program)
    out_val = final_regs[out_reg]
    diff = (out_val ^ packed_target) & target_mask
    bit_errors = jax.lax.population_count(diff).astype(jnp.int32)
    active_steps = jnp.sum(active_flags).astype(jnp.int32)
    reward = jnp.where(
        bit_errors == 0,
        1.0 - (active_steps.astype(jnp.float32) * 0.01),
        -1.0 * bit_errors.astype(jnp.float32),
    )
    return out_val, bit_errors, bit_errors, active_steps, reward


def _eval_single_program_alu64(
    program: jnp.ndarray,      # [L, 4] uint16
    test_inputs: jnp.ndarray,  # [T, NumInputs] uint64
    test_targets: jnp.ndarray, # [T] uint64
    out_reg: int,
    target_mask: jnp.ndarray,  # [] uint64
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Evaluate one program `[L, 4]` across `T` 64-bit integer word test cases."""
    num_tests = test_inputs.shape[0]
    num_in = min(test_inputs.shape[1], 4)

    regs = jnp.zeros((NUM_REGS, num_tests), dtype=jnp.uint64)
    for i in range(num_in):
        regs = regs.at[i].set(test_inputs[:, i])

    # Hardware constant registers in 64-bit word ALU mode
    regs = regs.at[REG_ZERO].set(ZERO64)
    regs = regs.at[REG_ONE].set(ONE64)
    regs = regs.at[REG_TWO].set(TWO64)
    regs = regs.at[REG_ALL_ONES].set(MASK64)

    def step_fn(reg_state: jnp.ndarray, instr: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        op_raw = instr[0].astype(jnp.int32)
        dest = (instr[1] & jnp.uint16(0x0F)).astype(jnp.int32)
        s1 = (instr[2] & jnp.uint16(0x0F)).astype(jnp.int32)
        s2_raw = instr[3].astype(jnp.int32)
        s2_lo = s2_raw & 0x0F
        s2_hi = (s2_raw >> 4) & 0x0F

        dense_op = HEX_TO_DENSE_LUT[op_raw & 0x3F]

        is_legacy_sel = s2_hi == 0
        cond_idx = jnp.where(is_legacy_sel, s1, (s2_hi - 1) & 0x0F)
        true_idx = jnp.where(is_legacy_sel, s2_lo, s1)
        false_idx = jnp.where(is_legacy_sel, 0, s2_lo)

        v1 = jnp.where(dense_op == 15, reg_state[true_idx], reg_state[s1])
        v2 = jnp.where(dense_op == 15, reg_state[false_idx], reg_state[s2_lo])
        v_cond = reg_state[cond_idx]

        res = _execute_opcode_alu64(dense_op, v1, v2, v_cond)

        is_valid_write = (dense_op != 0) & (dest < 12)
        new_val = jnp.where(is_valid_write, res, reg_state[dest])
        reg_state = reg_state.at[dest].set(new_val)
        is_active = (dense_op != 0).astype(jnp.int32)
        return reg_state, is_active

    final_regs, active_flags = jax.lax.scan(step_fn, regs, program)
    out_vals = final_regs[out_reg] & target_mask
    masked_targets = test_targets & target_mask
    diff = out_vals ^ masked_targets

    per_test_bit_errors = jax.lax.population_count(diff).astype(jnp.int32)
    bit_errors = jnp.sum(per_test_bit_errors).astype(jnp.int32)
    case_errors = jnp.sum(diff != ZERO64).astype(jnp.int32)
    active_steps = jnp.sum(active_flags).astype(jnp.int32)

    reward = jnp.where(
        bit_errors == 0,
        1.0 - (active_steps.astype(jnp.float32) * 0.01),
        -1.0 * bit_errors.astype(jnp.float32),
    )
    return out_vals, bit_errors, case_errors, active_steps, reward


@partial(jax.jit, static_argnames=("out_reg",))
def evaluate_bitpacked_kernel(
    programs: jnp.ndarray,
    packed_inputs: jnp.ndarray,
    packed_target: jnp.ndarray,
    out_reg: int = OUT_REG,
    target_mask: jnp.ndarray = MASK64,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JIT-compiled vectorized bit-packed Boolean kernel over batch dimension `B`."""
    vmapped = jax.vmap(
        _eval_single_program_bitpacked,
        in_axes=(0, None, None, None, None),
    )
    return vmapped(programs, packed_inputs, packed_target, out_reg, target_mask)


@partial(jax.jit, static_argnames=("out_reg",))
def evaluate_alu64_kernel(
    programs: jnp.ndarray,
    test_inputs: jnp.ndarray,
    test_targets: jnp.ndarray,
    out_reg: int = OUT_REG,
    target_mask: jnp.ndarray = MASK64,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JIT-compiled vectorized 64-bit ALU kernel over batch dimension `B`."""
    vmapped = jax.vmap(
        _eval_single_program_alu64,
        in_axes=(0, None, None, None, None),
    )
    return vmapped(programs, test_inputs, test_targets, out_reg, target_mask)


class LogicSandbox:
    """
    High-throughput JAX Vectorized Logic Sandbox (Tier 4).
    Automatically adapts to Windows multi-core CPU or Linux CUDA GPU via `LLREConfig`.
    """

    def __init__(self, out_reg: int = OUT_REG) -> None:
        self.config = get_config()
        self.out_reg = out_reg

    def evaluate_bitpacked(
        self,
        programs: jnp.ndarray | np.ndarray,
        packed_inputs: jnp.ndarray | np.ndarray,
        packed_target: int | jnp.ndarray | np.ndarray,
        out_reg: Optional[int] = None,
        target_mask: int = 0xFFFFFFFFFFFFFFFF,
    ) -> SandboxResult:
        """
        Execute a batch of `[B, L, 4]` programs in 64-way bit-packed Boolean mode.
        `packed_inputs`: shape `[NumInputs]` or scalar `uint64` (64 test cases packed per register).
        `packed_target`: scalar `uint64` packing the 64 expected Boolean outputs.
        """
        progs_jnp = jnp.asarray(programs, dtype=jnp.uint16)
        if progs_jnp.ndim == 2:
            progs_jnp = progs_jnp[None, ...]
        inputs_jnp = jnp.asarray(packed_inputs, dtype=jnp.uint64).reshape(-1)
        target_jnp = jnp.asarray(packed_target, dtype=jnp.uint64)
        mask_jnp = jnp.asarray(target_mask, dtype=jnp.uint64)
        reg = self.out_reg if out_reg is None else out_reg

        outputs, bit_err, case_err, steps, rewards = evaluate_bitpacked_kernel(
            progs_jnp, inputs_jnp, target_jnp, reg, mask_jnp
        )
        return SandboxResult(outputs, bit_err, case_err, steps, rewards)

    def evaluate_alu64(
        self,
        programs: jnp.ndarray | np.ndarray,
        test_inputs: jnp.ndarray | np.ndarray,
        test_targets: jnp.ndarray | np.ndarray,
        out_reg: Optional[int] = None,
        output_bits: int = 64,
    ) -> SandboxResult:
        """
        Execute a batch of `[B, L, 4]` programs across `T` 64-bit integer word test cases.
        `test_inputs`: shape `[T, NumInputs]` or `[T]` of `uint64`.
        `test_targets`: shape `[T]` of `uint64`.
        """
        progs_jnp = jnp.asarray(programs, dtype=jnp.uint16)
        if progs_jnp.ndim == 2:
            progs_jnp = progs_jnp[None, ...]
        inputs_jnp = jnp.asarray(test_inputs, dtype=jnp.uint64)
        if inputs_jnp.ndim == 1:
            inputs_jnp = inputs_jnp[:, None]
        if inputs_jnp.shape[1] < 4:
            inputs_jnp = jnp.pad(inputs_jnp, ((0, 0), (0, 4 - inputs_jnp.shape[1])))
        targets_jnp = jnp.asarray(test_targets, dtype=jnp.uint64).reshape(-1)
        mask_jnp = MASK64
        reg = self.out_reg if out_reg is None else out_reg

        outputs, bit_err, case_err, steps, rewards = evaluate_alu64_kernel(
            progs_jnp, inputs_jnp, targets_jnp, reg, mask_jnp
        )
        return SandboxResult(outputs, bit_err, case_err, steps, rewards)

    def benchmark(
        self,
        batch_size: Optional[int] = None,
        max_steps: Optional[int] = None,
        iterations: int = 20,
    ) -> Dict[str, float]:
        """
        Benchmark Tier 4 Sandbox execution speed in both 64-way bit-packed mode
        and 64-bit ALU mode.
        """
        B = batch_size or self.config.batch_size
        L = max_steps or self.config.max_steps
        rng = np.random.default_rng(42)

        # Generate random candidate programs [B, L, 4]
        valid_hex_ops = np.array(
            [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x10, 0x11, 0x30, 0x32],
            dtype=np.uint16,
        )
        ops = rng.choice(valid_hex_ops, size=(B, L, 1))
        dests = rng.integers(4, 12, size=(B, L, 1), dtype=np.uint16)
        s1s = rng.integers(0, 16, size=(B, L, 1), dtype=np.uint16)
        s2s = rng.integers(0, 16, size=(B, L, 1), dtype=np.uint16)
        progs = jnp.asarray(np.concatenate([ops, dests, s1s, s2s], axis=-1), dtype=jnp.uint16)

        packed_in = jnp.asarray([0xAAAAAAAAAAAAAAAA, 0xCCCCCCCCCCCCCCCC], dtype=jnp.uint64)
        packed_tgt = jnp.uint64(0x6666666666666666)

        # Warm-up XLA compilation
        res = evaluate_bitpacked_kernel(progs, packed_in, packed_tgt, OUT_REG, MASK64)
        jax.block_until_ready(res[0])

        t0 = time.perf_counter()
        for _ in range(iterations):
            res = evaluate_bitpacked_kernel(progs, packed_in, packed_tgt, OUT_REG, MASK64)
            jax.block_until_ready(res[0])
        elapsed = time.perf_counter() - t0

        programs_per_sec = (B * iterations) / elapsed
        test_evals_per_sec = programs_per_sec * 64.0
        gate_evals_per_sec = programs_per_sec * L * 64.0

        return {
            "batch_size": float(B),
            "max_steps": float(L),
            "elapsed_sec": elapsed,
            "programs_per_sec": programs_per_sec,
            "test_case_evals_per_sec": test_evals_per_sec,
            "bit_gate_ops_per_sec": gate_evals_per_sec,
        }
