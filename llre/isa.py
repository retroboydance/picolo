"""
Tier 4 Orthogonal Minimal Logic ISA & Register Bank Specification.

Defines the 16-instruction branchless ISA, 16-register bank layout (input, working,
output, and constant registers), and lookup tables mapping between canonical specification
hex opcodes (e.g., 0x01, 0x10, 0x30, 0x32) and dense neural indices (0..15).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Tuple

import jax.numpy as jnp
import numpy as np


class Opcode(IntEnum):
    """Canonical 16-instruction Orthogonal Logic ISA hex opcodes."""

    NOP = 0x00      # No operation (preserves register state; 0 gate penalty)
    AND = 0x01      # R_d <- R_s1 & R_s2
    OR = 0x02       # R_d <- R_s1 | R_s2
    XOR = 0x03      # R_d <- R_s1 ^ R_s2
    NOT = 0x04      # R_d <- ~R_s1
    SHR = 0x05      # R_d <- R_s1 >> (R_s2 & 63)
    SHL = 0x06      # R_d <- R_s1 << (R_s2 & 63)
    NAND = 0x07     # R_d <- ~(R_s1 & R_s2)
    NOR = 0x08      # R_d <- ~(R_s1 | R_s2)
    XNOR = 0x09     # R_d <- ~(R_s1 ^ R_s2)
    ADD = 0x10      # R_d <- R_s1 + R_s2 (64-bit modular addition)
    SUB = 0x11      # R_d <- R_s1 - R_s2 (64-bit modular subtraction)
    NEG = 0x12      # R_d <- -R_s1 (two's complement negation)
    CMP_EQ = 0x30   # R_d <- (R_s1 == R_s2) ? ~0 : 0 (or ~(R_s1 ^ R_s2) in bit-packed mode)
    CMP_ULT = 0x31  # R_d <- (R_s1 < R_s2)  ? ~0 : 0 (unsigned comparison mask)
    SELECT = 0x32   # R_d <- (R_c & R_s1) | (~R_c & R_s2) (branchless multiplexer)


@dataclass(frozen=True)
class OpcodeSpec:
    """Metadata for a single ISA instruction."""

    opcode: Opcode
    dense_id: int
    mnemonic: str
    lisp_symbol: str
    arity: int
    is_boolean_bitwise: bool
    description: str


OPCODE_SPECS: Tuple[OpcodeSpec, ...] = (
    OpcodeSpec(Opcode.NOP, 0, "NOP", "nop", 0, True, "No operation"),
    OpcodeSpec(Opcode.AND, 1, "AND", "bit-and", 2, True, "Bitwise AND"),
    OpcodeSpec(Opcode.OR, 2, "OR", "bit-or", 2, True, "Bitwise OR"),
    OpcodeSpec(Opcode.XOR, 3, "XOR", "bit-xor", 2, True, "Bitwise XOR"),
    OpcodeSpec(Opcode.NOT, 4, "NOT", "bit-not", 1, True, "Bitwise Invert"),
    OpcodeSpec(Opcode.SHR, 5, "SHR", "bit-shr", 2, False, "Logical Shift Right"),
    OpcodeSpec(Opcode.SHL, 6, "SHL", "bit-shl", 2, False, "Logical Shift Left"),
    OpcodeSpec(Opcode.NAND, 7, "NAND", "bit-nand", 2, True, "Bitwise NAND"),
    OpcodeSpec(Opcode.NOR, 8, "NOR", "bit-nor", 2, True, "Bitwise NOR"),
    OpcodeSpec(Opcode.XNOR, 9, "XNOR", "bit-xnor", 2, True, "Bitwise XNOR"),
    OpcodeSpec(Opcode.ADD, 10, "ADD", "add", 2, False, "64-bit Modular Addition"),
    OpcodeSpec(Opcode.SUB, 11, "SUB", "sub", 2, False, "64-bit Modular Subtraction"),
    OpcodeSpec(Opcode.NEG, 12, "NEG", "neg", 1, False, "64-bit Two's Complement Negation"),
    OpcodeSpec(Opcode.CMP_EQ, 13, "CMP_EQ", "cmp-eq", 2, True, "Bitwise/Word Equality Mask"),
    OpcodeSpec(Opcode.CMP_ULT, 14, "CMP_ULT", "cmp-ult", 2, True, "Unsigned Less-Than Mask"),
    OpcodeSpec(Opcode.SELECT, 15, "SELECT", "select", 3, True, "Branchless Multiplexer"),
)

NUM_OPCODES: int = len(OPCODE_SPECS)

# Lookup mappings
HEX_TO_SPEC: Dict[int, OpcodeSpec] = {int(s.opcode): s for s in OPCODE_SPECS}
DENSE_TO_SPEC: Dict[int, OpcodeSpec] = {s.dense_id: s for s in OPCODE_SPECS}
MNEMONIC_TO_SPEC: Dict[str, OpcodeSpec] = {s.mnemonic.upper(): s for s in OPCODE_SPECS}
LISP_TO_SPEC: Dict[str, OpcodeSpec] = {s.lisp_symbol: s for s in OPCODE_SPECS}

# Additional LISP aliases for ergonomic S-expressions
LISP_ALIASES: Dict[str, OpcodeSpec] = {
    "&": HEX_TO_SPEC[Opcode.AND],
    "|": HEX_TO_SPEC[Opcode.OR],
    "^": HEX_TO_SPEC[Opcode.XOR],
    "~": HEX_TO_SPEC[Opcode.NOT],
    ">>": HEX_TO_SPEC[Opcode.SHR],
    "<<": HEX_TO_SPEC[Opcode.SHL],
    "+": HEX_TO_SPEC[Opcode.ADD],
    "-": HEX_TO_SPEC[Opcode.SUB],
    "==": HEX_TO_SPEC[Opcode.CMP_EQ],
    "=": HEX_TO_SPEC[Opcode.CMP_EQ],
    "<": HEX_TO_SPEC[Opcode.CMP_ULT],
    "ite": HEX_TO_SPEC[Opcode.SELECT],
    "mux": HEX_TO_SPEC[Opcode.SELECT],
}
LISP_TO_SPEC.update(LISP_ALIASES)

# Constant-time array LUTs for XLA kernels
_hex_to_dense_np = np.zeros(64, dtype=np.int32)
_dense_to_hex_np = np.zeros(NUM_OPCODES, dtype=np.uint16)
for _spec in OPCODE_SPECS:
    _hex_to_dense_np[int(_spec.opcode) & 0x3F] = _spec.dense_id
    _dense_to_hex_np[_spec.dense_id] = int(_spec.opcode)
    # Also allow dense IDs (0..15) if passed directly where non-colliding
    if _spec.dense_id < 16 and _hex_to_dense_np[_spec.dense_id] == 0 and _spec.dense_id != 0:
        _hex_to_dense_np[_spec.dense_id] = _spec.dense_id

HEX_TO_DENSE_LUT = jnp.asarray(_hex_to_dense_np, dtype=jnp.int32)
DENSE_TO_HEX_LUT = jnp.asarray(_dense_to_hex_np, dtype=jnp.uint16)

# ============================================================================
# 16-Register Bank Architecture
# ============================================================================
NUM_REGS: int = 16

# Register partitions:
# R0..R3   : Read-only problem inputs (x, y, z, w)
# R4..R10  : Mutable scratch / intermediate registers (t0..t6)
# R11      : Mutable primary output register (OUT_REG)
# R12..R15 : Read-only hardware constant registers (0, 1, 2, ~0)
INPUT_REGS: Tuple[int, ...] = (0, 1, 2, 3)
SCRATCH_REGS: Tuple[int, ...] = (4, 5, 6, 7, 8, 9, 10)
OUT_REG: int = 11
WRITABLE_REGS: Tuple[int, ...] = (4, 5, 6, 7, 8, 9, 10, 11)
CONST_REGS: Tuple[int, ...] = (12, 13, 14, 15)

REG_ZERO: int = 12      # 0x0000000000000000 (0)
REG_ONE: int = 13       # 0x0000000000000001 (1)
REG_TWO: int = 14       # 0x0000000000000002 (2)
REG_ALL_ONES: int = 15  # 0xFFFFFFFFFFFFFFFF (~0 / True mask)

CONSTANT_VALUES: Dict[int, int] = {
    0: REG_ZERO,
    1: REG_ONE,
    2: REG_TWO,
    -1: REG_ALL_ONES,
    0xFFFFFFFFFFFFFFFF: REG_ALL_ONES,
}

REG_NAMES: Dict[int, str] = {
    0: "r0(x)",
    1: "r1(y)",
    2: "r2(z)",
    3: "r3(w)",
    4: "t0",
    5: "t1",
    6: "t2",
    7: "t3",
    8: "t4",
    9: "t5",
    10: "t6",
    11: "out",
    12: "const_0",
    13: "const_1",
    14: "const_2",
    15: "const_all1",
}


def encode_select_operands(cond_reg: int, true_reg: int, false_reg: int) -> Tuple[int, int]:
    """
    Pack ternary SELECT (cond ? true_reg : false_reg) into (Reg_Src1, Reg_Src2).
    - Reg_Src1 holds `true_reg` (or condition if using legacy mode).
    - Upper nibble of Reg_Src2 holds `cond_reg + 1` (1-indexed so 0 means legacy R0/Src1 mode),
      lower nibble holds `false_reg`.
    """
    packed_s2 = (((cond_reg & 0x0F) + 1) << 4) | (false_reg & 0x0F)
    return (true_reg & 0x0F), packed_s2


def decode_select_operands(s1: int, s2: int) -> Tuple[int, int, int]:
    """
    Unpack (Reg_Src1, Reg_Src2) into (cond_reg, true_reg, false_reg).
    Supports both packed ternary encoding and the specification's legacy 2-operand form
    where s1 is condition, s2 is true branch, and R0 is false branch.
    """
    hi = (s2 >> 4) & 0x0F
    lo = s2 & 0x0F
    if hi == 0:
        # Spec Section 5.3 line 164 compatibility: (regs[s1] & regs[s2]) | (~regs[s1] & regs[0])
        return (s1 & 0x0F), (lo & 0x0F), 0
    return (hi - 1) & 0x0F, (s1 & 0x0F), lo


def make_instruction(
    opcode: int | Opcode,
    dest: int = OUT_REG,
    src1: int = 0,
    src2: int = 0,
) -> List[int]:
    """Construct a single 4-tuple instruction `[Opcode, Reg_Dest, Reg_Src1, Reg_Src2]`."""
    return [int(opcode), int(dest) & 0x0F, int(src1) & 0x0F, int(src2) & 0xFF]


def pad_program(instructions: List[List[int]], max_steps: int) -> np.ndarray:
    """Pad a list of 4-tuple instructions with NOPs up to `max_steps` (`[L, 4]`, `uint16`)."""
    prog = np.zeros((max_steps, 4), dtype=np.uint16)
    n = min(len(instructions), max_steps)
    if n > 0:
        prog[:n] = np.asarray(instructions[:n], dtype=np.uint16)
    return prog
