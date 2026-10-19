"""
Tier 1: Peripheral I/O Adapter (`llre/io_adapter.py`).

Treats natural language strictly as an edge I/O peripheral (Sections 1.2, 3, 8.4):
1. `nl_to_challenge(query)`: Translates a natural language query or formal LISP spec
   into a structured `LogicChallenge` (ports + formal `(specification ...)`).
2. `explain_verified_circuit(primitive)`: Deterministic, zero-hallucination rule-based
   AST-to-English & AST-to-Python/C explanation renderer for formally verified circuits.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from llre.isa import (
    Opcode,
    REG_ALL_ONES,
    REG_ONE,
    REG_TWO,
    REG_ZERO,
)
from llre.lisp_compiler import (
    LispCompiler,
    PortDecl,
    parse_lisp_module,
)
from llre.self_play import (
    LogicChallenge,
    SelfPlayEngine,
    VerifiedPrimitive,
    build_expanded_curriculum,
)
import numpy as np


class PeripheralIOAdapter:
    """
    Tier 1 Edge Adapter translating between Human Natural Language / Formal Specs
    and the Token-Free Latent Reasoning Core.
    """

    def __init__(self, engine: Optional[SelfPlayEngine] = None) -> None:
        self.engine = engine or SelfPlayEngine()
        self.compiler = LispCompiler(max_steps=self.engine.max_steps)
        self.curriculum_map: Dict[str, LogicChallenge] = {
            ch.name: ch for ch in build_expanded_curriculum(max_level=3)
        }

    def parse_query_to_challenge(self, query: str) -> LogicChallenge:
        """
        Translate either a raw LISP `(module ...)` specification or a natural-language
        goal statement into a formal `LogicChallenge`.
        """
        stripped = query.strip()
        if stripped.startswith("(") and "module" in stripped[:24]:
            mod = parse_lisp_module(stripped)
            from llre.lisp_compiler import format_sexpr

            spec_str = (
                format_sexpr(mod.spec_expr)
                if mod.spec_expr is not None
                else "(assert (= result (= (mod x 2) 0)))"
            )
            return LogicChallenge(
                name=mod.name,
                description=f"Formal LISP module '{mod.name}'",
                inputs=mod.inputs,
                outputs=mod.outputs,
                spec_sexpr_str=spec_str,
            )

        q_lower = stripped.lower()
        # Semantic pattern routing to formal SMT specifications (Levels 1..3)
        patterns: List[Tuple[List[str], str]] = [
            (["two lowest", "clear two", "double kernighan"], "clear_two_lowest_set_bits"),
            (["trailing zero", "mask trailing", "(~x) & (x - 1)"], "mask_trailing_zeros"),
            (["isolate lowest zero", "lowest 0-bit", "(~x) & (x + 1)"], "isolate_lowest_zero_bit"),
            (["set lowest zero", "x | (x + 1)"], "set_lowest_zero_bit"),
            (["min(", "minimum", "cmp_ult", "unsigned min"], "branchless_min_u64"),
            (["2*x + y + 1", "congruential", "(x << 1) + y + 1"], "linear_congruential_step"),
            (["(x ^ y) - y", "abs_diff", "sign-mask"], "branchless_abs_diff_xor"),
            (["swap lsb", "(x & 1) << 1", "shift to bit position 1"], "swap_lsb_to_bit1"),
            (["even", "parity", "divisible by 2", "mod 2 == 0"], "parity_checker"),
            (["odd", "least significant bit", "lsb == 1"], "odd_lsb_detector"),
            (["clear lowest", "drop lowest", "x & (x - 1)", "kernighan"], "clear_lowest_set_bit"),
            (["isolate lowest", "lowest set", "x & (-x)", "x & -x"], "isolate_lowest_set_bit"),
            (["mux", "multiplex", "select", "ternary", "if mask"], "branchless_mux"),
            (["nand", "not and", "~(x & y)"], "de_morgan_nand"),
            (["xor", "symmetric difference", "differing bits"], "bitwise_xor_diff"),
            (["add", "sum", "increment", "x + y + 1"], "modular_add_inc"),
        ]
        for keywords, task_name in patterns:
            if any(kw in q_lower for kw in keywords):
                return self.curriculum_map[task_name]

        # Default fallback if query mentions a curriculum key directly
        for name, ch in self.curriculum_map.items():
            if name in q_lower:
                return ch

        return self.curriculum_map["parity_checker"]

    def render_english_explanation(
        self,
        primitive: VerifiedPrimitive,
        inputs: Optional[List[PortDecl]] = None,
    ) -> str:
        """
        Deterministic rule-based AST-to-English rendering template (Section 8.4).
        Explains each active SSA hardware instruction with 0% hallucination.
        """
        in_ports = inputs or [PortDecl("x", 64)]
        prog = np.asarray(primitive.bytecode, dtype=np.uint16)
        active = self.compiler.extract_active_instructions(prog, eliminate_dead_code=True)

        reg_desc: Dict[int, str] = {
            REG_ZERO: "constant 0 (0x0)",
            REG_ONE: "constant 1 (0x1)",
            REG_TWO: "constant 2 (0x2)",
            REG_ALL_ONES: "all-ones mask (~0)",
        }
        for i in range(4):
            reg_desc[i] = f"input `{in_ports[i].name}` (R{i})" if i < len(in_ports) else f"R{i}"

        english_steps: List[str] = []
        for idx, (_, spec, dest, s1, s2, cond_r) in enumerate(active):
            lhs = "output `result` (R11)" if (idx == len(active) - 1 and dest == 11) else f"register `t{idx}` (R{dest})"
            a1 = reg_desc.get(s1, f"R{s1}")
            a2 = reg_desc.get(s2, f"R{s2}")
            ac = reg_desc.get(cond_r, f"R{cond_r}")

            op = spec.opcode
            if op == Opcode.AND:
                desc = f"Compute bitwise AND of {a1} and {a2}, storing into {lhs}."
            elif op == Opcode.OR:
                desc = f"Compute bitwise OR of {a1} and {a2}, storing into {lhs}."
            elif op == Opcode.XOR:
                desc = f"Compute bitwise XOR of {a1} and {a2}, storing into {lhs}."
            elif op == Opcode.NOT:
                desc = f"Invert all bits (bitwise NOT) of {a1}, storing into {lhs}."
            elif op == Opcode.NAND:
                desc = f"Compute bitwise NAND ~( {a1} & {a2} ), storing into {lhs}."
            elif op == Opcode.SHL:
                desc = f"Shift {a1} left logically by {a2} bit(s), storing into {lhs}."
            elif op == Opcode.SHR:
                desc = f"Shift {a1} right logically by {a2} bit(s), storing into {lhs}."
            elif op == Opcode.ADD:
                desc = f"Compute 64-bit modular addition of {a1} and {a2}, storing into {lhs}."
            elif op == Opcode.SUB:
                desc = f"Compute 64-bit modular subtraction of {a2} from {a1}, storing into {lhs}."
            elif op == Opcode.NEG:
                desc = f"Compute two's complement negation (-{a1}), storing into {lhs}."
            elif op == Opcode.CMP_ULT:
                desc = f"Compare unsigned ({a1} < {a2}) to produce 64-bit mask (~0 if true else 0) in {lhs}."
            elif op == Opcode.SELECT:
                desc = f"Branchlessly multiplex {a1} (where mask {ac} is 1) and {a2} (where mask {ac} is 0) into {lhs}."
            else:
                desc = f"Apply `{spec.mnemonic}` to ({a1}, {a2}) and store into {lhs}."

            english_steps.append(f"  {idx + 1}. {desc}")
            reg_desc[dest] = f"`t{idx}` (R{dest})"

        header = [
            f"Verified Circuit Explanation for `{primitive.name}` ({primitive.status} via {primitive.backend}):",
            f"  Goal        : {primitive.description}",
            f"  Gate Count  : {primitive.active_gates} branchless ALU instruction(s)",
            f"  Latent Drop : E(z_0)={primitive.initial_energy:.4f} -> E(z*)={primitive.final_energy:.4f}",
            "  Execution Steps:",
        ]
        return "\n".join(header + english_steps)

    def query(self, natural_language_or_lisp: str) -> Dict[str, object]:
        """End-to-end query handler from Natural Language -> Latent Trajectory -> Verified AST."""
        challenge = self.parse_query_to_challenge(natural_language_or_lisp)
        primitive, trajectory = self.engine.solve_challenge(challenge)
        explanation = self.render_english_explanation(primitive, inputs=challenge.inputs)
        return {
            "challenge": challenge,
            "primitive": primitive,
            "trajectory": trajectory,
            "explanation": explanation,
        }
