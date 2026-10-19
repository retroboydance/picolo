"""
Tier 3: Symbolic Intermediate Logic Language (LISP / SMT-LIB AST Compiler).

Implements:
1. Homoiconic S-expression parser for `(module ...)` definitions (Section 4.2).
2. Deterministic O(L) bidirectional compiler: `LISP AST <-> Flat Tensor Bytecode [L, 4]`.
3. Backward dead-code elimination (DCE) for minimal canonical circuit decompilation.
4. Causal SSA grammar-constraint masks for neural decoding (making syntax/uninitialized-read
   errors mathematically impossible).
5. Standard SMT-LIB 2.0 (`QF_BV`) bit-vector code generator for Tier 5 formal solvers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from llre.isa import (
    CONSTANT_VALUES,
    DENSE_TO_SPEC,
    HEX_TO_SPEC,
    LISP_TO_SPEC,
    NUM_OPCODES,
    NUM_REGS,
    OUT_REG,
    Opcode,
    OpcodeSpec,
    REG_ALL_ONES,
    REG_ONE,
    REG_TWO,
    REG_ZERO,
    WRITABLE_REGS,
    decode_select_operands,
    encode_select_operands,
    make_instruction,
    pad_program,
)
import jax.numpy as jnp
import numpy as np

SExpr = Union[str, int, List["SExpr"]]


@dataclass
class PortDecl:
    """Input or output port declaration in a LISP module."""

    name: str
    bits: int = 64


@dataclass
class LispModule:
    """Parsed representation of a Tier 3 LISP Logic Module."""

    name: str
    inputs: List[PortDecl] = field(default_factory=lambda: [PortDecl("x", 64)])
    outputs: List[PortDecl] = field(default_factory=lambda: [PortDecl("result", 64)])
    circuit_expr: Optional[SExpr] = None
    spec_expr: Optional[SExpr] = None


def tokenize_sexpr(source: str) -> List[str]:
    """Tokenize a LISP S-expression string, stripping `;` line comments."""
    cleaned_lines = []
    for line in source.splitlines():
        if ";" in line:
            line = line.split(";", 1)[0]
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    text = text.replace("(", " ( ").replace(")", " ) ")
    return [tok for tok in text.split() if tok]


def parse_sexpr(source: str) -> SExpr:
    """Parse a LISP S-expression string into nested Python lists and atoms."""
    tokens = tokenize_sexpr(source)
    if not tokens:
        raise ValueError("Empty S-expression input")

    pos = 0

    def _parse() -> SExpr:
        nonlocal pos
        if pos >= len(tokens):
            raise SyntaxError("Unexpected EOF while parsing S-expression")
        tok = tokens[pos]
        pos += 1
        if tok == "(":
            items: List[SExpr] = []
            while pos < len(tokens) and tokens[pos] != ")":
                items.append(_parse())
            if pos >= len(tokens) or tokens[pos] != ")":
                raise SyntaxError("Missing closing parenthesis ')'")
            pos += 1
            return items
        if tok == ")":
            raise SyntaxError("Unexpected closing parenthesis ')'")
        # Parse integer literals (decimal or hex)
        try:
            if tok.startswith("0x") or tok.startswith("0X"):
                return int(tok, 16)
            return int(tok)
        except ValueError:
            return tok

    ast = _parse()
    return ast


def format_sexpr(expr: SExpr) -> str:
    """Render a nested SExpr back to a compact LISP string."""
    if isinstance(expr, list):
        return "(" + " ".join(format_sexpr(item) for item in expr) + ")"
    return str(expr)


def _parse_port_list(items: Sequence[SExpr]) -> List[PortDecl]:
    ports: List[PortDecl] = []
    for entry in items:
        if isinstance(entry, list) and len(entry) >= 1:
            name = str(entry[0])
            bits = 64
            if len(entry) >= 3 and str(entry[1]) == ":bits":
                bits = int(entry[2])
            ports.append(PortDecl(name=name, bits=bits))
        elif isinstance(entry, str):
            ports.append(PortDecl(name=entry, bits=64))
    return ports


def parse_lisp_module(source: str) -> LispModule:
    """
    Parse a `(module <name> (inputs ...) (outputs ...) (circuit ...) (specification ...))`
    S-expression into a structured `LispModule`.
    """
    ast = parse_sexpr(source)
    if not isinstance(ast, list) or len(ast) < 2 or ast[0] != "module":
        raise ValueError("Expected top-level `(module <name> ...)` S-expression")

    mod = LispModule(name=str(ast[1]), inputs=[], outputs=[])
    for section in ast[2:]:
        if not isinstance(section, list) or not section:
            continue
        head = str(section[0])
        if head == "inputs":
            mod.inputs = _parse_port_list(section[1:])
        elif head == "outputs":
            mod.outputs = _parse_port_list(section[1:])
        elif head == "circuit" and len(section) >= 2:
            mod.circuit_expr = section[1]
        elif head == "specification" and len(section) >= 2:
            mod.spec_expr = section[1]

    if not mod.inputs:
        mod.inputs = [PortDecl("x", 64)]
    if not mod.outputs:
        mod.outputs = [PortDecl("result", 64)]
    return mod


class LispCompiler:
    """
    Deterministic bidirectional compiler between LISP ASTs (`LispModule`),
    Flat Instruction Tensors `[L, 4]`, and SMT-LIB v2 specifications.
    """

    def __init__(self, max_steps: int = 16) -> None:
        self.max_steps = max_steps

    def compile_to_bytecode(self, module: LispModule | str) -> np.ndarray:
        """
        Compile a `LispModule` (or LISP source string) into a flat instruction
        tensor of shape `[max_steps, 4]` (`uint16`).
        """
        if isinstance(module, str):
            module = parse_lisp_module(module)
        if module.circuit_expr is None:
            raise ValueError(f"Module '{module.name}' has no (circuit ...) section")

        env: Dict[str, int] = {}
        for idx, port in enumerate(module.inputs[:4]):
            env[port.name] = idx
        # Standard register aliases
        for i in range(4):
            env.setdefault(f"r{i}", i)
        env["const_0"] = REG_ZERO
        env["const_1"] = REG_ONE
        env["const_2"] = REG_TWO
        env["const_all1"] = REG_ALL_ONES

        out_name = module.outputs[0].name if module.outputs else "result"
        instructions: List[List[int]] = []
        next_scratch_idx = 0
        scratch_pool = list(range(4, 11))  # R4..R10

        def alloc_scratch(preferred_name: Optional[str] = None) -> int:
            nonlocal next_scratch_idx
            if preferred_name == out_name:
                return OUT_REG
            reg = scratch_pool[next_scratch_idx % len(scratch_pool)]
            next_scratch_idx += 1
            return reg

        def resolve_atom(atom: Union[str, int]) -> int:
            if isinstance(atom, int):
                if atom in CONSTANT_VALUES:
                    return CONSTANT_VALUES[atom]
                raise ValueError(
                    f"Unsupported immediate literal {atom}; hardware constants are {list(CONSTANT_VALUES.keys())}"
                )
            if atom in env:
                return env[atom]
            raise KeyError(f"Unbound symbol '{atom}' in circuit expression")

        def compile_expr(expr: SExpr, target_reg: Optional[int] = None) -> int:
            if isinstance(expr, (str, int)):
                reg = resolve_atom(expr)
                if target_reg is not None and reg != target_reg:
                    # Emit identity move via OR with const_0
                    instructions.append(make_instruction(Opcode.OR, target_reg, reg, REG_ZERO))
                    return target_reg
                return reg

            if not expr:
                raise ValueError("Empty expression list in circuit")

            head = str(expr[0])
            if head == "let":
                # (let ((var1 val1) (var2 val2) ...) body)
                if len(expr) < 3 or not isinstance(expr[1], list):
                    raise ValueError(f"Malformed let expression: {expr}")
                bindings = expr[1]
                # Support both `(let ((v expr)) body)` and shorthand `(let (v expr) body)`
                if len(bindings) == 2 and isinstance(bindings[0], str):
                    bindings = [bindings]
                for bind in bindings:
                    if not isinstance(bind, list) or len(bind) != 2:
                        raise ValueError(f"Invalid let binding: {bind}")
                    var_name = str(bind[0])
                    val_expr = bind[1]
                    dest = OUT_REG if var_name == out_name else alloc_scratch(var_name)
                    actual_reg = compile_expr(val_expr, target_reg=dest)
                    env[var_name] = actual_reg
                return compile_expr(expr[2], target_reg=target_reg)

            if head not in LISP_TO_SPEC:
                raise ValueError(f"Unknown LISP operator '{head}'")

            spec = LISP_TO_SPEC[head]
            dest = target_reg if target_reg is not None else alloc_scratch()

            if spec.arity == 0:
                instructions.append(make_instruction(Opcode.NOP, dest, 0, 0))
                return dest
            elif spec.arity == 1:
                s1 = compile_expr(expr[1])
                instructions.append(make_instruction(spec.opcode, dest, s1, 0))
                return dest
            elif spec.arity == 2:
                s1 = compile_expr(expr[1])
                s2 = compile_expr(expr[2])
                instructions.append(make_instruction(spec.opcode, dest, s1, s2))
                return dest
            elif spec.arity == 3:
                # (select cond true_val false_val)
                cond_r = compile_expr(expr[1])
                true_r = compile_expr(expr[2])
                false_r = compile_expr(expr[3])
                s1_enc, s2_enc = encode_select_operands(cond_r, true_r, false_r)
                instructions.append(make_instruction(spec.opcode, dest, s1_enc, s2_enc))
                return dest
            else:
                raise ValueError(f"Unsupported operator arity {spec.arity}")

        final_reg = compile_expr(module.circuit_expr, target_reg=OUT_REG)
        if final_reg != OUT_REG:
            instructions.append(make_instruction(Opcode.OR, OUT_REG, final_reg, REG_ZERO))

        return pad_program(instructions, self.max_steps)

    def extract_active_instructions(
        self,
        program: np.ndarray | jnp.ndarray,
        eliminate_dead_code: bool = True,
    ) -> List[Tuple[int, OpcodeSpec, int, int, int, int]]:
        """
        Decode `[L, 4]` tensor into active instructions and perform backward
        Dead-Code Elimination (DCE) from `OUT_REG` (`R11`).

        Returns list of `(step_idx, OpcodeSpec, dest, src1, src2, cond_reg)`.
        """
        prog_np = np.asarray(program, dtype=np.uint16)
        decoded: List[Tuple[int, OpcodeSpec, int, int, int, int]] = []

        for step_idx, row in enumerate(prog_np):
            op_raw = int(row[0])
            dest = int(row[1]) & 0x0F
            s1 = int(row[2]) & 0x0F
            s2_raw = int(row[3]) & 0xFF

            spec = HEX_TO_SPEC.get(op_raw) or DENSE_TO_SPEC.get(op_raw)
            if spec is None or spec.opcode == Opcode.NOP or dest >= 12:
                continue

            if spec.opcode == Opcode.SELECT:
                cond_r, true_r, false_r = decode_select_operands(s1, s2_raw)
                decoded.append((step_idx, spec, dest, true_r, false_r, cond_r))
            else:
                decoded.append((step_idx, spec, dest, s1, s2_raw & 0x0F, 0))

        if not eliminate_dead_code:
            return decoded

        # Backward live-register tracking starting from OUT_REG (R11)
        needed_regs: Set[int] = {OUT_REG}
        kept_reversed: List[Tuple[int, OpcodeSpec, int, int, int, int]] = []

        for item in reversed(decoded):
            step_idx, spec, dest, s1, s2, cond_r = item
            if dest in needed_regs:
                kept_reversed.append(item)
                needed_regs.discard(dest)
                if spec.arity >= 1:
                    needed_regs.add(s1)
                if spec.arity >= 2:
                    needed_regs.add(s2)
                if spec.arity == 3:
                    needed_regs.add(cond_r)

        kept_reversed.reverse()
        return kept_reversed

    def decompile_to_lisp(
        self,
        program: np.ndarray | jnp.ndarray,
        module_name: str = "synthesized_circuit",
        inputs: Optional[List[PortDecl]] = None,
        outputs: Optional[List[PortDecl]] = None,
        spec_expr: Optional[SExpr] = None,
    ) -> str:
        """
        Decompile a `[L, 4]` instruction tensor back into a homoiconic LISP `(module ...)`
        string with Dead-Code Elimination.
        """
        in_ports = inputs or [PortDecl("x", 64)]
        out_ports = outputs or [PortDecl("result", 64)]
        out_name = out_ports[0].name

        reg_sym: Dict[int, str] = {
            REG_ZERO: "0",
            REG_ONE: "1",
            REG_TWO: "2",
            REG_ALL_ONES: "-1",
        }
        for i in range(4):
            reg_sym[i] = in_ports[i].name if i < len(in_ports) else f"r{i}"
        for r in range(4, 12):
            reg_sym[r] = "0"

        active = self.extract_active_instructions(program, eliminate_dead_code=True)
        let_bindings: List[Tuple[str, str]] = []

        for idx, (_, spec, dest, s1, s2, cond_r) in enumerate(active):
            is_last = idx == len(active) - 1
            var_name = out_name if (is_last and dest == OUT_REG) else f"t{idx}"
            a1 = reg_sym.get(s1, "0")
            a2 = reg_sym.get(s2, "0")
            ac = reg_sym.get(cond_r, "0")

            if spec.arity == 1:
                rhs = f"({spec.lisp_symbol} {a1})"
            elif spec.arity == 2:
                rhs = f"({spec.lisp_symbol} {a1} {a2})"
            elif spec.arity == 3:
                rhs = f"({spec.lisp_symbol} {ac} {a1} {a2})"
            else:
                rhs = a1

            let_bindings.append((var_name, rhs))
            reg_sym[dest] = var_name

        # Construct nested let expression
        final_val = reg_sym.get(OUT_REG, "0")
        if not let_bindings:
            circuit_body = f"    {final_val}"
        else:
            indent = "    "
            lines_open = []
            for depth, (vname, rhs) in enumerate(let_bindings):
                lines_open.append(f"{indent}{'  ' * depth}(let (({vname} {rhs}))")
            closing = ")" * len(let_bindings)
            lines_open.append(f"{indent}{'  ' * len(let_bindings)}{final_val}{closing}")
            circuit_body = "\n".join(lines_open)

        in_str = " ".join(f"({p.name} :bits {p.bits})" for p in in_ports)
        out_str = " ".join(f"({p.name} :bits {p.bits})" for p in out_ports)

        lines = [
            f"(module {module_name}",
            f"  (inputs {in_str})",
            f"  (outputs {out_str})",
            "  (circuit",
            circuit_body + ")",
        ]
        if spec_expr is not None:
            lines.append(f"  (specification\n    {format_sexpr(spec_expr)}))")
        else:
            lines[-1] = lines[-1] + ")"
        return "\n".join(lines)

    def to_smtlib2(
        self,
        program: np.ndarray | jnp.ndarray,
        inputs: Optional[List[PortDecl]] = None,
        spec_smt_expr: Optional[str] = None,
        output_bits: int = 64,
    ) -> str:
        """
        Emit a complete SMT-LIB 2.0 (`QF_BV`) verification script asserting
        `circuit(inputs) != spec(inputs)` to test for `UNSAT` (universal equivalence).
        """
        in_ports = inputs or [PortDecl("x", 64)]
        active = self.extract_active_instructions(program, eliminate_dead_code=True)

        smt_lines = [
            "(set-logic QF_BV)",
            "; Input Bit-Vector Declarations",
        ]
        for p in in_ports:
            smt_lines.append(f"(declare-const {p.name} (_ BitVec 64))")

        reg_expr: Dict[int, str] = {
            REG_ZERO: "(_ bv0 64)",
            REG_ONE: "(_ bv1 64)",
            REG_TWO: "(_ bv2 64)",
            REG_ALL_ONES: "(bvnot (_ bv0 64))",
        }
        for i in range(4):
            reg_expr[i] = in_ports[i].name if i < len(in_ports) else "(_ bv0 64)"
        for r in range(4, 12):
            reg_expr[r] = "(_ bv0 64)"

        smt_lines.append("; Synthesized Circuit SSA Instructions")
        for idx, (_, spec, dest, s1, s2, cond_r) in enumerate(active):
            v1 = reg_expr[s1]
            v2 = reg_expr[s2]
            vc = reg_expr[cond_r]
            ssa_name = f"step_{idx}_r{dest}"

            op = spec.opcode
            if op == Opcode.AND:
                expr = f"(bvand {v1} {v2})"
            elif op == Opcode.OR:
                expr = f"(bvor {v1} {v2})"
            elif op == Opcode.XOR:
                expr = f"(bvxor {v1} {v2})"
            elif op == Opcode.NOT:
                expr = f"(bvnot {v1})"
            elif op == Opcode.SHR:
                expr = f"(bvlshr {v1} (bvand {v2} (_ bv63 64)))"
            elif op == Opcode.SHL:
                expr = f"(bvshl {v1} (bvand {v2} (_ bv63 64)))"
            elif op == Opcode.NAND:
                expr = f"(bvnand {v1} {v2})"
            elif op == Opcode.NOR:
                expr = f"(bvnor {v1} {v2})"
            elif op == Opcode.XNOR:
                expr = f"(bvxnor {v1} {v2})"
            elif op == Opcode.ADD:
                expr = f"(bvadd {v1} {v2})"
            elif op == Opcode.SUB:
                expr = f"(bvsub {v1} {v2})"
            elif op == Opcode.NEG:
                expr = f"(bvneg {v1})"
            elif op == Opcode.CMP_EQ:
                expr = f"(ite (= {v1} {v2}) (bvnot (_ bv0 64)) (_ bv0 64))"
            elif op == Opcode.CMP_ULT:
                expr = f"(ite (bvult {v1} {v2}) (bvnot (_ bv0 64)) (_ bv0 64))"
            elif op == Opcode.SELECT:
                expr = f"(bvor (bvand {vc} {v1}) (bvand (bvnot {vc}) {v2}))"
            else:
                expr = v1

            smt_lines.append(f"(define-fun {ssa_name} () (_ BitVec 64) {expr})")
            reg_expr[dest] = ssa_name

        circuit_out = reg_expr[OUT_REG]
        smt_lines.append(f"(define-fun circuit_out () (_ BitVec 64) {circuit_out})")
        if spec_smt_expr:
            smt_lines.append(f"(define-fun spec_out () (_ BitVec 64) {spec_smt_expr})")
            smt_lines.append("; Assert negation of universal equivalence: exists x s.t. circuit(x) != spec(x)")
            smt_lines.append("(assert (not (= circuit_out spec_out)))")
            smt_lines.append("(check-sat)")
        return "\n".join(smt_lines)

    @staticmethod
    def build_grammar_masks(
        max_steps: int,
        num_inputs: int = 1,
        boolean_only: bool = False,
    ) -> Dict[str, jnp.ndarray]:
        """
        Construct deterministic boolean masks enforcing LISP/SSA grammar constraints
        for neural projection heads (Section 4.1):
        - `opcode_mask`: `[L, NUM_OPCODES]`
        - `dest_mask`:   `[L, NUM_REGS]` (only R4..R11 are writable; final step targets OUT_REG R11)
        - `src_mask`:    `[L, NUM_REGS]` (step l can only read declared inputs, hardware constants,
                         or scratch registers written in steps < l)
        """
        op_mask = np.ones((max_steps, NUM_OPCODES), dtype=bool)
        if boolean_only:
            for dense_id, spec in DENSE_TO_SPEC.items():
                if not spec.is_boolean_bitwise:
                    op_mask[:, dense_id] = False

        dest_mask = np.zeros((max_steps, NUM_REGS), dtype=bool)
        for l in range(max_steps):
            if l == max_steps - 1:
                dest_mask[l, OUT_REG] = True
            else:
                # Schedule SSA scratch register availability progressively
                avail_scratch = min(4 + l, 10)
                dest_mask[l, 4 : avail_scratch + 1] = True
                dest_mask[l, OUT_REG] = True

        src_mask = np.zeros((max_steps, NUM_REGS), dtype=bool)
        for l in range(max_steps):
            # Declared inputs R0..R(num_inputs-1)
            for i in range(min(max(num_inputs, 1), 4)):
                src_mask[l, i] = True
            # Hardware constants R12..R15
            for c in (REG_ZERO, REG_ONE, REG_TWO, REG_ALL_ONES):
                src_mask[l, c] = True
            # Causal SSA scratch registers written by preceding steps 0..l-1
            for prev in range(l):
                avail_reg = min(4 + prev, 10)
                src_mask[l, avail_reg] = True

        return {
            "opcode_mask": jnp.asarray(op_mask),
            "dest_mask": jnp.asarray(dest_mask),
            "src_mask": jnp.asarray(src_mask),
        }
