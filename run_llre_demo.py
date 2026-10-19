"""
End-to-End Execution & Benchmark Showcase for the Latent-Space Logic Reasoning Engine (LLRE).

Runs across both Windows (CPU) and Linux (CUDA GPU) using `.env` configuration:
1. Displays Cross-Platform Hardware & JAX XLA Runtime Configuration.
2. Benchmarks Tier 4 Vectorized Logic Sandbox throughput (programs/sec & bit-gate ops/sec).
3. Runs Tier 2/3/4/5 Autonomous Self-Play Curriculum with SIGReg & Dialectical Clamping.
4. Prints the Section 7 Observability Report (Formal Proof Rate, Energy Descent, Effective Dim, Probes).
5. Demonstrates Tier 1 Peripheral I/O Adapter translating Natural Language queries to formally
   verified LISP circuits, SMT-LIB 2.0 certificates, and deterministic English explanations.
"""

from __future__ import annotations

import argparse
from llre.config import get_config
from llre.io_adapter import PeripheralIOAdapter
from llre.metrics import build_observability_report
from llre.sandbox import LogicSandbox
from llre.self_play import SelfPlayEngine, build_core_curriculum


def _run_single_query(adapter: PeripheralIOAdapter, query_str: str, show_smt2: bool = False) -> None:
    print(f"\n>>> Query: \"{query_str}\"")
    out = adapter.query(query_str)
    prim = out["primitive"]
    print(out["explanation"])
    print("  Homoiconic LISP AST:")
    for line in prim.lisp_module.splitlines():
        print(f"    {line}")
    if show_smt2:
        print("  SMT-LIB 2.0 Certificate:")
        for line in prim.smt2_certificate.splitlines():
            print(f"    {line}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Latent-Space Logic Reasoning Engine (LLRE) CLI & Interactive Tester"
    )
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        default=None,
        help="Synthesize & formally verify a circuit for a natural language query or LISP (module ...)",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Launch interactive REPL to test natural language queries and LISP specifications",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Run only the Tier 4 Vectorized Logic Sandbox throughput benchmark",
    )
    parser.add_argument(
        "--smt2",
        action="store_true",
        help="Print the generated SMT-LIB 2.0 formal verification script for queries",
    )
    args = parser.parse_args()

    cfg = get_config()
    print(cfg.summary())

    if args.benchmark_only:
        print("\n[Phase 1] Benchmarking Tier 4 JAX Vectorized Logic Sandbox...")
        sandbox = LogicSandbox()
        bm = sandbox.benchmark(batch_size=cfg.batch_size, max_steps=cfg.max_steps, iterations=25)
        print(
            f"  -> Throughput : {bm['programs_per_sec']:,.0f} candidate programs/sec "
            f"({bm['test_case_evals_per_sec']:,.0f} test evals/sec | "
            f"{bm['bit_gate_ops_per_sec'] / 1e9:.2f} Gbit-gate ops/sec)"
        )
        return

    if args.query is not None or args.interactive:
        engine = SelfPlayEngine(max_steps=8, latent_dim=cfg.latent_dim, explore_batch_size=512)
        adapter = PeripheralIOAdapter(engine=engine)
        if args.query is not None:
            _run_single_query(adapter, args.query, show_smt2=args.smt2)
        if args.interactive:
            print("\n=== LLRE Interactive Mode (type 'exit' or 'quit' to stop) ===")
            while True:
                try:
                    user_q = input("\nLLRE> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not user_q or user_q.lower() in ("exit", "quit", "q"):
                    break
                _run_single_query(adapter, user_q, show_smt2=args.smt2)
        return

    # Full end-to-end showcase (default when no flags are passed)
    print("\n[Phase 1] Benchmarking Tier 4 JAX Vectorized Logic Sandbox...")
    sandbox = LogicSandbox()
    bm = sandbox.benchmark(batch_size=cfg.batch_size, max_steps=cfg.max_steps, iterations=20)
    print(
        f"  -> Throughput : {bm['programs_per_sec']:,.0f} candidate programs/sec "
        f"({bm['test_case_evals_per_sec']:,.0f} test evals/sec | "
        f"{bm['bit_gate_ops_per_sec'] / 1e9:.2f} Gbit-gate ops/sec)"
    )

    print("\n[Phase 2 & 3] Running Autonomous Self-Play Curriculum (JEPA + SIGReg + Formal Verifier)...")
    engine = SelfPlayEngine(max_steps=8, latent_dim=cfg.latent_dim, explore_batch_size=512)
    challenges = build_core_curriculum()
    sp_result = engine.run_self_play_curriculum(epochs=8, challenges=challenges)

    report = build_observability_report(
        challenges=challenges,
        primitives=sp_result["primitives"],
        trajectories=sp_result["trajectories"],
        rng_seed=cfg.random_seed,
    )

    print("\n============================================================")
    print(" Section 7: LLRE Metrics & Observability Dashboard")
    print("============================================================")
    print(f" Formal Proof Rate (UNSAT)    : {report.formal_proof_rate * 100:.1f}% (0% hallucination tolerance)")
    print(f" Bitwise Defect Density       : {report.mean_bitwise_defect_density:.4f}")
    print(f" Mean Latent Energy Drop      : -{report.mean_energy_drop:.4f} (Monotonic: {report.energy_monotonic_rate * 100:.1f}%)")
    print(f" Effective Dimensionality     : {report.effective_dimensionality:.2f} / {cfg.latent_dim} (SIGReg loss={report.sigreg_loss:.4f})")
    print(f" Linear Diagnostic Probe Acc  : {report.linear_probe_accuracy * 100:.1f}% (ALU vs Bitwise classification)")
    print("------------------------------------------------------------")
    print(f" {'Task Name':<26} | {'Status':<6} | {'Gates':<5} | {'Reward':<6} | {'E(z_0)->E(z*)':<18}")
    print("------------------------------------------------------------")
    for row in report.per_task_summary:
        e_str = f"{row['energy_init']:.3f} -> {row['energy_final']:.3f}"
        print(
            f" {row['task']:<26} | {row['status']:<6} | {row['active_gates']:<5} | "
            f"{row['reward']:<6.2f} | {e_str:<18}"
        )
    print("============================================================")

    print("\n[Phase 4] Tier 1 Peripheral I/O Adapter Demonstration:")
    adapter = PeripheralIOAdapter(engine=engine)
    sample_queries = [
        "Check whether an integer x is even.",
        "Clear the lowest set bit of x using Kernighan's trick.",
        "Branchless multiplexer: select x where mask z is 1 and y where z is 0.",
    ]
    for q in sample_queries:
        _run_single_query(adapter, q, show_smt2=args.smt2)


if __name__ == "__main__":
    main()
