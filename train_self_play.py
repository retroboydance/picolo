"""
Autonomous Self-Supervised Training & Validation Loop (`train_self_play.py`).

Trains the Tier 2 Continuous Latent Reasoning Core (JEPA + SIGReg + Dialectical Clamping)
and Tier 3 Grammar-Constrained Decoder via Asymmetric Self-Play, saving portable `.npz`
checkpoints (`checkpoints/llre_best.npz`, `checkpoints/llre_latest.npz`) and validating
pure-neural synthesis accuracy (`allow_hints=False`) at regular intervals.

Usage:
    python train_self_play.py --epochs 40 --val-every 10
    python train_self_play.py --resume --epochs 40
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any, Dict, List

from llre.checkpoint import (
    DEFAULT_CHECKPOINT_DIR,
    append_metrics_log,
    load_checkpoint,
    save_checkpoint,
)
from llre.config import get_config
from llre.isa import OUT_REG, Opcode, encode_select_operands, make_instruction, pad_program
from llre.latent_core import extract_spec_features, train_latent_core_step
from llre.self_play import LogicChallenge, SelfPlayEngine, build_core_curriculum
import jax
import jax.numpy as jnp
import numpy as np


def _canonicalize_elite_bytecode(engine: SelfPlayEngine, raw_bytecode: List[List[int]]) -> np.ndarray:
    """
    Canonicalize a verified circuit into a fixed-layout `[max_steps, 4]` target tensor
    compatible with `grammar_masks` (active instructions placed in steps `0..k-2` and final
    output write placed in step `max_steps - 1`).
    """
    active_tuples = engine.compiler.extract_active_instructions(
        np.asarray(raw_bytecode, dtype=np.uint16), eliminate_dead_code=True
    )
    compact_instrs: List[List[int]] = []
    for idx, (_, spec, _dest, s1, s2, cond_r) in enumerate(active_tuples):
        is_last = idx == len(active_tuples) - 1
        target_dest = OUT_REG if is_last else min(4 + idx, 10)
        if spec.opcode == Opcode.SELECT:
            es1, es2 = encode_select_operands(cond_r, s1, s2)
            compact_instrs.append(make_instruction(spec.opcode, target_dest, es1, es2))
        else:
            compact_instrs.append(make_instruction(spec.opcode, target_dest, s1, s2))

    canonical_prog = pad_program(compact_instrs, engine.max_steps)
    if len(compact_instrs) > 0 and len(compact_instrs) < engine.max_steps:
        last_op = compact_instrs[-1]
        canonical_prog = pad_program(compact_instrs[:-1], engine.max_steps)
        canonical_prog[engine.max_steps - 1] = np.asarray(
            [last_op[0], OUT_REG, last_op[2], last_op[3]], dtype=np.uint16
        )
    return canonical_prog


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLRE Self-Supervised Training Loop with Checkpointing & Pure-Neural Validation"
    )
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs to run")
    parser.add_argument("--lr", type=float, default=0.015, help="AdamW learning rate")
    parser.add_argument("--val-every", type=int, default=10, help="Validate and checkpoint every N epochs")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(DEFAULT_CHECKPOINT_DIR),
        help="Directory to store .npz checkpoints and training_metrics.jsonl",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoints/llre_latest.npz if present",
    )
    args = parser.parse_args()

    cfg = get_config()
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = ckpt_dir / "llre_latest.npz"
    best_ckpt = ckpt_dir / "llre_best.npz"
    metrics_log = ckpt_dir / "training_metrics.jsonl"

    print(cfg.summary())

    engine = SelfPlayEngine(max_steps=8, latent_dim=cfg.latent_dim, explore_batch_size=512)
    start_epoch = 0
    best_greedy_rate = -1.0

    if args.resume and latest_ckpt.exists():
        loaded_params, meta = load_checkpoint(latest_ckpt)
        engine.params = loaded_params
        from llre.latent_core import init_optimizer_state

        engine.opt_state = init_optimizer_state(engine.params)
        start_epoch = int(meta.get("epoch", 0))
        best_greedy_rate = float(meta.get("best_greedy_1shot_unsat_rate", -1.0))
        print(
            f"\n[Checkpoint] Resumed from {latest_ckpt} (epoch={start_epoch}, "
            f"best_greedy_unsat={best_greedy_rate * 100:.1f}%)"
        )

    challenges: List[LogicChallenge] = build_core_curriculum()

    # Phase A: Pre-Training Pure-Neural Baseline Validation (Epoch 0)
    print("\n[Validation @ Epoch 0 (Before Training)] Evaluating pure-neural policy (no seed hints)...")
    val0 = engine.validate_neural_policy(challenges)
    print(
        f"  -> 1-Shot Greedy Neural UNSAT Rate : {val0['greedy_1shot_unsat_rate'] * 100:5.1f}% "
        f"(Mean Bit Error: {val0['mean_greedy_bit_errors']:.1f})"
    )
    print(
        f"  -> Latent Cloud Neural UNSAT Rate  : {val0['neural_cloud_unsat_rate'] * 100:5.1f}% "
        f"(Mean Energy Drop: -{val0['mean_energy_drop']:.4f})"
    )

    # Phase B: Explorer Bob + Verifier discover and certify ground-truth circuits via Sandbox
    print("\n[Self-Play Discovery] Exploring & formally certifying target circuits in Sandbox...")
    spec_feats_list: List[jnp.ndarray] = []
    elite_progs_list: List[np.ndarray] = []
    defect_rates_list: List[float] = []

    for ch in challenges:
        test_in, test_out = engine._generate_sandbox_tests(ch, num_tests=64)
        s_feat = extract_spec_features(
            test_in, test_out, num_inputs=len(ch.inputs), output_bits=ch.outputs[0].bits
        )
        prim, _ = engine.solve_challenge(ch, allow_hints=True)
        canonical_prog = _canonicalize_elite_bytecode(engine, prim.bytecode)
        spec_feats_list.append(s_feat)
        elite_progs_list.append(canonical_prog)
        defect_rates_list.append(0.0 if prim.status == "UNSAT" else 1.0)

    spec_batch = jnp.stack(spec_feats_list, axis=0)
    elite_batch = jnp.asarray(np.stack(elite_progs_list, axis=0), dtype=jnp.uint16)
    defect_batch = jnp.asarray(defect_rates_list, dtype=jnp.float32)

    repeats = max(1, 32 // len(challenges))
    spec_aug = jnp.repeat(spec_batch, repeats, axis=0)
    elite_aug = jnp.repeat(elite_batch, repeats, axis=0)
    defect_aug = jnp.repeat(defect_batch, repeats, axis=0)

    print(
        f"\n[Self-Supervised Training] Running {args.epochs} epochs "
        f"(lr={args.lr}, SIGReg weight={cfg.sigreg_weight})..."
    )
    print("-----------------------------------------------------------------------------------------")
    print(
        f" {'Epoch':<7} | {'Total Loss':<10} | {'Decoder CE':<10} | {'SIGReg':<8} | "
        f"{'1-Shot UNSAT':<12} | {'Cloud UNSAT':<11} | {'Saved'}"
    )
    print("-----------------------------------------------------------------------------------------")

    t_start = time.perf_counter()
    for ep in range(1, args.epochs + 1):
        curr_epoch = start_epoch + ep
        engine.rng_key, step_key = jax.random.split(engine.rng_key)
        engine.params, metrics, engine.opt_state = train_latent_core_step(
            engine.params,
            spec_aug,
            elite_aug,
            defect_aug,
            engine.grammar_masks,
            step_key,
            lr=args.lr,
            sigreg_weight=cfg.sigreg_weight,
            max_steps=engine.max_steps,
            sigreg_projections=cfg.sigreg_projections,
            opt_state=engine.opt_state,
        )

        if ep % args.val_every == 0 or ep == args.epochs:
            val = engine.validate_neural_policy(challenges)
            greedy_rate = float(val["greedy_1shot_unsat_rate"])
            cloud_rate = float(val["neural_cloud_unsat_rate"])
            tot_loss = float(metrics["total_loss"])
            dec_ce = float(metrics["decoder_ce"])
            sig_l = float(metrics["sigreg_loss"])

            is_best = greedy_rate >= best_greedy_rate
            if is_best:
                best_greedy_rate = greedy_rate

            meta_record: Dict[str, Any] = {
                "epoch": curr_epoch,
                "total_loss": round(tot_loss, 5),
                "decoder_ce": round(dec_ce, 5),
                "sigreg_loss": round(sig_l, 5),
                "effective_dim": round(float(metrics["effective_dim"]), 3),
                "greedy_1shot_unsat_rate": round(greedy_rate, 4),
                "neural_cloud_unsat_rate": round(cloud_rate, 4),
                "mean_greedy_bit_errors": round(float(val["mean_greedy_bit_errors"]), 2),
                "best_greedy_1shot_unsat_rate": round(best_greedy_rate, 4),
                "per_task_validation": val["per_task"],
            }

            save_checkpoint(engine.params, latest_ckpt, metadata=meta_record)
            saved_tag = "latest"
            if is_best:
                save_checkpoint(engine.params, best_ckpt, metadata=meta_record)
                saved_tag = "latest + BEST"
            append_metrics_log(metrics_log, meta_record)

            print(
                f" {curr_epoch:<7} | {tot_loss:<10.4f} | {dec_ce:<10.4f} | {sig_l:<8.4f} | "
                f"{greedy_rate * 100:>10.1f}%  | {cloud_rate * 100:>9.1f}%  | {saved_tag}"
            )

    elapsed = time.perf_counter() - t_start
    print("-----------------------------------------------------------------------------------------")
    print(f"Training completed in {elapsed:.2f}s.")
    print(f"  -> Best Checkpoint   : {best_ckpt}")
    print(f"  -> Latest Checkpoint : {latest_ckpt}")
    print(f"  -> Validation Log    : {metrics_log}")
    print(f"  -> Primitive Library : {cfg.primitive_lib_path}")


if __name__ == "__main__":
    main()
