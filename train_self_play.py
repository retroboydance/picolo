"""
Autonomous Self-Supervised Training & Dynamic Curriculum Expansion Loop (`train_self_play.py`).

Trains the Tier 2 Continuous Latent Reasoning Core (JEPA + SIGReg + Dialectical Clamping)
and Tier 3 Grammar-Constrained Decoder via Asymmetric Self-Play:
1. Starts at Curriculum Level 1 (8 atomic & 2-gate tasks).
2. Validates pure-neural synthesis (`allow_hints=False`) every `--val-every` epochs.
3. Automatically triggers Dynamic Curriculum Expansion (Level 1 -> Level 2 -> Level 3,
   up to 16 compound 1-to-4 gate circuits) whenever 1-Shot Greedy Neural `UNSAT` hits 100%.
4. Persists both model weights (`params`) and AdamW momentum state (`opt_state`) in
   cross-OS `.npz` checkpoints (`checkpoints/llre_latest.npz`, `checkpoints/llre_best.npz`)
   for seamless `--resume` across Windows (CPU) and Linux (CUDA GPU).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

from llre.checkpoint import (
    DEFAULT_CHECKPOINT_DIR,
    append_metrics_log,
    load_checkpoint,
    save_checkpoint,
)
from llre.config import get_config
from llre.isa import OUT_REG, Opcode, encode_select_operands, make_instruction, pad_program
from llre.latent_core import extract_spec_features, init_optimizer_state, train_latent_core_step
from llre.self_play import LogicChallenge, SelfPlayEngine, build_expanded_curriculum
import jax
import jax.numpy as jnp
import numpy as np


def _canonicalize_elite_bytecode(engine: SelfPlayEngine, raw_bytecode: List[List[int]]) -> np.ndarray:
    """
    Canonicalize a verified circuit into a fixed-layout `[max_steps, 4]` target tensor
    compatible with causal `grammar_masks` (active instructions placed in steps `0..k-2`
    writing to `R4, R5, ...` with SSA register renaming, and final output placed in step `max_steps - 1`).
    """
    active_tuples = engine.compiler.extract_active_instructions(
        np.asarray(raw_bytecode, dtype=np.uint16), eliminate_dead_code=True
    )
    reg_remap: Dict[int, int] = {}
    compact_instrs: List[List[int]] = []

    for idx, (_, spec, orig_dest, s1, s2, cond_r) in enumerate(active_tuples):
        is_last = idx == len(active_tuples) - 1
        target_dest = OUT_REG if is_last else min(4 + idx, 10)

        mapped_s1 = reg_remap.get(s1, s1)
        mapped_s2 = reg_remap.get(s2, s2)
        mapped_cond = reg_remap.get(cond_r, cond_r)

        if spec.opcode == Opcode.SELECT:
            es1, es2 = encode_select_operands(mapped_cond, mapped_s1, mapped_s2)
            compact_instrs.append(make_instruction(spec.opcode, target_dest, es1, es2))
        else:
            compact_instrs.append(make_instruction(spec.opcode, target_dest, mapped_s1, mapped_s2))

        reg_remap[orig_dest] = target_dest

    canonical_prog = pad_program(compact_instrs, engine.max_steps)
    if len(compact_instrs) > 0 and len(compact_instrs) < engine.max_steps:
        last_op = compact_instrs[-1]
        canonical_prog = pad_program(compact_instrs[:-1], engine.max_steps)
        canonical_prog[engine.max_steps - 1] = np.asarray(
            [last_op[0], OUT_REG, last_op[2], last_op[3]], dtype=np.uint16
        )
    return canonical_prog


def _build_training_tensors(
    engine: SelfPlayEngine,
    challenges: List[LogicChallenge],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Discover/verify elite circuits for `challenges` and build batched JAX training tensors."""
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
    return spec_aug, elite_aug, defect_aug


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLRE Self-Supervised Training Loop with Dynamic Curriculum Expansion & Checkpointing"
    )
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs to run")
    parser.add_argument("--lr", type=float, default=0.012, help="AdamW learning rate")
    parser.add_argument("--val-every", type=int, default=10, help="Validate and checkpoint every N epochs")
    parser.add_argument(
        "--start-level",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Initial curriculum level (1=8 tasks, 2=12 tasks, 3=16 tasks)",
    )
    parser.add_argument(
        "--max-level",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="Maximum curriculum level to unlock dynamically via Proposer Alice",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(DEFAULT_CHECKPOINT_DIR),
        help="Directory to store .npz checkpoints and training_metrics.jsonl",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume weights, AdamW optimizer state, and curriculum level from checkpoints/llre_latest.npz",
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
    curr_level = args.start_level
    best_score = -1.0

    if args.resume and latest_ckpt.exists():
        loaded_params, meta, loaded_opt = load_checkpoint(latest_ckpt, return_opt_state=True)
        engine.params = loaded_params
        engine.opt_state = loaded_opt if loaded_opt is not None else init_optimizer_state(engine.params)
        start_epoch = int(meta.get("epoch", 0))
        curr_level = max(curr_level, int(meta.get("curriculum_level", 1)))
        best_score = float(meta.get("best_composite_score", -1.0))
        print(
            f"\n[Checkpoint] Resumed from {latest_ckpt} (epoch={start_epoch}, "
            f"level={curr_level}, opt_step={int(engine.opt_state['step'])})"
        )

    all_challenges = build_expanded_curriculum(max_level=args.max_level)
    active_challenges = [ch for ch in all_challenges if ch.level <= curr_level]

    print(
        f"\n[Curriculum Level {curr_level}] Active Tasks: {len(active_challenges)} / {len(all_challenges)}"
    )
    val0 = engine.validate_neural_policy(active_challenges)
    print(
        f"  -> Pre-Run 1-Shot Greedy UNSAT : {val0['greedy_1shot_unsat_rate'] * 100:5.1f}% "
        f"| Cloud UNSAT: {val0['neural_cloud_unsat_rate'] * 100:5.1f}% "
        f"(Mean Bit Error: {val0['mean_greedy_bit_errors']:.1f})"
    )

    spec_aug, elite_aug, defect_aug = _build_training_tensors(engine, active_challenges)

    print(
        f"\n[Self-Supervised Training] Running {args.epochs} epochs "
        f"(lr={args.lr}, max_level={args.max_level})..."
    )
    print("---------------------------------------------------------------------------------------------------")
    print(
        f" {'Epoch':<6} | {'Lvl (Tasks)':<11} | {'Total Loss':<10} | {'Decoder CE':<10} | "
        f"{'SIGReg':<7} | {'1-Shot UNSAT':<12} | {'Cloud UNSAT':<11} | {'Saved'}"
    )
    print("---------------------------------------------------------------------------------------------------")

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
            val = engine.validate_neural_policy(active_challenges)
            greedy_rate = float(val["greedy_1shot_unsat_rate"])
            cloud_rate = float(val["neural_cloud_unsat_rate"])
            tot_loss = float(metrics["total_loss"])
            dec_ce = float(metrics["decoder_ce"])
            sig_l = float(metrics["sigreg_loss"])

            # Composite score favors higher curriculum level + higher greedy UNSAT rate
            comp_score = float(curr_level) + greedy_rate
            is_best = comp_score >= best_score
            if is_best:
                best_score = comp_score

            meta_record: Dict[str, Any] = {
                "epoch": curr_epoch,
                "curriculum_level": curr_level,
                "active_task_count": len(active_challenges),
                "total_loss": round(tot_loss, 5),
                "decoder_ce": round(dec_ce, 5),
                "sigreg_loss": round(sig_l, 5),
                "effective_dim": round(float(metrics["effective_dim"]), 3),
                "greedy_1shot_unsat_rate": round(greedy_rate, 4),
                "neural_cloud_unsat_rate": round(cloud_rate, 4),
                "mean_greedy_bit_errors": round(float(val["mean_greedy_bit_errors"]), 2),
                "best_composite_score": round(best_score, 4),
                "per_task_validation": val["per_task"],
            }

            save_checkpoint(engine.params, latest_ckpt, metadata=meta_record, opt_state=engine.opt_state)
            saved_tag = "latest"
            if is_best:
                save_checkpoint(engine.params, best_ckpt, metadata=meta_record, opt_state=engine.opt_state)
                saved_tag = "latest + BEST"
            append_metrics_log(metrics_log, meta_record)

            lvl_str = f"L{curr_level} ({len(active_challenges):2d})"
            print(
                f" {curr_epoch:<6} | {lvl_str:<11} | {tot_loss:<10.4f} | {dec_ce:<10.4f} | "
                f"{sig_l:<7.4f} | {greedy_rate * 100:>10.1f}%  | {cloud_rate * 100:>9.1f}%  | {saved_tag}"
            )

            # Dynamic Curriculum Expansion: If 100% 1-Shot UNSAT achieved on current level, unlock next level!
            if greedy_rate >= 1.0 and curr_level < args.max_level and ep < args.epochs:
                curr_level += 1
                active_challenges = [ch for ch in all_challenges if ch.level <= curr_level]
                new_names = [ch.name for ch in active_challenges if ch.level == curr_level]
                print(
                    f"   >>> [Curriculum Expansion] Level {curr_level - 1} mastered (100% 1-Shot UNSAT)! "
                    f"Unlocking Level {curr_level} (+{len(new_names)} tasks: {', '.join(new_names)})"
                )
                spec_aug, elite_aug, defect_aug = _build_training_tensors(engine, active_challenges)

    elapsed = time.perf_counter() - t_start
    print("---------------------------------------------------------------------------------------------------")
    print(
        f"Training completed in {elapsed:.2f}s (Final Level: L{curr_level}, "
        f"Verified Primitives in Library: {len(engine.library.primitives)})."
    )
    print(f"  -> Best Checkpoint   : {best_ckpt}")
    print(f"  -> Latest Checkpoint : {latest_ckpt}")
    print(f"  -> Validation Log    : {metrics_log}")
    print(f"  -> Primitive Library : {cfg.primitive_lib_path}")


if __name__ == "__main__":
    main()
