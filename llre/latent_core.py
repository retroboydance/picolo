"""
Tier 2: Continuous Latent Reasoning Core (JEPA + SIGReg + Dialectical Clamping).

Implements in pure functional JAX (`jax.jit`, `jax.vmap`, `jax.value_and_grad`, `jax.lax.scan`):
1. Condition Encoder mapping input-output problem signatures into latent condition `c in R^D`.
2. JEPA State Predictor and Dialectical Energy Heads:
      E_total(z; c) = E_thesis(z; c) + E_antithesis(z; c)
3. Continuous Latent Trajectory Optimization:
      z_{t+1} = z_t - eta * grad_z E_total(z_t; c)
   tracking the full energy descent curve `E(z_0) -> ... -> E(z*)`.
4. Grammar-Constrained Amortized Projection Head decoding `z*` into `[B, L, 4]` instruction
   tensors where syntax errors and uninitialized register reads are masked to -inf.
"""

from __future__ import annotations

from functools import partial
from typing import Dict, NamedTuple, Optional, Tuple

from llre.isa import (
    DENSE_TO_HEX_LUT,
    NUM_OPCODES,
    NUM_REGS,
    OUT_REG,
)
from llre.lisp_compiler import LispCompiler
from llre.sigreg import compute_effective_dimensionality, sigreg_loss
import jax
import jax.numpy as jnp
import numpy as np

PyTree = Dict[str, Dict[str, jnp.ndarray]]


class LatentTrajectoryResult(NamedTuple):
    """Result of optimizing a continuous latent trajectory z_0 -> ... -> z* (native JAX PyTree)."""

    condition_c: jnp.ndarray         # [D] encoded problem condition
    z_init: jnp.ndarray              # [D] initial latent hypothesis z_0
    z_star: jnp.ndarray              # [D] converged equilibrium state z*
    trajectory: jnp.ndarray          # [K + 1, D] latent trajectory states
    energy_curve: jnp.ndarray        # [K + 1] E_total(z_t) at each step
    thesis_energy_curve: jnp.ndarray # [K + 1] E_thesis(z_t)
    antithesis_energy_curve: jnp.ndarray  # [K + 1] E_antithesis(z_t)


def _glorot_init(rng_key: jax.Array, in_dim: int, out_dim: int) -> jnp.ndarray:
    scale = jnp.sqrt(2.0 / float(in_dim + out_dim))
    return jax.random.normal(rng_key, (in_dim, out_dim), dtype=jnp.float32) * scale


def init_latent_core_params(
    rng_key: jax.Array,
    feature_dim: int = 68,
    latent_dim: int = 64,
    hidden_dim: int = 128,
    max_steps: int = 8,
) -> PyTree:
    """Initialize pure-JAX PyTree parameters for the Tier 2 Latent Core & Tier 3 Decoder."""
    keys = jax.random.split(rng_key, 16)
    out_per_step = NUM_OPCODES + 4 * NUM_REGS  # op, dest, s1, s2, cond

    params: PyTree = {
        # Condition encoder: maps problem signature features -> latent c in R^D
        "encoder": {
            "w1": _glorot_init(keys[0], feature_dim, hidden_dim),
            "b1": jnp.zeros((hidden_dim,), dtype=jnp.float32),
            "w2": _glorot_init(keys[1], hidden_dim, latent_dim),
            "b2": jnp.zeros((latent_dim,), dtype=jnp.float32),
        },
        # JEPA Latent Predictor: predicts target attractor center z_hat from c
        "jepa_predictor": {
            "w1": _glorot_init(keys[2], latent_dim, hidden_dim),
            "b1": jnp.zeros((hidden_dim,), dtype=jnp.float32),
            "w2": _glorot_init(keys[3], hidden_dim, latent_dim),
            "b2": jnp.zeros((latent_dim,), dtype=jnp.float32),
        },
        # Thesis Energy Head (Generator / Compatibility Energy)
        "thesis_head": {
            "w1": _glorot_init(keys[4], latent_dim * 3, hidden_dim),
            "b1": jnp.zeros((hidden_dim,), dtype=jnp.float32),
            "w2": _glorot_init(keys[5], hidden_dim, 1),
            "b2": jnp.zeros((1,), dtype=jnp.float32),
        },
        # Antithesis Energy Head (Adversarial Critic / Disagreement Clamping)
        "antithesis_head": {
            "w1": _glorot_init(keys[6], latent_dim * 2, hidden_dim),
            "b1": jnp.zeros((hidden_dim,), dtype=jnp.float32),
            "w2": _glorot_init(keys[7], hidden_dim, 1),
            "b2": jnp.zeros((1,), dtype=jnp.float32),
        },
        # Grammar-Constrained Projection Head: maps z* -> [max_steps, NUM_OPCODES + 4*NUM_REGS]
        "decoder": {
            "w1": _glorot_init(keys[8], latent_dim, hidden_dim),
            "b1": jnp.zeros((hidden_dim,), dtype=jnp.float32),
            "w2": _glorot_init(keys[9], hidden_dim, max_steps * out_per_step),
            "b2": jnp.zeros((max_steps * out_per_step,), dtype=jnp.float32),
        },
    }
    return params


@jax.jit
def encode_condition(params: PyTree, spec_features: jnp.ndarray) -> jnp.ndarray:
    """Encode problem specification features `[..., F]` into latent condition `c in R^D`."""
    enc = params["encoder"]
    h = jax.nn.silu(jnp.matmul(spec_features, enc["w1"]) + enc["b1"])
    c = jnp.matmul(h, enc["w2"]) + enc["b2"]
    return c


@jax.jit
def predict_jepa_target(params: PyTree, condition_c: jnp.ndarray) -> jnp.ndarray:
    """JEPA predictor mapping condition `c` to target attractor center `z_hat` in `R^D`."""
    pred = params["jepa_predictor"]
    h = jax.nn.silu(jnp.matmul(condition_c, pred["w1"]) + pred["b1"])
    z_hat = condition_c + (jnp.matmul(h, pred["w2"]) + pred["b2"])
    return z_hat


@jax.jit
def compute_dialectical_energy(
    params: PyTree,
    z: jnp.ndarray,
    condition_c: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute Dialectical Clamping Energies (Section 2.3):
        E_total(z; c) = E_thesis(z; c) + E_antithesis(z; c)
    Returns `(e_total, e_thesis, e_antithesis)`.
    """
    z_hat = predict_jepa_target(params, condition_c)
    diff = z - z_hat

    # Thesis energy: combines neural compatibility score + quadratic attractor basin around z_hat
    th = params["thesis_head"]
    th_in = jnp.concatenate([z, condition_c, z * condition_c], axis=-1)
    th_h = jax.nn.silu(jnp.matmul(th_in, th["w1"]) + th["b1"])
    th_net = jax.nn.softplus(jnp.squeeze(jnp.matmul(th_h, th["w2"]) + th["b2"], axis=-1))
    attractor_term = 0.5 * jnp.mean(diff ** 2, axis=-1)
    e_thesis = th_net + attractor_term

    # Antithesis energy: adversarial critic evaluating residual disagreement (z - z_hat, c)
    ant = params["antithesis_head"]
    ant_in = jnp.concatenate([diff, condition_c], axis=-1)
    ant_h = jax.nn.silu(jnp.matmul(ant_in, ant["w1"]) + ant["b1"])
    ant_net = jax.nn.softplus(jnp.squeeze(jnp.matmul(ant_h, ant["w2"]) + ant["b2"], axis=-1))
    norm_penalty = 0.05 * jnp.mean((jnp.linalg.norm(z, axis=-1, keepdims=True) - jnp.sqrt(z.shape[-1])) ** 2, axis=-1)
    e_antithesis = ant_net + norm_penalty

    e_total = e_thesis + e_antithesis
    return e_total, e_thesis, e_antithesis


@partial(jax.jit, static_argnames=("num_steps",))
def optimize_latent_trajectory(
    params: PyTree,
    condition_c: jnp.ndarray,
    rng_key: jax.Array,
    num_steps: int = 12,
    step_lr: float = 0.12,
) -> LatentTrajectoryResult:
    """
    Perform continuous latent reasoning as an energy-minimizing trajectory:
        z_0 -> z_1 -> ... -> z* = argmin_z [E_thesis(z; c) + E_antithesis(z; c)]
    """
    z_hat = predict_jepa_target(params, condition_c)
    noise = jax.random.normal(rng_key, shape=condition_c.shape, dtype=jnp.float32) * 0.65
    z_0 = z_hat + noise

    def energy_scalar(z_vec: jnp.ndarray) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        e_tot, e_th, e_ant = compute_dialectical_energy(params, z_vec, condition_c)
        return e_tot, (e_th, e_ant)

    grad_fn = jax.value_and_grad(energy_scalar, has_aux=True)

    def scan_step(
        z_curr: jnp.ndarray, _: None
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        (e_tot, (e_th, e_ant)), g = grad_fn(z_curr)
        # Normalized gradient step for smooth, stable descent
        g_norm = jnp.linalg.norm(g) + 1e-6
        clipped_g = jnp.where(g_norm > 2.0, g * (2.0 / g_norm), g)
        z_next = z_curr - step_lr * clipped_g
        return z_next, (z_curr, e_tot, e_th, e_ant)

    z_final, (traj_states, e_totals, e_ths, e_ants) = jax.lax.scan(
        scan_step, z_0, None, length=num_steps
    )
    e_final, e_th_final, e_ant_final = compute_dialectical_energy(params, z_final, condition_c)

    full_traj = jnp.concatenate([traj_states, z_final[None, :]], axis=0)
    full_e_tot = jnp.concatenate([e_totals, e_final[None]], axis=0)
    full_e_th = jnp.concatenate([e_ths, e_th_final[None]], axis=0)
    full_e_ant = jnp.concatenate([e_ants, e_ant_final[None]], axis=0)

    return LatentTrajectoryResult(
        condition_c=condition_c,
        z_init=z_0,
        z_star=z_final,
        trajectory=full_traj,
        energy_curve=full_e_tot,
        thesis_energy_curve=full_e_th,
        antithesis_energy_curve=full_e_ant,
    )


@partial(jax.jit, static_argnames=("max_steps",))
def decode_latent_to_logits(
    params: PyTree,
    z: jnp.ndarray,
    grammar_masks: Dict[str, jnp.ndarray],
    max_steps: int = 8,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Project continuous latent vector `z` (`[..., D]`) into grammar-masked instruction
    logits for `L = max_steps`:
      `(op_logits, dest_logits, s1_logits, s2_logits, cond_logits)`
    Invalid grammar transitions are masked to `-1e9`, making syntax errors impossible.
    """
    dec = params["decoder"]
    h = jax.nn.silu(jnp.matmul(z, dec["w1"]) + dec["b1"])
    raw = jnp.matmul(h, dec["w2"]) + dec["b2"]

    out_per_step = NUM_OPCODES + 4 * NUM_REGS
    reshaped = raw.reshape(*z.shape[:-1], max_steps, out_per_step)

    i0 = NUM_OPCODES
    i1 = i0 + NUM_REGS
    i2 = i1 + NUM_REGS
    i3 = i2 + NUM_REGS

    op_logits = reshaped[..., :i0]
    dest_logits = reshaped[..., i0:i1]
    s1_logits = reshaped[..., i1:i2]
    s2_logits = reshaped[..., i2:i3]
    cond_logits = reshaped[..., i3:]

    neg_inf = jnp.float32(-1e9)
    op_logits = jnp.where(grammar_masks["opcode_mask"], op_logits, neg_inf)
    dest_logits = jnp.where(grammar_masks["dest_mask"], dest_logits, neg_inf)
    s1_logits = jnp.where(grammar_masks["src_mask"], s1_logits, neg_inf)
    s2_logits = jnp.where(grammar_masks["src_mask"], s2_logits, neg_inf)
    cond_logits = jnp.where(grammar_masks["src_mask"], cond_logits, neg_inf)

    return op_logits, dest_logits, s1_logits, s2_logits, cond_logits


@partial(jax.jit, static_argnames=("batch_size", "max_steps"))
def sample_programs_from_latent(
    params: PyTree,
    z_star: jnp.ndarray,
    grammar_masks: Dict[str, jnp.ndarray],
    rng_key: jax.Array,
    batch_size: int = 512,
    max_steps: int = 8,
    temperature: float = 0.85,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sample a batch of `B` grammar-valid `[B, L, 4]` candidate programs around the
    converged equilibrium latent vector `z*`.
    - Sample 0 is the greedy deterministic argmax decoding of `z*`.
    - Samples 1..B-1 are temperature-scaled Gumbel samples from a local latent cloud around `z*`.
    Returns `(programs [B, L, 4] uint16, z_samples [B, D] float32)`.
    """
    k_cloud, k_op, k_dest, k_s1, k_s2, k_cond = jax.random.split(rng_key, 6)
    dim = z_star.shape[-1]

    # Construct latent cloud around z* (first item has 0 noise = exact z*)
    noise = jax.random.normal(k_cloud, (batch_size, dim), dtype=jnp.float32) * 0.25
    noise = noise.at[0].set(0.0)
    z_cloud = z_star[None, :] + noise

    op_l, dest_l, s1_l, s2_l, cond_l = decode_latent_to_logits(
        params, z_cloud, grammar_masks, max_steps=max_steps
    )

    def sample_cat(logits: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
        greedy = jnp.argmax(logits[0:1], axis=-1)
        sampled = jax.random.categorical(key, logits[1:] / jnp.maximum(temperature, 1e-3), axis=-1)
        return jnp.concatenate([greedy, sampled], axis=0)

    dense_ops = sample_cat(op_l, k_op)
    dests = sample_cat(dest_l, k_dest).astype(jnp.uint16)
    s1s = sample_cat(s1_l, k_s1).astype(jnp.uint16)
    s2s_base = sample_cat(s2_l, k_s2).astype(jnp.uint16)
    conds = sample_cat(cond_l, k_cond).astype(jnp.uint16)

    hex_ops = DENSE_TO_HEX_LUT[dense_ops]
    # Pack ternary condition register into upper nibble when opcode is SELECT (dense_op == 15)
    packed_s2 = jnp.where(
        dense_ops == 15,
        (((conds & jnp.uint16(0x0F)) + jnp.uint16(1)) << jnp.uint16(4)) | (s2s_base & jnp.uint16(0x0F)),
        s2s_base & jnp.uint16(0x0F),
    )

    programs = jnp.stack([hex_ops, dests, s1s, packed_s2], axis=-1).astype(jnp.uint16)
    return programs, z_cloud


def extract_spec_features(
    test_inputs: np.ndarray,
    test_targets: np.ndarray,
    num_inputs: int = 1,
    output_bits: int = 64,
    feature_dim: int = 68,
) -> jnp.ndarray:
    """
    Encode an input-output relation into a fixed-width continuous feature vector `[F]`
    without using natural language tokens:
    - First 32 dims: normalized LSB parity/bit pattern across the first 32 canonical test cases
    - Next 32 dims: normalized log-magnitude / high-bit correlation signature
    - Final 4 dims: structural metadata (num_inputs, output_bits, monotonicity, bitwise flag)
    """
    t_in = np.asarray(test_inputs, dtype=np.uint64)
    if t_in.ndim == 1:
        t_in = t_in[:, None]
    t_out = np.asarray(test_targets, dtype=np.uint64).reshape(-1)

    feats = np.zeros((feature_dim,), dtype=np.float32)
    n = min(len(t_out), 32)
    # Bit 0 signature in [-1, 1]
    feats[:n] = ((t_out[:n] & np.uint64(1)).astype(np.float32) * 2.0) - 1.0
    # Bit 1..7 / magnitude signature
    feats[32 : 32 + n] = np.tanh(np.log1p(t_out[:n].astype(np.float64)) / 10.0).astype(np.float32)

    feats[64] = float(num_inputs) / 4.0
    feats[65] = float(output_bits) / 64.0
    feats[66] = float(np.mean((t_out[:n] ^ t_in[:n, 0]) == 0))
    feats[67] = float(np.std(feats[:n]))
    return jnp.asarray(feats, dtype=jnp.float32)


@partial(jax.jit, static_argnames=("max_steps", "sigreg_projections"))
def train_latent_core_step(
    params: PyTree,
    spec_features_batch: jnp.ndarray,  # [N, F]
    elite_programs_batch: jnp.ndarray, # [N, L, 4] uint16
    defect_rates_batch: jnp.ndarray,   # [N] float32 in [0, 1]
    grammar_masks: Dict[str, jnp.ndarray],
    rng_key: jax.Array,
    lr: float = 0.015,
    sigreg_weight: float = 0.1,
    max_steps: int = 8,
    sigreg_projections: int = 64,
) -> Tuple[PyTree, Dict[str, jnp.ndarray]]:
    """
    Execute one self-supervised training step updating the Tier 2 Latent Core
    and Tier 3 Projection Head using:
        L_total = L_task (Decoder CE + Dialectical Energy Clamping) + lambda * R_SIGReg(Z)
    """
    from llre.isa import HEX_TO_DENSE_LUT

    k_sig, k_neg = jax.random.split(rng_key)

    def loss_fn(p: PyTree) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        # 1. Encode batch conditions c and predicted equilibrium states z_hat: [N, D]
        c_batch = encode_condition(p, spec_features_batch)
        z_hat = predict_jepa_target(p, c_batch)

        # 2. SIGReg Regularization on latent manifold Z (prevents collapse, enforces N(0, I))
        r_sigreg = sigreg_loss(z_hat, k_sig, num_projections=sigreg_projections)

        # 3. Dialectical Clamping Energy Loss (Thesis + Antithesis equilibrium)
        e_pos, e_th_pos, e_ant_pos = compute_dialectical_energy(p, z_hat, c_batch)
        z_corrupt = z_hat + jax.random.normal(k_neg, z_hat.shape, dtype=jnp.float32) * 1.25
        e_neg, _, e_ant_neg = compute_dialectical_energy(p, z_corrupt, c_batch)

        # Clamp equilibrium energy to empirical defect density, and force higher energy on off-manifold states
        energy_loss = jnp.mean((e_th_pos - defect_rates_batch) ** 2) + jnp.mean(
            jax.nn.softplus(1.0 + e_pos - e_neg)
        )

        # 4. Grammar-Constrained Decoder Cross-Entropy on verified/elite circuits
        op_l, dest_l, s1_l, s2_l, cond_l = decode_latent_to_logits(
            p, z_hat, grammar_masks, max_steps=max_steps
        )
        target_ops = HEX_TO_DENSE_LUT[elite_programs_batch[:, :, 0].astype(jnp.int32) & 0x3F]
        target_dest = (elite_programs_batch[:, :, 1] & jnp.uint16(0x0F)).astype(jnp.int32)
        target_s1 = (elite_programs_batch[:, :, 2] & jnp.uint16(0x0F)).astype(jnp.int32)
        s2_raw = elite_programs_batch[:, :, 3].astype(jnp.int32)
        target_s2 = s2_raw & 0x0F
        target_cond = jnp.maximum(((s2_raw >> 4) & 0x0F) - 1, 0)

        def ce(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            gathered = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)
            return -jnp.mean(gathered)

        decoder_loss = (
            ce(op_l, target_ops)
            + ce(dest_l, target_dest)
            + ce(s1_l, target_s1)
            + ce(s2_l, target_s2)
            + 0.5 * ce(cond_l, target_cond)
        )

        total_loss = decoder_loss + 0.5 * energy_loss + sigreg_weight * r_sigreg
        eff_dim = compute_effective_dimensionality(z_hat)

        metrics = {
            "total_loss": total_loss,
            "task_loss": decoder_loss + 0.5 * energy_loss,
            "sigreg_loss": r_sigreg,
            "effective_dim": eff_dim,
            "equilibrium_energy": jnp.mean(e_pos),
        }
        return total_loss, metrics

    (loss_val, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    # Pure-JAX normalized gradient update with weight decay
    new_params = jax.tree_util.tree_map(
        lambda w, g: w - lr * (jnp.clip(g, -2.0, 2.0) + 1e-4 * w),
        params,
        grads,
    )
    return new_params, metrics

