"""
Tier 2 Regularization: Sketched Isotropic Gaussian Regularization (SIGReg).

Implements Balestriero & LeCun's O(N * D) characteristic-function sketching regularizer
to prevent representation collapse in Joint-Embedding Predictive Architectures (JEPA)
without contrastive negative pairs or O(D^3) covariance matrix inversions.
Also provides the Effective Dimensionality (RankMe spectral entropy) observability metric.
"""

from __future__ import annotations

from functools import partial
from typing import Dict, Tuple

import jax
import jax.numpy as jnp

# Quadrature evaluation frequencies t_k in (0, 3.0] for Empirical Characteristic Function testing
DEFAULT_ECF_FREQUENCIES = jnp.linspace(0.25, 2.75, 11, dtype=jnp.float32)


@partial(jax.jit, static_argnames=("num_projections",))
def sigreg_loss(
    z: jnp.ndarray,
    rng_key: jax.Array,
    num_projections: int = 64,
) -> jnp.ndarray:
    """
    Compute Sketched Isotropic Gaussian Regularization loss R_SIGReg(Z).

    Args:
        z: Latent tensor of shape `[N, D]` (float32).
        rng_key: JAX PRNGKey for sampling 1D random projection directions.
        num_projections: Number of random 1D slices `M` (complexity O(N * D * M)).

    Returns:
        Scalar float32 SIGReg loss measuring departure from isotropic N(0, I).
    """
    if z.ndim == 1:
        z = z[None, :]
    n_samples, dim = z.shape

    # Sample M random unit vectors on the hypersphere S^{D-1}: [D, M]
    raw_dirs = jax.random.normal(rng_key, shape=(dim, num_projections), dtype=jnp.float32)
    norms = jnp.linalg.norm(raw_dirs, axis=0, keepdims=True) + 1e-8
    unit_dirs = raw_dirs / norms

    # 1D random projections: P = Z @ U -> [N, M]
    proj = jnp.matmul(z.astype(jnp.float32), unit_dirs)

    # Evaluate Empirical Characteristic Function (ECF) at frequencies t_k: [N, M, K]
    t = DEFAULT_ECF_FREQUENCIES[None, None, :]  # [1, 1, K]
    angles = proj[:, :, None] * t               # [N, M, K]

    ecf_real = jnp.mean(jnp.cos(angles), axis=0)  # [M, K]
    ecf_imag = jnp.mean(jnp.sin(angles), axis=0)  # [M, K]

    # Target standard Gaussian characteristic function: phi_0(t) = exp(-0.5 * t^2)
    target_real = jnp.exp(-0.5 * (DEFAULT_ECF_FREQUENCIES ** 2))[None, :]  # [1, K]
    weights = target_real  # Epps-Pulley Gaussian weighting w(t) = exp(-t^2 / 2)

    cf_error = weights * ((ecf_real - target_real) ** 2 + (ecf_imag ** 2))
    ecf_loss = jnp.mean(cf_error)

    # Sketched 1D mean & variance anchoring for fast convergence on small/medium batches
    proj_mean = jnp.mean(proj, axis=0)
    proj_var = jnp.mean(proj ** 2, axis=0)
    moment_loss = jnp.mean(proj_mean ** 2) + jnp.mean((proj_var - 1.0) ** 2)

    return ecf_loss + 0.25 * moment_loss


@jax.jit
def compute_effective_dimensionality(z: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the Effective Dimensionality (RankMe spectral entropy) of latent batch `Z`:
        D_eff = exp(- sum_i p_i log(p_i))
    where `p_i = sigma_i / sum_j sigma_j` are the normalized singular values of centered `Z`.
    Validates SIGReg health (Section 7): D_eff -> 1 indicates collapse; D_eff -> min(N, D)
    indicates healthy isotropic utilization of the latent manifold.
    """
    z_f32 = z.astype(jnp.float32)
    if z_f32.ndim == 1 or z_f32.shape[0] <= 1:
        return jnp.float32(1.0)
    centered = z_f32 - jnp.mean(z_f32, axis=0, keepdims=True)
    singular_vals = jnp.linalg.svd(centered, full_matrices=False, compute_uv=False)
    s_pos = jnp.maximum(singular_vals, 1e-12)
    probs = s_pos / jnp.sum(s_pos)
    entropy = -jnp.sum(probs * jnp.log(probs + 1e-12))
    return jnp.exp(entropy)


def diagnose_latent_manifold(z: jnp.ndarray, rng_key: jax.Array) -> Dict[str, float]:
    """Return a diagnostic dictionary of latent manifold health metrics."""
    loss_val = float(sigreg_loss(z, rng_key))
    d_eff = float(compute_effective_dimensionality(z))
    z_f32 = z.astype(jnp.float32)
    mean_norm = float(jnp.mean(jnp.linalg.norm(z_f32, axis=-1)))
    std_per_dim = float(jnp.mean(jnp.std(z_f32, axis=0)))
    return {
        "sigreg_loss": loss_val,
        "effective_dimensionality": d_eff,
        "mean_vector_norm": mean_norm,
        "mean_dim_std": std_per_dim,
    }
