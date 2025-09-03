"""
Some reusable utility functions
"""
import numpy as np
from scipy.spatial import cKDTree

def random_unit_vector(n_sample=1, gen=None, seed=42):
    """
    Generate random unit vectors on the unit sphere

    Parameters
    ----------
    n_sample : int, default=0
        Number of vectors to generate.
        If 0 → behaves like 1.
        Always returns shape (n, 3).
    gen : numpy Generator, optional
        Random generator. If None, a new one is created with `seed`.
    seed : int, default=42
        Seed used if `gen` is None.
    """
    if gen is None:
        gen = np.random.default_rng(seed)

    v = gen.normal(size=(n_sample, 3))
    norms = np.linalg.norm(v, axis=1)[:, None]
    return v / norms


def robust_stats(X):
    """Component-wise (median, dispersion) with MAD*1.4826; fallback to std if needed."""
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    sig = 1.4826 * mad
    zero = sig <= 0
    if np.any(zero):
        sig = np.where(zero, np.std(X, axis=0, ddof=1) + 1e-8, sig)
    return med, sig

def sample_sphere_surface(center, r, n, rng=None):
    """
    Return n points uniformly distributed on the surface of a sphere of radius r, centered at `center`.

    Parameters
    ----------
    center : array-like, shape (3,)
        Sphere center (x, y, z).
    r : float
        Sphere radius.
    n : int
        Number of surface points to sample.
    rng : np.random.Generator or int, optional
        Random generator or seed

    Returns
    -------
    P : ndarray, shape (n, 3)
        Sampled surface points.
    """
    gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    # Sample angles for uniform sphere:
    #    cos(theta) ~ U[-1,1]  and  phi ~ U[0, 2π)
    u   = gen.uniform(-1.0, 1.0, size=n)          # cos(theta)
    phi = gen.uniform(0.0, 2.0*np.pi, size=n)     # azimuth

    # Convert (u=cosθ, φ) to unit vectors (x,y,z)
    #    sinθ = sqrt(1 - cos^2θ) = sqrt(1 - u^2)
    s = np.sqrt(np.maximum(0.0, 1.0 - u*u))
    x = s * np.cos(phi)
    y = s * np.sin(phi)
    z = u

    # Scale by radius and translate by center
    P = np.column_stack((x, y, z)) * float(r)
    P += np.asarray(center, dtype=float)

    return P


def weights_by_density(pos, k=5, beta=1.0, tree = None):
    """
    Compute weighted probabilities based on local density
    beta = 0   -> uniform
    beta = 1   -> inverse density
    0<beta<1   -> flattened
    beta>1     -> higher weights for isolated particles
    """
    if tree is None:
        tree = cKDTree(pos)
    d, _ = tree.query(pos, k=k+1)
    d_k = d[:, -1]
    rho = k / (4/3 * np.pi * d_k**3)
    w = 1.0 / (rho + 1e-12)
    w = w ** beta
    return w / w.sum()
