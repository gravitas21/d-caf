#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D-CAF factory: minimal distance-based star generator (with fast CDF sampler)

Keep it **clear and tiny**:
- Default distance PDF: lognormal (mu=-2.15, sigma=0.9 in ln r)
  for ultra-fast, reusable sampling.
- Uniform random reference star (no density weighting)
- Place the new star at that distance in a random 3D direction (no box logic)
- Set velocity from nearest neighbors: per-axis MAD dispersion, median removed
- Enforce a **minimum separation** using a KD-tree (no close binaries)

Inputs: existing AMUSE Particles, number of new stars
Output: new AMUSE Particles (positions & velocities set; masses untouched)

Assumes AMUSE and SciPy are installed.
"""

import numpy as np
from scipy.spatial import cKDTree
from amuse.datamodel import Particles
from amuse.units import units
from dcaf.utilities.sampler import PDFSampler, lognormal_pdf
from dcaf.utilities.helpers import ( robust_stats, sample_sphere_surface )


def generate_stars(existing,
                   n_new,
                   box_size = None,
                   pdf_func=lognormal_pdf,
                   pdf_unit = units.parsec,
                   min_separation=0.01 | units.parsec,
                   max_separation = 100 |units.parsec,
                   neighbor_k_vel=20,
                   seed=42):
    """
    Generate stars based on existing set sampling distances to old stars from a
    pdf_func.

    pdf_func means to represent the star formation simulation we want to mimic.
    It could also draw from a tabulated pdf (TODO)

    Method: 
    1) Convert AMUSE Particles to plain NumPy arrays (positions/velocities)
    2) Build a PDFSampler once over [r_min, r_max] for the requested PDF
    3) Keep an evolving catalogue (cat_pos, cat_vel) and KD-tree for min-sep
    4) For each new star: sample distance r, pick a uniform reference, place in
       a random direction, enforce min-sep, set velocity from neighbors 
    5) Return the new AMUSE Particles
    """
    if len(existing) == 0:
        raise ValueError("existing Particles is empty")

    if n_new == 0 :
        return Particles()

    # construct arrays without units for now
    if pdf_unit is None:
        L = existing.x.unit
    else:
        L = pdf_unit
    S = existing.vx.unit
    #Original positions
    pos0 = np.column_stack([
        existing.x.value_in(L),
        existing.y.value_in(L),
        existing.z.value_in(L),
    ])
    vel0 = np.column_stack([
        existing.vx.value_in(S),
        existing.vy.value_in(S),
        existing.vz.value_in(S),
    ])
    # obtain PDF domain
    r_min = float(min_separation.value_in(L))

    if box_size is not None:
        r_max = np.sqrt(3)*box_size.value_in(L)
    else:
        r_max = float(max_separation.value_in(L))

    if not np.isfinite(r_max) or r_max <= r_min:
        raise ValueError('generate stars: rmin (%s) and rmax (%s) are not valid'%(r_min,r_max) )

    if pdf_func is None:
        pdf_func = lognormal_pdf

    sampler = PDFSampler(pdf_func, (r_min, r_max) , nsample=4096)

    # Keep existing stars updated
    gen = np.random.default_rng(seed)
    newP, newV = [], []
    cat_pos = pos0.copy()
    cat_vel = vel0.copy()
    tree = cKDTree(cat_pos)

    # Adding the stars
    for _ in range(n_new):
        placed = False
        while not placed :
            rtarget = float(sampler.sample(1, rng=gen)[0])
            #is the following line needed? just in case
            rtarget = max(rtarget, r_min)
            # remake the pool for each star. We allow new stars formed around the
            # same reference star.
            ref_pool = list(range(len(pos0)))
            while len(ref_pool) > 0 : 
                # --- choose a uniform reference among ORIGINAL stars ---
                ref = gen.choice(ref_pool)
                center = pos0[ref]

                # Find nearest neighbours on the surface of radius rtarget around the
                # reference star
                K = 256  
                C = sample_sphere_surface(center=center, r=rtarget, n=K, rng=gen) 
                dmin, _ = tree.query(C, k=1)

                # choose a rtarget within tolerance, no need is the same
                # reference star
                # defined by the minimum allowed separation
                valid = (dmin > r_min) & ( np.abs(dmin - rtarget) <= 0.5*r_min )
                if np.any(valid):

                    # pick one valid candidate at random
                    pick = gen.choice(np.flatnonzero(valid))
                    cand = C[pick]

                    # velocity from neighbors (median removed; per-axis MAD dispersion)
                    kk  = min(max(2, neighbor_k_vel), len(cat_pos))
                    nbr = tree.query(cand[None, :], k=kk)[1]
                    nbr = np.atleast_1d(nbr).reshape(-1)
                    _, v_sig = robust_stats(cat_vel[nbr])
                    v = gen.normal(0.0, 1.0, 3) * v_sig

                    # accept and update catalogue + tree
                    newP.append(cand); newV.append(v)
                    cat_pos = np.vstack([cat_pos, cand[None, :]])
                    cat_vel = np.vstack([cat_vel, v[None, :]])
                    tree = cKDTree(cat_pos)
                    placed = True
                    break
                else:
                    ref_pool.remove(ref)
                    continue

    new_parts = Particles(len(newP))
    if len(newP) > 0:
        newP = np.asarray(newP, float); newV = np.asarray(newV, float)
        new_parts.x = newP[:, 0] | L; new_parts.y = newP[:, 1] | L; new_parts.z = newP[:, 2] | L
        new_parts.vx = newV[:, 0] | S; new_parts.vy = newV[:, 1] | S; new_parts.vz = newV[:, 2] | S
    return new_parts

# tiny self-check
if __name__ == "__main__":
    N = 64
    p = Particles(N)
    p.x = np.random.uniform(0, 1, N) | units.parsec
    p.y = np.random.uniform(0, 1, N) | units.parsec
    p.z = np.random.uniform(0, 1, N) | units.parsec
    p.vx = np.random.normal(0, 1, N) | units.kms
    p.vy = np.random.normal(0, 1, N) | units.kms
    p.vz = np.random.normal(0, 1, N) | units.kms

    newp = generate_stars(p, 32, min_separation=0.02 | units.parsec)
    print("generated", len(newp))

